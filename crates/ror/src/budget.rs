//! Loads within the memory the machine has free. Parsing a model or place and extracting its scene takes
//! about 80 times the file's size at its peak (a 19.7 MB stitched world: +1.6 GB commit), and the service
//! starts every job of a batch at once: a hundred jobs on such a file parsed it a hundred times before any
//! of them showed in the machine's free memory, and filled a 64 GB machine (2026-10-17). A load now
//! reserves [`LOAD_FACTOR`] times its input's size while it runs, and starts only when the machine's
//! available memory covers what loads in flight have reserved, its own reservation and a floor left for
//! everything else ([`floor`]); a load with no other in flight always starts, so work goes on whatever the
//! machine holds. Loads of small inputs (a unit, an effect) reserve [`MIN_LOAD`] and never wait while
//! memory is plentiful.
//!
//! Jobs too ([`admit`]): a clip holds its memory for its whole length, not just its parse (a 1572 x 862
//! cutscene 2.5-4.5 GB of commit, a 4K one up to 8.5; GPU memory commits as much system memory), and
//! four cutscenes sent to one service at once took it to 10.4 GB. The first job of a process starts once
//! the machine has its floor free ([`crate::wait_for_memory`]); every other starts when the machine's
//! free memory covers the floor, the loads in flight, what the running jobs have yet to grow into and
//! its own need, and the process stays under 85 % of its cap ([`crate::memory_cap`]) with it. Short, it
//! first has the caches let go of what they keep (the caller's `relieve`), then queues until a job ends.
//! A job's need ([`Need`]) is what the same job took the last time it ran alone in this process, else a
//! guess from its pixels and its input files' bytes ([`clip_need`], [`still_need`]); the machine's free
//! memory pays for what it makes resident, the process's cap counts its commit.
//!
//! Draws within the GPU memory free ([`GpuGate`]): a draw (a still's group of views, a clip's frame)
//! starts once the GPU has free what its targets add, beside what the draws starting before it may still
//! take and a headroom, instead of the four draws at once the service allowed whatever the memory.
//!
//! Beside a game ([`crate::gaming`]) the floor counts the game's working set as taken, and every draw,
//! one alone too, waits for the GPU's memory to have the headroom left: past full, Windows makes room by
//! moving the game's own textures out, and every frame that samples one waits for it.

use parking_lot::{Condvar, Mutex};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

/// A load's reservation per byte of input: the measured parse-and-extract peak (~80x) with room.
pub const LOAD_FACTOR: u64 = 100;
/// The least a load reserves.
pub const MIN_LOAD: u64 = 32 << 20;
/// How often a waiting load looks at the machine's free memory again (others' processes free it too).
const RECHECK: Duration = Duration::from_millis(250);

struct State {
	reserved: u64,
	loads: usize,
}

static STATE: Mutex<State> = parking_lot::const_mutex(State { reserved: 0, loads: 0 });
static FREED: Condvar = Condvar::new();

thread_local! {
	/// Whether this thread holds a reservation: a load inside a load (a parse inside a scene's load) adds none.
	static HOLDING: Cell<bool> = const { Cell::new(false) };
}

/// Memory kept free for everything else: the floor a job waits for before it starts ([`crate::memory_floor`],
/// `ROR_MEMORY_FLOOR_GB` as the job's caller sets it, [`crate::setting`]; 0 when that is off). While a game
/// runs ([`crate::gaming`]), the default floor less the game's working set, and at least [`GAMING_FLOOR`]:
/// the floor is room for programs to start and grow, and the game took its share when it loaded. With the
/// game proxy's 10 GB held, 13.6-16.2 GB free and the floor of 7.4 GB, videos and sims asked for 4.2 GB
/// each beside two running and gave up after 60 s: 28 and 31 of two heavy mixes' 78 jobs (2026-11-02).
pub fn floor() -> u64 {
	let Some(total) = crate::total_memory() else {
		return 0;
	};
	let setting = crate::setting("ROR_MEMORY_FLOOR_GB");
	let Some(floor) = crate::memory_floor(total, setting.as_deref()) else {
		return 0;
	};
	match crate::gaming::now().and_then(|gaming| gaming.working_set) {
		Some(game) if setting.is_none() => floor.saturating_sub(game).max(GAMING_FLOOR.min(floor)),
		_ => floor,
	}
}

/// The least floor beside a game: its next level's loading and the file cache it streams through.
pub const GAMING_FLOOR: u64 = 4 << 30;

/// A reservation, given back when dropped, on the thread that took it.
pub struct LoadGuard {
	bytes: u64,
	_thread: std::marker::PhantomData<*const ()>,
}

impl Drop for LoadGuard {
	fn drop(&mut self) {
		if self.bytes == 0 {
			return;
		}
		HOLDING.with(|holding| holding.set(false));
		let mut state = STATE.lock();
		state.reserved = state.reserved.saturating_sub(self.bytes);
		state.loads = state.loads.saturating_sub(1);
		drop(state);
		FREED.notify_all();
	}
}

/// What a load of `input_bytes` reserves.
pub fn estimate(input_bytes: u64) -> u64 {
	input_bytes.saturating_mul(LOAD_FACTOR).max(MIN_LOAD)
}

/// Waits until a load of `input_bytes` fits (or none other is in flight) and reserves it; on a thread already
/// holding one, reserves nothing. `ROR_PROFILE` reports waits.
pub fn reserve(input_bytes: u64) -> LoadGuard {
	if HOLDING.with(Cell::get) {
		return LoadGuard { bytes: 0, _thread: std::marker::PhantomData };
	}
	let need = estimate(input_bytes);
	let floor = floor();
	let started = Instant::now();
	let mut state = STATE.lock();
	while state.loads > 0 {
		match crate::available_memory() {
			Some(free) if free >= state.reserved.saturating_add(need).saturating_add(floor) => break,
			None => break,
			Some(_) => {
				FREED.wait_for(&mut state, RECHECK);
			}
		}
	}
	state.reserved += need;
	state.loads += 1;
	drop(state);
	HOLDING.with(|holding| holding.set(true));
	let waited = started.elapsed();
	if waited > Duration::from_millis(50) && std::env::var_os("ROR_PROFILE").is_some() {
		eprintln!("profile: a {} MB load waited {:.2} s for memory", input_bytes >> 20, waited.as_secs_f64());
	}
	LoadGuard { bytes: need, _thread: std::marker::PhantomData }
}

/// Steps this thread's load aside while it waits for another load (a file a job before it is loading):
/// it stops counting as in flight, so the load it waits for can always start, then counts again.
pub fn park() -> Parked {
	if !HOLDING.with(Cell::get) {
		return Parked { active: false, _thread: std::marker::PhantomData };
	}
	STATE.lock().loads -= 1;
	FREED.notify_all();
	Parked { active: true, _thread: std::marker::PhantomData }
}

pub struct Parked {
	active: bool,
	_thread: std::marker::PhantomData<*const ()>,
}

impl Drop for Parked {
	fn drop(&mut self) {
		if self.active {
			STATE.lock().loads += 1;
		}
	}
}

/// Loads in flight and what they hold reserved (the service's health).
pub fn in_flight() -> (usize, u64) {
	let state = STATE.lock();
	(state.loads, state.reserved)
}

/// A job's memory, or a process's: what is resident in RAM (what the machine's free memory pays for) and
/// what is committed (what the process's cap counts; GPU memory commits as much system memory and takes
/// none of it while it stays on the GPU).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Need {
	pub resident: u64,
	pub commit: u64,
}

impl Need {
	fn now() -> Self {
		Self { resident: crate::process_resident().unwrap_or(0), commit: crate::process_commit().unwrap_or(0) }
	}

	fn at_least(self, least: u64) -> Self {
		Self { resident: self.resident.max(least), commit: self.commit.max(least) }
	}

	fn add(self, other: Self) -> Self {
		Self { resident: self.resident.saturating_add(other.resident), commit: self.commit.saturating_add(other.commit) }
	}

	fn less(self, other: Self) -> Self {
		Self { resident: self.resident.saturating_sub(other.resident), commit: self.commit.saturating_sub(other.commit) }
	}

	fn most(self, other: Self) -> Self {
		Self { resident: self.resident.max(other.resident), commit: self.commit.max(other.commit) }
	}
}

/// The jobs running in this process ([`admit`]).
struct Jobs {
	running: usize,
	/// What they were admitted for
	needs: Need,
	/// This process's memory when the first of them started, and the most since (sampled, [`watch`])
	baseline: Need,
	peak: Need,
	/// Counts jobs started beside another: a job that sees it unchanged from its start to its end ran alone
	overlaps: u64,
	/// Counts the baselines set (a job starting with none running): a sample read under an earlier one is not theirs
	baselines: u64,
	queued: usize,
}

const NONE: Need = Need { resident: 0, commit: 0 };
static JOBS: Mutex<Jobs> = parking_lot::const_mutex(Jobs { running: 0, needs: NONE, baseline: NONE, peak: NONE, overlaps: 0, baselines: 0, queued: 0 });
static JOB_ENDED: Condvar = Condvar::new();
/// What each job (by its caller's key) took the last time it ran alone: its peak over the process's
/// before it
static LEARNED: LazyLock<Mutex<HashMap<String, Need>>> = LazyLock::new(Default::default);

/// The least a job is admitted for, and the margin over what it took when it ran alone.
const MIN_JOB: u64 = 64 << 20;
const LEARNED_MARGIN_PERCENT: u64 = 115;
/// How often the running jobs' memory is sampled for their peak.
const SAMPLE: Duration = Duration::from_millis(100);

/// A guess at a clip's need from its pixels (width x height x supersample²) and its input files' bytes
/// (a place's scene stays in memory to the clip's end; a bundle's assets come by id, its file is KBs).
/// Measured alone through a service (2026-11-02, the commit its GPU memory included): a 1440p bundle
/// clip 178-319 MB resident and 411-1069 MB committed, a sim's 1440p clip 230-258 and 692-874; a
/// cutscene (1572 x 862) commits 2.5-4.5 GB, a 4K one 4.2-8.5. The guess, 1440p, a small file: 0.6
/// and 2.1 GB. It was 1.5 GB and 800 bytes a pixel of commit, asked of the machine's free memory: 4.2
/// GB for every 1440p clip.
pub fn clip_need(pixels: u64, input_bytes: u64) -> Need {
	Need {
		resident: (384u64 << 20).saturating_add(pixels.saturating_mul(64)).saturating_add(input_bytes.saturating_mul(SCENE_RESIDENT_FACTOR)),
		commit: (1u64 << 30).saturating_add(pixels.saturating_mul(300)).saturating_add(input_bytes.saturating_mul(SCENE_COMMIT_FACTOR)),
	}
}

/// A guess at a still job's need from its views' pixels and its input files' bytes. Measured as
/// [`clip_need`]'s: a model's four 1024² stills 94-112 MB resident, 297-325 MB committed; a map's
/// 32 views at 1080p peaked at 1.7 GB of commit (views draw eight at a time). The guess for the model:
/// 0.19 and 0.38 GB (it was 1.48 GB of commit).
pub fn still_need(pixels: u64, input_bytes: u64) -> Need {
	Need {
		resident: (96u64 << 20).saturating_add(pixels.saturating_mul(20)).saturating_add(input_bytes.saturating_mul(SCENE_RESIDENT_FACTOR)),
		commit: (192u64 << 20).saturating_add(pixels.saturating_mul(40)).saturating_add(input_bytes.saturating_mul(SCENE_COMMIT_FACTOR)),
	}
}

/// What a scene holds per byte of its file once loaded, resident and committed (its parse peaks at
/// [`LOAD_FACTOR`], reserved apart while it runs; a 19.7 MB stitched world committed 1.6 GB).
const SCENE_RESIDENT_FACTOR: u64 = 40;
const SCENE_COMMIT_FACTOR: u64 = 80;

/// The bytes of the files a job reads its scene from (0 for an asset id or a file not there).
pub fn file_bytes<P: AsRef<std::path::Path>>(paths: impl IntoIterator<Item = P>) -> u64 {
	paths.into_iter().filter_map(|path| std::fs::metadata(path).ok()).filter(|metadata| metadata.is_file()).map(|metadata| metadata.len()).sum()
}

/// A job's admission: given back when it ends, which lets queued jobs start.
pub struct JobGuard {
	need: Need,
	key: Option<String>,
	/// [`Jobs::overlaps`] at its start, `None` when another job was running then
	alone_from: Option<u64>,
	finished: bool,
}

impl JobGuard {
	/// The job ran to its end: what it took is learned (one that failed part way may have taken less).
	pub fn finished(&mut self) {
		self.finished = true;
	}
}

impl Drop for JobGuard {
	fn drop(&mut self) {
		let mut jobs = JOBS.lock();
		if let (true, Some(key), Some(overlaps)) = (self.finished, &self.key, self.alone_from)
			&& overlaps == jobs.overlaps
		{
			let took = jobs.peak.most(Need::now()).less(jobs.baseline);
			let learned = Need { resident: took.resident / 100 * LEARNED_MARGIN_PERCENT, commit: took.commit / 100 * LEARNED_MARGIN_PERCENT };
			if std::env::var_os("ROR_PROFILE").is_some() {
				eprintln!("profile: {key} took {} MB resident and {} MB of commit (admitted for {} and {})", took.resident >> 20, took.commit >> 20, self.need.resident >> 20, self.need.commit >> 20);
			}
			LEARNED.lock().insert(key.clone(), learned.at_least(MIN_JOB));
		}
		jobs.running = jobs.running.saturating_sub(1);
		jobs.needs = jobs.needs.less(self.need);
		drop(jobs);
		JOB_ENDED.notify_all();
	}
}

/// What the running jobs were admitted for and have not grown into yet: their needs less how far this
/// process's memory has risen (`now`) since the first of them started.
fn unrealized(jobs: &Jobs, now: Need) -> Need {
	jobs.needs.less(now.less(jobs.baseline))
}

/// Samples this process's memory while jobs run, for what a job alone took ([`JobGuard`]'s drop).
fn watch() {
	static STARTED: std::sync::Once = std::sync::Once::new();
	STARTED.call_once(|| {
		let _ = std::thread::Builder::new().name("ror-jobs-memory".into()).spawn(|| {
			loop {
				std::thread::sleep(SAMPLE);
				let baselines = {
					let jobs = JOBS.lock();
					if jobs.running == 0 {
						continue;
					}
					jobs.baselines
				};
				// read outside the lock: one read across a new baseline is the memory of the jobs before it
				let now = Need::now();
				let mut jobs = JOBS.lock();
				if jobs.running > 0 && jobs.baselines == baselines {
					jobs.peak = jobs.peak.most(now);
				}
			}
		});
	});
}

/// The highest this process may commit with a job's need added: 85 % of its cap ([`crate::memory_cap`],
/// `ROR_MAX_MEMORY_GB`), so the jobs queue before an allocation meets the cap and fails.
fn ceiling() -> Option<u64> {
	let cap = crate::memory_cap(crate::total_memory()?, std::env::var("ROR_MAX_MEMORY_GB").ok().as_deref())?;
	Some(cap / 100 * 85)
}

/// Waits until a job of `heuristic` bytes (or what the job under `key` took when it last ran alone) fits
/// beside the running ones and admits it; `relieve(bytes)` has the caches let go of up to that much and
/// returns what they freed. The first job of the process waits only for the machine's floor
/// ([`crate::wait_for_memory`]); one queued behind this process's own jobs waits for them. Either says
/// once why it waits ([`crate::say`]: what it needs, what is free, the floor and how to lower it) and
/// gives up with the same explanation [`crate::memory_wait`] after it came (60 s by default), or once its
/// caller has gone: the jobs that started 12-29 minutes after they came (2026-10-23) had lost their callers
/// long before.
pub fn admit(what: &str, key: Option<String>, heuristic: Need, relieve: &dyn Fn(u64) -> u64) -> anyhow::Result<JobGuard> {
	let need = key.as_ref().and_then(|key| LEARNED.lock().get(key).copied()).unwrap_or(heuristic).at_least(MIN_JOB);
	let ceiling = ceiling();
	let (started, limit) = (Instant::now(), crate::memory_wait());
	let (mut told, mut relieved_at) = (false, None::<Instant>);
	let mut jobs = JOBS.lock();
	loop {
		// read each round: a game starting or ending moves it (floor)
		let floor = floor();
		let now = Need::now();
		if jobs.running == 0 {
			if told {
				jobs.queued -= 1;
			}
			// holding the process's first place while it waits for the floor: the jobs after it queue
			jobs.running = 1;
			jobs.needs = need;
			(jobs.baseline, jobs.peak) = (now, now);
			jobs.baselines += 1;
			let guard = JobGuard { need, key, alone_from: Some(jobs.overlaps), finished: false };
			drop(jobs);
			watch();
			if told {
				crate::say(&format!("{what} starts after {:.1} s in the queue", started.elapsed().as_secs_f64()));
			}
			crate::wait_for_memory(what, need.resident, started)?;
			return Ok(guard);
		}
		let pending = unrealized(&jobs, now);
		let free = crate::available_memory();
		let short_machine = free.map_or(0, |free| floor.saturating_add(in_flight().1).saturating_add(pending.resident).saturating_add(need.resident).saturating_sub(free));
		let short_process = ceiling.map_or(0, |ceiling| now.commit.saturating_add(pending.commit).saturating_add(need.commit).saturating_sub(ceiling));
		let short = short_machine.max(short_process);
		if short == 0 {
			jobs.running += 1;
			jobs.needs = jobs.needs.add(need);
			jobs.overlaps += 1;
			if told {
				jobs.queued -= 1;
			}
			drop(jobs);
			if told {
				crate::say(&format!("{what} starts after {:.1} s in the queue", started.elapsed().as_secs_f64()));
			}
			return Ok(JobGuard { need, key, alone_from: None, finished: false });
		}
		// first the caches (again each few seconds while it waits: the running jobs keep adding to them)
		if relieved_at.is_none_or(|at| at.elapsed() > RELIEVE_AGAIN) {
			relieved_at = Some(Instant::now());
			drop(jobs);
			let freed = relieve(short);
			jobs = JOBS.lock();
			if freed > 0 {
				continue;
			}
		}
		let running = jobs.running;
		let why = || {
			let cap = match ceiling {
				Some(ceiling) if short_process > short_machine => format!(
					", and the process would pass {} (85 % of its cap, ROR_MAX_MEMORY_GB of the command that started it) with {} committed and {} more for this job",
					crate::gigabytes_text(ceiling),
					crate::gigabytes_text(now.commit),
					crate::gigabytes_text(need.commit)
				),
				_ => String::new(),
			};
			format!(
				"it needs {} and {} of that is not to be had: {} free, the floor of {} ror leaves to other programs (Studio, games, builds, other renders), {} the {} running job(s) of this process have yet to take{cap}",
				crate::gigabytes_text(need.resident),
				crate::gigabytes_text(short),
				free.map_or_else(|| "unknown".to_string(), crate::gigabytes_text),
				crate::gigabytes_text(floor),
				crate::gigabytes_text(pending.resident),
				running
			)
		};
		let waited = started.elapsed();
		if waited >= limit || crate::caller_gone() {
			let gave_up = if waited >= limit {
				format!("{what} gave up after waiting {:.0} s for memory: {}. {}", waited.as_secs_f64(), why(), crate::MEMORY_ADVICE)
			} else {
				format!("{what}: its caller went away while it waited for memory ({})", why())
			};
			if told {
				jobs.queued -= 1;
			}
			anyhow::bail!("{gave_up}");
		}
		if !told {
			told = true;
			jobs.queued += 1;
			let line = format!("{what} waits for memory: {}. It starts when they end or memory frees, and gives up after {:.0} s. {}", why(), limit.as_secs_f64(), crate::MEMORY_ADVICE);
			drop(jobs);
			crate::say(&line);
			jobs = JOBS.lock();
			continue;
		}
		JOB_ENDED.wait_for(&mut jobs, RECHECK);
	}
}

/// How long a queued job waits before it has the caches relieved again.
const RELIEVE_AGAIN: Duration = Duration::from_secs(3);

/// The targets a draw draws into for one size ([`crate::render::Renderer::target_sets`]): its size as
/// drawn and samples, their GPU memory, and whether a set of that size is kept from an earlier draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetSet {
	pub key: (u32, u32, u32),
	pub bytes: u64,
	pub kept: bool,
}

/// The GPU memory free for this process to draw into, and the GPU's own memory
/// ([`crate::render::gpu::Gpu::free_memory`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuFree {
	pub free: u64,
	pub total: u64,
}

/// The share of the GPU's memory left free beside the draws: what a draw takes past its targets' estimate
/// is not reserved. Measured (2026-11-02, a 16 GB GPU): a scene's first draw loads its
/// textures and meshes inside it (a large place's cut +1.38 GB, a map +0.5 GB, the place again after
/// it +0.35 GB) and a 4K view supersampled twice peaked 0.29 GB over what it kept; the
/// counters read show an allocation within ~15 ms. A tenth of 16 GB covers the largest; past it, the
/// renderer refuses meshes and textures anyway (`Renderer::gpu_room`, 90 % of its budget).
pub const GPU_HEADROOM_PERCENT: u64 = 10;
/// How often a draw short of GPU memory looks again (other programs free it too).
const GPU_RECHECK: Duration = Duration::from_millis(50);

/// Draws within the GPU memory free. A draw starts when the GPU has free (`free`: within this process's
/// budget and what the whole adapter has free, every program's use counted) what its targets add, beside
/// what the draws starting before it have reserved and [`GPU_HEADROOM_PERCENT`] of the GPU's memory. A set
/// kept from an earlier draw, or one a draw starting now makes, adds nothing: renders recording at once
/// draw into the same targets (`render::reuse`'s scratch targets), so eight 1080p frames take one set. A
/// draw with none other starting always starts, and so does one adding nothing; short, a draw first has
/// the renderer let go of what no render holds (`relieve`), then waits for a draw to end or memory to free.
/// A draw reserves from its start until its commands are submitted: what it allocated shows in the GPU's
/// use from then on, and its readbacks (system memory) wait outside.
///
/// It replaces a fixed four draws at once (`ror serve --gpu-slots`): a fifth render waited for a slot with
/// the GPU half empty, its scene's loading and its views' recording (CPU work) held up too.
#[derive(Clone, Default)]
pub struct GpuGate {
	inner: Arc<GateInner>,
}

#[derive(Default)]
struct GateInner {
	state: Mutex<Drawing>,
	ended: Condvar,
}

/// The draws started and not yet submitted, what they reserved and the sets they make.
#[derive(Default)]
struct Drawing {
	draws: usize,
	reserved: u64,
	sets: Vec<((u32, u32, u32), usize)>,
	/// Draws waiting for memory now, and since the start: those that waited and how long in all
	waiting: usize,
	admitted: u64,
	waited: u64,
	waited_ms: f64,
}

impl Drawing {
	/// What `sets` add: those neither kept nor made by a draw starting now.
	fn adds(&self, sets: &[TargetSet]) -> u64 {
		sets.iter().filter(|set| !set.kept && !self.sets.iter().any(|(key, _)| *key == set.key)).map(|set| set.bytes).sum()
	}
}

/// A draw's admission: given back when dropped, once its commands are submitted.
pub struct GpuAdmission {
	inner: Arc<GateInner>,
	reserved: u64,
	keys: Vec<(u32, u32, u32)>,
	/// How long it waited for memory
	pub waited: Duration,
}

impl Drop for GpuAdmission {
	fn drop(&mut self) {
		let mut state = self.inner.state.lock();
		state.draws = state.draws.saturating_sub(1);
		state.reserved = state.reserved.saturating_sub(self.reserved);
		for key in &self.keys {
			if let Some(index) = state.sets.iter().position(|(set, _)| set == key) {
				state.sets[index].1 -= 1;
				if state.sets[index].1 == 0 {
					state.sets.swap_remove(index);
				}
			}
		}
		drop(state);
		self.inner.ended.notify_all();
	}
}

impl GpuGate {
	/// Waits until a draw into `sets` fits ([`GpuGate`]) and admits it. Beside a game ([`crate::gaming`]) a
	/// draw alone waits for the headroom too, and gives up after [`crate::memory_wait`] saying why: past
	/// full, Windows makes room by moving the game's textures out (a game proxy's 7 GB set beside many
	/// renders and Studios: 4.0 of its 7 GB resident at times, the System process's paging 50 % of a GPU
	/// engine, 25 fps, 2026-11-02).
	pub fn admit(&self, sets: &[TargetSet], free: &dyn Fn() -> Option<GpuFree>, relieve: &dyn Fn(u64) -> u64) -> anyhow::Result<GpuAdmission> {
		self.admit_beside(sets, free, relieve, &crate::gaming::now)
	}

	/// [`GpuGate::admit`] with the game running given by `game` (each time it looks again).
	pub fn admit_beside(&self, sets: &[TargetSet], free: &dyn Fn() -> Option<GpuFree>, relieve: &dyn Fn(u64) -> u64, game: &dyn Fn() -> Option<crate::gaming::Gaming>) -> anyhow::Result<GpuAdmission> {
		let started = Instant::now();
		let (mut relieved_at, mut waiting, mut told) = (None::<Instant>, false, false);
		let mut state = self.inner.state.lock();
		loop {
			let adds = state.adds(sets);
			if adds == 0 {
				break;
			}
			let gaming = game();
			if state.draws == 0 && gaming.is_none() {
				break;
			}
			let Some(GpuFree { free, total }) = free() else {
				break;
			};
			let headroom = total / 100 * GPU_HEADROOM_PERCENT;
			let short = state.reserved.saturating_add(adds).saturating_add(headroom).saturating_sub(free);
			if short == 0 {
				break;
			}
			if relieved_at.is_none_or(|at| at.elapsed() > RELIEVE_AGAIN) {
				relieved_at = Some(Instant::now());
				drop(state);
				let freed = relieve(short);
				state = self.inner.state.lock();
				if freed > 0 {
					continue;
				}
			}
			if let Some(gaming) = &gaming {
				let reserved = state.reserved;
				let why = || {
					let game = match (&gaming.image, gaming.gpu) {
						(Some(image), Some(gpu)) => format!("{image} holds {}, ", crate::gigabytes_text(gpu)),
						_ => String::new(),
					};
					format!(
						"it adds {} of GPU memory and {} of that is not to be had: {}{} of {} free, {} kept free beside the game, {} the draws starting with it take",
						crate::gigabytes_text(adds),
						crate::gigabytes_text(short),
						game,
						crate::gigabytes_text(free),
						crate::gigabytes_text(total),
						crate::gigabytes_text(headroom),
						crate::gigabytes_text(reserved)
					)
				};
				// not on the caller going: a clip's frames after its early sheet draw with the caller gone
				let (waited, limit) = (started.elapsed(), crate::memory_wait());
				if waited >= limit {
					if waiting {
						state.waiting -= 1;
					}
					anyhow::bail!("a draw gave up after waiting {:.0} s for GPU memory while a game runs: {}. ROR_MEMORY_WAIT=<seconds> sets the wait", waited.as_secs_f64(), why());
				}
				if !told {
					told = true;
					let line = format!("a draw waits for GPU memory while a game runs: {}. It starts when memory frees, and gives up after {:.0} s", why(), limit.as_secs_f64());
					drop(state);
					crate::say(&line);
					state = self.inner.state.lock();
					continue;
				}
			}
			if !waiting {
				waiting = true;
				state.waiting += 1;
			}
			self.inner.ended.wait_for(&mut state, GPU_RECHECK);
		}
		let waited = started.elapsed();
		if waiting {
			state.waiting -= 1;
			state.waited += 1;
			state.waited_ms += waited.as_secs_f64() * 1000.0;
		}
		let reserved = state.adds(sets);
		state.draws += 1;
		state.reserved += reserved;
		state.admitted += 1;
		let mut keys = Vec::with_capacity(sets.len());
		for set in sets {
			match state.sets.iter_mut().find(|(key, _)| *key == set.key) {
				Some((_, count)) => *count += 1,
				None => state.sets.push((set.key, 1)),
			}
			keys.push(set.key);
		}
		if told {
			drop(state);
			crate::say(&format!("the draw starts after {:.1} s waiting for GPU memory", waited.as_secs_f64()));
		}
		Ok(GpuAdmission { inner: self.inner.clone(), reserved, keys, waited })
	}

	/// The draws starting now and what they reserved, those waiting for memory, and those that waited
	/// since the service started (the service's health).
	pub fn status(&self) -> serde_json::Value {
		let state = self.inner.state.lock();
		serde_json::json!({
			"drawing": state.draws,
			"reservedBytes": state.reserved,
			"waiting": state.waiting,
			"admitted": state.admitted,
			"waited": state.waited,
			"waitedMs": state.waited_ms,
			"headroomPercent": GPU_HEADROOM_PERCENT,
		})
	}
}

/// The jobs running and queued, what they were admitted for and have yet to grow into, and the jobs whose
/// needs were learned (the service's health).
pub fn jobs() -> serde_json::Value {
	let now = Need::now();
	let jobs = JOBS.lock();
	let pending = if jobs.running > 0 { unrealized(&jobs, now) } else { NONE };
	serde_json::json!({
		"running": jobs.running,
		"queued": jobs.queued,
		"needBytes": jobs.needs.resident,
		"needCommitBytes": jobs.needs.commit,
		"unrealizedBytes": pending.resident,
		"unrealizedCommitBytes": pending.commit,
		"ceilingBytes": ceiling(),
		"floorBytes": floor(),
		"learned": LEARNED.lock().len(),
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	// the budget is one for the process: these tests take turns
	static TURN: Mutex<()> = parking_lot::const_mutex(());

	#[test]
	fn nested_loads_reserve_once_and_give_back() {
		let _turn = TURN.lock();
		let before = in_flight();
		{
			let _outer = reserve(1 << 20);
			let _inner = reserve(1 << 20);
			let (loads, reserved) = in_flight();
			assert_eq!(loads, before.0 + 1);
			assert_eq!(reserved, before.1 + estimate(1 << 20));
		}
		assert_eq!(in_flight(), before);
	}

	#[test]
	fn a_parked_load_lets_the_one_it_waits_for_start() {
		let _turn = TURN.lock();
		let holder = reserve(1 << 20);
		let (loads, _) = in_flight();
		{
			let _parked = park();
			assert_eq!(in_flight().0, loads - 1);
		}
		assert_eq!(in_flight().0, loads);
		drop(holder);
	}

	#[test]
	fn small_inputs_reserve_the_minimum() {
		assert_eq!(estimate(10), MIN_LOAD);
		assert_eq!(estimate(20 << 20), 2000 << 20);
	}

	/// A need of `n` bytes, resident and committed.
	fn bytes(n: u64) -> Need {
		Need { resident: n, commit: n }
	}

	// Admission reads the machine's memory on Windows only (crate::available_memory): elsewhere every job is admitted
	#[cfg(windows)]
	#[test]
	fn a_job_that_does_not_fit_has_the_caches_relieved_then_queues_until_the_running_ones_end() {
		let _turn = TURN.lock();
		let first = admit("first", None, bytes(MIN_JOB), &|_| 0).unwrap();
		let relieved = std::sync::atomic::AtomicUsize::new(0);
		let (started, admitted) = (std::sync::atomic::AtomicBool::new(false), std::sync::atomic::AtomicBool::new(false));
		std::thread::scope(|scope| {
			scope.spawn(|| {
				started.store(true, std::sync::atomic::Ordering::SeqCst);
				// more than any machine has: it fits only once it is the process's first job
				let _second = admit(
					"second",
					None,
					bytes(u64::MAX / 4),
					&|_| {
						relieved.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
						0
					},
				)
				.unwrap();
				admitted.store(true, std::sync::atomic::Ordering::SeqCst);
			});
			while !started.load(std::sync::atomic::Ordering::SeqCst) {
				std::thread::yield_now();
			}
			std::thread::sleep(Duration::from_millis(400));
			assert!(!admitted.load(std::sync::atomic::Ordering::SeqCst), "admitted beside a running job it does not fit beside");
			assert_eq!(jobs()["queued"], 1);
			assert!(relieved.load(std::sync::atomic::Ordering::SeqCst) >= 1, "queued without relieving the caches first");
			drop(first);
		});
		assert!(admitted.load(std::sync::atomic::Ordering::SeqCst));
		assert_eq!(jobs()["queued"], 0);
	}

	#[test]
	fn a_job_that_ran_alone_to_its_end_is_admitted_for_what_it_took_next_time() {
		let _turn = TURN.lock();
		let key = "test: a job alone".to_string();
		let mut job = admit("alone", Some(key.clone()), bytes(1 << 40), &|_| 0).unwrap();
		// written to, so resident as well as committed
		let held = vec![1u8; 128 << 20];
		std::thread::sleep(SAMPLE * 3);
		job.finished();
		drop(job);
		drop(held);
		let learned = LEARNED.lock().get(&key).copied().expect("learned");
		assert!((MIN_JOB..1 << 40).contains(&learned.resident) && (MIN_JOB..1 << 40).contains(&learned.commit), "{learned:?}");
		// one that failed part way teaches nothing
		let failed = "test: a job that failed".to_string();
		drop(admit("failed", Some(failed.clone()), bytes(1 << 40), &|_| 0).unwrap());
		assert!(!LEARNED.lock().contains_key(&failed));
	}

	/// A caller that hears what its job says, with its own settings, gone or not.
	struct Listening {
		heard: Mutex<Vec<String>>,
		wait: &'static str,
		gone: bool,
	}

	impl crate::Caller for Listening {
		fn hear(&self, text: &str) {
			self.heard.lock().push(text.to_string());
		}
		fn gone(&self) -> bool {
			self.gone
		}
		fn setting(&self, name: &str) -> Option<String> {
			(name == "ROR_MEMORY_WAIT").then(|| self.wait.to_string())
		}
	}

	// Admission reads the machine's memory on Windows only (crate::available_memory): elsewhere every job is admitted
	#[cfg(windows)]
	#[test]
	fn a_job_short_of_memory_tells_its_caller_why_once_and_gives_up_after_the_wait() {
		let _turn = TURN.lock();
		let first = admit("first", None, bytes(MIN_JOB), &|_| 0).unwrap();
		let caller = std::rc::Rc::new(Listening { heard: Mutex::new(Vec::new()), wait: "0.6", gone: false });
		let started = Instant::now();
		// more than any machine has, behind a job that does not end
		let error = crate::for_caller(caller.clone(), || admit("the second", None, bytes(u64::MAX / 4), &|_| 0)).err().expect("admitted");
		let waited = started.elapsed();
		drop(first);
		assert!(waited >= Duration::from_millis(600) && waited < Duration::from_secs(10), "{waited:?}");
		let heard = caller.heard.lock().clone();
		assert_eq!(heard.len(), 1, "said once: {heard:?}");
		// what it needs, what is free, the floor and how to lower it, to the caller and in the failure
		for line in [heard[0].as_str(), &format!("{error:#}")] {
			for part in ["it needs", "GB free", "the floor of", "running job", "ROR_MEMORY_FLOOR_GB=", "ROR_MEMORY_WAIT="] {
				assert!(line.contains(part), "{part:?} missing from {line:?}");
			}
		}
		assert!(heard[0].starts_with("the second waits for memory: ") && heard[0].contains("gives up after 1 s"), "{}", heard[0]);
		assert!(format!("{error:#}").starts_with("the second gave up after waiting 1 s for memory: "), "{error:#}");
		assert_eq!(jobs()["queued"], 0);
	}

	// Admission reads the machine's memory on Windows only (crate::available_memory): elsewhere every job is admitted
	#[cfg(windows)]
	#[test]
	fn a_job_whose_caller_has_gone_stops_waiting_for_memory() {
		let _turn = TURN.lock();
		let first = admit("first", None, bytes(MIN_JOB), &|_| 0).unwrap();
		let caller = std::rc::Rc::new(Listening { heard: Mutex::new(Vec::new()), wait: "60", gone: true });
		let started = Instant::now();
		let error = crate::for_caller(caller, || admit("the second", None, bytes(u64::MAX / 4), &|_| 0)).err().expect("admitted");
		drop(first);
		assert!(started.elapsed() < Duration::from_secs(5));
		assert!(format!("{error:#}").contains("its caller went away while it waited for memory"), "{error:#}");
		assert_eq!(jobs()["queued"], 0);
	}

	const GB: u64 = 1 << 30;

	fn set(width: u32, bytes: u64, kept: bool) -> TargetSet {
		TargetSet { key: (width, width, 4), bytes, kept }
	}

	/// [`GpuGate::admit`] with no game running, whatever this machine runs while the tests do.
	fn draw(gate: &GpuGate, sets: &[TargetSet], free: &dyn Fn() -> Option<GpuFree>, relieve: &dyn Fn(u64) -> u64) -> GpuAdmission {
		gate.admit_beside(sets, free, relieve, &|| None).unwrap()
	}

	#[test]
	fn draws_that_fit_start_together_however_many() {
		// a 16 GB GPU with 8 free: twelve draws of 0.3 GB each, sizes of their own, all at once
		let gate = GpuGate::default();
		let free = || Some(GpuFree { free: 8 * GB, total: 16 * GB });
		let admissions: Vec<GpuAdmission> = (0..12).map(|index| draw(&gate, &[set(1000 + index, 300 << 20, false)], &free, &|_| 0)).collect();
		assert!(admissions.iter().all(|admission| admission.waited < Duration::from_millis(40)));
		assert_eq!(gate.status()["drawing"], 12);
		assert_eq!(gate.status()["reservedBytes"], 12 * (300u64 << 20));
		drop(admissions);
		assert_eq!(gate.status()["drawing"], 0);
		assert_eq!(gate.status()["reservedBytes"], 0);
	}

	#[test]
	fn a_set_kept_or_being_made_adds_nothing() {
		// 2 GB free, under the headroom alone: a draw into a kept set, or into the set another draw makes
		// now, starts; a new size waits
		let gate = GpuGate::default();
		let free = || Some(GpuFree { free: 2 * GB, total: 16 * GB });
		let first = draw(&gate, &[set(3840, 2 * GB, false)], &free, &|_| 0);
		let same = draw(&gate, &[set(3840, 2 * GB, false)], &free, &|_| 0);
		let kept = draw(&gate, &[set(1920, GB, true)], &free, &|_| 0);
		assert_eq!(gate.status()["reservedBytes"], 2 * GB);
		assert!(same.waited < Duration::from_millis(40) && kept.waited < Duration::from_millis(40));
		drop((first, same, kept));
	}

	#[test]
	fn a_draw_that_does_not_fit_has_memory_relieved_then_waits_for_one_to_end() {
		let gate = GpuGate::default();
		let free_bytes = std::sync::atomic::AtomicU64::new(3 * GB);
		let free = || Some(GpuFree { free: free_bytes.load(std::sync::atomic::Ordering::SeqCst), total: 16 * GB });
		let relieved = std::sync::atomic::AtomicUsize::new(0);
		let relieve = |_| {
			relieved.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
			0
		};
		let first = draw(&gate, &[set(1024, GB, false)], &free, &relieve);
		let admitted = std::sync::atomic::AtomicBool::new(false);
		std::thread::scope(|scope| {
			scope.spawn(|| {
				// 1 GB reserved + 1 GB + 1.6 GB headroom > 3 GB free
				let second = draw(&gate, &[set(2048, GB, false)], &free, &relieve);
				admitted.store(true, std::sync::atomic::Ordering::SeqCst);
				assert!(second.waited >= Duration::from_millis(200), "{:?}", second.waited);
			});
			std::thread::sleep(Duration::from_millis(300));
			assert!(!admitted.load(std::sync::atomic::Ordering::SeqCst), "started beside a draw it does not fit beside");
			assert_eq!(gate.status()["waiting"], 1);
			assert_eq!(relieved.load(std::sync::atomic::Ordering::SeqCst), 1, "waited without relieving first");
			drop(first);
		});
		assert!(admitted.load(std::sync::atomic::Ordering::SeqCst));
		assert_eq!(gate.status()["waited"], 1);
		// memory freed by others (no draw ending) lets a waiting draw start too
		let first = draw(&gate, &[set(1024, GB, false)], &free, &relieve);
		std::thread::scope(|scope| {
			let waiter = scope.spawn(|| draw(&gate, &[set(4096, GB, false)], &free, &relieve).waited);
			std::thread::sleep(Duration::from_millis(150));
			free_bytes.store(8 * GB, std::sync::atomic::Ordering::SeqCst);
			assert!(waiter.join().unwrap() >= Duration::from_millis(100));
		});
		drop(first);
	}

	#[test]
	fn a_draw_alone_starts_whatever_the_memory() {
		let gate = GpuGate::default();
		let admission = draw(&gate, &[set(8192, 12 * GB, false)], &|| Some(GpuFree { free: GB, total: 16 * GB }), &|_| 0);
		assert!(admission.waited < Duration::from_millis(40));
		// and with no memory known (no Vulkan budget, no adapter counter), every draw starts
		let unknown = draw(&gate, &[set(4096, 12 * GB, false)], &|| None, &|_| 0);
		assert!(unknown.waited < Duration::from_millis(40));
	}

	fn a_game() -> Option<crate::gaming::Gaming> {
		Some(crate::gaming::Gaming { image: Some("game.exe".into()), working_set: Some(9 * GB), private: Some(11 * GB), gpu: Some(7 * GB) })
	}

	#[test]
	fn beside_a_game_a_draw_alone_waits_for_the_headroom_and_starts_when_memory_frees() {
		let gate = GpuGate::default();
		let free_bytes = std::sync::atomic::AtomicU64::new(GB);
		let free = || Some(GpuFree { free: free_bytes.load(std::sync::atomic::Ordering::SeqCst), total: 16 * GB });
		let relieved = std::sync::atomic::AtomicUsize::new(0);
		let relieve = |_| {
			relieved.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
			0
		};
		std::thread::scope(|scope| {
			// 0.5 GB + 1.6 GB headroom > 1 GB free: it has the renderer's caches relieved, then waits
			let waiter = scope.spawn(|| gate.admit_beside(&[set(1024, GB / 2, false)], &free, &relieve, &a_game).map(|admission| admission.waited));
			std::thread::sleep(Duration::from_millis(200));
			assert_eq!(gate.status()["waiting"], 1, "a draw alone started beside a game without the memory");
			assert!(relieved.load(std::sync::atomic::Ordering::SeqCst) >= 1, "waited without relieving first");
			free_bytes.store(4 * GB, std::sync::atomic::Ordering::SeqCst);
			assert!(waiter.join().unwrap().unwrap() >= Duration::from_millis(150));
		});
		// a set kept from an earlier draw adds nothing and starts at once
		let kept = gate.admit_beside(&[set(2048, GB, true)], &|| Some(GpuFree { free: 0, total: 16 * GB }), &relieve, &a_game).unwrap();
		assert!(kept.waited < Duration::from_millis(40));
	}

	#[test]
	fn beside_a_game_a_draw_that_never_fits_gives_up_after_the_wait_saying_why() {
		let gate = GpuGate::default();
		let caller = std::rc::Rc::new(Listening { heard: Mutex::new(Vec::new()), wait: "0.4", gone: false });
		let started = Instant::now();
		let result = crate::for_caller(caller.clone(), || gate.admit_beside(&[set(3840, 2 * GB, false)], &|| Some(GpuFree { free: GB, total: 16 * GB }), &|_| 0, &a_game));
		let error = format!("{:#}", result.err().expect("admitted"));
		assert!(started.elapsed() >= Duration::from_millis(400) && started.elapsed() < Duration::from_secs(10));
		assert!(error.starts_with("a draw gave up after waiting 0 s for GPU memory while a game runs: ") && error.contains("game.exe holds 7.0 GB"), "{error}");
		let heard = caller.heard.lock().clone();
		assert!(heard.len() == 1 && heard[0].starts_with("a draw waits for GPU memory while a game runs: "), "{heard:?}");
		assert_eq!(gate.status()["waiting"], 0);
		assert_eq!(gate.status()["drawing"], 0);
	}

	#[test]
	fn beside_a_game_a_draw_whose_caller_has_gone_still_waits_for_the_memory() {
		// a clip's frames after its early sheet draw for a caller that has gone: they wait, they do not fail
		let gate = GpuGate::default();
		let caller = std::rc::Rc::new(Listening { heard: Mutex::new(Vec::new()), wait: "0.4", gone: true });
		let started = Instant::now();
		let result = crate::for_caller(caller, || gate.admit_beside(&[set(3840, 2 * GB, false)], &|| Some(GpuFree { free: GB, total: 16 * GB }), &|_| 0, &a_game));
		assert!(result.is_err() && started.elapsed() >= Duration::from_millis(400), "gave up after {:?}", started.elapsed());
	}

	#[test]
	fn needs_grow_with_the_pixels_and_the_input() {
		for need in [|need: Need| need.resident, |need: Need| need.commit] {
			assert!(need(clip_need(3840 * 2160, 0)) > need(clip_need(1572 * 862, 0)));
			assert!(need(clip_need(1572 * 862, 0)) > need(still_need(1572 * 862, 0)));
			assert!(need(clip_need(1572 * 862, 20 << 20)) > need(clip_need(1572 * 862, 0)));
		}
		assert_eq!(still_need(0, 0), Need { resident: 96 << 20, commit: 192 << 20 });
		// what a clip makes resident is a fraction of what it commits (its GPU memory stays on the GPU),
		// and both cover what one took alone (2026-11-02: 319 MB resident, 1069 MB committed at most)
		let clip = clip_need(2560 * 1440, 20 << 10);
		assert!(clip.resident < clip.commit / 2 && clip.resident > 319 << 20 && clip.commit > 1069 << 20, "{clip:?}");
	}

	#[test]
	fn file_bytes_counts_the_files_there() {
		let file = std::env::temp_dir().join(format!("ror-file-bytes-{}", std::process::id()));
		std::fs::write(&file, [0u8; 1000]).unwrap();
		assert_eq!(file_bytes([file.as_path(), std::path::Path::new("rbxassetid://1"), std::env::temp_dir().as_path()]), 1000);
		let _ = std::fs::remove_file(&file);
	}
}
