//! ror: renders Roblox 3D objects (models, parts, meshes, characters) offline.
//!
//! - [`scene`]: a Roblox model file (or Studio export) → the scene description the renderer draws
//! - [`render`]: the wgpu renderer (shared GPU device, cached meshes and textures, headless views)
//! - [`job`]: a render request (input, views, output directory) run against a renderer
//! - [`output`]: PNG encoding and the state / output paths
//! - [`service`]: the warm render service many agents share, and its client
//! - [`studio`]: the Studio parity harness (capture Studio, compare, calibration experiments)
//! - [`clean`]: disk use (identical files linked into one, old derived images removed)
//! - [`inspect`]: what a render shows, read back as text (parts by coverage, a pixel's part, two
//!   renders compared)
//! - [`lod`]: Roblox's levels of detail and distance culling for a camera (`--lod roblox`)
//! - [`lint`]: authoring lints over what a render or a video draws (weightless particles; over a
//!   video, linear fades, restraint within a form, frozen clouds, refused images, rigid beams, counts
//!   per effect)

pub mod budget;
pub mod clean;
pub mod cost;
pub mod gaming;
pub mod inspect;
pub mod job;
pub mod lint;
pub mod lod;
pub mod mp4;
pub mod nvenc;
pub mod output;
pub(crate) mod par;
pub mod render;
pub mod scene;
pub mod script;
pub mod service;
pub mod studio;
pub mod video;
#[cfg(ror_x264)]
pub mod x264;

/// The worker threads of each of ror's thread pools (rayon's global pool, [`init_threads`], and the
/// renderer's inner one): half the logical processors and at most 8, unless `RAYON_NUM_THREADS` says.
/// The work they share is bound by memory and the GPU more than by cores: on 32 logical processors
/// (2026-10-14, alternating runs, images identical), 8 drew a map's 32 views, its 2 s clip and
/// a GUI clip as fast as 32 or faster (0.90-0.96 / 1.59-1.67 / 1.74-1.90 s), with 20-35 % less CPU
/// time, a third of the peak cores (the clip's 8.8 -> 5.1) and a peak commit 28 % lower (the views'
/// 2.4 -> 1.7 GB: each worker held its own textures' decoding); only a cost grid took longer (1.46 ->
/// 1.77 s). The machine is shared with games and Studios: ror never takes every core.
pub fn worker_threads() -> usize {
	static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*THREADS.get_or_init(|| {
		if let Some(threads) = std::env::var("RAYON_NUM_THREADS").ok().and_then(|value| value.parse::<usize>().ok()).filter(|&threads| threads > 0) {
			return threads;
		}
		let logical = std::thread::available_parallelism().map_or(2, |threads| threads.get());
		(logical / 2).clamp(1, 8)
	})
}

/// Starts rayon's global pool with [`worker_threads`] threads (rayon's own default is one a logical
/// processor): the first thing a program on this library does, before any parallel work.
pub fn init_threads() {
	let _ = rayon::ThreadPoolBuilder::new().num_threads(worker_threads()).thread_name(|index| format!("ror-pool-{index}")).build_global();
}

/// Gives the system back the memory the allocator holds free: mimalloc purges free pages only on
/// allocator activity in the thread that owns them, so a process gone quiet (the render service
/// between jobs, rayon's workers parked) kept every page its last job freed. Each pool thread
/// collects its own, then this one, forcing the arenas' purge.
pub fn release_free_memory() {
	#[cfg(windows)]
	{
		let collect = || unsafe { libmimalloc_sys::mi_collect(true) };
		rayon::broadcast(|_| collect());
		render::inner_pool().broadcast(|_| collect());
		collect();
	}
}

/// `ROR_MEMORY_REPORT`: this process's commit and working set now, on stderr with `label` (where a
/// command's memory goes, step by step).
/// Milliseconds since the first [`memory_checkpoint`] (a command's "start"), for `ROR_MEMORY_REPORT`.
pub fn since_first_checkpoint_ms() -> f64 {
	static FIRST: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
	FIRST.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// Physical memory free for use now (Windows' available bytes; `None` where it is not known).
pub fn available_memory() -> Option<u64> {
	#[cfg(windows)]
	// SAFETY: the system's counters, read into a zeroed struct whose length field is set
	unsafe {
		use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
		let mut status: MEMORYSTATUSEX = std::mem::zeroed();
		status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
		if GlobalMemoryStatusEx(&mut status) != 0 {
			return Some(status.ullAvailPhys);
		}
	}
	None
}

#[cfg(test)]
mod memory_guard_tests {
	use super::{memory_cap, memory_floor};
	const GIB: u64 = 1 << 30;

	#[test]
	fn floor_and_cap_follow_the_machine_and_the_settings() {
		// a 64 GB machine: 12 % and 40 %; a 16 GB one: the minimums
		assert_eq!(memory_floor(62 * GIB, None), Some((62.0 * 0.12 * GIB as f64) as u64));
		assert_eq!(memory_cap(62 * GIB, None), Some((62.0 * 0.4 * GIB as f64) as u64));
		assert_eq!(memory_floor(16 * GIB, None), Some(4 * GIB));
		assert_eq!(memory_cap(8 * GIB, None), Some(6 * GIB));
		// settings in gigabytes, 0 turning each off, anything else ignored
		assert_eq!(memory_floor(62 * GIB, Some("2.5")), Some((2.5 * GIB as f64) as u64));
		assert_eq!(memory_floor(62 * GIB, Some("0")), None);
		assert_eq!(memory_cap(62 * GIB, Some(" 10 ")), Some(10 * GIB));
		assert_eq!(memory_cap(62 * GIB, Some("0")), None);
		assert_eq!(memory_cap(62 * GIB, Some("lots")), Some((62.0 * 0.4 * GIB as f64) as u64));
	}
}

/// This process's committed memory now (bytes; `None` where it is not known): its heap and, through the
/// driver, the GPU memory it holds.
pub fn process_commit() -> Option<u64> {
	#[cfg(windows)]
	// SAFETY: this process's own counters, read into a zeroed struct of the size passed
	unsafe {
		use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
		use windows_sys::Win32::System::Threading::GetCurrentProcess;
		let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
		counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
		if GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0 {
			return Some(counters.PagefileUsage as u64);
		}
	}
	None
}

/// This process's private memory resident in RAM now (bytes; `None` where it is not known): what it takes
/// from the machine's free memory. Its commit ([`process_commit`]) also counts GPU memory, which takes
/// none of the machine's RAM while it stays on the GPU (a 1440p bundle clip's service: 1.3 GB of commit,
/// 0.5 GB resident).
pub fn process_resident() -> Option<u64> {
	#[cfg(windows)]
	// SAFETY: this process's own counters, read into a zeroed struct of the size passed
	unsafe {
		use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX2};
		use windows_sys::Win32::System::Threading::GetCurrentProcess;
		let mut counters: PROCESS_MEMORY_COUNTERS_EX2 = std::mem::zeroed();
		counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX2>() as u32;
		if GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut counters).cast::<PROCESS_MEMORY_COUNTERS>(), counters.cb) != 0 {
			return Some(counters.PrivateWorkingSetSize as u64);
		}
		// before Windows 10 1809: the whole working set
		let mut plain: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
		plain.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
		if GetProcessMemoryInfo(GetCurrentProcess(), &mut plain, plain.cb) != 0 {
			return Some(plain.WorkingSetSize as u64);
		}
	}
	None
}

/// Physical memory installed (bytes; `None` where it is not known).
pub fn total_memory() -> Option<u64> {
	#[cfg(windows)]
	// SAFETY: the system's counters, read into a zeroed struct whose length field is set
	unsafe {
		use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
		let mut status: MEMORYSTATUSEX = std::mem::zeroed();
		status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
		if GlobalMemoryStatusEx(&mut status) != 0 {
			return Some(status.ullTotalPhys);
		}
	}
	None
}

const GB: f64 = (1u64 << 30) as f64;

/// A setting in gigabytes (`ROR_MEMORY_FLOOR_GB`, `ROR_MAX_MEMORY_GB`): `None` when unset or not a
/// number, `Some(None)` for 0 (off), else the bytes.
fn gigabytes(setting: Option<&str>) -> Option<Option<u64>> {
	let value: f64 = setting?.trim().parse().ok()?;
	Some(if value > 0.0 { Some((value * GB) as u64) } else { None })
}

/// How much memory must be free before a job starts: `ROR_MEMORY_FLOOR_GB` (0: no floor), else 12 %
/// of the machine's memory and at least 4 GB.
pub fn memory_floor(total: u64, setting: Option<&str>) -> Option<u64> {
	gigabytes(setting).unwrap_or_else(|| Some(((total as f64 * 0.12) as u64).max(4 << 30)))
}

/// The most memory one ror process may commit: `ROR_MAX_MEMORY_GB` (0: no cap), else 40 % of the
/// machine's memory and at least 6 GB.
pub fn memory_cap(total: u64, setting: Option<&str>) -> Option<u64> {
	gigabytes(setting).unwrap_or_else(|| Some(((total as f64 * 0.4) as u64).max(6 << 30)))
}

/// Whom a job runs for on this thread, beyond the job itself: the client of a render service request
/// (service::Reply), which hears what the job says as it says it ([`say`]) and may set the environment
/// variables that shape its job ([`setting`]). Without one: this process's stderr and environment.
pub trait Caller {
	/// A line said now (on stderr too: a service's is its serve.log).
	fn hear(&self, text: &str);
	/// The caller has gone (its connection closed): a job waiting for it gives up.
	fn gone(&self) -> bool;
	/// The caller's value of an environment variable that shapes its job.
	fn setting(&self, name: &str) -> Option<String>;
}

thread_local! {
	static CALLER: std::cell::RefCell<Option<std::rc::Rc<dyn Caller>>> = const { std::cell::RefCell::new(None) };
}

/// Runs `run` for `caller` on this thread ([`say`], [`setting`] and [`caller_gone`] are its).
pub fn for_caller<R>(caller: std::rc::Rc<dyn Caller>, run: impl FnOnce() -> R) -> R {
	struct Restore(Option<std::rc::Rc<dyn Caller>>);
	impl Drop for Restore {
		fn drop(&mut self) {
			CALLER.with(|current| *current.borrow_mut() = self.0.take());
		}
	}
	let _restore = Restore(CALLER.with(|current| current.replace(Some(caller))));
	run()
}

/// Tells whoever asked for the job on this thread `text`, now: on stderr (`ror: <text>`), and to a
/// render service request's client as it happens (the client prints it on its own stderr).
pub fn say(text: &str) {
	eprintln!("ror: {text}");
	if let Some(caller) = CALLER.with(|current| current.borrow().clone()) {
		caller.hear(text);
	}
}

/// An environment variable as the job's caller has it set (a service request's client's), else as this
/// process has it.
pub fn setting(name: &str) -> Option<String> {
	CALLER.with(|current| current.borrow().as_ref().and_then(|caller| caller.setting(name))).or_else(|| std::env::var(name).ok())
}

/// Whether the caller of the job on this thread has gone.
pub fn caller_gone() -> bool {
	CALLER.with(|current| current.borrow().as_ref().is_some_and(|caller| caller.gone()))
}

/// The settings a job's caller may give it ([`setting`]): a render service's client sends its own.
pub const CALLER_SETTINGS: [&str; 2] = ["ROR_MEMORY_FLOOR_GB", "ROR_MEMORY_WAIT"];

/// How long a job waits for memory before it gives up: `ROR_MEMORY_WAIT` seconds ([`setting`]), 60 by
/// default. Commands waited 5-29 minutes behind the floor (2026-10-23: two Studios held 10.5 GB
/// of a 32 GB machine), most of it after their callers had given up: a minute shows the wait, and why,
/// while the caller can still act on it.
pub fn memory_wait() -> std::time::Duration {
	let seconds = setting("ROR_MEMORY_WAIT").and_then(|value| value.trim().parse::<f64>().ok()).filter(|seconds| seconds.is_finite() && *seconds >= 0.0);
	std::time::Duration::from_secs_f64(seconds.unwrap_or(60.0))
}

/// How to have a job wait less, at the end of every line about a wait for memory.
pub const MEMORY_ADVICE: &str = "ROR_MEMORY_FLOOR_GB=<GB> lowers the floor (0 turns it off) and ROR_MEMORY_WAIT=<seconds> sets the wait, in the environment of the command that asks for the job (a render service takes its caller's)";

/// Gigabytes, as the lines about memory give them.
pub fn gigabytes_text(bytes: u64) -> String {
	format!("{:.1} GB", bytes as f64 / GB)
}

/// Holds a job back while the machine is short of memory: many callers use ror at once, and a
/// map's or a cutscene's job commits gigabytes, so jobs started on a full machine filled it until it
/// paged to a stall (2026-10-17: a 64 GB machine at 100 %). Waits while less than [`memory_floor`] is
/// free, saying so once ([`say`]: what it needs, what is free, the floor and how to lower it), and gives
/// up [`memory_wait`] after `since` with the same explanation rather than pile on; also when its caller
/// has gone. `need` is what the job itself is expected to take.
pub fn wait_for_memory(what: &str, need: u64, since: std::time::Instant) -> anyhow::Result<()> {
	let (Some(total), Some(_)) = (total_memory(), available_memory()) else {
		return Ok(());
	};
	// beside a game, the floor less what the game holds (budget::floor)
	let floor = budget::floor();
	if floor == 0 {
		return Ok(());
	}
	let limit = memory_wait();
	let start = std::time::Instant::now();
	let mut told = false;
	loop {
		let free = available_memory().unwrap_or(u64::MAX);
		if free >= floor {
			if told {
				say(&format!("{what} starts after {:.0} s waiting for memory ({} free)", start.elapsed().as_secs_f64(), gigabytes_text(free)));
			}
			return Ok(());
		}
		let why = format!(
			"{} of {} free, under the floor of {} ror leaves to other programs (Studio, games, builds, other renders); it needs that much free to start, and about {} more for itself",
			gigabytes_text(free),
			gigabytes_text(total),
			gigabytes_text(floor),
			gigabytes_text(need)
		);
		let waited = since.elapsed();
		if waited >= limit {
			anyhow::bail!("{what} gave up after waiting {:.0} s for memory: {why}. {MEMORY_ADVICE}", waited.as_secs_f64());
		}
		if caller_gone() {
			anyhow::bail!("{what}: its caller went away while it waited for memory ({why})");
		}
		if !told {
			say(&format!("{what} waits for memory: {why}. It gives up after {:.0} s. {MEMORY_ADVICE}", limit.as_secs_f64()));
			told = true;
		}
		std::thread::sleep(std::time::Duration::from_millis(250));
	}
}

/// Has a crash the process cannot report itself (an access violation in a driver, say) say so on stderr
/// as it ends: the exception, its address and the process (a render service's stderr is its serve.log).
/// Panics and allocation failures say so already; a process that ended silent left its caller a failure
/// with nothing after it ("ror render failed:").
pub fn report_crashes() {
	#[cfg(windows)]
	// SAFETY: installs a function that only formats into a stack buffer and writes it to stderr
	unsafe {
		windows_sys::Win32::System::Diagnostics::Debug::SetUnhandledExceptionFilter(Some(crashed));
	}
}

#[cfg(all(test, windows))]
mod crash_report_tests {
	#[test]
	fn a_crash_says_so_on_stderr() {
		// the test binary run again with this test alone, crashing on purpose (no error box)
		if std::env::var_os("ROR_TEST_CRASH").is_some() {
			super::report_crashes();
			// SAFETY: none; an access violation is the point
			unsafe {
				use windows_sys::Win32::System::Diagnostics::Debug::{SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX, SetErrorMode};
				SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
				std::ptr::write_volatile(std::ptr::without_provenance_mut::<u8>(0x10), 1);
			}
			return;
		}
		let output = std::process::Command::new(std::env::current_exe().unwrap())
			.args(["--exact", "crash_report_tests::a_crash_says_so_on_stderr", "--nocapture", "--test-threads", "1"])
			.env("ROR_TEST_CRASH", "1")
			.output()
			.unwrap();
		assert!(!output.status.success());
		let stderr = String::from_utf8_lossy(&output.stderr);
		assert!(stderr.contains("ror: crashed (pid ") && stderr.contains("an access violation, exception 0xC0000005 at 0x"), "{stderr}");
	}
}

#[cfg(windows)]
unsafe extern "system" fn crashed(info: *const windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS) -> i32 {
	use std::io::Write;
	// SAFETY: the system hands the filter the exception's record, read once if present
	let (code, address) = unsafe { info.as_ref().and_then(|info| info.ExceptionRecord.as_ref()).map_or((0, 0), |record| (record.ExceptionCode as u32, record.ExceptionAddress as usize)) };
	let name = match code {
		0xC000_0005 => "an access violation",
		0xC000_00FD => "a stack overflow",
		0xC000_0017 | 0xC000_012D => "out of memory",
		0xC000_001D => "an illegal instruction",
		0xC000_0094 => "an integer division by zero",
		0xC000_0409 => "a fail-fast abort",
		_ => "an exception",
	};
	// no allocation here: the heap may be what failed
	let mut buffer = [0u8; 256];
	let length = {
		let mut cursor: &mut [u8] = &mut buffer;
		let _ = writeln!(cursor, "ror: crashed (pid {}): {name}, exception 0x{code:08X} at 0x{address:X}", std::process::id());
		256 - cursor.len()
	};
	// SAFETY: this process's stderr handle and a buffer of `length` bytes
	unsafe {
		use windows_sys::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE};
		let mut written = 0u32;
		windows_sys::Win32::Storage::FileSystem::WriteFile(GetStdHandle(STD_ERROR_HANDLE), buffer.as_ptr(), length as u32, &mut written, std::ptr::null_mut());
	}
	// EXCEPTION_CONTINUE_SEARCH: the process ends as it would have
	0
}

/// Caps this process's committed memory at [`memory_cap`] with a Windows job object, so a job that
/// grows past it fails with an allocation error instead of the machine paging until it stalls. The
/// processes it starts (ffmpeg, rur, a service) are each capped alike; a process already in a job
/// keeps the stricter of the two caps. The job lets a child break away when it asks to: the render
/// service does (service::client::spawn_detached) to outlive its caller, and refused, it was started
/// inside this job, so a caller's kill-on-close job (Node's, around every process it spawns) ended the
/// service with the caller and each command started a new one (~2 s instead of under 0.1).
pub fn cap_memory() {
	#[cfg(windows)]
	// SAFETY: a job object made, set and joined by this process; its handle stays open for its life
	unsafe {
		use windows_sys::Win32::Foundation::CloseHandle;
		use windows_sys::Win32::System::JobObjects::{
			AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
			JOB_OBJECT_LIMIT_BREAKAWAY_OK, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
		};
		use windows_sys::Win32::System::Threading::GetCurrentProcess;
		let Some(total) = total_memory() else { return };
		let Some(cap) = memory_cap(total, std::env::var("ROR_MAX_MEMORY_GB").ok().as_deref()) else {
			return;
		};
		let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
		if job.is_null() {
			return;
		}
		let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
		info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_BREAKAWAY_OK;
		info.ProcessMemoryLimit = cap as usize;
		let size = std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32;
		if SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const std::ffi::c_void, size) == 0
			|| AssignProcessToJobObject(job, GetCurrentProcess()) == 0
		{
			CloseHandle(job);
		}
	}
}

#[cfg(all(test, windows))]
mod cap_memory_tests {
	#[test]
	fn a_capped_process_can_start_a_child_outside_its_job() {
		// the test binary run again with this test alone, capped, starting a child that breaks away
		if std::env::var_os("ROR_TEST_BREAKAWAY").is_some() {
			use std::os::windows::process::CommandExt;
			super::cap_memory();
			let mut in_job = 0;
			// SAFETY: this process's pseudo handle and an out value
			unsafe {
				windows_sys::Win32::System::JobObjects::IsProcessInJob(windows_sys::Win32::System::Threading::GetCurrentProcess(), std::ptr::null_mut(), &mut in_job);
			}
			assert!(in_job != 0, "cap_memory made no job");
			const CREATE_BREAKAWAY_FROM_JOB: u32 = 0x0100_0000;
			let status = std::process::Command::new("cmd").args(["/c", "exit"]).creation_flags(CREATE_BREAKAWAY_FROM_JOB).status();
			assert!(status.as_ref().is_ok_and(|status| status.success()),"breaking away from the memory cap's job was refused: {status:?}");
			return;
		}
		let output = std::process::Command::new(std::env::current_exe().unwrap())
			.args(["--exact", "cap_memory_tests::a_capped_process_can_start_a_child_outside_its_job", "--nocapture", "--test-threads", "1"])
			.env("ROR_TEST_BREAKAWAY", "1")
			.env("ROR_MAX_MEMORY_GB", "64")
			.output()
			.unwrap();
		assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stdout));
	}
}

pub fn memory_checkpoint(label: &str) {
	if std::env::var_os("ROR_MEMORY_REPORT").is_none() {
		return;
	}
	#[cfg(windows)]
	// SAFETY: this process's own counters, read into a zeroed struct of the size passed
	unsafe {
		use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
		use windows_sys::Win32::System::Threading::GetCurrentProcess;
		let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
		counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
		if GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0 {
			eprintln!("ror memory at {label} ({:.0} ms): commit {} MB, working set {} MB", since_first_checkpoint_ms(), counters.PagefileUsage >> 20, counters.WorkingSetSize >> 20);
		}
	}
}

/// Windows hands every inheritable handle to every child process, whatever its own stdio: a child
/// that outlives this process (the render service, Studio, rur's native helper behind Node) would
/// keep a caller's stdout / stderr pipes open, and a caller reading them to the end waits for that
/// child. This process's own std handles are never meant for children (`Stdio::inherit` passes
/// duplicates). `ror` calls it first thing; the renderer calls it before starting rur, for the
/// examples and other programs built on this library.
pub fn private_std_handles() {
	#[cfg(windows)]
	{
		use windows_sys::Win32::Foundation::{HANDLE_FLAG_INHERIT, SetHandleInformation};
		use windows_sys::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE};
		for which in [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
			let handle = unsafe { GetStdHandle(which) };
			if !handle.is_null() {
				unsafe {
					SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0);
				}
			}
		}
	}
}
