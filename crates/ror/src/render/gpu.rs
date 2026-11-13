//! The headless GPU device, shared by every render in the process (and by a window drawing what ror
//! renders: its instance and adapter make the window's surface on the same device).
//!
//! Starting it (`ror serve`'s first seconds; measured on a desktop with a discrete and an integrated
//! GPU, 2026-10-10, `examples/gpu_start.rs`): an instance of every backend and its adapters took
//! 4.0-6.5 s, DX12's alone 2.2-3.0 s, Vulkan's 2.3-3.1 s; ror never chose DX12 over a discrete Vulkan
//! adapter. So Vulkan is asked first, and every backend only when it has no discrete GPU. The pipelines then take the driver's compiles: NVIDIA keeps those per
//! executable, and the render service runs a copy of ror named by its build, so the first service
//! after every build compiled them all again (3.7-5.1 s; 0.12-0.19 s from the driver's cache). A
//! pipeline cache of ror's own (`<state dir>/pipelines/<adapter and driver>`, Vulkan's) keeps them
//! across builds.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use parking_lot::{Condvar, Mutex};

pub struct Gpu {
	pub instance: wgpu::Instance,
	pub adapter: wgpu::Adapter,
	pub device: wgpu::Device,
	pub queue: wgpu::Queue,
	pub adapter_info: wgpu::AdapterInfo,
	pub features: wgpu::Features,
	/// The driver's compiled pipelines, kept on disk between processes ([`Gpu::save_pipeline_cache`]);
	/// `None` where the backend has none (DX12) or `ROR_PIPELINE_CACHE=0`.
	pub pipeline_cache: Option<wgpu::PipelineCache>,
	cache_file: Option<PathBuf>,
	/// The size of what was last written (written again only when it grew).
	cache_written: Mutex<usize>,
	/// The driver refused the cache this process started from: the content store's copy is written
	/// again whatever its size.
	store_stale: std::sync::atomic::AtomicBool,
	/// Bytes written through the queue since its last submission ([`Gpu::uploaded`])
	pending_uploads: std::sync::atomic::AtomicU64,
	/// The submissions made through [`Gpu::submit`] and which the GPU has done, and the one thread that
	/// polls the device for them ([`Tracker`])
	tracker: Arc<Tracker>,
	poller: Mutex<Option<std::thread::JoinHandle<()>>>,
	/// What the whole adapter has in use, opened by the first [`Gpu::free_memory`]
	adapter_memory: std::sync::OnceLock<Option<AdapterMemory>>,
}

/// The adapter's own memory in use by every program, as Windows counts it (the "GPU Adapter Memory"
/// counter of the adapter's LUID, what Task Manager shows as dedicated GPU memory: within ~15 ms of an
/// allocation, and matching nvidia-smi's to 0.2 %), and the size of its largest device-local heap.
#[cfg_attr(not(windows), allow(dead_code))]
struct AdapterMemory {
	#[cfg(windows)]
	counter: Mutex<pdh::Counter>,
	size: u64,
}

impl AdapterMemory {
	/// The memory free on the adapter.
	fn free(&self) -> Option<u64> {
		#[cfg(windows)]
		{
			Some(self.size.saturating_sub(self.counter.lock().read()?))
		}
		#[cfg(not(windows))]
		{
			let _ = self.size;
			None
		}
	}
}

/// One PDH counter read on demand (Win32 performance counters).
#[cfg(windows)]
mod pdh {
	use windows_sys::Win32::System::Performance::{PDH_FMT_COUNTERVALUE, PDH_FMT_LARGE, PDH_HCOUNTER, PDH_HQUERY, PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterValue, PdhOpenQueryW};

	pub struct Counter {
		query: PDH_HQUERY,
		counter: PDH_HCOUNTER,
	}

	// SAFETY: the query's handles are used under the owner's mutex only
	unsafe impl Send for Counter {}

	impl Counter {
		/// `path` (English names), if the counter exists and answers.
		pub fn open(path: &str) -> Option<Self> {
			let wide: Vec<u16> = path.encode_utf16().chain([0]).collect();
			let mut query = std::ptr::null_mut();
			// SAFETY: out-pointers to locals, a NUL-terminated path; the query is closed on drop
			unsafe {
				if PdhOpenQueryW(std::ptr::null(), 0, &mut query) != 0 {
					return None;
				}
				// closed on drop from here on
				let mut opened = Self { query, counter: std::ptr::null_mut() };
				if PdhAddEnglishCounterW(query, wide.as_ptr(), 0, &mut opened.counter) != 0 {
					return None;
				}
				opened.read()?;
				Some(opened)
			}
		}

		/// Its value now.
		pub fn read(&mut self) -> Option<u64> {
			let mut value = PDH_FMT_COUNTERVALUE::default();
			// SAFETY: the query and counter this owns, an out-pointer to a local
			unsafe {
				if PdhCollectQueryData(self.query) != 0 || PdhGetFormattedCounterValue(self.counter, PDH_FMT_LARGE, std::ptr::null_mut(), &mut value) != 0 {
					return None;
				}
				// PDH_CSTATUS_VALID_DATA or PDH_CSTATUS_NEW_DATA
				(value.CStatus <= 1).then(|| value.Anonymous.largeValue.max(0) as u64)
			}
		}
	}

	impl Drop for Counter {
		fn drop(&mut self) {
			// SAFETY: the query this owns, closed once (its counter with it)
			unsafe { PdhCloseQuery(self.query) };
		}
	}
}

/// A submission made through [`Gpu::submit`]: its number in the order the queue took them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Submitted(u64);

/// What became of the submissions: each numbered as the queue takes it and marked done by the queue's
/// own callback once the GPU has run it. One thread polls the device for that ([`poll_loop`]); every
/// wait for the GPU (a readback's, an upload's) waits on [`Tracker::changed`] instead of polling.
///
/// Before, every waiter polled the device itself, 0.5 ms at a time: wgpu holds the device's fence
/// through a poll's wait and every submission takes that fence first, so with eight clips' encoders
/// waiting on their readbacks each frame's submission waited ~1 ms for a poll to let go of it. The
/// poller waits on a Vulkan timeline semaphore of its own, signalled with each submission
/// ([`Timeline`]), holding no wgpu lock, and polls only once that has moved: a poll that does not wait
/// takes the fence for microseconds.
///
/// Metal has no such semaphore, and its wait in wgpu is a loop that reads the command buffer's status
/// and sleeps 1 ms, the fence held throughout: the poller polls without waiting there and naps between
/// polls holding nothing ([`poll_loop`]).
struct Tracker {
	state: Mutex<Track>,
	/// a submission was done (for its waiters)
	changed: Condvar,
	/// a submission was made, or the device is going (for the poller)
	work: Condvar,
	/// Taken through every `device.poll` ([`Gpu::poll`])
	poll_lock: Mutex<()>,
	/// Held while a submission is numbered and made, so the numbers follow the queue's order
	submitting: Mutex<u64>,
	timeline: Option<Timeline>,
	/// The timeline is not trusted (a submission panicked half-way): the poller polls on its own
	timeline_off: std::sync::atomic::AtomicBool,
}

#[derive(Default)]
struct Track {
	/// The last submission made and the last done (0: none)
	submitted: u64,
	done: u64,
	stopped: bool,
}

impl Tracker {
	fn finished(&self, number: u64) {
		let mut state = self.state.lock();
		state.done = state.done.max(number);
		state.submitted = state.submitted.max(number);
		drop(state);
		self.changed.notify_all();
	}

	fn timeline(&self) -> Option<&Timeline> {
		self.timeline.as_ref().filter(|_| !self.timeline_off.load(std::sync::atomic::Ordering::Relaxed))
	}
}

/// A Vulkan timeline semaphore of ror's own, signalled with each submission's number by the batch that
/// carries it (`add_signal_semaphore`, under [`Tracker::submitting`]): the poller waits on it without
/// any of wgpu's locks.
struct Timeline {
	device: ash::Device,
	semaphore: ash::vk::Semaphore,
}

impl Timeline {
	#[cfg(target_vendor = "apple")]
	fn new(_adapter: &wgpu::Adapter, _device: &wgpu::Device) -> Option<Self> {
		None
	}

	#[cfg(not(target_vendor = "apple"))]
	fn new(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Option<Self> {
		if std::env::var("ROR_GPU_TIMELINE").is_ok_and(|value| value == "0") {
			return None;
		}
		// SAFETY: the adapter's and device's own Vulkan objects, read and used without changing wgpu's
		unsafe {
			let hal_adapter = adapter.as_hal::<wgpu::hal::api::Vulkan>()?;
			let capabilities = hal_adapter.physical_device_capabilities();
			if capabilities.properties().api_version < ash::vk::API_VERSION_1_2 {
				return None;
			}
			// wgpu turns the feature on where the device has it (its own fence is then one)
			let instance = hal_adapter.shared_instance().raw_instance();
			let mut timeline = ash::vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
			let mut features = ash::vk::PhysicalDeviceFeatures2::default().push_next(&mut timeline);
			instance.get_physical_device_features2(hal_adapter.raw_physical_device(), &mut features);
			if timeline.timeline_semaphore == ash::vk::FALSE {
				return None;
			}
			let hal_device = device.as_hal::<wgpu::hal::api::Vulkan>()?;
			let raw = hal_device.raw_device().clone();
			let mut kind = ash::vk::SemaphoreTypeCreateInfo::default().semaphore_type(ash::vk::SemaphoreType::TIMELINE).initial_value(0);
			let semaphore = raw.create_semaphore(&ash::vk::SemaphoreCreateInfo::default().push_next(&mut kind), None).ok()?;
			Some(Self { device: raw, semaphore })
		}
	}

	/// Has the batch carrying submission `number`'s signal by `timeout`?
	fn wait(&self, number: u64, timeout: Duration) -> Result<bool, ash::vk::Result> {
		let semaphores = [self.semaphore];
		let values = [number];
		let info = ash::vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
		// SAFETY: a semaphore of this device, alive until the poller has stopped (Gpu's drop)
		match unsafe { self.device.wait_semaphores(&info, timeout.as_nanos() as u64) } {
			Ok(()) => Ok(true),
			Err(ash::vk::Result::TIMEOUT) => Ok(false),
			Err(error) => Err(error),
		}
	}
}

/// The poller: polls the device each time the GPU has done a submission someone made, until the
/// device goes. Each poll fires the callbacks of what is done (readbacks' mappings, [`Tracker::finished`],
/// GPU slots going back).
fn poll_loop(device: wgpu::Device, tracker: Arc<Tracker>) {
	let mut early = 0u32;
	#[cfg(target_vendor = "apple")]
	let mut naps = 0u32;
	loop {
		let next = {
			let mut state = tracker.state.lock();
			loop {
				if state.stopped {
					return;
				}
				if state.done < state.submitted {
					break state.done + 1;
				}
				tracker.work.wait(&mut state);
			}
		};
		let reached = match tracker.timeline() {
			Some(timeline) => match timeline.wait(next, Duration::from_millis(20)) {
				Ok(reached) => reached,
				// a lost device: wgpu's poll reports it
				Err(_) => {
					std::thread::sleep(Duration::from_millis(1));
					true
				}
			},
			// Metal: a wait inside the poll holds wgpu's fence through a 1 ms sleep (wgpu-hal's Metal
			// `wait` reads the command buffer's status, then sleeps), and every submission takes that
			// fence first: eight threads drawing small views got through under two submissions a
			// millisecond between them, each waiting out the rest of a sleep. A poll that does not wait
			// reads the same status in microseconds, so the sleep is taken here, holding nothing: short
			// while work finishes quickly, longer once the GPU is on something long.
			#[cfg(target_vendor = "apple")]
			None => {
				{
					let _one = tracker.poll_lock.lock();
					let _ = device.poll(wgpu::PollType::Poll);
				}
				if tracker.state.lock().done < next {
					naps += 1;
					std::thread::sleep(if naps > BRIEF_NAPS { LONG_NAP } else { BRIEF_NAP });
				} else {
					naps = 0;
				}
				continue;
			}
			#[cfg(not(target_vendor = "apple"))]
			None => {
				// no timeline: a short wait inside the poll (holding wgpu's fence), as every waiter did
				let _one = tracker.poll_lock.lock();
				let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_micros(500)) });
				continue;
			}
		};
		{
			let _one = tracker.poll_lock.lock();
			let _ = device.poll(wgpu::PollType::Poll);
		}
		// the semaphore can be signalled a moment before wgpu's own fence of the same batch reads so
		if reached && tracker.state.lock().done < next {
			early += 1;
			if early > 64 {
				std::thread::sleep(Duration::from_micros(50));
			} else {
				std::thread::yield_now();
			}
		} else {
			early = 0;
		}
	}
}

/// Metal's poller sleeps this long between polls that found the next submission not done
/// ([`poll_loop`]), for its first [`BRIEF_NAPS`] such polls in a row: a small view is back within a
/// few of them.
#[cfg(target_vendor = "apple")]
const BRIEF_NAP: Duration = Duration::from_micros(50);
#[cfg(target_vendor = "apple")]
const BRIEF_NAPS: u32 = 40;
/// And this long after them: the GPU is on something long, and a poll four times a millisecond still
/// answers sooner than the 1 ms sleeps of wgpu's own wait did.
#[cfg(target_vendor = "apple")]
const LONG_NAP: Duration = Duration::from_micros(250);

/// Queued writes past this are submitted and waited for ([`Gpu::uploaded`]).
const UPLOAD_FLUSH_BYTES: u64 = 64 << 20;

/// Device-local GPU memory in bytes ([`Gpu::vram`]).
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Vram {
	pub usage: u64,
	pub budget: u64,
}

/// A pipeline cache file past this is started again (it keeps every shader version it has seen).
const MAX_PIPELINE_CACHE: usize = 64 << 20;

/// Whether this process turned Vulkan's implicit layers off itself ([`headless_layers`]): the
/// processes it starts (rur) get the environment they had.
static LAYERS_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The instance ror's devices come from: wgpu's defaults without indirect-call validation, which a
/// release build turns on and which builds its validation pipelines with every device; ror issues no
/// indirect draws or dispatches.
fn instance_descriptor(backends: wgpu::Backends) -> wgpu::InstanceDescriptor {
	wgpu::InstanceDescriptor { backends, flags: wgpu::InstanceFlags::default() - wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL, ..Default::default() }
}

/// Vulkan's implicit layers are what other programs install for every Vulkan application: overlays and
/// capture tools (game overlays, shader-cache and screen-capture tools, GPU profilers). They hook a
/// window's presentation and a headless device has none, but the loader loads them all into every
/// process: ~140 MB of commit and ~0.15 s of the device's start (a viewport worker's, measured).
/// Off unless the caller chose layers (`VK_LOADER_LAYERS_DISABLE`) or keeps them (`ROR_VULKAN_LAYERS=1`,
/// for an Nsight capture; `ROR_RENDERDOC` names RenderDoc's layer itself, render::renderdoc).
fn headless_layers() {
	if std::env::var_os("VK_LOADER_LAYERS_DISABLE").is_none() && std::env::var("ROR_VULKAN_LAYERS").map_or(true, |value| value != "1") {
		// SAFETY: Windows' environment block is guarded by the system; read by the loader below
		unsafe { std::env::set_var("VK_LOADER_LAYERS_DISABLE", "~implicit~") };
		LAYERS_OFF.store(true, std::sync::atomic::Ordering::Relaxed);
	}
}

/// The environment variables of [`headless_layers`] this process set itself, for a child process
/// that should load Vulkan as it would have.
pub(crate) fn own_loader_settings() -> &'static [&'static str] {
	if LAYERS_OFF.load(std::sync::atomic::Ordering::Relaxed) { &["VK_LOADER_LAYERS_DISABLE"] } else { &[] }
}

/// A process the render service does not start (`--no-service`, examples, calibration) loads only the
/// Vulkan driver of the GPU a service last started on, as the client starts the service
/// (service::RememberedGpu: the loader otherwise starts every GPU's driver, an integrated GPU's for 0.5-2 s
/// beside a discrete one), unless the caller chose drivers or a backend or `ROR_VULKAN_FILTER=0`.
/// Returns the adapter that driver should give; [`Gpu::create`] loads every driver when it does not.
fn remembered_driver() -> Option<String> {
	if ["VK_LOADER_DRIVERS_SELECT", "ROR_BACKEND"].iter().any(|name| std::env::var_os(name).is_some()) || std::env::var("ROR_VULKAN_FILTER").is_ok_and(|value| value == "0") {
		return None;
	}
	let gpu = crate::service::remembered_gpu(&crate::output::state_dir())?;
	if gpu.drivers.is_empty() {
		return None;
	}
	// SAFETY: as in headless_layers; the same pair the client gives a filtered service, which rur's
	// start takes off again (render::rur)
	unsafe {
		std::env::set_var("VK_LOADER_DRIVERS_SELECT", &gpu.drivers);
		std::env::set_var("ROR_VULKAN_ADAPTER", &gpu.adapter);
	}
	Some(gpu.adapter)
}

/// The instance and adapter a device is made on ([`Gpu::on_adapter`]): a process that makes its device
/// again keeps them (the render service, dormant between jobs), and so skips the loader's and the
/// driver's start.
#[derive(Clone)]
pub struct GpuAdapter {
	pub instance: wgpu::Instance,
	pub adapter: wgpu::Adapter,
}

impl GpuAdapter {
	/// The adapter [`Gpu::new`] would choose.
	pub fn find() -> Result<Self> {
		let (instance, adapter) = pollster::block_on(find_adapter())?;
		Ok(Self { instance, adapter })
	}
}

/// The allocator's block sizes devices are made with (`ROR_MEMORY_HINTS`).
fn memory_hints() -> wgpu::MemoryHints {
	match std::env::var("ROR_MEMORY_HINTS").ok().as_deref() {
		Some("usage") => wgpu::MemoryHints::MemoryUsage,
		// `manual:<first MB>-<last MB>`: the allocator's blocks from the first size doubling up to the last
		Some(manual) if manual.starts_with("manual:") => {
			let (first, last) = manual["manual:".len()..].split_once('-').unwrap_or(("128", "512"));
			let mb = |text: &str| text.trim().parse::<u64>().unwrap_or(128) << 20;
			wgpu::MemoryHints::Manual { suballocated_device_memory_block_size: mb(first)..mb(last) }
		}
		Some("performance") => wgpu::MemoryHints::Performance,
		// The allocator's shared blocks from 16 MiB doubling up to 128 MiB (Performance: 128 to 512):
		// each memory type's first block is committed as soon as anything small lands in it (a
		// 200-byte staging buffer at device creation, a uniform, a 64 x 64 image's textures), and a
		// block goes back only once empty. Every perf suite job's peak commit fell 80-165 MB with
		// these (a 64 x 64 still 580 -> 489, a unit 723 -> 609, a VFX clip 876 -> 712), the idle
		// service's 413 -> 312 MB, none slower; MemoryUsage (8 to 64, and smaller blocks of their
		// own for render targets) made clips 5-15 % slower.
		_ => wgpu::MemoryHints::Manual { suballocated_device_memory_block_size: (16 << 20)..(128 << 20) },
	}
}

/// The instance and adapter of ror's devices: Vulkan's discrete GPU when there is one (the remembered
/// GPU's driver alone first), else what every backend gives.
async fn find_adapter() -> Result<(wgpu::Instance, wgpu::Adapter)> {
	let options = wgpu::RequestAdapterOptions {
		power_preference: wgpu::PowerPreference::HighPerformance,
		compatible_surface: None,
		force_fallback_adapter: false,
	};
	let given = match std::env::var("ROR_BACKEND").ok().as_deref() {
		Some("vulkan") => Some(wgpu::Backends::VULKAN),
		Some("dx12") => Some(wgpu::Backends::DX12),
		Some("all") => Some(wgpu::Backends::PRIMARY),
		_ => None,
	};
	super::renderdoc::load();
	headless_layers();
	// Vulkan first: a discrete GPU there is what every backend together would choose (wgpu prefers
	// Vulkan, and a discrete GPU for HighPerformance)
	if given.is_none() {
		let remembered = remembered_driver();
		let instance = wgpu::Instance::new(&instance_descriptor(wgpu::Backends::VULKAN));
		if let Ok(adapter) = instance.request_adapter(&options).await
			&& adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu
			&& remembered.as_ref().is_none_or(|name| adapter.get_info().name == *name)
		{
			return Ok((instance, adapter));
		} else if remembered.is_some() {
			// the remembered GPU's driver did not give it: every driver, as before
			drop(instance);
			// SAFETY: Windows' environment block is guarded by the system; nothing here reads it concurrently
			unsafe {
				std::env::remove_var("VK_LOADER_DRIVERS_SELECT");
				std::env::remove_var("ROR_VULKAN_ADAPTER");
			}
			let instance = wgpu::Instance::new(&instance_descriptor(wgpu::Backends::VULKAN));
			if let Ok(adapter) = instance.request_adapter(&options).await
				&& adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu
			{
				return Ok((instance, adapter));
			}
		}
	}
	let instance = wgpu::Instance::new(&instance_descriptor(given.unwrap_or(wgpu::Backends::PRIMARY)));
	let adapter = match instance.request_adapter(&options).await {
		Ok(adapter) => adapter,
		Err(_) => instance
			.request_adapter(&wgpu::RequestAdapterOptions { force_fallback_adapter: true, ..options })
			.await
			.context("no GPU adapter (native or software) is available")?,
	};
	Ok((instance, adapter))
}

/// Whether the device's queue asks for a low priority among every program's GPU work
/// (VK_KHR_global_priority): `ROR_GPU_PRIORITY=low`, or `gaming` for while a game runs
/// ([`crate::gaming`]); medium, the driver's default, otherwise. NVIDIA's driver puts a low queue in
/// Windows' lower hardware-queue band (band 0 against 1, DxgKrnl's ChangeHwContextPriority), but beside
/// the game proxy with a ror clip every 5 s it did not keep the proxy's frames shorter (2026-11-02, two
/// runs each, others' GPU work in both: GPU-frame p99 22.4 and 14.7 ms low, 13.9 and 16.4 medium), and
/// a monitor's idle GPU priority for background work, the same band, had stretched them (21.5-22.0 against
/// 15.4-15.7 ms). The priority is fixed for a device's life: a service makes its device again on its
/// next wake.
pub fn wants_low_priority() -> bool {
	match std::env::var("ROR_GPU_PRIORITY").ok().as_deref().map(str::trim) {
		Some("low") => true,
		Some("gaming") => crate::gaming::active(),
		_ => false,
	}
}

/// A device whose queue has the low global priority ([`wants_low_priority`]), where Vulkan's driver
/// offers it; `None` to make the device as usual.
fn low_priority_device(adapter: &wgpu::Adapter, descriptor: &wgpu::DeviceDescriptor<'_>) -> Option<(wgpu::Device, wgpu::Queue)> {
	#[cfg(target_vendor = "apple")]
	{
		let _ = (adapter, descriptor);
		None
	}
	#[cfg(not(target_vendor = "apple"))]
	{
		use ash::vk;
		if !wants_low_priority() {
			return None;
		}
		// SAFETY: the adapter's own Vulkan objects; the callback only adds an extension the device supports
		// and a structure to the queue's creation, which lives until the device is made
		unsafe {
			let opened = {
				let hal = adapter.as_hal::<wgpu::hal::api::Vulkan>()?;
				let capabilities = hal.physical_device_capabilities();
				let extension = [ash::khr::global_priority::NAME, ash::ext::global_priority::NAME].into_iter().find(|name| capabilities.supports_extension(name))?;
				let mut priority = vk::DeviceQueueGlobalPriorityCreateInfoKHR::default().global_priority(vk::QueueGlobalPriorityKHR::LOW);
				let priority = &mut priority;
				let callback: Box<wgpu::hal::vulkan::CreateDeviceCallback<'_>> = Box::new(move |args| {
					args.extensions.push(extension);
					if let Some(queue) = args.queue_create_infos.first_mut() {
						*queue = queue.push_next(priority);
					}
				});
				hal.open_with_callback(descriptor.required_features, &descriptor.memory_hints, Some(callback))
					.inspect_err(|error| eprintln!("ror: a low-priority GPU queue: {error}; the device is made as usual"))
					.ok()?
			};
			adapter.create_device_from_hal(opened, descriptor).inspect_err(|error| eprintln!("ror: a low-priority GPU device: {error}; the device is made as usual")).ok()
		}
	}
}

impl Gpu {
	pub fn new() -> Result<Self> {
		Self::with_memory_hints(memory_hints())
	}

	/// A device on `found` as [`Gpu::new`] makes it.
	pub fn on_adapter(found: &GpuAdapter) -> Result<Self> {
		pollster::block_on(Self::create(found.instance.clone(), found.adapter.clone(), memory_hints()))
	}

	/// A device whose allocator sizes its memory blocks by `memory_hints` (`MemoryUsage`: small blocks, for a process that renders small images and stays up between them).
	pub fn with_memory_hints(memory_hints: wgpu::MemoryHints) -> Result<Self> {
		let found = GpuAdapter::find()?;
		pollster::block_on(Self::create(found.instance, found.adapter, memory_hints))
	}

	async fn create(instance: wgpu::Instance, adapter: wgpu::Adapter, memory_hints: wgpu::MemoryHints) -> Result<Self> {
		let mut wanted = wgpu::Features::TEXTURE_COMPRESSION_BC | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | wgpu::Features::PIPELINE_CACHE;
		if std::env::var("ROR_GPU_PROFILE").is_ok_and(|value| value == "1") {
			wanted |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
		}
		if std::env::var("ROR_PIPELINE_CACHE").is_ok_and(|value| value == "0") {
			wanted.remove(wgpu::Features::PIPELINE_CACHE);
		}
		let features = adapter.features() & wanted;
		let mut limits = wgpu::Limits::default();
		let supported = adapter.limits();
		limits.max_texture_dimension_2d = supported.max_texture_dimension_2d.min(16384);
		limits.max_buffer_size = supported.max_buffer_size;
		limits.max_storage_buffer_binding_size = supported.max_storage_buffer_binding_size;
		// local lights' shadow arrays take six layers a light, up to MAX_SHADOW_LIGHTS (wgpu's default
		// 256 held 42: a map's gameplay views reach more and the render panicked)
		limits.max_texture_array_layers = supported.max_texture_array_layers.min(2048);
		let descriptor = wgpu::DeviceDescriptor { label: Some("ror device"), required_features: features, required_limits: limits, memory_hints, ..Default::default() };
		let (device, queue) = match low_priority_device(&adapter, &descriptor) {
			Some(made) => made,
			None => adapter.request_device(&descriptor).await.context("creating the GPU device")?,
		};
		let adapter_info = adapter.get_info();
		let mut read_len = 0;
		let (pipeline_cache, cache_file) = if features.contains(wgpu::Features::PIPELINE_CACHE)
			&& let Some(key) = wgpu::util::pipeline_cache_key(&adapter_info)
		{
			let name = format!("pipelines/{key}");
			let state = crate::output::state_dir();
			let file = state.join(&name);
			// a state dir without one starts from the content store's (what another build or checkout
			// made on this machine: the driver checks it is its own)
			let data = crate::output::read_state_file(&state, &name).filter(|data| data.len() <= MAX_PIPELINE_CACHE);
			read_len = data.as_ref().map_or(0, Vec::len);
			// SAFETY: the data is what `get_data` wrote for an adapter with the same key (the file name);
			// `fallback` makes a corrupt or foreign file an empty cache
			let cache = unsafe { device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor { label: Some("ror pipelines"), data: data.as_deref(), fallback: true }) };
			(Some(cache), Some(file))
		} else {
			(None, None)
		};
		// what the file holds, unless the driver refused it (a driver update's pipelineCacheUUID: it starts
		// from an empty cache, and the file is written again once this process's pipelines are in it; kept
		// as written, a stale file that no later process outgrew made every process compile every
		// pipeline, ~0.6 s, as a driver given a cache of the application's own keeps none of its own)
		let file_len = cache_file.as_ref().and_then(|file| std::fs::metadata(file).ok()).map_or(0, |metadata| metadata.len() as usize);
		let loaded = pipeline_cache.as_ref().and_then(|cache| cache.get_data()).map_or(0, |data| data.len());
		let written = if loaded >= file_len { file_len } else { 0 };
		let store_stale = std::sync::atomic::AtomicBool::new(loaded < read_len);
		let timeline = if adapter_info.backend == wgpu::Backend::Vulkan { Timeline::new(&adapter, &device) } else { None };
		let tracker = Arc::new(Tracker {
			state: Mutex::new(Track::default()),
			changed: Condvar::new(),
			work: Condvar::new(),
			poll_lock: Mutex::new(()),
			submitting: Mutex::new(0),
			timeline,
			timeline_off: std::sync::atomic::AtomicBool::new(false),
		});
		Ok(Self {
			adapter_info,
			instance,
			adapter,
			device,
			queue,
			features,
			pipeline_cache,
			cache_file,
			cache_written: Mutex::new(written),
			store_stale,
			pending_uploads: std::sync::atomic::AtomicU64::new(0),
			tracker,
			poller: Mutex::new(None),
			adapter_memory: std::sync::OnceLock::new(),
		})
	}

	/// Counts `bytes` written through the queue (`write_texture`, `write_buffer`); past
	/// UPLOAD_FLUSH_BYTES since the last submission, submits and waits so wgpu frees their staging
	/// copies. Each queued write keeps a copy in staging memory until a submission carries it to the
	/// GPU: a map's textures loaded in one go held 423 MB of them, on top of the textures themselves.
	pub fn uploaded(&self, bytes: u64) {
		use std::sync::atomic::Ordering;
		if self.pending_uploads.fetch_add(bytes, Ordering::Relaxed) + bytes < UPLOAD_FLUSH_BYTES {
			return;
		}
		let submission = self.submit([]);
		let _ = self.wait_for(submission);
	}

	/// Submits `buffers` (with the writes queued before them): every submission of ror's goes through
	/// here, numbered in the queue's order, so that [`Gpu::wait_for`] can wait for it without polling.
	pub fn submit(&self, buffers: impl IntoIterator<Item = wgpu::CommandBuffer>) -> Submitted {
		self.submit_then(buffers, None::<fn()>)
	}

	/// [`Gpu::submit`], `done` called once the GPU has run the submission (from the thread that polls
	/// then: keep it short).
	pub fn submit_then<F: FnOnce() + Send + 'static>(&self, buffers: impl IntoIterator<Item = wgpu::CommandBuffer>, done: Option<F>) -> Submitted {
		use std::sync::atomic::Ordering;
		let tracker = &self.tracker;
		let mut last = tracker.submitting.lock();
		let number = *last + 1;
		#[cfg(not(target_vendor = "apple"))]
		if let Some(timeline) = tracker.timeline()
			// SAFETY: the queue's own Vulkan object; the semaphore goes with the next batch, this one
			// (every submission is made under `submitting`)
			&& let Some(queue) = unsafe { self.queue.as_hal::<wgpu::hal::api::Vulkan>() }
		{
			queue.add_signal_semaphore(timeline.semaphore, Some(number));
		}
		let buffers: Vec<wgpu::CommandBuffer> = buffers.into_iter().collect();
		if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.queue.submit(buffers))) {
			// its signal may be left for the next batch: the poller stops trusting the semaphore
			tracker.timeline_off.store(true, Ordering::Relaxed);
			drop(last);
			std::panic::resume_unwind(panic);
		}
		*last = number;
		let marked = tracker.clone();
		self.queue.on_submitted_work_done(move || {
			if let Some(done) = done {
				done();
			}
			marked.finished(number);
		});
		drop(last);
		// the writes queued before went with it
		self.pending_uploads.store(0, Ordering::Relaxed);
		{
			let mut state = tracker.state.lock();
			state.submitted = state.submitted.max(number);
		}
		self.start_poller();
		tracker.work.notify_one();
		Submitted(number)
	}

	fn start_poller(&self) {
		let mut poller = self.poller.lock();
		if poller.is_none() {
			let (device, tracker) = (self.device.clone(), self.tracker.clone());
			*poller = Some(std::thread::Builder::new().name("ror-gpu-poller".into()).spawn(move || poll_loop(device, tracker)).expect("the GPU poller thread"));
		}
	}

	/// Waits until the GPU has done `submission` (60 s at most), its callbacks fired: the readbacks it
	/// carried can be mapped. The poller polls for it; nothing here holds wgpu's locks.
	pub fn wait_for(&self, submission: Submitted) -> Result<(), wgpu::PollError> {
		let deadline = Instant::now() + Duration::from_secs(60);
		let mut state = self.tracker.state.lock();
		while state.done < submission.0 {
			if self.tracker.changed.wait_until(&mut state, deadline).timed_out() && state.done < submission.0 {
				return Err(wgpu::PollError::Timeout);
			}
		}
		Ok(())
	}

	/// Waits for `submission`, then for the mapping of a buffer it wrote (`ready`, the receiver of its
	/// `map_async` callback, asked for after the submission): a mapping asked for once its submission
	/// was already done fires at the next poll, made here.
	pub fn wait_mapped<T>(&self, submission: Submitted, ready: &std::sync::mpsc::Receiver<T>) -> Result<T> {
		self.wait_for(submission).map_err(|error| anyhow::anyhow!("GPU wait: {error}"))?;
		match ready.try_recv() {
			Ok(result) => Ok(result),
			Err(std::sync::mpsc::TryRecvError::Disconnected) => Err(anyhow::anyhow!("readback channel closed")),
			Err(std::sync::mpsc::TryRecvError::Empty) => {
				let _ = self.poll(wgpu::PollType::Poll);
				ready.recv_timeout(Duration::from_secs(60)).map_err(|_| anyhow::anyhow!("readback: no answer"))
			}
		}
	}

	/// `device.poll`, one at a time in the process. wgpu-core 27 asserts after a wait that the queue it
	/// finds empty has finished the submission waited for; two polls at once can break that: one reads
	/// the device's fence just before the GPU finishes the last submission, the other just after, and
	/// the second empties the queue before the first looks at it, which then panics with its stale
	/// value ("If the queue is empty, the current submission index (N) should be at least the wait
	/// submission index (N + 1)"). Eight clips at once through the service lost one job in six that way
	/// ("the encoder panicked"). A queue submission still runs its own poll beside these, but only
	/// right after its own work is queued, which keeps the queue from being empty.
	pub fn poll(&self, poll_type: wgpu::PollType) -> Result<wgpu::PollStatus, wgpu::PollError> {
		let _one = self.tracker.poll_lock.lock();
		self.device.poll(poll_type)
	}

	pub fn supports_bc(&self) -> bool {
		self.features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
	}

	/// The GPU's own memory as the system shares it out (Vulkan's VK_EXT_memory_budget): what this
	/// process uses and the budget the system gives it, which shrinks as other programs (a game) take
	/// more. `None` on other backends or without the extension.
	#[cfg(target_vendor = "apple")]
	pub fn vram(&self) -> Option<Vram> {
		None
	}

	#[cfg(not(target_vendor = "apple"))]
	pub fn vram(&self) -> Option<Vram> {
		// SAFETY: the adapter's own instance and physical device, read without changing anything
		unsafe {
			let adapter = self.adapter.as_hal::<wgpu::hal::api::Vulkan>()?;
			if !adapter.physical_device_capabilities().supports_extension(ash::ext::memory_budget::NAME) {
				return None;
			}
			let instance = adapter.shared_instance().raw_instance();
			let mut budget = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
			let mut properties = ash::vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
			instance.get_physical_device_memory_properties2(adapter.raw_physical_device(), &mut properties);
			let memory = properties.memory_properties;
			let mut vram = Vram::default();
			for (index, heap) in memory.memory_heaps[..memory.memory_heap_count as usize].iter().enumerate() {
				if heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL) {
					vram.usage += budget.heap_usage[index];
					vram.budget += budget.heap_budget[index];
				}
			}
			(vram.budget > 0).then_some(vram)
		}
	}

	/// The GPU memory free for this process to draw into: what its budget leaves ([`Gpu::vram`]) and, on
	/// Windows, what the adapter has free with every program's use counted ([`AdapterMemory`]); with the
	/// GPU's own memory. The budget alone stays near the whole GPU while other programs hold much of it
	/// (15.3 GB of a 16 GB GPU beside 6.5 GB of theirs, 2026-11-02): several services
	/// drawing by it would take the same memory twice. None when neither is known.
	pub fn free_memory(&self) -> Option<crate::budget::GpuFree> {
		let budget = self.vram().map(|vram| crate::budget::GpuFree { free: vram.budget.saturating_sub(vram.usage), total: vram.budget });
		let adapter = self.adapter_memory.get_or_init(|| self.open_adapter_memory()).as_ref().and_then(|memory| Some(crate::budget::GpuFree { free: memory.free()?, total: memory.size }));
		match (budget, adapter) {
			(Some(budget), Some(adapter)) => Some(crate::budget::GpuFree { free: budget.free.min(adapter.free), total: adapter.total }),
			(budget, adapter) => budget.or(adapter),
		}
	}

	/// The adapter's memory counter, by the LUID Vulkan gives (VkPhysicalDeviceIDProperties), and its
	/// largest device-local heap.
	#[cfg(windows)]
	fn open_adapter_memory(&self) -> Option<AdapterMemory> {
		// SAFETY: the adapter's own instance and physical device, read without changing anything
		let (luid, size) = unsafe {
			let adapter = self.adapter.as_hal::<wgpu::hal::api::Vulkan>()?;
			let instance = adapter.shared_instance().raw_instance();
			let mut id = ash::vk::PhysicalDeviceIDProperties::default();
			let mut properties = ash::vk::PhysicalDeviceProperties2::default().push_next(&mut id);
			instance.get_physical_device_properties2(adapter.raw_physical_device(), &mut properties);
			if id.device_luid_valid != ash::vk::TRUE {
				return None;
			}
			let memory = instance.get_physical_device_memory_properties(adapter.raw_physical_device());
			let size = memory.memory_heaps[..memory.memory_heap_count as usize].iter().filter(|heap| heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL)).map(|heap| heap.size).max()?;
			(id.device_luid, size)
		};
		let low = u32::from_le_bytes([luid[0], luid[1], luid[2], luid[3]]);
		let high = u32::from_le_bytes([luid[4], luid[5], luid[6], luid[7]]);
		let counter = pdh::Counter::open(&format!("\\GPU Adapter Memory(luid_0x{high:08X}_0x{low:08X}_phys_0)\\Dedicated Usage"))?;
		Some(AdapterMemory { counter: Mutex::new(counter), size })
	}

	#[cfg(not(windows))]
	fn open_adapter_memory(&self) -> Option<AdapterMemory> {
		None
	}

	/// Stops the poller and lets go of the timeline once the GPU is idle (a process's renderer is
	/// dropped only by tests and examples; the service exits without).
	fn stop(&mut self) {
		self.tracker.state.lock().stopped = true;
		self.tracker.work.notify_all();
		if let Some(poller) = self.poller.get_mut().take() {
			let _ = poller.join();
		}
		if let Some(timeline) = &self.tracker.timeline {
			let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_secs(10)) });
			// SAFETY: no batch left to signal it (the queue is idle) and no wait on it (the poller ended)
			unsafe { timeline.device.destroy_semaphore(timeline.semaphore, None) };
		}
	}

	/// Writes the pipeline cache to disk when it holds more than the file (after the renderer made its
	/// pipelines, and after any made later).
	pub fn save_pipeline_cache(&self) {
		let (Some(cache), Some(file)) = (&self.pipeline_cache, &self.cache_file) else {
			return;
		};
		let Some(data) = cache.get_data() else {
			return;
		};
		let mut written = self.cache_written.lock();
		if data.len() <= *written || data.len() > MAX_PIPELINE_CACHE {
			return;
		}
		if let Some(parent) = file.parent() {
			let _ = std::fs::create_dir_all(parent);
		}
		if crate::output::write_atomic(file, &data).is_ok() {
			*written = data.len();
		}
		// the content store keeps the largest a process made (and one the driver took over one it refused)
		if let Some(store) = crate::output::store_dir()
			&& let Some(name) = file.file_name()
		{
			let kept = store.join("pipelines").join(name);
			let stale = self.store_stale.load(std::sync::atomic::Ordering::Relaxed);
			if (stale || std::fs::metadata(&kept).map_or(0, |metadata| metadata.len() as usize) < data.len()) && crate::output::write_atomic(&kept, &data).is_ok() {
				self.store_stale.store(false, std::sync::atomic::Ordering::Relaxed);
			}
		}
	}
}

impl Drop for Gpu {
	fn drop(&mut self) {
		self.stop();
	}
}

#[cfg(all(test, target_vendor = "apple"))]
mod tests {
	use std::time::{Duration, Instant};

	use super::Gpu;

	/// A little work is waited for in a fraction of a millisecond, alone and beside another thread's:
	/// the poller looks every 50 µs and holds wgpu's fence only while it looks. Waiting inside the
	/// poll, it slept 1 ms with the fence held, so no wait was shorter than that and a submission made
	/// meanwhile waited for the fence first.
	#[test]
	fn a_small_submission_is_back_within_a_millisecond() {
		let gpu = Gpu::new().expect("a GPU");
		let buffer = |usage| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("poller test"), size: 4096, usage, mapped_at_creation: false });
		let (from, to) = (buffer(wgpu::BufferUsages::COPY_SRC), buffer(wgpu::BufferUsages::COPY_DST));
		// something for the GPU to do, submitted and waited for: the middle one of 200 such waits
		let median = || {
			let mut took: Vec<Duration> = (0..200)
				.map(|_| {
					let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
					encoder.copy_buffer_to_buffer(&from, 0, &to, 0, Some(4096));
					let started = Instant::now();
					gpu.wait_for(gpu.submit([encoder.finish()])).expect("the GPU did the work");
					started.elapsed()
				})
				.collect();
			took.sort();
			took[took.len() / 2]
		};
		let alone = median();
		let beside = std::thread::scope(|scope| {
			let other = scope.spawn(median);
			(median(), other.join().expect("the other thread"))
		});
		assert!(alone < Duration::from_micros(600), "half of 200 waits took {alone:?} or more");
		assert!(beside.0.max(beside.1) < Duration::from_micros(600), "half of 200 waits took {beside:?} or more beside another thread's");
	}
}
