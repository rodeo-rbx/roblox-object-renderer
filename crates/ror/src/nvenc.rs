//! NVENC in the process: NVIDIA's hardware H.264 encoder, through the driver's own library and a CUDA
//! context, taking an MP4 clip's NV12 frames straight from their readback (docs/video.md). The encoding
//! is the GPU's encoder block's work, so a frame costs the CPU one copy into the driver's buffer, where
//! x264 takes ~12 ms of a core for a 1440p frame and a copy through ffmpeg a pipe and a process more.
//!
//! The driver's structures are large and mostly reserved space: each is a zeroed block here, its fields
//! written and read at their offsets in NVENC API 12.0's nvEncodeAPI.h (drivers 522.25 and later take
//! that version), as a C program including that header printed them (sizeof, offsetof, a bit set to
//! find each bitfield).

use std::ffi::{CStr, c_char, c_void};
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// How an NVENC clip spends its bits (`ror video --nvenc`): constant quality, under a ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rate {
	/// The constant quality aimed for, 1-51 (lower is better; 19-21 looks like the clip drawn).
	pub quality: u32,
	/// The most kilobits a second it may spend (a file that must fit a size); none: what the quality takes.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub kbps: Option<u32>,
}

impl Rate {
	/// `--nvenc`'s text: `<quality>` or `<quality>:<kbps>`.
	pub fn parse(text: &str) -> Result<Rate> {
		let (quality, kbps) = match text.split_once(':') {
			Some((quality, kbps)) => (quality, Some(kbps.trim().parse::<u32>().with_context(|| format!("--nvenc: {kbps:?} is not a bitrate in kbit/s"))?)),
			None => (text, None),
		};
		let quality = quality.trim().parse::<u32>().with_context(|| format!("--nvenc: {quality:?} is not a quality (1-51)"))?;
		if !(1..=51).contains(&quality) || kbps == Some(0) {
			bail!("--nvenc takes a quality of 1-51 and a bitrate above 0 (got {text:?})");
		}
		Ok(Rate { quality, kbps })
	}
}

/// What an encoder is opened with.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
	pub width: u32,
	pub height: u32,
	/// Frames a second, as a fraction.
	pub fps: (u32, u32),
	pub rate: Rate,
}

/// The stream's parameter sets, without start codes or lengths.
pub struct Headers {
	pub sps: Vec<u8>,
	pub pps: Vec<u8>,
}

/// A frame out of the encoder, in decode order: its NAL units with their 4-byte lengths (parameter sets
/// left out) and its place among the clip's frames as shown.
pub struct Packet<'a> {
	pub data: &'a [u8],
	pub keyframe: bool,
	pub display: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Guid(u32, u16, u16, [u8; 8]);

const API_VERSION: u32 = 0x0000_000c;
const FUNCTION_LIST_VER: u32 = 0x7002_000c;
const SESSION_VER: u32 = 0x7001_000c;
const PRESET_CONFIG_VER: u32 = 0xf004_000c;
const CONFIG_VER: u32 = 0xf008_000c;
const INITIALIZE_VER: u32 = 0xf005_000c;
const CREATE_INPUT_VER: u32 = 0x7001_000c;
const CREATE_BITSTREAM_VER: u32 = 0x7001_000c;
const LOCK_INPUT_VER: u32 = 0x7001_000c;
const PIC_VER: u32 = 0xf006_000c;
const LOCK_BITSTREAM_VER: u32 = 0x7002_000c;
const SEQUENCE_VER: u32 = 0x7001_000c;

// each structure's size in 8-byte words
const FUNCTION_LIST_WORDS: usize = 2552 / 8;
const SESSION_WORDS: usize = 1552 / 8;
const PRESET_CONFIG_WORDS: usize = 5128 / 8;
const CONFIG_WORDS: usize = 3584 / 8;
const INITIALIZE_WORDS: usize = 1808 / 8;
const CREATE_BUFFER_WORDS: usize = 776 / 8;
const LOCK_WORDS: usize = 1544 / 8;
const PIC_WORDS: usize = 3360 / 8;

// NV_ENCODE_API_FUNCTION_LIST's entries
const FN_INITIALIZE: usize = 96;
const FN_CREATE_INPUT: usize = 104;
const FN_DESTROY_INPUT: usize = 112;
const FN_CREATE_BITSTREAM: usize = 120;
const FN_DESTROY_BITSTREAM: usize = 128;
const FN_ENCODE: usize = 136;
const FN_LOCK_BITSTREAM: usize = 144;
const FN_UNLOCK_BITSTREAM: usize = 152;
const FN_LOCK_INPUT: usize = 160;
const FN_UNLOCK_INPUT: usize = 168;
const FN_SEQUENCE_PARAMS: usize = 184;
const FN_DESTROY: usize = 224;
const FN_OPEN_SESSION_EX: usize = 240;
const FN_LAST_ERROR: usize = 304;
const FN_PRESET_CONFIG_EX: usize = 320;

const SESSION_DEVICE_TYPE: usize = 4;
const SESSION_DEVICE: usize = 8;
const SESSION_API_VERSION: usize = 24;
const PRESET_CONFIG: usize = 8;
const CONFIG_PROFILE: usize = 4;
const CONFIG_GOP_LENGTH: usize = 20;
const CONFIG_FRAME_INTERVAL_P: usize = 24;
const CONFIG_RC: usize = 40;
const CONFIG_CODEC: usize = 168;
const RC_MODE: usize = 4;
const RC_AVERAGE_BITRATE: usize = 20;
const RC_MAX_BITRATE: usize = 24;
const RC_VBV_BUFFER: usize = 28;
const RC_VBV_INITIAL_DELAY: usize = 32;
const RC_ENABLE_AQ: (usize, u32) = (36, 3);
const RC_ENABLE_LOOKAHEAD: (usize, u32) = (36, 5);
const RC_AQ_STRENGTH: (usize, u32) = (36, 12);
const RC_TARGET_QUALITY: usize = 88;
const RC_TARGET_QUALITY_LSB: usize = 89;
const RC_LOOKAHEAD_DEPTH: usize = 90;
const RC_MULTI_PASS: usize = 100;
const H264_OUTPUT_AUD: (usize, u32) = (0, 6);
const H264_REPEAT_SPSPPS: (usize, u32) = (0, 12);
const H264_IDR_PERIOD: usize = 8;
const H264_VUI: usize = 72;
const VUI_SIGNAL_PRESENT: usize = 8;
const VUI_VIDEO_FORMAT: usize = 12;
const VUI_FULL_RANGE: usize = 16;
const VUI_COLOUR_PRESENT: usize = 20;
const VUI_PRIMARIES: usize = 24;
const VUI_TRANSFER: usize = 28;
const VUI_MATRIX: usize = 32;
const VUI_CHROMA_LOCATION_PRESENT: usize = 36;
const VUI_CHROMA_LOCATION_TOP: usize = 40;
const VUI_CHROMA_LOCATION_BOTTOM: usize = 44;
const INIT_ENCODE_GUID: usize = 4;
const INIT_PRESET_GUID: usize = 20;
const INIT_WIDTH: usize = 36;
const INIT_HEIGHT: usize = 40;
const INIT_DAR_WIDTH: usize = 44;
const INIT_DAR_HEIGHT: usize = 48;
const INIT_RATE_NUM: usize = 52;
const INIT_RATE_DEN: usize = 56;
const INIT_PTD: usize = 64;
const INIT_CONFIG: usize = 88;
const INIT_MAX_WIDTH: usize = 96;
const INIT_MAX_HEIGHT: usize = 100;
const INIT_TUNING: usize = 136;
const INPUT_WIDTH: usize = 4;
const INPUT_HEIGHT: usize = 8;
const INPUT_FORMAT: usize = 16;
const INPUT_BUFFER: usize = 24;
const BITSTREAM_BUFFER: usize = 16;
const LOCK_INPUT_BUFFER: usize = 8;
const LOCK_INPUT_DATA: usize = 16;
const LOCK_INPUT_PITCH: usize = 24;
const PIC_WIDTH: usize = 4;
const PIC_HEIGHT: usize = 8;
const PIC_PITCH: usize = 12;
const PIC_FLAGS: usize = 16;
const PIC_FRAME_INDEX: usize = 20;
const PIC_TIMESTAMP: usize = 24;
const PIC_DURATION: usize = 32;
const PIC_INPUT: usize = 40;
const PIC_OUTPUT: usize = 48;
const PIC_FORMAT: usize = 64;
const PIC_STRUCT: usize = 68;
const LOCKED_OUTPUT: usize = 8;
const LOCKED_SIZE: usize = 36;
const LOCKED_TIMESTAMP: usize = 40;
const LOCKED_DATA: usize = 56;
const LOCKED_PICTURE_TYPE: usize = 64;
const SEQUENCE_BUFFER_SIZE: usize = 4;
const SEQUENCE_BUFFER: usize = 16;
const SEQUENCE_OUT_SIZE: usize = 24;

const CODEC_H264: Guid = Guid(0x6bc82762, 0x4e63, 0x4ca4, [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf]);
const PROFILE_HIGH: Guid = Guid(0xe7cbc309, 0x4f7a, 0x4b89, [0xaf, 0x2a, 0xd5, 0x37, 0xc9, 0x2b, 0xe3, 0x10]);
const PRESET_P3: Guid = Guid(0x36850110, 0x3a07, 0x441f, [0x94, 0xd5, 0x36, 0x70, 0x63, 0x1f, 0x91, 0xf6]);
const DEVICE_TYPE_CUDA: u32 = 1;
const BUFFER_FORMAT_NV12: u32 = 1;
const PIC_STRUCT_FRAME: u32 = 1;
const PIC_FLAG_EOS: u32 = 8;
const RC_VBR: u32 = 1;
const TUNING_HIGH_QUALITY: u32 = 1;
const PIC_TYPE_IDR: u32 = 3;
const TWO_PASS_QUARTER_RESOLUTION: u32 = 1;
const NEED_MORE_INPUT: u32 = 17;
/// Frames between keyframes, as ffmpeg's encoders have it.
const GOP: u32 = 250;

/// A zeroed structure of the driver's, `N` 8-byte words long, its fields at byte offsets.
#[repr(C, align(8))]
struct Raw<const N: usize>([u64; N]);

impl<const N: usize> Raw<N> {
	fn new(version: u32) -> Box<Self> {
		let mut raw = Box::new(Raw([0; N]));
		raw.put(0, version);
		raw
	}

	fn put<T: Copy>(&mut self, offset: usize, value: T) {
		assert!(offset + size_of::<T>() <= N * 8);
		// SAFETY: inside the block, as checked
		unsafe { self.0.as_mut_ptr().cast::<u8>().add(offset).cast::<T>().write_unaligned(value) }
	}

	fn get<T: Copy>(&self, offset: usize) -> T {
		assert!(offset + size_of::<T>() <= N * 8);
		// SAFETY: inside the block, as checked; T is plain data
		unsafe { self.0.as_ptr().cast::<u8>().add(offset).cast::<T>().read_unaligned() }
	}

	/// Sets a bitfield `width` bits wide in the 32-bit word at its offset.
	fn bits(&mut self, (offset, shift): (usize, u32), width: u32, value: u32) {
		let mask = ((1u32 << width) - 1) << shift;
		let word: u32 = self.get(offset);
		self.put(offset, (word & !mask) | ((value << shift) & mask));
	}

	fn pointer(&mut self) -> *mut c_void {
		self.0.as_mut_ptr().cast()
	}
}

type Status = u32;
type OpenSession = unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> Status;
type PresetConfig = unsafe extern "system" fn(*mut c_void, Guid, Guid, u32, *mut c_void) -> Status;
/// An entry taking the encoder and a structure or a buffer's handle.
type Call = unsafe extern "system" fn(*mut c_void, *mut c_void) -> Status;
type Destroy = unsafe extern "system" fn(*mut c_void) -> Status;
type LastError = unsafe extern "system" fn(*mut c_void) -> *const c_char;

/// The driver's encoder library and CUDA, loaded once, with the encoder's entry points.
struct Api {
	_encode: libloading::Library,
	cuda: libloading::Library,
	functions: Box<Raw<FUNCTION_LIST_WORDS>>,
}

// The libraries' entry points are plain functions, callable from any thread.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
	fn load() -> Result<Api> {
		let (encode_name, cuda_name) = if cfg!(windows) { ("nvEncodeAPI64.dll", "nvcuda.dll") } else { ("libnvidia-encode.so.1", "libcuda.so.1") };
		// SAFETY: NVIDIA's driver libraries, whose initialisers load nothing of ours
		let (encode, cuda) = unsafe { (libloading::Library::new(encode_name), libloading::Library::new(cuda_name)) };
		let encode = encode.map_err(|error| anyhow!("no NVIDIA encoder library ({encode_name}): {error}"))?;
		let cuda = cuda.map_err(|error| anyhow!("no CUDA library ({cuda_name}): {error}"))?;
		let mut functions = Raw::<FUNCTION_LIST_WORDS>::new(FUNCTION_LIST_VER);
		// SAFETY: the two exports as nvEncodeAPI.h declares them
		unsafe {
			let highest: libloading::Symbol<unsafe extern "system" fn(*mut u32) -> Status> = encode.get(b"NvEncodeAPIGetMaxSupportedVersion\0")?;
			let mut version = 0u32;
			if highest(&mut version) != 0 || version < (12 << 4) {
				bail!("the NVIDIA driver's encoder is version {}.{}: 12.0 or later is needed (driver 522.25)", version >> 4, version & 0xf);
			}
			let create: libloading::Symbol<unsafe extern "system" fn(*mut c_void) -> Status> = encode.get(b"NvEncodeAPICreateInstance\0")?;
			let status = create(functions.pointer());
			if status != 0 {
				bail!("NvEncodeAPICreateInstance failed ({status})");
			}
			let init: libloading::Symbol<unsafe extern "system" fn(u32) -> i32> = cuda.get(b"cuInit\0")?;
			let status = init(0);
			if status != 0 {
				bail!("cuInit failed ({status})");
			}
		}
		Ok(Api { _encode: encode, cuda, functions })
	}

	fn get() -> Result<&'static Api> {
		static API: OnceLock<Result<Api, String>> = OnceLock::new();
		API.get_or_init(|| Api::load().map_err(|error| format!("{error:#}"))).as_ref().map_err(|error| anyhow!("{error}"))
	}

	fn entry<T: Copy>(&self, offset: usize) -> Result<T> {
		let pointer: *const c_void = self.functions.get(offset);
		if pointer.is_null() {
			bail!("the NVIDIA driver's encoder lacks an entry point (offset {offset})");
		}
		// SAFETY: T is one of the function pointer types above, the entry at this offset its function
		Ok(unsafe { std::mem::transmute_copy::<*const c_void, T>(&pointer) })
	}
}

/// The process's CUDA context while any encoder is open: made for the first, destroyed with the last
/// (it holds ~100 MB of the process's memory and some of the GPU's).
struct CudaContext;

static CONTEXT: Mutex<(usize, usize)> = Mutex::new((0, 0));

impl CudaContext {
	fn take(api: &Api) -> Result<*mut c_void> {
		let mut held = CONTEXT.lock();
		if held.0 == 0 {
			// SAFETY: CUDA's driver API as cuda.h declares it; the context is popped off this thread at once
			unsafe {
				let device_get: libloading::Symbol<unsafe extern "system" fn(*mut i32, i32) -> i32> = api.cuda.get(b"cuDeviceGet\0")?;
				let create: libloading::Symbol<unsafe extern "system" fn(*mut *mut c_void, u32, i32) -> i32> = api.cuda.get(b"cuCtxCreate_v2\0")?;
				let pop: libloading::Symbol<unsafe extern "system" fn(*mut *mut c_void) -> i32> = api.cuda.get(b"cuCtxPopCurrent_v2\0")?;
				let mut device = 0i32;
				let status = device_get(&mut device, 0);
				if status != 0 {
					bail!("cuDeviceGet failed ({status}): no CUDA device");
				}
				let mut context = std::ptr::null_mut();
				let status = create(&mut context, 0, device);
				if status != 0 || context.is_null() {
					bail!("cuCtxCreate failed ({status})");
				}
				let mut popped = std::ptr::null_mut();
				pop(&mut popped);
				held.1 = context as usize;
			}
		}
		held.0 += 1;
		Ok(held.1 as *mut c_void)
	}

	fn release(api: &Api) {
		let mut held = CONTEXT.lock();
		held.0 = held.0.saturating_sub(1);
		if held.0 == 0 && held.1 != 0 {
			// SAFETY: the context made in `take`, no encoder left on it
			unsafe {
				if let Ok(destroy) = api.cuda.get::<unsafe extern "system" fn(*mut c_void) -> i32>(b"cuCtxDestroy_v2\0") {
					destroy(held.1 as *mut c_void);
				}
			}
			held.1 = 0;
		}
	}
}

/// B-frames between the frames they refer to, and the frames the rate control looks ahead: with both
/// a 1440p attack clip under a ceiling keeps 1.5 dB more than with neither, in a file a fifth smaller.
const B_FRAMES: u32 = 3;
const LOOKAHEAD: u16 = 8;
/// Buffers over the frames the encoder holds, as NVIDIA's own encoder class keeps.
const SPARE: usize = 3;

pub struct Encoder {
	api: &'static Api,
	session: *mut c_void,
	/// Each frame's input buffer and the bitstream buffer an encoded frame lands in, taken in turn.
	slots: Vec<(*mut c_void, *mut c_void)>,
	/// Frames sent, and read; a frame is read once the encoder has `delay` more (those it looks ahead
	/// and reorders), or the clip has ended.
	frames: usize,
	read: usize,
	delay: usize,
	ended: bool,
	width: u32,
	height: u32,
	/// The last frame read, as [`Packet`] gives it.
	packet: Vec<u8>,
}

// Used from one thread at a time; the driver's session is not bound to the thread that opened it.
unsafe impl Send for Encoder {}

impl Encoder {
	/// An encoder with B-frames and lookahead, or without where the GPU's encoder has neither.
	pub fn open(settings: &Settings) -> Result<Encoder> {
		match Encoder::open_with(settings, B_FRAMES, LOOKAHEAD) {
			Ok(encoder) => Ok(encoder),
			Err(first) => Encoder::open_with(settings, 0, 0).map_err(|_| first),
		}
	}

	fn open_with(settings: &Settings, b_frames: u32, lookahead: u16) -> Result<Encoder> {
		let Settings { width, height, fps: (fps_num, fps_den), rate } = *settings;
		if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) || fps_num == 0 || fps_den == 0 {
			bail!("nvenc: {width}x{height} at {fps_num}/{fps_den} fps (even sizes and a rate above 0)");
		}
		let api = Api::get()?;
		let context = CudaContext::take(api)?;
		let mut encoder = Encoder { api, session: std::ptr::null_mut(), slots: Vec::new(), frames: 0, read: 0, delay: b_frames as usize + lookahead as usize + SPARE, ended: false, width, height, packet: Vec::new() };
		let mut session = Raw::<SESSION_WORDS>::new(SESSION_VER);
		session.put(SESSION_DEVICE_TYPE, DEVICE_TYPE_CUDA);
		session.put(SESSION_DEVICE, context);
		session.put(SESSION_API_VERSION, API_VERSION);
		// SAFETY (every call below): the driver's entry points with the structures they take, each of its
		// declared size and version, alive across the call
		let status = unsafe { api.entry::<OpenSession>(FN_OPEN_SESSION_EX)?(session.pointer(), &mut encoder.session) };
		if status != 0 {
			// a session that failed to open is still destroyed (nvEncodeAPI.h), which Drop does
			bail!("nvenc: no encoder session ({status}: {})", if status == 10 { "the GPU's encoder has no session free" } else { "no NVIDIA GPU with an H.264 encoder, or a driver too old" });
		}
		let mut preset = Raw::<PRESET_CONFIG_WORDS>::new(PRESET_CONFIG_VER);
		preset.put(PRESET_CONFIG, CONFIG_VER);
		encoder.check(unsafe { api.entry::<PresetConfig>(FN_PRESET_CONFIG_EX)?(encoder.session, CODEC_H264, PRESET_P3, TUNING_HIGH_QUALITY, preset.pointer()) }, "the preset")?;
		let mut config = Raw::<CONFIG_WORDS>::new(CONFIG_VER);
		config.0.copy_from_slice(&preset.0[PRESET_CONFIG / 8..PRESET_CONFIG / 8 + CONFIG_WORDS]);
		config.put(0, CONFIG_VER);
		config.put(CONFIG_PROFILE, PROFILE_HIGH);
		config.put(CONFIG_GOP_LENGTH, GOP);
		config.put(CONFIG_FRAME_INTERVAL_P, b_frames as i32 + 1);
		// constant quality under the ceiling, as ffmpeg's `-rc vbr -cq Q -maxrate R -spatial-aq 1 -multipass qres -rc-lookahead L`
		let rc = CONFIG_RC;
		config.put(rc + RC_MODE, RC_VBR);
		config.put(rc + RC_AVERAGE_BITRATE, 0u32);
		config.put(rc + RC_VBV_BUFFER, 0u32);
		config.put(rc + RC_VBV_INITIAL_DELAY, 0u32);
		if let Some(kbps) = rate.kbps {
			config.put(rc + RC_MAX_BITRATE, kbps.saturating_mul(1000));
		}
		config.put(rc + RC_TARGET_QUALITY, rate.quality as u8);
		config.put(rc + RC_TARGET_QUALITY_LSB, 0u8);
		config.bits((rc + RC_ENABLE_AQ.0, RC_ENABLE_AQ.1), 1, 1);
		config.bits((rc + RC_AQ_STRENGTH.0, RC_AQ_STRENGTH.1), 4, 8);
		config.bits((rc + RC_ENABLE_LOOKAHEAD.0, RC_ENABLE_LOOKAHEAD.1), 1, u32::from(lookahead > 0));
		config.put(rc + RC_LOOKAHEAD_DEPTH, lookahead);
		config.put(rc + RC_MULTI_PASS, TWO_PASS_QUARTER_RESOLUTION);
		let h264 = CONFIG_CODEC;
		config.put(h264 + H264_IDR_PERIOD, GOP);
		config.bits((h264 + H264_OUTPUT_AUD.0, H264_OUTPUT_AUD.1), 1, 0);
		config.bits((h264 + H264_REPEAT_SPSPPS.0, H264_REPEAT_SPSPPS.1), 1, 0);
		// the colour as the frames are: BT.709 primaries and matrix, the sRGB transfer, limited range, centred chroma
		let vui = h264 + H264_VUI;
		config.put(vui + VUI_SIGNAL_PRESENT, 1u32);
		config.put(vui + VUI_VIDEO_FORMAT, 5u32);
		config.put(vui + VUI_FULL_RANGE, 0u32);
		config.put(vui + VUI_COLOUR_PRESENT, 1u32);
		config.put(vui + VUI_PRIMARIES, 1u32);
		config.put(vui + VUI_TRANSFER, 13u32);
		config.put(vui + VUI_MATRIX, 1u32);
		config.put(vui + VUI_CHROMA_LOCATION_PRESENT, 1u32);
		config.put(vui + VUI_CHROMA_LOCATION_TOP, 1u32);
		config.put(vui + VUI_CHROMA_LOCATION_BOTTOM, 1u32);
		let mut initialize = Raw::<INITIALIZE_WORDS>::new(INITIALIZE_VER);
		initialize.put(INIT_ENCODE_GUID, CODEC_H264);
		initialize.put(INIT_PRESET_GUID, PRESET_P3);
		initialize.put(INIT_WIDTH, width);
		initialize.put(INIT_HEIGHT, height);
		initialize.put(INIT_DAR_WIDTH, width);
		initialize.put(INIT_DAR_HEIGHT, height);
		initialize.put(INIT_RATE_NUM, fps_num);
		initialize.put(INIT_RATE_DEN, fps_den);
		initialize.put(INIT_PTD, 1u32);
		initialize.put(INIT_CONFIG, config.pointer());
		initialize.put(INIT_MAX_WIDTH, width);
		initialize.put(INIT_MAX_HEIGHT, height);
		initialize.put(INIT_TUNING, TUNING_HIGH_QUALITY);
		encoder.check(unsafe { api.entry::<Call>(FN_INITIALIZE)?(encoder.session, initialize.pointer()) }, "its settings")?;
		// a buffer for every frame sent and not read yet
		for _ in 0..encoder.delay + 1 {
			let mut input = Raw::<CREATE_BUFFER_WORDS>::new(CREATE_INPUT_VER);
			input.put(INPUT_WIDTH, width);
			input.put(INPUT_HEIGHT, height);
			input.put(INPUT_FORMAT, BUFFER_FORMAT_NV12);
			encoder.check(unsafe { api.entry::<Call>(FN_CREATE_INPUT)?(encoder.session, input.pointer()) }, "an input buffer")?;
			encoder.slots.push((input.get(INPUT_BUFFER), std::ptr::null_mut()));
			let mut output = Raw::<CREATE_BUFFER_WORDS>::new(CREATE_BITSTREAM_VER);
			encoder.check(unsafe { api.entry::<Call>(FN_CREATE_BITSTREAM)?(encoder.session, output.pointer()) }, "an output buffer")?;
			encoder.slots.last_mut().expect("just pushed").1 = output.get(BITSTREAM_BUFFER);
		}
		Ok(encoder)
	}

	fn check(&self, status: Status, what: &str) -> Result<()> {
		if status == 0 {
			return Ok(());
		}
		// SAFETY: the driver's own text for the session's last error, copied at once
		let said = match self.api.entry::<LastError>(FN_LAST_ERROR) {
			Ok(last) => unsafe {
				let text = last(self.session);
				if text.is_null() { String::new() } else { CStr::from_ptr(text).to_string_lossy().into_owned() }
			},
			Err(_) => String::new(),
		};
		bail!("nvenc: {what} failed ({status}{}{})", if said.is_empty() { "" } else { ": " }, said.trim())
	}

	pub fn headers(&mut self) -> Result<Headers> {
		let mut buffer = [0u8; 1024];
		let mut size = 0u32;
		let mut payload = Raw::<LOCK_WORDS>::new(SEQUENCE_VER);
		payload.put(SEQUENCE_BUFFER_SIZE, buffer.len() as u32);
		payload.put(SEQUENCE_BUFFER, buffer.as_mut_ptr());
		payload.put(SEQUENCE_OUT_SIZE, &mut size as *mut u32);
		// SAFETY: as in `open`; the buffer and the size outlive the call
		self.check(unsafe { self.api.entry::<Call>(FN_SEQUENCE_PARAMS)?(self.session, payload.pointer()) }, "the stream's parameter sets")?;
		let (mut sps, mut pps) = (Vec::new(), Vec::new());
		for unit in units(&buffer[..(size as usize).min(buffer.len())]) {
			match unit[0] & 0x1f {
				7 => sps = unit.to_vec(),
				8 => pps = unit.to_vec(),
				_ => {}
			}
		}
		if sps.len() < 4 || pps.is_empty() {
			bail!("nvenc gave no parameter sets");
		}
		Ok(Headers { sps, pps })
	}

	/// Gives the encoder the clip's next NV12 frame (`width` x `height` luma, then the interleaved chroma
	/// at half the height); [`Encoder::receive`] then has the frames it finished.
	pub fn send(&mut self, nv12: &[u8]) -> Result<()> {
		let (width, height) = (self.width as usize, self.height as usize);
		if nv12.len() < width * height * 3 / 2 {
			bail!("nvenc: a frame of {} bytes for {width}x{height}", nv12.len());
		}
		if self.frames - self.read >= self.slots.len() {
			bail!("nvenc: frame {} sent with {} unread", self.frames, self.frames - self.read);
		}
		let (input, output) = self.slots[self.frames % self.slots.len()];
		let api = self.api;
		let mut lock = Raw::<LOCK_WORDS>::new(LOCK_INPUT_VER);
		lock.put(LOCK_INPUT_BUFFER, input);
		// SAFETY: as in `open`; the locked buffer holds `pitch` bytes a row for 3/2 of the height
		let pitch = unsafe {
			self.check(api.entry::<Call>(FN_LOCK_INPUT)?(self.session, lock.pointer()), "locking an input buffer")?;
			let (data, pitch): (*mut u8, u32) = (lock.get(LOCK_INPUT_DATA), lock.get(LOCK_INPUT_PITCH));
			if data.is_null() || (pitch as usize) < width {
				api.entry::<Call>(FN_UNLOCK_INPUT)?(self.session, input);
				bail!("nvenc: an input buffer of {pitch} bytes a row for {width}");
			}
			for (row, pixels) in nv12[..width * height * 3 / 2].chunks_exact(width).enumerate() {
				std::ptr::copy_nonoverlapping(pixels.as_ptr(), data.add(row * pitch as usize), width);
			}
			self.check(api.entry::<Call>(FN_UNLOCK_INPUT)?(self.session, input), "unlocking an input buffer")?;
			pitch
		};
		let mut picture = Raw::<PIC_WORDS>::new(PIC_VER);
		picture.put(PIC_WIDTH, self.width);
		picture.put(PIC_HEIGHT, self.height);
		picture.put(PIC_PITCH, pitch);
		picture.put(PIC_FRAME_INDEX, self.frames as u32);
		picture.put(PIC_TIMESTAMP, self.frames as u64);
		picture.put(PIC_DURATION, 1u64);
		picture.put(PIC_INPUT, input);
		picture.put(PIC_OUTPUT, output);
		picture.put(PIC_FORMAT, BUFFER_FORMAT_NV12);
		picture.put(PIC_STRUCT, PIC_STRUCT_FRAME);
		// SAFETY: as in `open`
		let status = unsafe { api.entry::<Call>(FN_ENCODE)?(self.session, picture.pointer()) };
		// held back (a B-frame's reference to come, the lookahead filling) is no failure
		if status != NEED_MORE_INPUT {
			self.check(status, "encoding a frame")?;
		}
		self.frames += 1;
		Ok(())
	}

	/// The clip has no more frames: those the encoder holds are finished for [`Encoder::receive`].
	pub fn finish(&mut self) -> Result<()> {
		if !self.ended && self.frames > 0 {
			let mut end = Raw::<PIC_WORDS>::new(PIC_VER);
			end.put(PIC_FLAGS, PIC_FLAG_EOS);
			// SAFETY: as in `open`
			self.check(unsafe { self.api.entry::<Call>(FN_ENCODE)?(self.session, end.pointer()) }, "finishing the clip")?;
		}
		self.ended = true;
		Ok(())
	}

	/// The next finished frame in decode order, if one waits. The packet borrows the encoder until its next call.
	pub fn receive(&mut self) -> Result<Option<Packet<'_>>> {
		if self.read >= if self.ended { self.frames } else { self.frames.saturating_sub(self.delay) } {
			return Ok(None);
		}
		let output = self.slots[self.read % self.slots.len()].1;
		let api = self.api;
		let mut locked = Raw::<LOCK_WORDS>::new(LOCK_BITSTREAM_VER);
		locked.put(LOCKED_OUTPUT, output);
		// SAFETY: as in `open`; the locked bitstream is the driver's until unlocked, copied before
		let (keyframe, display) = unsafe {
			self.check(api.entry::<Call>(FN_LOCK_BITSTREAM)?(self.session, locked.pointer()), "reading a frame")?;
			let (data, size): (*const u8, u32) = (locked.get(LOCKED_DATA), locked.get(LOCKED_SIZE));
			self.packet.clear();
			if !data.is_null() {
				for unit in units(std::slice::from_raw_parts(data, size as usize)) {
					// the parameter sets are the file's own (its avcC); no access unit delimiters in an MP4
					if !matches!(unit[0] & 0x1f, 7..=9) {
						self.packet.extend_from_slice(&(unit.len() as u32).to_be_bytes());
						self.packet.extend_from_slice(unit);
					}
				}
			}
			let (kind, display): (u32, u64) = (locked.get(LOCKED_PICTURE_TYPE), locked.get(LOCKED_TIMESTAMP));
			self.check(api.entry::<Call>(FN_UNLOCK_BITSTREAM)?(self.session, output), "releasing a frame")?;
			(kind == PIC_TYPE_IDR, display as usize)
		};
		self.read += 1;
		if self.packet.is_empty() {
			bail!("nvenc put out an empty frame");
		}
		Ok(Some(Packet { data: &self.packet, keyframe, display }))
	}
}

impl Drop for Encoder {
	fn drop(&mut self) {
		let api = self.api;
		// SAFETY: the session's own buffers and the session, each destroyed once
		unsafe {
			if !self.session.is_null() {
				// frames never read (a clip that failed): the encoder told to end before its buffers go
				if !self.ended
					&& self.frames > 0
					&& let Ok(encode) = api.entry::<Call>(FN_ENCODE)
				{
					let mut end = Raw::<PIC_WORDS>::new(PIC_VER);
					end.put(PIC_FLAGS, PIC_FLAG_EOS);
					encode(self.session, end.pointer());
				}
				for &(input, output) in &self.slots {
					if !input.is_null()
						&& let Ok(destroy) = api.entry::<Call>(FN_DESTROY_INPUT)
					{
						destroy(self.session, input);
					}
					if !output.is_null()
						&& let Ok(destroy) = api.entry::<Call>(FN_DESTROY_BITSTREAM)
					{
						destroy(self.session, output);
					}
				}
				if let Ok(destroy) = api.entry::<Destroy>(FN_DESTROY) {
					destroy(self.session);
				}
			}
		}
		CudaContext::release(api);
	}
}

/// An Annex B stream's NAL units, without their start codes.
fn units(stream: &[u8]) -> impl Iterator<Item = &[u8]> {
	let mut starts = Vec::new();
	let mut at = 0;
	while at + 3 <= stream.len() {
		if stream[at] == 0 && stream[at + 1] == 0 && stream[at + 2] == 1 {
			starts.push(at + 3);
			at += 3;
		} else {
			at += 1;
		}
	}
	let ends: Vec<usize> = starts.iter().skip(1).map(|next| next - 3).chain([stream.len()]).collect();
	starts.into_iter().zip(ends).filter_map(move |(start, mut end)| {
		// a four-byte start code's first zero, and zeros a unit is padded with, are not the unit's
		while end > start && stream[end - 1] == 0 {
			end -= 1;
		}
		(end > start).then(|| &stream[start..end])
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn an_annex_b_stream_splits_into_its_units() {
		let stream = [0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 4, 5, 6, 0];
		let found: Vec<&[u8]> = units(&stream).collect();
		assert_eq!(found, vec![&[0x67, 1, 2][..], &[0x68, 3][..], &[0x65, 4, 5, 6][..]]);
		assert!(units(&[]).next().is_none());
	}

	#[test]
	fn the_rate_reads_a_quality_and_a_ceiling() {
		assert_eq!(Rate::parse("21").unwrap(), Rate { quality: 21, kbps: None });
		assert_eq!(Rate::parse("21:8300").unwrap(), Rate { quality: 21, kbps: Some(8300) });
		assert!(Rate::parse("0").is_err() && Rate::parse("21:0").is_err() && Rate::parse("x").is_err());
	}

	/// Fields set in a block land where the driver reads them.
	#[test]
	fn a_block_keeps_its_fields_and_bits() {
		let mut raw = Raw::<4>::new(7);
		raw.put(12, 0xffff_ffffu32);
		raw.bits((12, 12), 4, 8);
		raw.bits((12, 3), 1, 0);
		assert_eq!(raw.get::<u32>(0), 7);
		assert_eq!(raw.get::<u32>(12), 0xffff_8ff7);
	}

	/// Where the machine has the encoder: every frame comes back once, a keyframe first, each frame's
	/// place in the clip told, with the encoder holding frames back as it reorders them.
	#[test]
	fn a_clip_comes_back_whole_and_in_decode_order() {
		let (width, height, frames) = (320usize, 240usize, 40usize);
		let Ok(mut encoder) = Encoder::open(&Settings { width: width as u32, height: height as u32, fps: (30, 1), rate: Rate { quality: 21, kbps: Some(2000) } }) else {
			return;
		};
		let headers = encoder.headers().unwrap();
		assert_eq!((headers.sps[0] & 0x1f, headers.pps[0] & 0x1f), (7, 8));
		let mut shown = Vec::new();
		let mut keep = |encoder: &mut Encoder| {
			while let Some(packet) = encoder.receive().unwrap() {
				assert!(!packet.data.is_empty() && packet.keyframe == shown.is_empty());
				shown.push(packet.display);
			}
		};
		for frame in 0..frames {
			// a bar moving across grey: something for the encoder to predict
			let mut nv12 = vec![128u8; width * height * 3 / 2];
			for row in 0..height {
				nv12[row * width + frame * 4..row * width + frame * 4 + 32].fill(235);
			}
			encoder.send(&nv12).unwrap();
			keep(&mut encoder);
		}
		encoder.finish().unwrap();
		keep(&mut encoder);
		let mut sorted = shown.clone();
		sorted.sort_unstable();
		assert_eq!(sorted, (0..frames).collect::<Vec<_>>(), "decode order {shown:?}");
	}
}
