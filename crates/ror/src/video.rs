//! Video: a model or an effect over a stretch of its timeline, drawn frame by frame by one warm
//! renderer and encoded while it draws: an H.264 MP4 through ffmpeg (a VP9 WebM keeping a transparent
//! background, a GIF on request), an animated PNG where ffmpeg is missing; and a contact sheet of
//! moments for a reader that cannot play video. `ror video`; the render service's `POST /video`.
//!
//! The work frames share is done once: the model is read once, and a playback that only emits (an
//! effect played by its emitters' attributes) has its scene extracted once, each
//! frame then working out its particles and trails alone; any other playback gives each thread a
//! document of its own, its changes undone after every moment. Scenes are built ahead on every core while the GPU draws,
//! and the encoder takes each frame while the next one draws.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use glam::Vec3;
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::job::{SceneCache, ViewResult, input_bytes, input_name};
use crate::output;
use crate::render::camera::CameraSpec;
use crate::render::guis::FrameGuis;
use crate::render::{Background, PartsKey, PendingView, PreparedSky, RenderStats, Renderer, VideoReadback, ViewRequest};
use crate::scene::load::Document;
use crate::scene::{self, Playback, Scene};
use crate::budget::GpuGate;

/// The most frames a video may have (two minutes at 60 a second).
const MAX_FRAMES: usize = 7200;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoFormat {
	/// H.264 in MP4 (ffmpeg): plays everywhere, no alpha.
	#[default]
	Mp4,
	/// VP9 in WebM (ffmpeg), with alpha over a transparent background.
	Webm,
	/// GIF (ffmpeg, one palette for the clip).
	Gif,
	/// Animated PNG, written by ror: lossless with alpha, large; what ror writes without ffmpeg.
	Apng,
	/// One PNG per frame in `frames/`.
	Png,
}

impl VideoFormat {
	pub fn parse(text: &str) -> Result<Self> {
		Ok(match text.trim().to_ascii_lowercase().as_str() {
			"mp4" | "h264" => VideoFormat::Mp4,
			"webm" | "vp9" => VideoFormat::Webm,
			"gif" => VideoFormat::Gif,
			"apng" => VideoFormat::Apng,
			"png" | "frames" => VideoFormat::Png,
			other => bail!("--format {other:?}: mp4, webm, gif, apng or png"),
		})
	}

	fn needs_ffmpeg(self) -> bool {
		matches!(self, VideoFormat::Mp4 | VideoFormat::Webm | VideoFormat::Gif)
	}

	fn file_name(self) -> &'static str {
		match self {
			VideoFormat::Mp4 => "video.mp4",
			VideoFormat::Webm => "video.webm",
			VideoFormat::Gif => "video.gif",
			VideoFormat::Apng => "video.png",
			VideoFormat::Png => "frames",
		}
	}
}

fn one() -> f32 {
	1.0
}

fn default_sheet() -> usize {
	12
}

fn default_name() -> String {
	"video".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoJob {
	/// A model or place file, or an asset URI / id (as [`crate::job::Job::input`]).
	pub input: String,
	#[serde(default)]
	pub select: Option<String>,
	/// Output directory; `<state dir>/out/<input name>-video` when absent.
	#[serde(default)]
	pub out: Option<PathBuf>,
	/// The view drawn every frame: size, camera (an orbit is fitted once to what the whole clip
	/// draws), background, MSAA, bloom.
	pub view: ViewRequest,
	/// The view's name in the summary (`ror render --views <name> --camera-from <out>` draws a still
	/// from the video's camera).
	#[serde(default = "default_name")]
	pub name: String,
	/// What happens when (docs/playback.md); each frame draws it at its own moment.
	#[serde(default)]
	pub playback: Option<Playback>,
	/// The file the playback was read from, when it was one (the summary's `playback`: a recording's
	/// other files beside it).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub playback_file: Option<PathBuf>,
	/// The first frame's moment, seconds after the playback's start.
	#[serde(default)]
	pub start: f32,
	/// Seconds of the playback drawn from `start`.
	pub duration: f32,
	/// Frames per second of video.
	pub fps: f32,
	/// The speed the video plays the effect at: 0.25 shows `duration` four times slower.
	#[serde(default = "one")]
	pub speed: f32,
	/// Exactly these moments, seconds after the playback's start, one frame each in order, in place of
	/// `start` + `duration` at `fps` (the video still plays at `fps`): stills taken at their own times.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub times: Option<Vec<f32>>,
	/// Degrees an orbit camera turns about what it frames over the clip (360: once round).
	#[serde(default)]
	pub orbit: f32,
	#[serde(default)]
	pub format: VideoFormat,
	/// Moments on the contact sheet (0: none).
	#[serde(default = "default_sheet")]
	pub sheet: usize,
	#[serde(default)]
	pub materials_from: Option<String>,
	#[serde(default)]
	pub lighting_from: Option<String>,
	/// Another file's world drawn around the model, never framed (crate::job::MapSetting).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub map: Option<crate::job::MapSetting>,
	/// The orbit frames this box (min, max) instead of what the clip draws.
	#[serde(default)]
	pub frame: Option<[[f32; 3]; 2]>,
	/// The orbit frames what is at or under this instance over the clip.
	#[serde(default)]
	pub focus: Option<String>,
	/// ffmpeg's executable (the client's: a service may run with another PATH); [`find_ffmpeg`] when
	/// absent, an animated PNG when there is none.
	#[serde(default)]
	pub ffmpeg: Option<PathBuf>,
	/// A moving camera, in place of the view's: each key from its moment until the next (the first
	/// before its own), as a recording's game camera moved (ree's `camera.json`).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub camera_track: Option<Vec<CameraKey>>,
	/// Through the service: the sheet's moments drawn first, and the result sent back once `sheet.png`
	/// is written (the clip itself finished after it: [`run_with`]).
	#[serde(default, skip_serializing_if = "std::ops::Not::not")]
	pub sheet_first: bool,
	/// Other files made from the same frames while the clip encodes ([`VideoCopy`]): an MP4 clip's.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub copies: Vec<VideoCopy>,
	/// An MP4 encoded by the GPU's hardware encoder in the process ([`crate::nvenc`]) at this quality and
	/// ceiling, in place of x264; the clip fails where there is none.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub nvenc: Option<crate::nvenc::Rate>,
	/// The folder a relative local path in the input's content is read from (as
	/// [`crate::job::Job::relative_to`]).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub relative_to: Option<PathBuf>,
}

/// A file ffmpeg makes from the clip's frames as they are drawn (the NV12 the clip's encoder takes),
/// its arguments between that input and the file: a smaller copy to share without decoding the clip
/// again once it is written (a 4K clip's: ready with the clip, where encoding it after took 2.4-7.8 s).
/// The file's extension names the container.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VideoCopy {
	pub file: PathBuf,
	#[serde(default)]
	pub args: Vec<String>,
}

/// What became of a [`VideoCopy`]: written, or why not, and how long the clip waited for it once its
/// own file was done.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CopyResult {
	pub file: PathBuf,
	pub written: bool,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub error: Option<String>,
	pub waited_ms: f64,
}

/// One key of a camera track: from `at` (seconds of the playback) the camera is this Roblox CFrame
/// (position, then the rotation by rows) with this vertical field of view.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CameraKey {
	pub at: f32,
	pub cframe: [f32; 12],
	#[serde(default = "crate::render::camera::default_fov")]
	pub fov: f32,
}

/// The job's playback, watched through its camera track when it has one (Studio updates only the
/// emitters that camera could draw: [`scene::playback::ViewerKey`]).
fn watched(job: &VideoJob) -> Playback {
	let mut playback = job.playback.clone().unwrap_or_default();
	if let Some(track) = job.camera_track.as_deref().filter(|_| playback.viewer.is_empty()) {
		let aspect = job.view.width.max(1) as f32 / job.view.height.max(1) as f32;
		let mut keys: Vec<scene::playback::ViewerKey> = track.iter().map(|key| scene::playback::ViewerKey { at: key.at, cframe: key.cframe, fov: key.fov, aspect }).collect();
		keys.sort_by(|a, b| a.at.total_cmp(&b.at));
		playback.viewer = keys;
	}
	playback
}

/// The track's camera at a moment: the latest key at or before it, else the first.
fn track_camera(track: &[CameraKey], time: f32) -> CameraSpec {
	let key = track.iter().filter(|key| key.at <= time + 1e-4).max_by(|a, b| a.at.total_cmp(&b.at)).unwrap_or(&track[0]);
	CameraSpec::CFrame { components: key.cframe, fov: key.fov, near: None }
}

/// A moment on the contact sheet.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Moment {
	pub time: f32,
	pub frame: usize,
	pub particles: usize,
}

/// How bright one encoded frame is (a summary's `frameStats`, one per frame): what a review reads a
/// hit frame's peak from. Measured on the pixels the encoder takes, after the tone curve and every
/// post effect: an MP4's BT.709 NV12 as the video decodes it (luma from each pixel's Y, in steps of
/// 1/219; red, green and blue with its 2 x 2 block's chroma), any other format's RGBA, both over
/// black where the background is transparent.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FrameStats {
	/// The frame's moment (seconds of the playback, to four places) and index.
	pub time: f64,
	pub frame: usize,
	/// Pixels whose red, green and blue are all 230 or more of 255.
	pub near_white: u64,
	/// Pixels whose Rec.709 luma is 0.8 or more.
	pub bright: u64,
	/// The mean Rec.709 luma, 0-1, to four places.
	pub luma: f64,
}

/// A frame's pixels measured ([`FrameStats`] without its moment).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Brightness {
	near_white: u64,
	bright: u64,
	luma: f64,
}

/// A near-white pixel's red, green and blue are all at least this (8-bit display values).
const NEAR_WHITE: u32 = 230;

/// Rec.709 luma weights in ten-thousandths (red, green, blue), and 0.8 of their full 255 x 10,000.
const LUMA_WEIGHTS: [u32; 3] = [2126, 7152, 722];
const BRIGHT_LUMA: u32 = 2_040_000;

/// Limited-range BT.709 Y (16-235): the least at luma 0.8 ((192 - 16) / 219 = 0.804; 191's 0.799),
/// and the least whose decoded red, green and blue can all be 229.5 or more (Y decodes to their
/// Rec.709 mean: 213 to 229.4, 214 to 230.5).
const BRIGHT_Y: u8 = 192;
const NEAR_WHITE_Y: u8 = 214;

/// An RGBA frame's brightness, each pixel over black by its alpha as MP4 flattens a transparent
/// background (`(c * a + 127) / 255`, the colour itself at alpha 255).
fn rgba_brightness(rgba: &[u8]) -> Brightness {
	let (pixels, _) = rgba.as_chunks::<4>();
	let (mut near_white, mut bright, mut sum) = (0u64, 0u64, 0u64);
	// u32 sums of 1,024 pixels' luma (at most 2,550,000 each) cannot overflow
	for block in pixels.chunks(1024) {
		let (mut white, mut lit, mut total) = (0u32, 0u32, 0u32);
		for &[red, green, blue, alpha] in block {
			// (c * a + 127) / 255 without a division: c * a / 255 rounded (never a half)
			let over = |value: u8| {
				let scaled = value as u32 * alpha as u32 + 128;
				(scaled + (scaled >> 8)) >> 8
			};
			let (red, green, blue) = (over(red), over(green), over(blue));
			let luma = red * LUMA_WEIGHTS[0] + green * LUMA_WEIGHTS[1] + blue * LUMA_WEIGHTS[2];
			total += luma;
			lit += (luma >= BRIGHT_LUMA) as u32;
			white += (red.min(green).min(blue) >= NEAR_WHITE) as u32;
		}
		(near_white, bright, sum) = (near_white + white as u64, bright + lit as u64, sum + total as u64);
	}
	Brightness { near_white, bright, luma: sum as f64 / (pixels.len().max(1) as f64 * 2_550_000.0) }
}

/// A limited-range BT.709 NV12 frame's brightness (`width` x `height`, even): luma from Y, near white
/// decoded with the chroma of each pixel's 2 x 2 block.
fn nv12_brightness(nv12: &[u8], width: usize, height: usize) -> Brightness {
	let pixels = width * height;
	if width == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) || nv12.len() < pixels + pixels / 2 {
		return Brightness::default();
	}
	use rayon::prelude::*;
	let (luma, chroma) = nv12.split_at(pixels);
	// bands of rows, on the thread pool unless a clip's own thread measures: whole-number sums, so the same
	// however the rows are split
	const BAND: usize = 64;
	let measure = |(band, rows): (usize, &[u8])| {
		let (mut sum, mut bright, mut near_white) = (0u64, 0u64, 0u64);
		for (offset, line) in rows.chunks_exact(width).enumerate() {
			let row = band * BAND + offset;
			let pairs = &chroma[row / 2 * width..(row / 2 + 1) * width];
			let (total, lit, white) = row_levels(line, pairs);
			(sum, bright, near_white) = (sum + total, bright + lit, near_white + white);
		}
		(sum, bright, near_white)
	};
	let add = |a: (u64, u64, u64), b: (u64, u64, u64)| (a.0 + b.0, a.1 + b.1, a.2 + b.2);
	let (sum, bright, near_white) = if crate::par::spread(0, usize::MAX) {
		luma.par_chunks(width * BAND).enumerate().map(measure).reduce(|| (0, 0, 0), add)
	} else {
		luma.chunks(width * BAND).enumerate().map(measure).fold((0, 0, 0), add)
	};
	let mean = sum as f64 / pixels as f64;
	Brightness { near_white, bright, luma: ((mean - 16.0) / 219.0).clamp(0.0, 1.0) }
}

/// A row of luma's sum (each pixel within 16-235), its pixels at least [`BRIGHT_Y`] and those near white
/// ([`near_white_yuv`] with the row's chroma `pairs`; only pixels light enough are decoded).
fn row_levels_plain(line: &[u8], pairs: &[u8], from: usize) -> (u64, u64, u64) {
	let (mut total, mut lit, mut white) = (0u64, 0u64, 0u64);
	for (column, &y) in line.iter().enumerate().skip(from) {
		total += y.clamp(16, 235) as u64;
		lit += (y >= BRIGHT_Y) as u64;
		white += (y >= NEAR_WHITE_Y && near_white_yuv(y, pairs[column & !1], pairs[column | 1])) as u64;
	}
	(total, lit, white)
}

/// [`row_levels_plain`] sixteen pixels at a time: a 1440p frame's 3.7 million took 1.25 ms one by one on
/// the thread that hands the frame to the encoder, a third of that thread's time.
#[cfg(target_arch = "x86_64")]
fn row_levels(line: &[u8], pairs: &[u8]) -> (u64, u64, u64) {
	use std::arch::x86_64::*;
	let blocks = line.len() / 16;
	let mut white = 0u64;
	// SAFETY: SSE2 is part of x86_64; each load reads 16 bytes inside `line`
	let (total, lit) = unsafe {
		let (zero, one) = (_mm_setzero_si128(), _mm_set1_epi8(1));
		let (low, high) = (_mm_set1_epi8(16), _mm_set1_epi8(235u8 as i8));
		let (bright, light) = (_mm_set1_epi8(BRIGHT_Y as i8), _mm_set1_epi8(NEAR_WHITE_Y as i8));
		let (mut total, mut lit) = (zero, zero);
		for block in 0..blocks {
			let pixels = _mm_loadu_si128(line.as_ptr().add(block * 16).cast());
			total = _mm_add_epi64(total, _mm_sad_epu8(_mm_min_epu8(_mm_max_epu8(pixels, low), high), zero));
			// unsigned y >= level: the larger of the two is y
			let is_bright = _mm_cmpeq_epi8(_mm_max_epu8(pixels, bright), pixels);
			lit = _mm_add_epi64(lit, _mm_sad_epu8(_mm_and_si128(is_bright, one), zero));
			let mut is_light = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_max_epu8(pixels, light), pixels)) as u32;
			while is_light != 0 {
				let column = block * 16 + is_light.trailing_zeros() as usize;
				white += near_white_yuv(line[column], pairs[column & !1], pairs[column | 1]) as u64;
				is_light &= is_light - 1;
			}
		}
		let lanes = |sums: __m128i| _mm_cvtsi128_si64(sums) as u64 + _mm_cvtsi128_si64(_mm_unpackhi_epi64(sums, sums)) as u64;
		(lanes(total), lanes(lit))
	};
	let rest = row_levels_plain(line, pairs, blocks * 16);
	(total + rest.0, lit + rest.1, white + rest.2)
}

#[cfg(not(target_arch = "x86_64"))]
fn row_levels(line: &[u8], pairs: &[u8]) -> (u64, u64, u64) {
	row_levels_plain(line, pairs, 0)
}

/// Whether a limited-range BT.709 pixel decodes to red, green and blue all at least 229.5 of 255
/// (the inverse of the NV12 conversion's matrix).
fn near_white_yuv(y: u8, u: u8, v: u8) -> bool {
	// grey, as most of a white flash is
	if u == 128 && v == 128 {
		return y >= NEAR_WHITE_Y;
	}
	let luma = (y as f32 - 16.0) * (255.0 / 219.0);
	let blue = luma + (u as f32 - 128.0) * (255.0 * 1.8556 / 224.0);
	let red = luma + (v as f32 - 128.0) * (255.0 * 1.5748 / 224.0);
	let green = (luma - 0.2126 * red - 0.0722 * blue) / 0.7152;
	let least = NEAR_WHITE as f32 - 0.5;
	red >= least && green >= least && blue >= least
}

/// `value` to four places.
fn four_places(value: f64) -> f64 {
	(value * 10_000.0).round() / 10_000.0
}

/// The summary's frames line: the frame with the most near-white pixels and the one with the highest
/// mean luma, the moments a hit frame should be.
fn frames_line(frames: &[FrameStats], pixels: u64) -> Option<String> {
	let share = |count: u64| format!("{:.2} %", count as f64 * 100.0 / pixels.max(1) as f64);
	let lightest = frames.iter().max_by(|a, b| a.luma.total_cmp(&b.luma).then(b.frame.cmp(&a.frame)))?;
	let luma = format!("mean luma peaks at {:.2} s (frame {}, {:.3}, {} bright)", lightest.time, lightest.frame, lightest.luma, share(lightest.bright));
	match frames.iter().filter(|frame| frame.near_white > 0).max_by(|a, b| a.near_white.cmp(&b.near_white).then(b.frame.cmp(&a.frame))) {
		Some(whitest) => Some(format!("frames: near white peaks at {:.2} s (frame {}, {} of pixels), {luma}", whitest.time, whitest.frame, share(whitest.near_white))),
		None => Some(format!("frames: no near-white pixel in any frame, {luma}")),
	}
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoStats {
	/// The parts, ParticleEmitters (the legacy effects' among them), beams and trails in the world in
	/// any frame, each counted once (`Census`): an effect a playback parents in after the first frame
	/// counts too.
	pub parts: usize,
	pub emitters: usize,
	pub beams: usize,
	pub trails: usize,
	/// The most particles a frame drew, and that frame's moment.
	pub particles_max: usize,
	pub particles_max_at: f32,
	/// Frames that drew no particle while the scene has emitters (an effect not started or over).
	pub frames_without_particles: usize,
	pub missing: Vec<String>,
}

/// The parts, ParticleEmitters, beams and trails a clip's frames hold, each counted once however many
/// frames it is in. Frames' scenes are extracted from documents of their own (a file read again names
/// its instances anew), so an instance is known by its path and its place among those of that path,
/// as `crate::lint::Motion` knows emitters.
#[derive(Default)]
struct Census {
	parts: Copies,
	emitters: Copies,
	beams: Copies,
	trails: Copies,
}

impl Census {
	fn add(&mut self, scene: &Scene) {
		self.add_after(scene, None);
	}

	/// [`Census::add`] of the frame after `last`'s, both counted here: a kind whose paths are the last
	/// frame's in the same order adds nothing (a map clip hashed its 2,100 part paths every frame, 0.55 ms).
	fn add_after(&mut self, scene: &Scene, last: Option<&Scene>) {
		fn paths<'a, T: 'a>(items: &'a [T], path: fn(&T) -> &str) -> impl Iterator<Item = &'a str> + Clone {
			items.iter().map(path)
		}
		fn add<'a, T: 'a>(copies: &mut Copies, items: &'a [T], last: Option<&'a [T]>, path: fn(&T) -> &str) {
			if !last.is_some_and(|last| last.len() == items.len() && paths(last, path).eq(paths(items, path))) {
				copies.add(paths(items, path));
			}
		}
		add(&mut self.parts, &scene.parts, last.map(|last| &last.parts[..]), |part| part.path.as_str());
		add(&mut self.emitters, &scene.particles, last.map(|last| &last.particles[..]), |emitter| emitter.path.as_str());
		add(&mut self.beams, &scene.beams, last.map(|last| &last.beams[..]), |beam| beam.path.as_str());
		add(&mut self.trails, &scene.trails, last.map(|last| &last.trails[..]), |trail| trail.path.as_str());
	}

	/// (parts, emitters, beams, trails)
	fn counts(&self) -> (usize, usize, usize, usize) {
		(self.parts.count(), self.emitters.count(), self.beams.count(), self.trails.count())
	}
}

/// Instances by path: the most of one path a frame held (copies, same-named siblings).
#[derive(Default)]
struct Copies(HashMap<String, usize>);

impl Copies {
	fn add<'a>(&mut self, paths: impl Iterator<Item = &'a str>) {
		let mut frame: HashMap<&str, usize> = HashMap::new();
		for path in paths {
			*frame.entry(path).or_default() += 1;
		}
		for (path, copies) in frame {
			match self.0.get_mut(path) {
				Some(most) => *most = (*most).max(copies),
				None => {
					self.0.insert(path.to_owned(), copies);
				}
			}
		}
	}

	fn count(&self) -> usize {
		self.0.values().sum()
	}
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoTimings {
	/// Reading the input and its first scene.
	pub load_ms: f64,
	/// Fitting the camera to what the clip draws; the warm runs beside it.
	pub framing_ms: f64,
	/// Fetching the clip's files not yet on disk ahead of its frames.
	pub warm_ms: f64,
	/// Preparing and submitting every frame, including waits for the bounded output queue.
	pub render_ms: f64,
	/// Draining the remaining readbacks and encoder after the last frame was submitted.
	pub encode_ms: f64,
	pub sheet_ms: f64,
	pub total_ms: f64,
	/// Preparing and submitting per frame, on average; `total_ms` also includes the output tail.
	pub frame_ms: f64,
}

/// Accumulated output stages; waits overlap the producer and encoder's work.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OutputProfile {
	pub scene_wait_ms: f64,
	/// Uploading each frame's GUI images (found ahead by the GUI stage), while earlier frames draw.
	pub gui_upload_ms: f64,
	/// Waiting for the GPU memory a frame's targets add ([`GpuGate`])
	pub gpu_wait_ms: f64,
	pub prepare_ms: f64,
	pub draw_ms: f64,
	pub readback_ms: f64,
	pub thumbnail_ms: f64,
	pub encoder_wait_ms: f64,
	pub pipe_ms: f64,
	pub flatten_ms: f64,
	/// Measuring each frame's brightness (`frameStats`), after its pixels went to the encoder.
	pub measure_ms: f64,
	/// Frames that were the frame before's image again ([`same_scene`]): not drawn (drawn and compared
	/// with `ROR_FRAME_REUSE=check`).
	pub repeated: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoResult {
	/// A sheet-first result sent ahead of its clip ([`run_with`]): this file stands in the output folder
	/// until the clip and `summary.json` are written (`clip.failed` replaces it if they are not), and
	/// what needs every frame (the video, `frameStats`, the lints, the census) is still to come; `stats`
	/// and `moments` are the sheet's moments'.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub pending: Option<PathBuf>,
	/// Stage totals when `ROR_VIDEO_PROFILE` is set on the renderer process.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub output_profile: Option<OutputProfile>,
	/// What was drawn: the model or place file (absolute), else the asset as given; and the playback
	/// file, when the playback was read from one.
	pub input: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub playback: Option<PathBuf>,
	pub out: PathBuf,
	/// The video file (the frames' folder for `png`).
	pub video: PathBuf,
	pub format: VideoFormat,
	/// What encoded it: ffmpeg's encoder (libx264, libvpx-vp9, gif), else ror's own (apng, png).
	pub encoder: String,
	pub sheet: Option<PathBuf>,
	pub width: u32,
	pub height: u32,
	pub frames: usize,
	pub fps: f32,
	/// The playback drawn: from `start`, `duration` seconds, at `speed`; the video lasts `seconds`.
	pub start: f32,
	pub duration: f32,
	pub speed: f32,
	pub seconds: f32,
	pub moments: Vec<Moment>,
	/// The box the camera keeps in view: what every moment sampled draws, parts and effects.
	pub framed: Option<[[f32; 3]; 2]>,
	/// The first frame's camera, as `ror render --camera-from <out>` reads it.
	pub views: Vec<ViewResult>,
	pub stats: VideoStats,
	/// The ParticleEmitters emitting in some frame whose particles fly weightless (crate::lint).
	pub weightless: crate::lint::Weightless,
	/// The other authoring lints over the clip (crate::lint::Lint): linear fades, restraint within a
	/// form, frozen clouds, refused images.
	pub lint: crate::lint::Lint,
	/// Each effect's emitters, textures, beams and trails over the clip (crate::lint::Counts).
	pub counts: crate::lint::Counts,
	/// Every encoded frame's brightness, in order.
	pub frame_stats: Vec<FrameStats>,
	/// The summary's lines, which `ror video` prints to stderr: the frames' brightness peaks, the
	/// effects' counts and each lint that found something.
	pub lines: Vec<String>,
	pub warnings: Vec<String>,
	pub timings: VideoTimings,
	/// The job's copies, in its order.
	#[serde(skip_serializing_if = "Vec::is_empty")]
	pub copies: Vec<CopyResult>,
}

/// ffmpeg's executable: `ROR_FFMPEG`, else the first on PATH, else winget's (Gyan.FFmpeg) install.
pub fn find_ffmpeg() -> Option<PathBuf> {
	if let Some(path) = std::env::var_os("ROR_FFMPEG").map(PathBuf::from).filter(|path| path.is_file()) {
		return Some(path);
	}
	let name = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
	if let Some(found) = std::env::var_os("PATH").and_then(|paths| std::env::split_paths(&paths).map(|dir| dir.join(name)).find(|path| path.is_file())) {
		return Some(found);
	}
	let local = std::env::var_os("LOCALAPPDATA").map(PathBuf::from)?.join("Microsoft").join("WinGet");
	if local.join("Links").join(name).is_file() {
		return Some(local.join("Links").join(name));
	}
	let packages = std::fs::read_dir(local.join("Packages")).ok()?;
	for package in packages.flatten().filter(|entry| entry.file_name().to_string_lossy().to_ascii_lowercase().contains("ffmpeg")) {
		for build in std::fs::read_dir(package.path()).into_iter().flatten().flatten() {
			let candidate = build.path().join("bin").join(name);
			if candidate.is_file() {
				return Some(candidate);
			}
		}
	}
	None
}

/// The encoders an ffmpeg has (`ffmpeg -encoders`): asked once per executable, and kept on disk
/// (`<state dir>/ffmpeg/<path, size and time of the executable>.txt`) for every process after,
/// until ffmpeg itself changes (a launch each, 15-55 ms; the 2.76 s one once seen was the network
/// lookup [`crate::output::work_from`] explains, paid by every ffmpeg the service started).
fn ffmpeg_encoders(ffmpeg: &Path) -> Result<Arc<String>> {
	static KNOWN: OnceLock<Mutex<std::collections::HashMap<PathBuf, Arc<String>>>> = OnceLock::new();
	let known = KNOWN.get_or_init(Default::default);
	if let Some(listed) = known.lock().get(ffmpeg) {
		return Ok(listed.clone());
	}
	let file = ffmpeg_list_file(ffmpeg);
	if let Some(listed) = file.as_ref().and_then(|file| std::fs::read_to_string(file).ok()).filter(|text| text.contains("libx264") || text.contains("Encoders:")) {
		let listed = Arc::new(listed);
		known.lock().insert(ffmpeg.to_path_buf(), listed.clone());
		return Ok(listed);
	}
	let mut command = Command::new(ffmpeg);
	command.args(["-hide_banner", "-encoders"]).stdin(Stdio::null());
	hide_console(&mut command);
	let output = command.output().with_context(|| format!("running {}", ffmpeg.display()))?;
	let listed = Arc::new(String::from_utf8_lossy(&output.stdout).into_owned());
	if output.status.success()
		&& let Some(file) = file
	{
		if let Some(parent) = file.parent() {
			let _ = std::fs::create_dir_all(parent);
		}
		let _ = crate::output::write_atomic(&file, listed.as_bytes());
	}
	known.lock().insert(ffmpeg.to_path_buf(), listed.clone());
	Ok(listed)
}

/// Where an ffmpeg's encoder list is kept: named by its path, size and modification time.
fn ffmpeg_list_file(ffmpeg: &Path) -> Option<PathBuf> {
	use std::hash::{Hash, Hasher};
	let metadata = std::fs::metadata(ffmpeg).ok()?;
	let modified = metadata.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos();
	let mut hasher = std::collections::hash_map::DefaultHasher::new();
	(ffmpeg, metadata.len(), modified).hash(&mut hasher);
	Some(output::state_dir().join("ffmpeg").join(format!("{:016x}.txt", hasher.finish())))
}

/// Readies the ffmpeg clips would use, off the caller's thread: its encoder list (from disk, else
/// asked), and one short run of it, so the first clip's encoder starts from an executable already
/// read in. The render service does this as it starts.
pub fn warm_ffmpeg() {
	std::thread::Builder::new()
		.name("ror-ffmpeg-warm".into())
		.spawn(|| {
			let Some(ffmpeg) = find_ffmpeg() else { return };
			let _ = ffmpeg_encoders(&ffmpeg);
			let mut command = Command::new(&ffmpeg);
			command.args(["-hide_banner", "-version"]).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
			hide_console(&mut command);
			let _ = command.status();
		})
		.ok();
}

/// The encoder a format takes from what an ffmpeg has, best first.
fn pick_encoder(format: VideoFormat, listed: &str) -> Option<&'static str> {
	let has = |name: &str| listed.lines().any(|line| line.split_whitespace().nth(1) == Some(name));
	let choices: &[&'static str] = match format {
		VideoFormat::Mp4 => &["libx264", "h264_nvenc", "h264_amf", "h264_qsv", "h264_mf", "mpeg4"],
		VideoFormat::Webm => &["libvpx-vp9", "libvpx"],
		VideoFormat::Gif => &["gif"],
		_ => &[],
	};
	choices.iter().copied().find(|name| has(name))
}

fn nvenc_options(command: &mut Command) {
	command.args(["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-bf", "3", "-pix_fmt", "nv12"]);
	// Spatial AQ preserves smooth skies and faint effects that a fast uniform quantizer loses.
	command.args(["-spatial-aq", "1", "-aq-strength", "8"]);
	// NVENC does not propagate ffmpeg's chroma-location option into the H.264 VUI.
	command.args(["-bsf:v", "h264_metadata=chroma_sample_loc_type=1"]);
}

#[derive(Hash, PartialEq, Eq)]
struct NvencProbe { executable: PathBuf, width: u32, height: u32 }

fn nvenc_probes() -> &'static Mutex<std::collections::HashMap<NvencProbe, bool>> {
	static KNOWN: OnceLock<Mutex<std::collections::HashMap<NvencProbe, bool>>> = OnceLock::new();
	KNOWN.get_or_init(Default::default)
}

/// An encoder in ffmpeg's list may have no usable device or driver. Open it at the requested size
/// before choosing it, once per executable and size in the warm service.
fn nvenc_available(ffmpeg: &Path, width: u32, height: u32) -> bool {
	let known = nvenc_probes();
	let key = NvencProbe { executable: ffmpeg.to_path_buf(), width, height };
	if let Some(&works) = known.lock().get(&key) {
		return works;
	}
	let mut command = Command::new(ffmpeg);
	command.args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", &format!("color=c=black:s={width}x{height}:r=1"), "-frames:v", "1"]);
	nvenc_options(&mut command);
	command.args(["-f", "null", "-"]).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
	hide_console(&mut command);
	let works = command.status().is_ok_and(|status| status.success());
	known.lock().insert(key, works);
	works
}

/// libx264's arguments: `ultrafast` at CRF 19 on a thread count set by the frame size (the same
/// bytes on any machine), unless `ROR_VIDEO_X264=veryfast` asks for the arguments before 2026-10-15
/// (`veryfast`, CRF 17, as many threads as x264 takes for the machine's cores).
///
/// Measured on the suite's clips (1080p, 60-78 frames; notes/video-encode.md "Fast preset"): one clip
/// encodes 1.6-2x faster (map 240 -> 124 ms, still 247 -> 123, VFX 199 -> 122, GUI 218 -> 153) and
/// eight at once 1.4-2.1x (map 961 -> 464 ms, 24.6 -> 10.6 core-s), 4K 240 frames 968 ms against
/// NVENC's 1,324; PSNR against ror's own NV12 input as high or higher on every clip (map 43.8 -> 44.6
/// dB, still 45.6 -> 46.9, VFX 53.1 -> 53.3, GUI 50.3 -> 51.1), SSIM 0.982-0.996 (today's
/// 0.984-0.997, no frame under 0.978 where today's had 0.978), files 1.9-3.3 times as large.
fn x264_arguments(width: u32, height: u32) -> Vec<String> {
	let mut arguments: Vec<String> = Vec::new();
	if x264_veryfast() {
		arguments.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "nv12"].map(String::from));
	} else {
		let threads = x264_threads(width, height).to_string();
		arguments.extend(["-c:v", "libx264", "-preset", X264_PRESET, "-crf", "19", "-threads", &threads, "-pix_fmt", "nv12"].map(String::from));
	}
	arguments
}

/// libx264's preset and CRF for an MP4, through ffmpeg or in the process ([`crate::x264`]).
const X264_PRESET: &str = "ultrafast";
#[cfg(ror_x264)]
const X264_CRF: f32 = 19.0;

/// `ROR_VIDEO_X264=veryfast`: the arguments before 2026-10-15 (through ffmpeg only).
fn x264_veryfast() -> bool {
	std::env::var("ROR_VIDEO_X264").is_ok_and(|value| value == "veryfast")
}

/// libx264's frame threads: the fewest that encode as fast as sixteen (each holds a frame set: 26 MB at 1440p, 57 at 4K).
fn x264_threads(width: u32, height: u32) -> u32 {
	if width as u64 * height as u64 >= 6_000_000 { 8 } else { 6 }
}

/// Whether an MP4 is encoded in the process ([`crate::x264`], built into this ror): unless
/// `ROR_VIDEO_ENCODER` names one of ffmpeg's (`libx264`, `h264_nvenc`) or `ROR_VIDEO_X264=veryfast`
/// asks for the old arguments. `ROR_VIDEO_ENCODER=x264` insists on it.
fn x264_in_process(format: VideoFormat) -> Result<bool> {
	if format != VideoFormat::Mp4 {
		return Ok(false);
	}
	let choice = std::env::var("ROR_VIDEO_ENCODER").unwrap_or_else(|_| "auto".into());
	match choice.as_str() {
		"auto" => Ok(cfg!(ror_x264) && !x264_veryfast()),
		"x264" if !cfg!(ror_x264) => bail!("ROR_VIDEO_ENCODER=x264: this ror is built without x264 (the x264 feature, vendor/x264)"),
		"x264" if x264_veryfast() => bail!("ROR_VIDEO_ENCODER=x264 takes ror's preset: ROR_VIDEO_X264=veryfast goes through ffmpeg"),
		"x264" => Ok(true),
		_ => Ok(false),
	}
}

fn video_encoder(format: VideoFormat, ffmpeg: &Path, listed: &str, width: u32, height: u32, frames: usize, choice: &str) -> Result<&'static str> {
	let preferred = pick_encoder(format, listed).with_context(|| format!("{} has no encoder for {format:?}", ffmpeg.display()))?;
	if format != VideoFormat::Mp4 {
		return Ok(preferred);
	}
	// libx264's fast preset beats NVENC here at every size measured (4K, 240 frames: 968 against 1,324
	// ms; 1080p clips 122-153 against 296-341 ms): hardware only when asked for
	let _ = frames;
	let hardware = match choice {
		"auto" | "x264" => false,
		"libx264" => return listed.lines().any(|line| line.split_whitespace().nth(1) == Some("libx264")).then_some("libx264").context("ffmpeg has no libx264 encoder"),
		"h264_nvenc" => true,
		_ => bail!("ROR_VIDEO_ENCODER: auto, x264, libx264 or h264_nvenc (got {choice:?})"),
	};
	if hardware && listed.lines().any(|line| line.split_whitespace().nth(1) == Some("h264_nvenc")) && nvenc_available(ffmpeg, width, height) {
		return Ok("h264_nvenc");
	}
	Ok(preferred)
}

/// Builds the scene at any moment of the job's playback.
/// A clip's input as every frame starts from, for a playback that changes no property and names no
/// Lighting: its document, what it draws, where its effects hang and whether it holds GUIs. The
/// service keeps the last few while their files are unchanged ([`SceneCache::video_input`]): a map
/// clip read and extracted the map again for every job, 65-70 ms of it.
pub(crate) struct FixedInput {
	document: Document,
	scene: Scene,
	roots: crate::lint::EffectRoots,
	guis: bool,
}

struct Builder<'a> {
	/// The input's bytes (none when its [`FixedInput`] was kept)
	bytes: Vec<u8>,
	select: Option<&'a str>,
	playback: Playback,
	scenes: &'a SceneCache,
	/// Where the input's relative local paths are read from (VideoJob::relative_to)
	relative_to: Option<&'a Path>,
	/// A world drawn around the model (`--map`)
	map: Option<&'a crate::job::MapSetting>,
	/// Another place's MaterialService and Lighting (the map's own unless others are named)
	lend: (Option<&'a str>, Option<&'a str>),
	/// Whether the documents hold that place's Lighting and Terrain's Clouds, for a playback that changes them
	lent: bool,
	/// What every moment draws, extracted once with its document, for a playback that changes no
	/// property (`fixed`) or only the Lighting (`lit`: each moment reads the Lighting of a copy of
	/// that document, changed, into it; the copies keep its referents, which the scene names)
	fixed: Option<Arc<FixedInput>>,
	lit: Option<(Document, Scene)>,
	/// Otherwise documents for threads to take, change, put back as they were and return.
	documents: Mutex<Vec<Document>>,
	/// Whether the model holds SurfaceGuis or BillboardGuis, whose images are found for several
	/// frames together.
	guis: bool,
	/// The playback's tweens that fade linearly, and where the effects hang (crate::lint).
	fades: Vec<crate::lint::Fade>,
	roots: crate::lint::EffectRoots,
}

impl<'a> Builder<'a> {
	/// The builder of `job`'s frames, its input's bytes from `read` unless `scenes` kept its
	/// [`FixedInput`].
	fn new(read: impl FnOnce() -> Result<Vec<u8>>, job: &'a VideoJob, scenes: &'a SceneCache) -> Result<Self> {
		let mut builder = Builder {
			bytes: Vec::new(),
			select: job.select.as_deref(),
			playback: watched(job),
			scenes,
			relative_to: job.relative_to.as_deref(),
			map: job.map.as_ref(),
			lend: scenes.lenders(job.map.as_ref(), job.materials_from.as_deref(), job.lighting_from.as_deref())?,
			lent: false,
			fixed: None,
			lit: None,
			documents: Mutex::new(Vec::new()),
			guis: false,
			fades: Vec::new(),
			roots: Default::default(),
		};
		let fixed = builder.playback.changes_nothing() && !builder.playback.touches_lighting() && !builder.playback.play;
		let key = fixed.then(|| scenes.video_key(&job.input, builder.select, builder.map, builder.lend, builder.relative_to)).flatten();
		if let Some(input) = key.as_deref().and_then(|key| scenes.video_input(key)) {
			builder.fades = crate::lint::tween_fades(&input.document, &builder.playback);
			builder.roots = input.roots.clone();
			builder.guis = input.guis;
			builder.fixed = Some(input);
			return Ok(builder);
		}
		builder.bytes = read()?;
		let (document, lent) = builder.document()?;
		builder.lent = lent;
		builder.fades = crate::lint::tween_fades(&document, &builder.playback);
		builder.roots = crate::lint::EffectRoots::of(&document);
		builder.guis = document.dom.descendants().any(|instance| matches!(instance.class.as_str(), "SurfaceGui" | "BillboardGui"));
		if builder.playback.changes_nothing() {
			let scene = builder.extracted(&document)?;
			let input = Arc::new(FixedInput { document, scene, roots: builder.roots.clone(), guis: builder.guis });
			if let Some(key) = key {
				scenes.keep_video_input(key, input.clone());
			}
			builder.fixed = Some(input);
		} else if builder.playback.resolve(&document)?.changes_only_lighting(&document) {
			let scene = builder.extracted(&document)?;
			builder.lit = Some((document, scene));
		} else {
			builder.documents.lock().push(document);
		}
		Ok(builder)
	}

	/// The input as a document, with the map drawn around it and the Lighting the playback changes.
	fn document(&self) -> Result<(Document, bool)> {
		let mut document = scene::load::read_bytes(&self.bytes)?;
		document.play = self.playback.play;
		document.salt = self.playback.salt.clone();
		if let Some(select) = self.select {
			document.select(select)?;
		}
		if let Some(folder) = self.relative_to {
			document.read_relative_to(folder);
		}
		if let Some(map) = self.map {
			self.scenes.add_map(&mut document, map)?;
		}
		let lent = self.scenes.give_lighting(&mut document, &self.playback, self.lend.1)?;
		Ok((document, lent))
	}

	/// Fetches the files the clip draws that are not on disk yet, many at a time
	/// ([`ror_assets::Assets::warm`], as `ror render` does: job::run_admitted): those of the scene a
	/// fixed or lit clip draws, else of the input before its playback. Not for an input the service
	/// kept (`bytes` empty), whose files came with its first clip.
	fn warm(&self, assets: &ror_assets::Assets) {
		// offline there is nothing to fetch, and the document and scene below are all this step would make
		if self.bytes.is_empty() || assets.offline() {
			return;
		}
		let fresh;
		let scene = match (&self.fixed, &self.lit) {
			(Some(input), _) => &input.scene,
			(None, Some((_, scene))) => scene,
			(None, None) => match self.document().and_then(|(document, _)| self.extracted(&document)) {
				Ok(scene) => {
					fresh = scene;
					&fresh
				}
				Err(_) => return,
			},
		};
		assets.warm(&crate::render::downloads(scene), &crate::render::transcode_packs);
	}

	/// What a document draws as it stands, in the other places' MaterialService and Lighting.
	fn extracted(&self, document: &Document) -> Result<Scene> {
		let mut scene = scene::extract::extract(document);
		self.finish(&mut scene)?;
		Ok(scene)
	}

	/// The other places' MaterialService and Lighting (unless the documents carry that Lighting).
	fn finish(&self, scene: &mut Scene) -> Result<()> {
		self.scenes.lend(scene, self.lend.0, self.lend.1.filter(|_| !self.lent))
	}

	/// The scene `time` seconds after the playback's start, on a document of its own.
	#[cfg(test)]
	fn scene_at(&self, time: f32) -> Result<Scene> {
		self.scene_with(time, None)
	}

	/// [`Builder::scene_at`], a document and its resolved playback taken from `kept` and put back
	/// there: a later moment of the same document resolves no target or event again
	/// ([`scene::playback::Resolved::at`]).
	fn scene_with<'s>(&'s self, time: f32, kept: Option<&Mutex<Vec<Kept<'s>>>>) -> Result<Scene> {
		self.scene_checked(time, kept, scene_reuse())
	}

	/// [`Builder::scene_with`] reusing the untouched roots' entries as `reuse` says (`Check`: and the
	/// whole extraction compared with it, an error where they differ).
	fn scene_checked<'s>(&'s self, time: f32, kept: Option<&Mutex<Vec<Kept<'s>>>>, reuse: SceneReuse) -> Result<Scene> {
		let taken = kept.and_then(|kept| kept.lock().pop());
		let resolved_for = |document: &Document, earlier: Option<scene::playback::Resolved<'s>>| -> Result<scene::playback::Resolved<'s>> {
			match earlier {
				Some(mut resolved) => {
					if resolved.at(time) {
						Ok(resolved)
					} else {
						self.playback.resolve_at(document, time)
					}
				}
				None => self.playback.resolve_at(document, time),
			}
		};
		if let Some(input) = &self.fixed {
			let resolved = resolved_for(&input.document, taken.map(|kept| kept.resolved))?;
			let mut scene = input.scene.clone();
			resolved.apply_history(&input.document, &mut scene);
			if let Some(kept) = kept {
				kept.lock().push(Kept { document: None, resolved, roots: None });
			}
			return Ok(scene);
		}
		let (taken, earlier, mut roots) = match taken {
			Some(Kept { document: Some(document), resolved, roots }) => (Some(document), Some(resolved), roots),
			_ => (self.documents.lock().pop(), None, None),
		};
		let mut document = match (taken, &self.lit) {
			(Some(document), _) => document,
			(None, Some((template, _))) => template.copy(),
			(None, None) => self.document()?.0,
		};
		let resolved = resolved_for(&document, earlier)?;
		// a document whose state failed half-way is dropped, not reused
		let undo = resolved.apply_state(&mut document)?;
		let mut scene = match &self.lit {
			// what is drawn stays as in the file: only the Lighting is read again
			Some((_, base)) => {
				let mut scene = base.clone();
				if self.playback.touches_lighting() {
					scene.environment = scene::extract::environment(&document).unwrap_or_default();
				}
				scene
			}
			// the roots no moment changes extracted once for this document (kept with it)
			None if kept.is_some() && reuse != SceneReuse::Off => {
				let roots = roots.get_or_insert_with(|| scene::extract::RootReuse::new(&document, &resolved.touched()));
				let scene = scene::extract::extract_reusing(&document, roots, &undo);
				if reuse == SceneReuse::Check {
					let whole = scene::extract::extract(&document);
					if scene != whole {
						if let Some(dir) = std::env::var_os("ROR_SCENE_REUSE_DUMP") {
							let _ = std::fs::write(Path::new(&dir).join("reused.txt"), format!("{scene:#?}"));
							let _ = std::fs::write(Path::new(&dir).join("whole.txt"), format!("{whole:#?}"));
						}
						bail!("ROR_SCENE_REUSE=check: the scene at {time} s with {} of {} roots extracted once differs from the whole extraction", roots.count(), document.roots.len());
					}
				}
				scene
			}
			None => scene::extract::extract(&document),
		};
		resolved.apply_history(&document, &mut scene);
		undo.restore(&mut document);
		match kept {
			Some(kept) => kept.lock().push(Kept { document: Some(document), resolved, roots }),
			None => self.documents.lock().push(document),
		}
		if self.lit.is_none() {
			self.finish(&mut scene)?;
		}
		Ok(scene)
	}

	/// Every scene of `times` in order, each with the sky's maps for its Lighting made, sent on as it
	/// is made while the frames before it draw. One document makes them one after another (each
	/// moment's emitters read and played on rayon's threads); while the draws wait for scenes, a helper
	/// with a document of its own makes later ones too ([`Ahead`]), up to one a core. A document per
	/// thread from the start, sixteen frames side by side, held a clip's model (8-10 MB for a
	/// cosmetic's) and its playback (5-6 MB) sixteen times, most of them idle while the GPU drew.
	fn produce<'s>(&'s self, renderer: &Renderer, times: &[f32], kept: &Mutex<Vec<Kept<'s>>>, sender: SyncSender<Result<(Scene, PreparedSky)>>) {
		// a fixed input's frames all have its parts: one preparation key for the clip
		let parts = PartsKey::default();
		let build = |index: usize| {
			self.scene_with(times[index], Some(kept)).map(|scene| {
				let sky = if self.fixed.is_some() { renderer.prepare_environment_sharing(&scene, &parts) } else { renderer.prepare_environment(&scene) };
				(scene, sky)
			})
		};
		let ahead = Ahead::new(times.len());
		let _producing = Producing::new();
		std::thread::scope(|scope| {
			let (mut helpers, mut unblocked) = (0, 0);
			let most = rayon::current_num_threads().clamp(2, 16) - 1;
			let mut second = f64::INFINITY;
			for index in 0..times.len() {
				let started = Instant::now();
				let built = ahead.take(index, &build);
				// slow second and third frames (the first reads the untouched roots once, and the second
				// alone may still settle: an attack clip's took 12 ms, its others 2.4): helpers from now, as
				// many as it takes for scenes to come about as fast as the draws take them, rather than
				// one at a time each after the draws ran dry (a 5,053-part map sim's frames take ~25 ms to
				// make, a helper's own document ~130 ms before its first)
				if index == 1 {
					second = started.elapsed().as_secs_f64() * 1000.0;
				}
				if index == 2 && helpers == 0 {
					let ms = second.min(started.elapsed().as_secs_f64() * 1000.0);
					let (free, others) = (crate::available_memory(), PRODUCING.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(1));
					let roomy = others == 0 && free.is_some_and(|free| free > EARLY_HELPERS_FREE);
					if std::env::var_os("ROR_PROFILE").is_some() {
						eprintln!("profile: the producer's second and third frames {ms:.1} ms or more; {} GB free, {others} other clips making scenes: {}", free.map_or(0, |free| free >> 30), if roomy { "helpers now" } else { "helpers only as the draws wait" });
					}
					if ms > EARLY_HELPERS_FROM_MS && roomy {
						for _ in 0..((ms / EARLY_HELPERS_FROM_MS).ceil() as usize).min(EARLY_HELPERS).min(most) {
							let Some(slot) = HelperSlot::take(most) else { break };
							helpers += 1;
							scope.spawn(|| {
								let (_slot, _clip) = (slot, crate::par::ClipThread::enter());
								ahead.help(&build)
							});
						}
					}
				}
				match sender.try_send(built) {
					Ok(()) => unblocked += 1,
					Err(std::sync::mpsc::TrySendError::Full(built)) => {
						unblocked = 0;
						if sender.send(built).is_err() {
							break;
						}
					}
					Err(std::sync::mpsc::TrySendError::Disconnected(_)) => break,
				}
				ahead.sent(index);
				// the draws took every scene as it came, twice the queue's length in a row: the
				// producer is what they wait for
				if unblocked >= 2 * SCENES_AHEAD
					&& helpers < most
					&& let Some(slot) = HelperSlot::take(most)
				{
					helpers += 1;
					unblocked = 0;
					scope.spawn(|| {
						let (_slot, _clip) = (slot, crate::par::ClipThread::enter());
						ahead.help(&build)
					});
				}
			}
			ahead.stop();
		});
	}
}

/// The frames a clip's producer has made ahead of the one it sends next, by helpers
/// ([`Builder::produce`]): each takes the next frame nobody has, a few past the last sent at most.
struct Ahead<T> {
	count: usize,
	state: Mutex<AheadState<T>>,
	changed: parking_lot::Condvar,
}

struct AheadState<T> {
	/// The next frame nobody has taken
	next: usize,
	/// Frames made, waiting to be sent
	made: std::collections::BTreeMap<usize, T>,
	/// The last frame sent, plus one
	sent: usize,
	stopped: bool,
}

impl<T> Ahead<T> {
	fn new(count: usize) -> Self {
		Self { count, state: Mutex::new(AheadState { next: 0, made: Default::default(), sent: 0, stopped: false }), changed: parking_lot::Condvar::new() }
	}

	/// Frame `index`: made by a helper (waiting for it while one makes it), else made here.
	fn take(&self, index: usize, build: &(dyn Fn(usize) -> T + Sync)) -> T {
		let mut state = self.state.lock();
		loop {
			if let Some(built) = state.made.remove(&index) {
				return built;
			}
			if state.next <= index {
				state.next = index + 1;
				drop(state);
				return build(index);
			}
			self.changed.wait(&mut state);
		}
	}

	fn sent(&self, index: usize) {
		self.state.lock().sent = index + 1;
		self.changed.notify_all();
	}

	fn stop(&self) {
		self.state.lock().stopped = true;
		self.changed.notify_all();
	}

	/// A helper's loop: the next frame nobody has, while it is within [`SCENES_AHEAD`] of the queue's
	/// end, until the clip ends or the producer stops.
	fn help(&self, build: &(dyn Fn(usize) -> T + Sync)) {
		let mut state = self.state.lock();
		loop {
			if state.stopped || state.next >= self.count {
				return;
			}
			if state.next >= state.sent + 2 * SCENES_AHEAD {
				self.changed.wait(&mut state);
				continue;
			}
			let index = state.next;
			state.next += 1;
			drop(state);
			let built = build(index);
			state = self.state.lock();
			state.made.insert(index, built);
			self.changed.notify_all();
		}
	}
}

/// A document a clip's producer drew a moment of, with the playback resolved against it (none: the
/// builder's own document, for a playback that changes no property), and its roots no moment changes
/// once extracted.
struct Kept<'s> {
	document: Option<Document>,
	resolved: scene::playback::Resolved<'s>,
	roots: Option<scene::extract::RootReuse>,
}

/// Whether a clip's frames reuse the extraction of the roots its playback never changes
/// (`ROR_SCENE_REUSE`: `off`, or `check`: compare every frame's scene with its whole extraction).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SceneReuse {
	On,
	Off,
	Check,
}

fn scene_reuse() -> SceneReuse {
	static REUSE: OnceLock<SceneReuse> = OnceLock::new();
	*REUSE.get_or_init(|| match std::env::var("ROR_SCENE_REUSE").as_deref() {
		Ok("off") => SceneReuse::Off,
		Ok("check") => SceneReuse::Check,
		_ => SceneReuse::On,
	})
}

/// Whether a frame whose scene and camera are the frame before's ([`same_scene`]) is that frame's image
/// again, not drawn (`ROR_FRAME_REUSE`: `off`, or `check`: drawn all the same and its pixels compared
/// with the frame before's, an error where they differ).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FrameReuse {
	On,
	Off,
	Check,
}

fn frame_reuse() -> FrameReuse {
	static REUSE: OnceLock<FrameReuse> = OnceLock::new();
	*REUSE.get_or_init(|| match std::env::var("ROR_FRAME_REUSE").as_deref() {
		Ok("off") => FrameReuse::Off,
		Ok("check") => FrameReuse::Check,
		_ => FrameReuse::On,
	})
}

/// Every moment of `times` of `job` made with its untouched roots extracted once and compared with
/// the whole extraction (`ROR_SCENE_REUSE=check`, without a GPU): the frames compared, the roots
/// extracted once, and both ways' milliseconds (examples/scene_reuse.rs over recordings on disk).
#[doc(hidden)]
pub fn check_scene_reuse(job: &VideoJob, scenes: &SceneCache, times: &[f32]) -> Result<(usize, usize, usize, f64, f64)> {
	let builder = Builder::new(|| Ok(std::fs::read(&job.input)?), job, scenes)?;
	if builder.fixed.is_some() || builder.lit.is_some() {
		return Ok((0, 0, 0, 0.0, 0.0));
	}
	let held = Mutex::new(Vec::new());
	let (mut whole_ms, mut reused_ms) = (0.0, 0.0);
	for &time in times {
		let started = Instant::now();
		let _ = builder.scene_checked(time, Some(&held), SceneReuse::Off)?;
		whole_ms += started.elapsed().as_secs_f64() * 1000.0;
		let started = Instant::now();
		let _ = builder.scene_checked(time, Some(&held), SceneReuse::On)?;
		reused_ms += started.elapsed().as_secs_f64() * 1000.0;
		builder.scene_checked(time, Some(&held), SceneReuse::Check)?;
	}
	let kept = held.lock();
	let (reused, roots) = kept.iter().find_map(|kept| kept.roots.as_ref().zip(kept.document.as_ref()).map(|(roots, document)| (roots.count(), document.roots.len()))).unwrap_or((0, 0));
	Ok((times.len(), reused, roots, whole_ms, reused_ms))
}

/// Whether two frames' scenes draw the same image under the same camera: every field but the warnings
/// (they are merged either way), compared value for value; the cheap ones first. Scenes of different
/// documents (a producer's helpers each read the file again) name their instances by other referents
/// and never compare equal: the frame is drawn.
fn same_scene(a: &Scene, b: &Scene) -> bool {
	let Scene { parts, lights, beams, highlights, kept_glass, particles, trails, guis, environment, materials, warnings: _, lineage, live, setting, play } = a;
	*live == b.live
		&& *play == b.play
		&& *setting == b.setting
		&& parts.len() == b.parts.len()
		&& *environment == b.environment
		&& *materials == b.materials
		&& *kept_glass == b.kept_glass
		&& *lights == b.lights
		&& *highlights == b.highlights
		&& *beams == b.beams
		&& *trails == b.trails
		&& *guis == b.guis
		&& *particles == b.particles
		&& *parts == b.parts
		&& *lineage == b.lineage
}

/// The most frames whose GUIs are found together ([`find_guis`]), and how many found wait for their
/// upload: their images are in memory meanwhile (13 telegraphs: 34 MB a frame).
const GUI_FRAMES: usize = 16;
const FOUND_FRAMES: usize = 4;
/// Scenes made ahead of the GUI stage: as many as it finds together for a clip with GUIs, a few
/// otherwise (each cosmetic's 2-3 MB; the producer outpaces the draws).
const SCENES_AHEAD: usize = 4;
/// A clip's second and third frames both taking longer than this to make start a helper for every such
/// stretch of the quicker one's time, up to [`EARLY_HELPERS`] ([`Builder::produce`]): a cosmetic's take 2-4 ms and start none.
/// Only with more than [`EARLY_HELPERS_FREE`] of memory free and no other clip making scenes in this
/// process: their documents sooner cost a map sim ~200 MB more at its peak.
const EARLY_HELPERS_FROM_MS: f64 = 8.0;
const EARLY_HELPERS: usize = 4;
const EARLY_HELPERS_FREE: u64 = 16 << 30;

/// Clips making scenes in this process now (the service's jobs side by side).
static PRODUCING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Producer helpers in this process now, every clip's together ([`HelperSlot`]).
static HELPERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// One of the process's producer helpers: as many in all as one clip may start (one a worker thread less
/// one), not that many a clip. Each holds a document of its own (a cutscene's ~135 MB), and the
/// scenes they make share rayon's threads: four cutscenes side by side in one service started 7 each,
/// 1.2-1.5 GB more at the peak than with one each, in the same wall time (66.4-66.7 s against 63.8-68.4).
struct HelperSlot;

impl HelperSlot {
	fn take(most: usize) -> Option<Self> {
		HELPERS.fetch_update(std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Acquire, |helpers| (helpers < most).then_some(helpers + 1)).ok().map(|_| Self)
	}
}

impl Drop for HelperSlot {
	fn drop(&mut self) {
		HELPERS.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
	}
}

/// One clip counted in [`PRODUCING`] while it lives.
struct Producing;

impl Producing {
	fn new() -> Self {
		PRODUCING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
		Self
	}
}

impl Drop for Producing {
	fn drop(&mut self) {
		PRODUCING.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
	}
}

/// The GUI stage between the producer and the draws: the GUI images of every batch of scenes the
/// producer has ready (as many as wait, up to [`GUI_FRAMES`]), found in one go
/// ([`Renderer::resolve_frame_guis`]: from the GPU cache, disk or one set of rur calls), so that an
/// animated GUI costs a share of a few rur calls rather than one each frame, and no draw waits on
/// rur with its GPU memory admitted. Frames go on in order, each with its camera's GUIs.
fn find_guis(renderer: &Renderer, view: &ViewRequest, cameras: &[CameraSpec], receiver: Receiver<Result<(Scene, PreparedSky)>>, sender: SyncSender<Result<(Scene, PreparedSky, FrameGuis)>>) {
	let mut index = 0;
	while let Ok(first) = receiver.recv() {
		let mut batch = vec![first];
		while batch.len() < GUI_FRAMES
			&& let Ok(more) = receiver.try_recv()
		{
			batch.push(more);
		}
		let requests: Vec<ViewRequest> = (0..batch.len())
			.map(|offset| {
				let mut request = view.clone();
				if let Some(camera) = cameras.get(index + offset) {
					request.camera = camera.clone();
				}
				request
			})
			.collect();
		let frames: Vec<(&Scene, &ViewRequest)> = batch.iter().zip(&requests).filter_map(|(built, request)| built.as_ref().ok().map(|(scene, _)| (scene, request))).collect();
		let mut found = renderer.resolve_frame_guis(&frames).into_iter();
		for built in batch {
			let frame = built.map(|(scene, sky)| (scene, sky, found.next().unwrap_or_default()));
			if sender.send(frame).is_err() {
				return;
			}
			index += 1;
		}
	}
}

/// A frame ready to draw: its scene, its sky's maps, its GUI textures on the GPU and what went wrong
/// finding them.
type ReadyFrame = (Scene, PreparedSky, crate::render::guis::GuiTextures, Vec<String>);

/// The upload stage between the GUI stage and the draws: each frame's GUI images onto the GPU while
/// the frames before it draw (7.7 ms a frame of 13 telegraphs in the draw loop before), a frame or
/// two ahead, so that only the frames about to draw hold them in GPU memory.
fn upload_guis(renderer: &Renderer, receiver: Receiver<Result<(Scene, PreparedSky, FrameGuis)>>, sender: SyncSender<Result<(ReadyFrame, f64)>>) {
	for found in receiver {
		let ready = found.map(|(scene, sky, guis)| {
			let started = Instant::now();
			let mut warnings = Vec::new();
			let textures = renderer.upload_frame_guis(guis, &mut warnings);
			((scene, sky, textures, warnings), started.elapsed().as_secs_f64() * 1000.0)
		});
		if sender.send(ready).is_err() {
			return;
		}
	}
}

/// What a moment draws, for framing a clip: the box of its parts as they draw and of its beams' and
/// trails' ends and points, and its live particles (centre, side).
struct Drawn {
	solid: Option<(Vec3, Vec3)>,
	particles: Vec<(Vec3, f32)>,
}

fn drawn(renderer: &Renderer, scene: &Scene) -> Drawn {
	let mut boxes: Vec<(Vec3, Vec3)> = Vec::new();
	if !scene.parts.is_empty()
		&& let Some(parts) = scene.framing_bounds_drawn(&renderer.drawn_boxes(scene))
	{
		boxes.push(parts);
	}
	let mut points: Vec<Vec3> = scene.beams.iter().flat_map(|beam| [Vec3::from(beam.attachment0.translation), Vec3::from(beam.attachment1.translation)]).collect();
	points.extend(scene.trails.iter().flat_map(|trail| trail.points.iter().flat_map(|point| [point.position0, point.position1])));
	if let Some(first) = points.first() {
		boxes.push(points.iter().fold((*first, *first), |(low, high), point| (low.min(*point), high.max(*point))));
	}
	let particles = scene.particles.par_iter().flat_map_iter(|emitter| crate::render::particles::simulate(emitter).into_iter().filter(|particle| particle.opacity > 0.0).map(|particle| (particle.position, particle.size))).collect();
	Drawn { solid: boxes.into_iter().reduce(|(low, high), (min, max)| (low.min(min), high.max(max))), particles }
}

/// The box an orbit frames for one scene, as a clip's moment frames it ([`drawn`]): `render --map`
/// frames the model alone this way.
pub(crate) fn subject_box(renderer: &Renderer, scene: &Scene) -> Option<(Vec3, Vec3)> {
	clip_box(vec![drawn(renderer, scene)])
}

/// The box a clip's moments draw: their parts, beams and trails, and their particles pooled, the
/// farthest few each way left out as stray sparks would shrink the rest (the few drifting off a burst
/// late are a few of the clip's particles, where one moment's own would count them all).
fn clip_box(moments: Vec<Drawn>) -> Option<(Vec3, Vec3)> {
	let mut boxes: Vec<(Vec3, Vec3)> = moments.iter().filter_map(|moment| moment.solid).collect();
	let particles: Vec<(Vec3, f32)> = moments.into_iter().flat_map(|moment| moment.particles).collect();
	if !particles.is_empty() {
		boxes.push(robust_box(&particles));
	}
	boxes.into_iter().reduce(|(low, high), (min, max)| (low.min(min), high.max(max)))
}

/// The box of particles (centre, side) without the farthest 3 % each way, padded by half the median
/// particle's side: their images' soft edges and empty margins leave most of a square clear.
fn robust_box(particles: &[(Vec3, f32)]) -> (Vec3, Vec3) {
	let trim = if particles.len() >= 30 { particles.len() * 3 / 100 } else { 0 };
	let mut low = Vec3::ZERO;
	let mut high = Vec3::ZERO;
	for axis in 0..3 {
		let mut values: Vec<f32> = particles.iter().map(|(position, _)| position[axis]).collect();
		values.sort_by(f32::total_cmp);
		low[axis] = values[trim];
		high[axis] = values[values.len() - 1 - trim];
	}
	let mut sizes: Vec<f32> = particles.iter().map(|(_, size)| size.max(0.0)).collect();
	sizes.sort_by(f32::total_cmp);
	let half = sizes[sizes.len() / 2] * 0.5;
	(low - Vec3::splat(half), high + Vec3::splat(half))
}

/// Up to `count` indices spread over `0..frames`, the first and the last among them.
fn spread(frames: usize, count: usize) -> Vec<usize> {
	if frames == 0 || count == 0 {
		return Vec::new();
	}
	if count == 1 || frames == 1 {
		return vec![0];
	}
	let mut indices: Vec<usize> = (0..count).map(|index| ((index as f64 * (frames - 1) as f64) / (count - 1) as f64).round() as usize).collect();
	indices.dedup();
	indices
}

/// Whether two frames' cameras are the same (their specs, as JSON: a camera track's keys and an orbit's
/// fitted values alike).
fn same_camera(a: &CameraSpec, b: &CameraSpec) -> bool {
	serde_json::to_string(a).ok().is_some_and(|a| serde_json::to_string(b).ok().is_some_and(|b| a == b))
}

/// Every frame's camera, and the box they frame.
type Framing = (Vec<CameraSpec>, Option<(Vec3, Vec3)>);

/// Each frame's camera: a camera track's key at its moment; else an orbit fitted once to the box every
/// sampled moment draws (or the box or instance given), the same distance all round when it turns;
/// any other camera as given. Also the box framed.
fn cameras<'s>(renderer: &Renderer, builder: &'s Builder, times: &[f32], job: &VideoJob, held: &Mutex<Vec<Kept<'s>>>) -> Result<Framing> {
	if let Some(track) = job.camera_track.as_deref().filter(|track| !track.is_empty()) {
		return Ok((times.iter().map(|&time| track_camera(track, time)).collect(), None));
	}
	let (width, height) = (job.view.width.max(1), job.view.height.max(1));
	let CameraSpec::Orbit { yaw, pitch, distance, padding, target, fov } = job.view.camera else {
		return Ok((vec![job.view.camera.clone(); times.len()], None));
	};
	let framed = match (&job.frame, &job.focus) {
		(Some([min, max]), _) => Some((Vec3::from(*min).min(Vec3::from(*max)), Vec3::from(*min).max(Vec3::from(*max)))),
		_ if distance.is_some() && target.is_some() => None,
		(None, focus) => {
			let samples = spread(times.len(), 9);
			// on the clip's own document, one moment after another (each spread over the cores)
			let moments: Vec<Result<Drawn>> = samples
				.iter()
				.map(|&index| {
					let scene = builder.scene_with(times[index], Some(held))?;
					Ok(match focus {
						Some(focus) => Drawn { solid: scene.bounds_at(focus), particles: Vec::new() },
						None => drawn(renderer, scene.framing_subject().as_ref().unwrap_or(&scene)),
					})
				})
				.collect();
			let union = clip_box(moments.into_iter().collect::<Result<Vec<_>>>()?);
			if union.is_none()
				&& let Some(focus) = focus
			{
				bail!("focus {focus:?}: no part or effect is at or under it at any moment");
			}
			union
		}
	};
	let Some(bounds) = framed else {
		return Ok((vec![job.view.camera.clone(); times.len()], None));
	};
	let orbit_at = |index: usize| yaw + job.orbit * index as f32 / times.len().max(1) as f32;
	// one distance for the clip: the farthest any yaw it turns through needs
	let yaws: Vec<f32> = if job.orbit == 0.0 { vec![yaw] } else { (0..24).map(|step| yaw + job.orbit * step as f32 / 24.0).collect() };
	let mut fitted = distance;
	let mut center = target;
	for sample in yaws {
		if let CameraSpec::Orbit { distance: Some(needed), target: Some(middle), .. } = (CameraSpec::Orbit { yaw: sample, pitch, distance, padding, target, fov }).framing(bounds, width, height) {
			fitted = Some(fitted.map_or(needed, |known: f32| known.max(needed)));
			center = Some(middle);
		}
	}
	let cameras = (0..times.len()).map(|index| CameraSpec::Orbit { yaw: orbit_at(index), pitch, distance: fitted, padding, target: center, fov }).collect();
	Ok((cameras, Some(bounds)))
}

/// The playback of a clip whose model has parts the game keeps under the camera (scene::follow): a
/// motion key for each such part at every frame's moment, where it stands for that frame's camera
/// (the first key holds before the clip starts), so their emitters' particles stay in the world where
/// they were emitted as the camera moves on, as the game's are. None when no emitter rides on one.
fn follow_playback<'s>(builder: &'s Builder, held: &Mutex<Vec<Kept<'s>>>, job: &VideoJob, cameras: &[CameraSpec], framed: Option<(Vec3, Vec3)>, times: &[f32]) -> Result<Option<Playback>> {
	let Some(&first) = times.first() else { return Ok(None) };
	let scene = builder.scene_with(first, Some(held))?;
	let mut parts: Vec<(String, scene::follow::Follow)> = Vec::new();
	for emitter in &scene.particles {
		let Some(follow) = emitter.follow else { continue };
		let Some(part) = scene.lineage.get(&emitter.path).and_then(|lineage| lineage.len().checked_sub(1 + follow.levels as usize).map(|index| &lineage[index])) else { continue };
		if !parts.iter().any(|(path, _)| path == part) {
			parts.push((part.clone(), follow));
		}
	}
	if parts.is_empty() {
		return Ok(None);
	}
	let (width, height) = (job.view.width.max(1), job.view.height.max(1));
	let placed: Vec<crate::render::camera::Camera> = cameras.iter().map(|spec| spec.place(framed, width, height)).collect();
	let mut playback = watched(job);
	for (path, follow) in parts {
		// a playback that moves the part already says where it is
		if playback.motion.iter().any(|motion| motion.target == path) {
			continue;
		}
		let keys = times
			.iter()
			.zip(&placed)
			.map(|(&at, camera)| {
				let at_camera = follow.position(camera.eye, -Vec3::from(camera.world.matrix3.z_axis));
				scene::playback::MotionKey { at, cframe: vec![at_camera.x, at_camera.y, at_camera.z], easing: None, direction: None }
			})
			.collect();
		playback.motion.push(scene::playback::Motion { target: path, keys, relative: false, tagged: Some(scene::follow::TAG.to_string()), physics_from: None });
	}
	Ok(Some(playback))
}

/// Where frames go: x264 in the process into ror's MP4, ffmpeg's stdin, ror's animated PNG, or a folder
/// of PNGs.
enum Encoder {
	#[cfg(ror_x264)]
	X264 { encoder: crate::x264::Encoder, mp4: crate::mp4::Mp4Writer, sei: Vec<u8>, temporary: PathBuf, target: PathBuf },
	Nvenc { encoder: crate::nvenc::Encoder, mp4: crate::mp4::Mp4Writer, temporary: PathBuf, target: PathBuf },
	Ffmpeg { child: std::process::Child, input: Option<std::fs::File>, errors: std::thread::JoinHandle<String>, temporary: PathBuf, target: PathBuf, name: String, nv12: bool },
	Apng { writer: Box<png::Writer<std::io::BufWriter<std::fs::File>>>, temporary: PathBuf, target: PathBuf },
	Frames { dir: PathBuf, width: u32, height: u32 },
}

/// An encoder started with its clip ([`run`]): one the clip never fed (it failed before its frames)
/// has its process stopped and its partial file removed.
struct EncoderStarted(Option<Encoder>);

impl EncoderStarted {
	fn take(mut self) -> Encoder {
		self.0.take().expect("the encoder, once")
	}
}

impl Drop for EncoderStarted {
	fn drop(&mut self) {
		match self.0.take() {
			Some(Encoder::Ffmpeg { mut child, input, temporary, .. }) => {
				drop(input);
				let _ = child.kill();
				let _ = child.wait();
				let _ = std::fs::remove_file(&temporary);
			}
			Some(Encoder::Apng { writer, temporary, .. }) => {
				drop(writer);
				let _ = std::fs::remove_file(&temporary);
			}
			#[cfg(ror_x264)]
			Some(Encoder::X264 { encoder, mp4, temporary, .. }) => {
				drop((encoder, mp4));
				let _ = std::fs::remove_file(&temporary);
			}
			Some(Encoder::Nvenc { encoder, mp4, temporary, .. }) => {
				drop((encoder, mp4));
				let _ = std::fs::remove_file(&temporary);
			}
			_ => {}
		}
	}
}

/// The job's copies being made ([`VideoCopy`]): an ffmpeg each, fed every frame's NV12 by a writer
/// thread of its own, so the clip's encoder only copies the frame once and goes on (writing on the
/// encoder's thread slowed a 4K clip's frames 14.2 -> 16.1 ms). One that stops (no NVENC, a bad
/// argument) takes no more frames and says why; the clip goes on. Dropped unfinished (the clip
/// failed), each is stopped and its partial file removed.
struct CopyRuns {
	runs: Vec<CopyRun>,
	/// Frames' buffers, each taken again once no writer holds it
	buffers: Vec<Arc<Vec<u8>>>,
}

/// Frames a copy's writer may have queued before the clip's encoder waits for it.
const COPY_QUEUE: usize = 4;

struct CopyRun {
	child: std::process::Child,
	/// To the writer, which holds ffmpeg's stdin; none once it stopped
	frames: Option<SyncSender<Arc<Vec<u8>>>>,
	writer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
	errors: Option<std::thread::JoinHandle<String>>,
	temporary: PathBuf,
	target: PathBuf,
}

impl CopyRuns {
	fn none() -> CopyRuns {
		CopyRuns { runs: Vec::new(), buffers: Vec::new() }
	}

	fn start(copies: &[VideoCopy], ffmpeg: Option<&Path>, (width, height): (u32, u32), fps: f32) -> Result<CopyRuns> {
		let mut runs = Vec::new();
		for copy in copies {
			let ffmpeg = ffmpeg.context("a copy needs ffmpeg (ROR_FFMPEG, PATH or winget's Gyan.FFmpeg)")?;
			let muxer = match copy.file.extension().and_then(|extension| extension.to_str()).map(str::to_ascii_lowercase).as_deref() {
				Some("mp4" | "m4v") => "mp4",
				Some("mov") => "mov",
				Some("mkv") => "matroska",
				Some("webm") => "webm",
				_ => bail!("copy {}: an .mp4, .m4v, .mov, .mkv or .webm file", copy.file.display()),
			};
			if let Some(parent) = copy.file.parent().filter(|parent| !parent.as_os_str().is_empty()) {
				std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
			}
			let temporary = copy.file.with_extension(format!("tmp{}", std::process::id()));
			let rate = format!("{}", (fps * 1000.0).round() / 1000.0);
			let tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-color_range", "tv", "-chroma_sample_location", "center"];
			let mut command = Command::new(ffmpeg);
			command.args(["-hide_banner", "-nostats", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "nv12", "-s", &format!("{width}x{height}"), "-framerate", &rate]);
			command.args(tags).args(["-i", "-"]).args(&copy.args).args(tags);
			command.args(["-f", muxer]).arg(&temporary).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());
			let input = encoder_input(&mut command);
			below_normal(&mut command);
			let mut child = command.spawn().with_context(|| format!("starting {}", ffmpeg.display()))?;
			let mut stderr = child.stderr.take().context("ffmpeg's stderr")?;
			let errors = std::thread::spawn(move || {
				let mut text = String::new();
				let _ = std::io::Read::read_to_string(&mut stderr, &mut text);
				text
			});
			let mut input: Box<dyn Write + Send> = match input {
				Some(file) => Box::new(file),
				None => Box::new(child.stdin.take().context("ffmpeg's stdin")?),
			};
			let (frames, queue) = sync_channel::<Arc<Vec<u8>>>(COPY_QUEUE);
			let writer = std::thread::Builder::new().name("ror-copy".into()).spawn(move || -> std::io::Result<()> {
				for frame in queue {
					input.write_all(&frame)?;
				}
				Ok(())
			})?;
			runs.push(CopyRun { child, frames: Some(frames), writer: Some(writer), errors: Some(errors), temporary, target: copy.file.clone() });
		}
		Ok(CopyRuns { runs, buffers: Vec::new() })
	}

	fn is_empty(&self) -> bool {
		self.runs.is_empty()
	}

	/// One frame's NV12 copied once, into a buffer every copy's writer takes.
	fn write(&mut self, nv12: &[u8]) {
		if self.runs.iter().all(|run| run.frames.is_none()) {
			return;
		}
		let mut frame = match self.buffers.iter().position(|buffer| Arc::strong_count(buffer) == 1) {
			Some(index) => self.buffers.swap_remove(index),
			None => Arc::new(Vec::with_capacity(nv12.len())),
		};
		let bytes = Arc::get_mut(&mut frame).expect("a buffer no writer holds");
		bytes.clear();
		bytes.extend_from_slice(nv12);
		for run in &mut self.runs {
			if let Some(frames) = &run.frames
				&& frames.send(frame.clone()).is_err()
			{
				run.frames = None;
			}
		}
		self.buffers.push(frame);
	}

	/// Each copy's input closed, its ffmpeg waited for and its file put in place.
	fn finish(mut self) -> Vec<CopyResult> {
		let started = Instant::now();
		std::mem::take(&mut self.runs)
			.into_iter()
			.map(|mut run| {
				drop(run.frames.take());
				let broken = run.writer.take().and_then(|writer| writer.join().unwrap_or_else(|_| Err(std::io::Error::other("its writer panicked"))).err());
				let status = run.child.wait();
				let said = run.errors.take().and_then(|errors| errors.join().ok()).unwrap_or_default();
				let error = match status {
					Ok(status) if status.success() && broken.is_none() => std::fs::rename(&run.temporary, &run.target).err().map(|error| format!("writing {}: {error}", run.target.display())),
					Ok(status) => Some(format!("ffmpeg ({status}): {}{}", said.trim().lines().rev().take(6).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(" / "), broken.map(|broken| format!(" / {broken}")).unwrap_or_default())),
					Err(error) => Some(format!("ffmpeg: {error}")),
				};
				if error.is_some() {
					let _ = std::fs::remove_file(&run.temporary);
				}
				CopyResult { file: run.target.clone(), written: error.is_none(), error, waited_ms: started.elapsed().as_secs_f64() * 1000.0 }
			})
			.collect()
	}
}

impl Drop for CopyRuns {
	fn drop(&mut self) {
		for mut run in self.runs.drain(..) {
			drop(run.frames.take());
			// a writer stuck on a full pipe fails once its ffmpeg is gone
			let _ = run.child.kill();
			if let Some(writer) = run.writer.take() {
				let _ = writer.join();
			}
			let _ = run.child.wait();
			let _ = std::fs::remove_file(&run.temporary);
		}
	}
}

enum OutputFrame<'a> {
	/// A frame drawn: its pixels once the GPU has them, its contact-sheet index, whether the encoder keeps
	/// its pixels for a frame repeating it, and (`ROR_FRAME_REUSE=check`) whether they must be the kept
	/// ones.
	Drawn { view: PendingView<'a>, sheet: Option<usize>, keep: bool, compare: bool },
	/// The frame before's image again: its scene and camera the same ([`same_scene`]).
	Repeat,
}

struct EncodedVideo {
	name: String,
	profile: OutputProfile,
	thumbnails: Vec<(usize, Vec<u8>)>,
	/// Each frame's, in order.
	brightness: Vec<Brightness>,
	copies: Vec<CopyResult>,
}

impl Encoder {
	/// What encodes the clip, as the summary's `encoder` names it.
	fn name(&self) -> String {
		match self {
			#[cfg(ror_x264)]
			Encoder::X264 { .. } => "x264".into(),
			Encoder::Nvenc { .. } => "nvenc".into(),
			Encoder::Ffmpeg { name, .. } => name.clone(),
			Encoder::Apng { .. } => "apng".into(),
			Encoder::Frames { .. } => "png".into(),
		}
	}

	/// Whether it takes each frame as NV12 (an MP4's), which a [`VideoCopy`] is fed.
	fn takes_nv12(&self) -> bool {
		match self {
			#[cfg(ror_x264)]
			Encoder::X264 { .. } => true,
			Encoder::Nvenc { .. } => true,
			Encoder::Ffmpeg { nv12, .. } => *nv12,
			_ => false,
		}
	}

	fn new(format: VideoFormat, ffmpeg: Option<&Path>, out: &Path, (width, height): (u32, u32), fps: f32, frames: usize, nvenc: Option<crate::nvenc::Rate>) -> Result<Encoder> {
		match format {
			VideoFormat::Mp4 if nvenc.is_some() => {
				let target = out.join(format.file_name());
				let temporary = target.with_extension(format!("tmp{}", std::process::id()));
				let rate = crate::mp4::frame_rate(fps);
				let mut encoder = crate::nvenc::Encoder::open(&crate::nvenc::Settings { width, height, fps: rate, rate: nvenc.expect("checked") })?;
				let headers = encoder.headers()?;
				let mp4 = crate::mp4::Mp4Writer::create(&temporary, width, height, rate, &headers.sps, &headers.pps, frames, "ror nvenc", true)?;
				Ok(Encoder::Nvenc { encoder, mp4, temporary, target })
			}
			VideoFormat::Png => {
				let dir = out.join(format.file_name());
				if dir.is_dir() {
					std::fs::remove_dir_all(&dir).with_context(|| format!("clearing {}", dir.display()))?;
				}
				std::fs::create_dir_all(&dir)?;
				Ok(Encoder::Frames { dir, width, height })
			}
			VideoFormat::Apng => {
				let target = out.join(format.file_name());
				let temporary = target.with_extension(format!("tmp{}", std::process::id()));
				let file = std::fs::File::create(&temporary).with_context(|| format!("creating {}", temporary.display()))?;
				let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
				encoder.set_color(png::ColorType::Rgba);
				encoder.set_depth(png::BitDepth::Eight);
				encoder.set_compression(png::Compression::Fast);
				encoder.set_animated(frames as u32, 0)?;
				encoder.set_frame_delay(100, (fps * 100.0).round().clamp(1.0, 65535.0) as u16)?;
				let writer = Box::new(encoder.write_header()?);
				Ok(Encoder::Apng { writer, temporary, target })
			}
			#[cfg(ror_x264)]
			VideoFormat::Mp4 if x264_in_process(format)? => {
				let target = out.join(format.file_name());
				let temporary = target.with_extension(format!("tmp{}", std::process::id()));
				let rate = crate::mp4::frame_rate(fps);
				let settings = crate::x264::Settings { width, height, fps: rate, preset: X264_PRESET.into(), crf: X264_CRF, threads: x264_threads(width, height) };
				let mut encoder = crate::x264::Encoder::open(&settings)?;
				let headers = encoder.headers()?;
				let mp4 = crate::mp4::Mp4Writer::create(&temporary, width, height, rate, &headers.sps, &headers.pps, frames, "ror x264", false)?;
				Ok(Encoder::X264 { encoder, mp4, sei: headers.sei, temporary, target })
			}
			_ => {
				let ffmpeg = ffmpeg.context("no ffmpeg")?;
				let listed = ffmpeg_encoders(ffmpeg)?;
				let choice = std::env::var("ROR_VIDEO_ENCODER").unwrap_or_else(|_| "auto".into());
				let name = video_encoder(format, ffmpeg, &listed, width, height, frames, &choice)?;
				let target = out.join(format.file_name());
				let temporary = target.with_extension(format!("tmp{}", std::process::id()));
				let rate = format!("{}", (fps * 1000.0).round() / 1000.0);
				let mut command = Command::new(ffmpeg);
				let nv12 = format == VideoFormat::Mp4;
				command.args(["-hide_banner", "-nostats", "-loglevel", "error", "-y", "-filter_threads", "2", "-f", "rawvideo", "-pix_fmt", if nv12 { "nv12" } else { "rgba" }, "-s", &format!("{width}x{height}"), "-framerate", &rate]);
				if nv12 {
					command.args(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-color_range", "tv", "-chroma_sample_location", "center"]);
				}
				command.args(["-i", "-"]);
				match (format, name) {
					(VideoFormat::Mp4, "libx264") => command.args(x264_arguments(width, height)),
					(VideoFormat::Mp4, "h264_nvenc") => { nvenc_options(&mut command); &mut command },
					(VideoFormat::Mp4, other) => command.args(["-c:v", other, "-b:v", "8M", "-pix_fmt", "yuv420p"]),
					(VideoFormat::Webm, "libvpx-vp9") => command.args(["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-crf", "30", "-b:v", "0", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1", "-auto-alt-ref", "0"]),
					(VideoFormat::Webm, other) => command.args(["-c:v", other, "-pix_fmt", "yuva420p", "-crf", "10", "-b:v", "4M", "-auto-alt-ref", "0"]),
					(_, _) => command.args(["-filter_complex", "[0:v]split[a][b];[a]palettegen=stats_mode=diff:reserve_transparent=1[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle:alpha_threshold=128", "-loop", "0"]),
				};
				let muxer = match format {
					VideoFormat::Mp4 => "mp4",
					VideoFormat::Webm => "webm",
					_ => "gif",
				};
				if format == VideoFormat::Mp4 {
					command.args(["-movflags", "+faststart", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-color_range", "tv", "-chroma_sample_location", "center"]);
				}
				command.args(["-f", muxer]).arg(&temporary).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());
				let input = encoder_input(&mut command);
				below_normal(&mut command);
				let mut child = command.spawn().with_context(|| format!("starting {}", ffmpeg.display()))?;
				let mut stderr = child.stderr.take().context("ffmpeg's stderr")?;
				let errors = std::thread::spawn(move || {
					let mut text = String::new();
					let _ = std::io::Read::read_to_string(&mut stderr, &mut text);
					text
				});
				Ok(Encoder::Ffmpeg { child, input, errors, temporary, target, name: name.to_string(), nv12 })
			}
		}
	}

	/// Takes every frame `receiver` gives (to `copies` too), then finishes the file and theirs.
	fn consume<'r>(self, receiver: Receiver<OutputFrame<'r>>, (width, height): (u32, u32), mut copies: CopyRuns) -> Result<EncodedVideo> {
		let mut profile = OutputProfile::default();
		let mut kept = Vec::new();
		let mut brightness = Vec::new();
		let (tile_width, tile_height) = thumbnail_size(width, height);
		let nv12 = self.takes_nv12();
		// NVENC takes a frame in the time of a copy: its thread measures the frame itself, where x264's
		// hands that to the pool to encode meanwhile
		let inline = matches!(&self, Encoder::Nvenc { .. }) && copies.is_empty();
		// the last drawn frame's pixels, while a frame may repeat it
		let mut held: Option<crate::render::Held<'r>> = None;
		let mut consume = |frame: OutputFrame<'r>, write: &mut dyn FnMut(&[u8], &mut OutputProfile) -> Result<()>| -> Result<()> {
			let (view, sheet_index, keep, compare) = match frame {
				OutputFrame::Drawn { view, sheet, keep, compare } => (view, sheet, keep, compare),
				OutputFrame::Repeat => {
					// the same pixels, so the same brightness
					let measured = *brightness.last().context("a repeated first frame")?;
					brightness.push(measured);
					return held.as_ref().context("a repeated frame with none kept")?.with(|pixels| {
						copies.write(pixels);
						write(pixels, &mut profile)
					})?;
				}
			};
			let mut stats = RenderStats::default();
			let earlier = held.take();
			let mut consume_pixels = |pixels: &[u8], rgba: Option<&[u8]>| -> Result<()> {
				if compare
					&& let Some(earlier) = &earlier
					&& !earlier.with(|kept| kept == pixels)?
				{
					bail!("ROR_FRAME_REUSE=check: frame {} (its scene and camera the frame before's) drew other pixels", brightness.len());
				}
				let sheet = match sheet_index {
					Some(_) => Some(rgba.context("sheet RGBA readback")?),
					None => None,
				};
				// the frame's brightness and its sheet thumbnail made on the thread pool while the encoder
				// takes it: on this thread they held up the next frame (a 1440p clip's brightness took
				// 0.93 ms a frame, 234 of its 564 ms, with a free encoder no less)
				let (mut measured, mut thumbnail, mut measure_ms, mut thumbnail_ms) = (Brightness::default(), None, 0.0, 0.0);
				let written = if inline {
					let started = Instant::now();
					measured = nv12_brightness(pixels, width as usize, height as usize);
					measure_ms = started.elapsed().as_secs_f64() * 1000.0;
					if let Some(rgba) = sheet {
						let started = Instant::now();
						thumbnail = Some(downscale(rgba, width, height, tile_width, tile_height));
						thumbnail_ms = started.elapsed().as_secs_f64() * 1000.0;
					}
					write(pixels, &mut profile)
				} else {
					rayon::in_place_scope(|scope| {
						scope.spawn(|_| {
							let started = Instant::now();
							measured = if nv12 { nv12_brightness(pixels, width as usize, height as usize) } else { rgba_brightness(pixels) };
							measure_ms = started.elapsed().as_secs_f64() * 1000.0;
						});
						if let Some(rgba) = sheet {
							scope.spawn(|_| {
								let started = Instant::now();
								thumbnail = Some(downscale(rgba, width, height, tile_width, tile_height));
								thumbnail_ms = started.elapsed().as_secs_f64() * 1000.0;
							});
						}
						if !copies.is_empty() {
							scope.spawn(|_| copies.write(pixels));
						}
						write(pixels, &mut profile)
					})
				};
				profile.measure_ms += measure_ms;
				profile.thumbnail_ms += thumbnail_ms;
				if let (Some(index), Some(thumbnail)) = (sheet_index, thumbnail) {
					kept.push((index, thumbnail));
				}
				brightness.push(measured);
				written
			};
			let result = match (keep, nv12) {
				(false, true) => view.with_nv12(&mut stats, consume_pixels),
				(false, false) => view.with_rgba(&mut stats, |_, rgba| consume_pixels(rgba, Some(rgba))),
				(true, true) => view.with_nv12_held(&mut stats, consume_pixels).map(|(result, kept)| {
					held = Some(kept);
					result
				}),
				(true, false) => view.with_rgba_held(&mut stats, |_, rgba| consume_pixels(rgba, Some(rgba))).map(|(result, kept)| {
					held = Some(kept);
					result
				}),
			};
			drop(earlier);
			profile.readback_ms += stats.readback_ms;
			result
		};
		let name = match self {
			#[cfg(ror_x264)]
			Encoder::X264 { mut encoder, mut mp4, sei, temporary, target } => {
				let mut flush_ms = 0.0;
				let written = (|| -> Result<()> {
					// x264's version SEI goes before the first frame's units, as ffmpeg puts it (its MP4
					// keeps only the parameter sets aside)
					let mut first = Some(sei);
					let mut put = |packet: Option<crate::x264::Packet<'_>>, mp4: &mut crate::mp4::Mp4Writer| -> Result<()> {
						let Some(packet) = packet else { return Ok(()) };
						if packet.pts != packet.dts {
							bail!("x264 put out frame {} at {}: frames out of order", packet.pts, packet.dts);
						}
						match first.take().filter(|sei| !sei.is_empty()) {
							Some(sei) => mp4.write(&[sei.as_slice(), packet.data].concat(), packet.keyframe),
							None => mp4.write(packet.data, packet.keyframe),
						}
					};
					for (index, frame) in receiver.into_iter().enumerate() {
						consume(frame, &mut |nv12, profile| {
							let started = Instant::now();
							let packet = encoder.encode(Some(nv12), index as i64)?;
							put(packet, &mut mp4)?;
							profile.pipe_ms += started.elapsed().as_secs_f64() * 1000.0;
							Ok(())
						})?;
					}
					let started = Instant::now();
					while encoder.delayed() > 0 {
						let packet = encoder.encode(None, 0)?;
						put(packet, &mut mp4)?;
					}
					flush_ms = started.elapsed().as_secs_f64() * 1000.0;
					drop(encoder);
					mp4.finish()
				})();
				profile.pipe_ms += flush_ms;
				if let Err(error) = written {
					let _ = std::fs::remove_file(&temporary);
					return Err(error);
				}
				std::fs::rename(&temporary, &target).with_context(|| format!("writing {}", target.display()))?;
				"x264".into()
			}
			Encoder::Nvenc { mut encoder, mut mp4, temporary, target } => {
				let written = (|| -> Result<()> {
					for frame in receiver {
						consume(frame, &mut |nv12, profile| {
							let started = Instant::now();
							encoder.send(nv12)?;
							while let Some(packet) = encoder.receive()? {
								mp4.write_at(packet.data, packet.keyframe, packet.display)?;
							}
							profile.pipe_ms += started.elapsed().as_secs_f64() * 1000.0;
							Ok(())
						})?;
					}
					encoder.finish()?;
					while let Some(packet) = encoder.receive()? {
						mp4.write_at(packet.data, packet.keyframe, packet.display)?;
					}
					drop(encoder);
					mp4.finish()
				})();
				if let Err(error) = written {
					let _ = std::fs::remove_file(&temporary);
					return Err(error);
				}
				std::fs::rename(&temporary, &target).with_context(|| format!("writing {}", target.display()))?;
				"nvenc".into()
			}
			Encoder::Frames { dir, width, height } => {
				for (index, frame) in receiver.into_iter().enumerate() {
					consume(frame, &mut |rgba, _| output::write_png(&dir.join(format!("{index:04}.png")), width, height, rgba, true))?;
				}
				"png".into()
			}
			Encoder::Apng { mut writer, temporary, target } => {
				let written = (|| -> Result<()> {
					for frame in receiver {
						consume(frame, &mut |rgba, _| Ok(writer.write_image_data(rgba)?))?;
					}
					writer.finish()?;
					Ok(())
				})();
				if let Err(error) = written {
					let _ = std::fs::remove_file(&temporary);
					return Err(error);
				}
				std::fs::rename(&temporary, &target).with_context(|| format!("writing {}", target.display()))?;
				"apng".into()
			}
			Encoder::Ffmpeg { mut child, input, errors, temporary, target, name, .. } => {
				let mut stdin: Box<dyn Write> = match input {
					Some(file) => Box::new(file),
					None => Box::new(child.stdin.take().context("ffmpeg's stdin")?),
				};
				let mut broken = None;
				for frame in receiver {
					let written = consume(frame, &mut |pixels, profile| {
						let pipe_start = Instant::now();
						let written = stdin.write_all(pixels);
						profile.pipe_ms += pipe_start.elapsed().as_secs_f64() * 1000.0;
						Ok(written?)
					});
					if let Err(error) = written {
						broken = Some(error);
						break;
					}
				}
				drop(stdin);
				let status = child.wait()?;
				let errors = errors.join().unwrap_or_default();
				if !status.success() || broken.is_some() {
					let _ = std::fs::remove_file(&temporary);
					bail!("ffmpeg ({name}) failed: {}{}", errors.trim().lines().rev().take(6).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(" / "), broken.map(|error| format!(" / {error:#}")).unwrap_or_default());
				}
				std::fs::rename(&temporary, &target).with_context(|| format!("writing {}", target.display()))?;
				name
			}
		};
		Ok(EncodedVideo { name, profile, thumbnails: kept, brightness, copies: copies.finish() })
	}
}

#[cfg(windows)]
fn encoder_input(command: &mut Command) -> Option<std::fs::File> {
	use std::os::windows::io::FromRawHandle;
	use windows_sys::Win32::System::Pipes::CreatePipe;
	let (mut read, mut write) = (std::ptr::null_mut(), std::ptr::null_mut());
	// Large raw frames otherwise cross a small system pipe buffer, repeatedly waking both threads.
	// Command duplicates the read handle for its child; these handles remain owned by their files.
	if unsafe { CreatePipe(&mut read, &mut write, std::ptr::null(), 1 << 20) } == 0 {
		return None;
	}
	let read = unsafe { std::fs::File::from_raw_handle(read) };
	let write = unsafe { std::fs::File::from_raw_handle(write) };
	command.stdin(Stdio::from(read));
	Some(write)
}

#[cfg(not(windows))]
fn encoder_input(_: &mut Command) -> Option<std::fs::File> {
	None
}

#[cfg(windows)]
fn hide_console(command: &mut Command) {
	use std::os::windows::process::CommandExt;
	const CREATE_NO_WINDOW: u32 = 0x0800_0000;
	command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_console(_: &mut Command) {}

/// The encoder below the draws: libx264's threads otherwise took the cores the draws' threads needed
/// (eight clips at once: every frame's recording 2.3 -> 4.8 ms). Its output is the same whatever
/// gets the CPU first. Without a window too, as [`hide_console`] (one call sets both).
#[cfg(windows)]
fn below_normal(command: &mut Command) {
	use std::os::windows::process::CommandExt;
	const CREATE_NO_WINDOW: u32 = 0x0800_0000;
	command.creation_flags(CREATE_NO_WINDOW | windows_sys::Win32::System::Threading::BELOW_NORMAL_PRIORITY_CLASS);
}

#[cfg(not(windows))]
fn below_normal(_: &mut Command) {}

/// The calling thread below the draws (the encoder's: measuring, thumbnails, the pipe).
fn thread_below_normal() {
	#[cfg(windows)]
	// SAFETY: the calling thread's own pseudo-handle
	unsafe {
		use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	}
}

/// Draws `job`: every frame on `renderer` (each admitted within the GPU's free memory by `gpu`), the
/// video, the contact sheet and `summary.json` into its folder.
pub fn run(renderer: &Renderer, scenes: &SceneCache, job: &VideoJob, gpu: Option<&GpuGate>) -> Result<VideoResult> {
	run_with(renderer, scenes, job, gpu, None)
}

/// The marker a sheet-first clip leaves in its folder while it is finished ([`VideoResult::pending`]),
/// and the one a clip that failed after its sheet leaves (its error). Not `video.*`: an atomic write's
/// temporary is `<stem>.tmp<pid>` (output::write_atomic), the in-process encoder's file's too.
pub const PENDING_FILE: &str = "clip.pending";
pub const FAILED_FILE: &str = "clip.failed";

/// The clips one output folder takes are drawn one after another ([`run_with`]: a sheet-first clip
/// finishing in the background holds its folder until it is done).
fn folder_lock(out: &Path) -> Arc<Mutex<()>> {
	static FOLDERS: OnceLock<Mutex<HashMap<PathBuf, std::sync::Weak<Mutex<()>>>>> = OnceLock::new();
	let key = std::path::absolute(out).unwrap_or_else(|_| out.to_path_buf());
	let key = PathBuf::from(key.to_string_lossy().to_lowercase());
	let mut folders = FOLDERS.get_or_init(Default::default).lock();
	folders.retain(|_, lock| lock.strong_count() > 0);
	if let Some(lock) = folders.get(&key).and_then(std::sync::Weak::upgrade) {
		return lock;
	}
	let lock = Arc::new(Mutex::new(()));
	folders.insert(key, Arc::downgrade(&lock));
	lock
}

/// A sheet-first clip that failed after its sheet was sent: its error in `clip.failed` beside the
/// sheet, in place of `clip.pending` (what a reader waiting for the clip finds instead).
pub fn fail_pending(pending: &Path, error: &str) {
	if let Some(dir) = pending.parent() {
		let _ = output::write_atomic(&dir.join(FAILED_FILE), serde_json::json!({ "error": error }).to_string().as_bytes());
	}
	let _ = std::fs::remove_file(pending);
}

/// [`run`], and for a job that asks for its sheet first (`sheet_first`) with `early` given: the sheet's
/// moments drawn ahead of the clip, `sheet.png` written and `early` given the result so far (its
/// `pending` the marker standing in the folder) before the clip is drawn and encoded; the marker goes
/// once `summary.json` is written (a caller whose job then fails writes `clip.failed`: [`fail_pending`]).
/// The sheet's frames are drawn twice (once ahead, once in the clip), the same pixels both times, and
/// the clip's own sheet, the same bytes, written again over the early one.
pub fn run_with(renderer: &Renderer, scenes: &SceneCache, job: &VideoJob, gpu: Option<&GpuGate>, early: Option<&mut dyn FnMut(&VideoResult)>) -> Result<VideoResult> {
	// within the memory the machine has free, beside the process's other jobs (crate::budget::admit): a
	// clip holds its memory to its end
	let view = &job.view;
	let pixels = view.width as u64 * view.height as u64 * (view.supersample.max(1) as u64).pow(2);
	let key = format!("video {} {:?} {}x{}x{} {:?}", job.input, job.playback_file, view.width, view.height, view.supersample, job.format);
	let input_bytes = crate::budget::file_bytes([Some(std::path::Path::new(&job.input)), job.playback_file.as_deref(), job.map.as_ref().map(|map| std::path::Path::new(&map.file))].into_iter().flatten());
	let mut admitted = crate::job::admit(renderer, scenes, &format!("video of {}", job.input), key, crate::budget::clip_need(pixels, input_bytes))?;
	let result = run_admitted(renderer, scenes, job, gpu, early);
	if result.is_ok() {
		admitted.finished();
	}
	result
}

fn run_admitted(renderer: &Renderer, scenes: &SceneCache, job: &VideoJob, gpu: Option<&GpuGate>, early: Option<&mut dyn FnMut(&VideoResult)>) -> Result<VideoResult> {
	let started = Instant::now();
	let mut timings = VideoTimings::default();
	let mut profile = OutputProfile::default();
	if !(job.fps > 0.0 && job.fps <= 240.0) {
		bail!("fps must be above 0 and at most 240 (got {})", job.fps);
	}
	let times: Vec<f32> = match &job.times {
		Some(times) => {
			if times.is_empty() || times.iter().any(|time| !time.is_finite()) {
				bail!("times must be a list of finite seconds (got {} of them)", times.len());
			}
			times.clone()
		}
		None => {
			if !(job.duration > 0.0 && job.duration.is_finite()) {
				bail!("duration must be above 0 (got {})", job.duration);
			}
			if !(job.speed > 0.0 && job.speed.is_finite()) {
				bail!("speed must be above 0 (got {})", job.speed);
			}
			let count = ((job.duration / job.speed * job.fps).round() as usize).max(1);
			(0..count).map(|index| job.start + index as f32 * job.speed / job.fps).collect()
		}
	};
	let count = times.len();
	if count > MAX_FRAMES {
		bail!("{count} frames is more than {MAX_FRAMES}: a shorter duration, a lower fps or a higher speed");
	}
	let mut warnings = Vec::new();

	// what the files are: ffmpeg's formats need it, else an animated PNG
	let ffmpeg = job.ffmpeg.clone().filter(|path| path.is_file()).or_else(find_ffmpeg);
	let mut format = job.format;
	if format.needs_ffmpeg() && ffmpeg.is_none() && !x264_in_process(format)? {
		warnings.push(format!("no ffmpeg (ROR_FFMPEG, PATH or winget's Gyan.FFmpeg): an animated PNG instead of {format:?}").to_lowercase());
		format = VideoFormat::Apng;
	}
	let transparent = job.view.background == Background::Transparent;
	let flatten = transparent && format == VideoFormat::Mp4;
	if flatten {
		warnings.push("mp4 has no alpha: the transparent background is black (--format webm keeps it)".into());
	}
	let mut view = job.view.clone();
	view.ids = false;
	view.texture = false;
	// H.264 and VP9 take even sizes
	if matches!(format, VideoFormat::Mp4 | VideoFormat::Webm) {
		view.width += view.width % 2;
		view.height += view.height % 2;
	}
	let out = job.out.clone().unwrap_or_else(|| output::state_dir().join("out").join(format!("{}-video", input_name(&job.input))));
	std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
	let folder = folder_lock(&out);
	let _folder = folder.lock();
	// what an earlier sheet-first clip in this folder left, should its service have ended mid-way
	let pending = out.join(PENDING_FILE);
	let _ = std::fs::remove_file(&pending);
	let _ = std::fs::remove_file(out.join(FAILED_FILE));
	// started now, while the input loads and the camera is fitted (its process's start, a cold
	// executable's first launch among it, off the clip's path); let go of if the clip fails first
	let encoder = EncoderStarted(Some(Encoder::new(format, ffmpeg.as_deref(), &out, (view.width, view.height), job.fps, count, job.nvenc)?));
	let copies = if job.copies.is_empty() || encoder.0.as_ref().is_some_and(Encoder::takes_nv12) {
		CopyRuns::start(&job.copies, ffmpeg.as_deref(), (view.width, view.height), job.fps)?
	} else {
		warnings.push(format!("copies come from an mp4 clip's frames: none made for {format:?}").to_lowercase());
		CopyRuns::none()
	};

	let builder = Builder::new(|| input_bytes(renderer, &job.input), job, scenes)?;
	// the document the clip's moments are made on, with its playback resolved, from the framing on
	let held = Mutex::new(Vec::new());
	timings.load_ms = started.elapsed().as_secs_f64() * 1000.0;

	let framing_start = Instant::now();
	// the clip's files fetched many at a time beside the framing, not one round trip per file and
	// transcode pack as the first frames ask for each
	let (framing, warm_ms) = std::thread::scope(|scope| {
		let warming = scope.spawn(|| {
			let start = Instant::now();
			builder.warm(&renderer.assets);
			start.elapsed().as_secs_f64() * 1000.0
		});
		let framing = cameras(renderer, &builder, &times, job, &held);
		(framing, warming.join().unwrap_or(0.0))
	});
	let (cameras, framed) = framing?;
	timings.framing_ms = framing_start.elapsed().as_secs_f64() * 1000.0;
	timings.warm_ms = warm_ms;
	// parts the game keeps under the camera move with this clip's camera: the frames are made from a
	// playback that keys their motion (`follow_playback`)
	let followed_job;
	let followed = match follow_playback(&builder, &held, job, &cameras, framed, &times)? {
		Some(playback) => {
			followed_job = VideoJob { playback: Some(playback), ..job.clone() };
			Some(Builder::new(|| input_bytes(renderer, &job.input), &followed_job, scenes)?)
		}
		None => None,
	};
	let followed_held = Mutex::new(Vec::new());
	let (builder, held) = match &followed {
		Some(followed) => (followed, &followed_held),
		None => (&builder, &held),
	};

	let encoder = encoder.take();

	let sheet_frames = spread(count, job.sheet);
	let mut sent = false;
	if job.sheet_first
		&& !sheet_frames.is_empty()
		&& let Some(early) = early
	{
		let ahead = Instant::now();
		let first = sheet_first(renderer, &builder, &held, &times, &cameras, &view, &sheet_frames, gpu)?;
		let mut early_stats = VideoStats { particles_max: 0, ..Default::default() };
		(early_stats.parts, early_stats.emitters, early_stats.beams, early_stats.trails) = first.census.counts();
		for (&index, &particles) in sheet_frames.iter().zip(&first.particles) {
			if particles > early_stats.particles_max {
				early_stats.particles_max = particles;
				early_stats.particles_max_at = times[index];
			}
		}
		early_stats.frames_without_particles = if early_stats.emitters > 0 { first.particles.iter().filter(|&&particles| particles == 0).count() } else { 0 };
		early_stats.missing = first.missing.clone();
		let tiles: Vec<(String, &[u8])> = sheet_frames.iter().zip(&first.thumbnails).zip(&first.particles).map(|((&index, rgba), &particles)| (sheet_label(times[index], particles, early_stats.emitters > 0), rgba.as_slice())).collect();
		let (tile_width, tile_height) = thumbnail_size(view.width, view.height);
		let (rgba, sheet_width, sheet_height) = contact_sheet(&tiles, tile_width, tile_height, transparent);
		let sheet = out.join("sheet.png");
		output::write_png(&sheet, sheet_width, sheet_height, &rgba, true)?;
		output::write_atomic(&pending, serde_json::json!({ "pid": std::process::id() }).to_string().as_bytes())?;
		let mut early_warnings = warnings.clone();
		for warning in &first.warnings {
			if !early_warnings.contains(warning) {
				early_warnings.push(warning.clone());
			}
		}
		let early_timings = VideoTimings { load_ms: timings.load_ms, framing_ms: timings.framing_ms, sheet_ms: ahead.elapsed().as_secs_f64() * 1000.0, total_ms: started.elapsed().as_secs_f64() * 1000.0, ..Default::default() };
		early(&VideoResult {
			pending: Some(pending.clone()),
			output_profile: None,
			input: if Path::new(&job.input).is_file() { std::path::absolute(&job.input).map_or_else(|_| job.input.clone(), |path| path.display().to_string()) } else { job.input.clone() },
			playback: job.playback_file.as_deref().map(|file| std::path::absolute(file).unwrap_or_else(|_| file.to_path_buf())),
			video: out.join(format.file_name()),
			out: out.clone(),
			format,
			encoder: encoder.name(),
			sheet: Some(sheet),
			width: view.width,
			height: view.height,
			frames: count,
			fps: job.fps,
			start: job.start,
			duration: job.duration,
			speed: job.speed,
			seconds: count as f32 / job.fps,
			moments: sheet_frames.iter().zip(&first.particles).map(|(&frame, &particles)| Moment { time: times[frame], frame, particles }).collect(),
			framed: framed.map(|(min, max)| [min.into(), max.into()]),
			views: first.view.iter().map(|image| ViewResult { name: job.name.clone(), ..image.clone() }).collect(),
			stats: early_stats,
			weightless: Default::default(),
			lint: Default::default(),
			counts: Default::default(),
			frame_stats: Vec::new(),
			lines: vec![format!("sheet first: the clip, every frame's brightness, the lints and the counts over every frame follow in summary.json; {} stands in the folder until then", PENDING_FILE)],
			warnings: early_warnings,
			timings: early_timings,
			copies: Vec::new(),
		});
		sent = true;
	}
	let mut particles_per_frame = vec![0usize; count];
	let mut stats = VideoStats::default();
	let mut first_view = None;
	let render_start = Instant::now();
	let mut drawn_at = None;
	let (EncodedVideo { name: encoded, profile: encoder_profile, thumbnails: kept, brightness, copies }, (motion, census, effects)) = std::thread::scope(|scope| -> Result<_> {
		let (batch_sender, batch_receiver) = sync_channel::<Result<(Scene, PreparedSky)>>(if builder.guis { GUI_FRAMES } else { SCENES_AHEAD });
		let (found_sender, found_receiver) = sync_channel::<Result<(Scene, PreparedSky, FrameGuis)>>(FOUND_FRAMES);
		let (scene_sender, scene_receiver) = sync_channel::<Result<(ReadyFrame, f64)>>(2);
		let (frame_sender, frame_receiver) = sync_channel::<OutputFrame<'_>>(2);
		let (builder, held) = (&builder, &held);
		let times_ref = &times;
		let (view_ref, cameras_ref) = (&view, &cameras);
		scope.spawn(move || {
			let _clip = crate::par::ClipThread::enter();
			builder.produce(renderer, times_ref, held, batch_sender)
		});
		scope.spawn(move || {
			let _clip = crate::par::ClipThread::enter();
			find_guis(renderer, view_ref, cameras_ref, batch_receiver, found_sender)
		});
		scope.spawn(move || {
			let _clip = crate::par::ClipThread::enter();
			upload_guis(renderer, found_receiver, scene_sender)
		});
		let size = (view.width, view.height);
		let encoding = scope.spawn(move || {
			thread_below_normal();
			let _clip = crate::par::ClipThread::enter();
			encoder.consume(frame_receiver, size, copies)
		});
		// the authoring lints and effects' counts, and dropping each frame's scene, off the drawing
		// thread: in frame order, as they were taken (a map clip's 3,710 part paths hashed per frame)
		let (lint_sender, lint_receiver) = sync_channel::<Scene>(2);
		let linting = scope.spawn(move || {
			let (mut motion, mut census, mut effects) = (crate::lint::Motion::default(), Census::default(), crate::lint::Effects::default());
			let mut last: Option<Scene> = None;
			for scene in lint_receiver {
				// they read the model, not a map around it
				let subject = scene.framing_subject();
				motion.add(subject.as_ref().unwrap_or(&scene));
				census.add_after(&scene, last.as_ref());
				effects.add(subject.as_ref().unwrap_or(&scene));
				last = Some(scene);
			}
			(motion, census, effects)
		});
		// the occlusion and the residency's parts kept from frame to frame while the camera holds still
		let clip_kept = crate::render::ClipKept::new();
		let reuse = frame_reuse();
		let drawn = (|| -> Result<()> {
			let _clip = crate::par::ClipThread::enter();
			let mut scene_start = Instant::now();
			// the frame before's scene while the next may repeat it (to the lints once it is compared), and
			// whether the encoder kept its pixels
			let mut last: Option<(Scene, bool)> = None;
			for (index, built) in scene_receiver.iter().enumerate() {
				profile.scene_wait_ms += scene_start.elapsed().as_secs_f64() * 1000.0;
				let ((scene, sky, guis, gui_warnings), upload_ms) = built?;
				profile.gui_upload_ms += upload_ms;
				let mut request = view.clone();
				request.camera = cameras[index].clone();
				for warning in gui_warnings {
					if !warnings.contains(&warning) {
						warnings.push(warning);
					}
				}
				// the frame before's image again: the same scene under the same camera (a sheet frame drawn
				// for its thumbnail)
				let sheet = sheet_frames.contains(&index);
				let repeats = reuse != FrameReuse::Off && !sheet && index > 0 && same_camera(&cameras[index], &cameras[index - 1]) && last.as_ref().is_some_and(|(last, kept)| *kept && same_scene(last, &scene));
				if let Some((earlier, _)) = last.take()
					&& lint_sender.send(earlier).is_err()
				{
					bail!("the lint thread stopped");
				}
				profile.repeated += usize::from(repeats);
				if repeats && reuse == FrameReuse::On {
					drop((sky, guis));
					particles_per_frame[index] = particles_per_frame[index - 1];
					for warning in &scene.warnings {
						if !warnings.contains(warning) {
							warnings.push(warning.clone());
						}
					}
					last = Some((scene, true));
					let encoder_start = Instant::now();
					let sent = frame_sender.send(OutputFrame::Repeat);
					profile.encoder_wait_ms += encoder_start.elapsed().as_secs_f64() * 1000.0;
					if sent.is_err() {
						return Ok(());
					}
					scene_start = Instant::now();
					continue;
				}
				let readback = (format == VideoFormat::Mp4).then(|| VideoReadback { rgba: sheet, flatten });
				// admitted within the GPU's free memory until submitted (a count of draws at once made clips
				// wait: four slots held through eight clips' CPU work cost each ~1 s)
				// what the clip keeps from frame to frame while the camera holds still (render::ClipKept)
				let keep = cameras.get(index + 1).is_some_and(|next| same_camera(next, &cameras[index]));
				let (mut images, frame_stats) = renderer.submit_frame(&scene, std::slice::from_ref(&request), sky, readback, guis, gpu, Some((&clip_kept, keep)))?;
				profile.gpu_wait_ms += frame_stats.gpu_wait_ms;
				let pending = images.pop().context("the render gave no image")?;
				let image = pending.image();
				profile.prepare_ms += frame_stats.prepare_ms;
				profile.draw_ms += frame_stats.draw_ms;
				profile.readback_ms += frame_stats.readback_ms;
				particles_per_frame[index] = frame_stats.particles;
				if frame_stats.particles > stats.particles_max {
					stats.particles_max = frame_stats.particles;
					stats.particles_max_at = times[index];
				}
				let complete = frame_stats.missing.is_empty();
				for missing in frame_stats.missing {
					if !stats.missing.contains(&missing) {
						stats.missing.push(missing);
					}
				}
				for warning in frame_stats.warnings.iter().chain(scene.warnings.iter()) {
					if !warnings.contains(warning) {
						warnings.push(warning.clone());
					}
				}
				// a frame drawn with an asset missing is not repeated: the asset may be there for the next
				let keep = reuse != FrameReuse::Off && complete;
				if reuse == FrameReuse::Off {
					if lint_sender.send(scene).is_err() {
						bail!("the lint thread stopped");
					}
				} else {
					last = Some((scene, keep));
				}
				if index == 0 {
					first_view = Some(ViewResult {
						name: job.name.clone(),
						file: None,
						ids_file: None,
						width: image.width,
						height: image.height,
						camera: image.camera.cframe_components(),
						fov: image.camera.fov,
						near: image.camera.near,
						lod: image.lod.clone(),
						alone: None,
					});
				}
				let encoder_start = Instant::now();
				let sent = frame_sender.send(OutputFrame::Drawn { view: pending, sheet: sheet.then_some(index), keep, compare: repeats });
				profile.encoder_wait_ms += encoder_start.elapsed().as_secs_f64() * 1000.0;
				if sent.is_err() {
					// the encoder stopped: its error says why
					return Ok(());
				}
				scene_start = Instant::now();
			}
			if let Some((earlier, _)) = last.take()
				&& lint_sender.send(earlier).is_err()
			{
				bail!("the lint thread stopped");
			}
			Ok(())
		})();
		drawn_at = Some(Instant::now());
		// the producer, the encoder and the lints end once their channels close
		drop(scene_receiver);
		drop(frame_sender);
		drop(lint_sender);
		let encoded = encoding.join().map_err(|_| anyhow!("the encoder panicked"));
		let linted = linting.join().map_err(|_| anyhow!("the lint thread panicked"));
		drawn?;
		Ok((encoded??, linted?))
	})?;
	profile.readback_ms += encoder_profile.readback_ms;
	profile.pipe_ms = encoder_profile.pipe_ms;
	profile.flatten_ms = encoder_profile.flatten_ms;
	profile.thumbnail_ms = encoder_profile.thumbnail_ms;
	profile.measure_ms = encoder_profile.measure_ms;
	let drawn_at = drawn_at.unwrap_or_else(Instant::now);
	timings.render_ms = drawn_at.duration_since(render_start).as_secs_f64() * 1000.0;
	timings.encode_ms = drawn_at.elapsed().as_secs_f64() * 1000.0;
	timings.frame_ms = timings.render_ms / count as f64;
	(stats.parts, stats.emitters, stats.beams, stats.trails) = census.counts();
	stats.frames_without_particles = if stats.emitters > 0 { particles_per_frame.iter().filter(|&&particles| particles == 0).count() } else { 0 };

	let sheet_start = Instant::now();
	let moments: Vec<Moment> = kept.iter().map(|(index, _)| Moment { time: times[*index], frame: *index, particles: particles_per_frame[*index] }).collect();
	let sheet = if kept.is_empty() {
		None
	} else {
		let tiles: Vec<(String, &[u8])> = kept.iter().map(|(index, rgba)| (sheet_label(times[*index], particles_per_frame[*index], stats.emitters > 0), rgba.as_slice())).collect();
		let (width, height) = thumbnail_size(view.width, view.height);
		let (rgba, width, height) = contact_sheet(&tiles, width, height, transparent);
		let file = out.join("sheet.png");
		output::write_png(&file, width, height, &rgba, true)?;
		Some(file)
	};
	timings.sheet_ms = sheet_start.elapsed().as_secs_f64() * 1000.0;
	let frame_stats: Vec<FrameStats> = brightness
		.iter()
		.zip(&times)
		.enumerate()
		.map(|(frame, (measured, &time))| FrameStats { time: four_places(f64::from(time)), frame, near_white: measured.near_white, bright: measured.bright, luma: four_places(measured.luma) })
		.collect();
	let lint = effects.lint(&builder.fades, &stats.missing);
	let counts = effects.counts(&builder.roots);
	let mut lines: Vec<String> = frames_line(&frame_stats, view.width as u64 * view.height as u64).into_iter().chain(counts.line()).collect();
	lines.extend(lint.lines());
	timings.total_ms = started.elapsed().as_secs_f64() * 1000.0;
	if stats.emitters > 0 && stats.particles_max == 0 {
		warnings.push("no frame drew a particle: the effect's emitters never emitted in this stretch (an emit, or another --start / --duration)".into());
	}
	for copy in copies.iter().filter(|copy| !copy.written) {
		warnings.push(format!("copy {} not written: {}", copy.file.display(), copy.error.as_deref().unwrap_or("no reason given")));
	}
	let result = VideoResult {
		pending: None,
		output_profile: std::env::var_os("ROR_VIDEO_PROFILE").map(|_| profile),
		input: if Path::new(&job.input).is_file() { std::path::absolute(&job.input).map_or_else(|_| job.input.clone(), |path| path.display().to_string()) } else { job.input.clone() },
		playback: job.playback_file.as_deref().map(|file| std::path::absolute(file).unwrap_or_else(|_| file.to_path_buf())),
		video: out.join(format.file_name()),
		out: out.clone(),
		format,
		encoder: encoded,
		sheet,
		width: view.width,
		height: view.height,
		frames: count,
		fps: job.fps,
		start: job.start,
		duration: job.duration,
		speed: job.speed,
		seconds: count as f32 / job.fps,
		moments,
		framed: framed.map(|(min, max)| [min.into(), max.into()]),
		views: first_view.into_iter().collect(),
		stats,
		weightless: motion.weightless(),
		lint,
		counts,
		frame_stats,
		lines,
		warnings,
		timings,
		copies,
	};
	output::write_json_atomic(&out.join("summary.json"), &result, true)?;
	if sent {
		std::fs::remove_file(&pending).with_context(|| format!("removing {}", pending.display()))?;
	}
	Ok(result)
}

/// What the sheet's moments drew ahead of the clip ([`run_with`]).
struct SheetFirst {
	/// Each moment's thumbnail and particles drawn, in the sheet's order
	thumbnails: Vec<Vec<u8>>,
	particles: Vec<usize>,
	census: Census,
	missing: Vec<String>,
	warnings: Vec<String>,
	/// The first frame's camera (the sheet's first moment is the clip's first frame)
	view: Option<ViewResult>,
}

/// The sheet's moments drawn as the clip draws them (each its frame's scene, sky and camera, no
/// occlusion kept between them, as the clip's first frame draws), a few submitted ahead of the one read
/// back, each thumbnail made as [`Encoder::consume`] makes it.
#[allow(clippy::too_many_arguments)]
fn sheet_first<'s>(renderer: &Renderer, builder: &'s Builder, held: &Mutex<Vec<Kept<'s>>>, times: &[f32], cameras: &[CameraSpec], view: &ViewRequest, frames: &[usize], gpu: Option<&GpuGate>) -> Result<SheetFirst> {
	let (tile_width, tile_height) = thumbnail_size(view.width, view.height);
	let mut first = SheetFirst { thumbnails: Vec::new(), particles: Vec::new(), census: Census::default(), missing: Vec::new(), warnings: Vec::new(), view: None };
	let parts = PartsKey::default();
	let mut queue: std::collections::VecDeque<PendingView<'_>> = std::collections::VecDeque::new();
	let take = |pending: PendingView<'_>, first: &mut SheetFirst| -> Result<()> {
		let mut stats = RenderStats::default();
		let image = pending.finish(&mut stats)?;
		first.thumbnails.push(downscale(&image.rgba, image.width, image.height, tile_width, tile_height));
		Ok(())
	};
	for &index in frames {
		let scene = builder.scene_with(times[index], Some(held))?;
		let sky = if builder.fixed.is_some() { renderer.prepare_environment_sharing(&scene, &parts) } else { renderer.prepare_environment(&scene) };
		let mut request = view.clone();
		request.camera = cameras[index].clone();
		let (mut images, frame_stats) = renderer.submit_frame(&scene, std::slice::from_ref(&request), sky, None, Default::default(), gpu, None)?;
		let pending = images.pop().context("the render gave no image")?;
		if first.view.is_none() {
			let image = pending.image();
			first.view = Some(ViewResult { name: String::new(), file: None, ids_file: None, width: image.width, height: image.height, camera: image.camera.cframe_components(), fov: image.camera.fov, near: image.camera.near, lod: image.lod.clone(), alone: None });
		}
		first.particles.push(frame_stats.particles);
		for missing in frame_stats.missing {
			if !first.missing.contains(&missing) {
				first.missing.push(missing);
			}
		}
		for warning in frame_stats.warnings.iter().chain(scene.warnings.iter()) {
			if !first.warnings.contains(warning) {
				first.warnings.push(warning.clone());
			}
		}
		first.census.add(&scene);
		queue.push_back(pending);
		// two frames on the GPU while the oldest is read back
		if queue.len() > 2 {
			take(queue.pop_front().expect("queued"), &mut first)?;
		}
	}
	while let Some(pending) = queue.pop_front() {
		take(pending, &mut first)?;
	}
	Ok(first)
}

fn sheet_label(time: f32, particles: usize, effects: bool) -> String {
	if effects { format!("t {time:.2} s · {particles} particles") } else { format!("t {time:.2} s") }
}

/// Columns for a sheet of `count` tiles: up to four across, few empty cells.
fn sheet_columns(count: usize) -> usize {
	match count {
		0..=4 => count.max(1),
		5 | 6 | 9 => 3,
		_ => 4,
	}
}

fn thumbnail_size(width: u32, height: u32) -> (u32, u32) {
	let tile_width = width.clamp(1, 480);
	let tile_height = ((tile_width as f64 * height as f64 / width.max(1) as f64).round() as u32).max(1);
	(tile_width, tile_height)
}

/// The frames on one image, each under its label, at most 480 pixels wide each.
fn contact_sheet(tiles: &[(String, &[u8])], width: u32, height: u32, transparent: bool) -> (Vec<u8>, u32, u32) {
	const LABEL: u32 = 24;
	const GAP: u32 = 4;
	let columns = sheet_columns(tiles.len()) as u32;
	let rows = (tiles.len() as u32).div_ceil(columns);
	let (tile_width, tile_height) = thumbnail_size(width, height);
	let (sheet_width, sheet_height) = (columns * tile_width + (columns + 1) * GAP, rows * (tile_height + LABEL) + (rows + 1) * GAP);
	let mut sheet = vec![0u8; (sheet_width * sheet_height * 4) as usize];
	for pixel in sheet.chunks_exact_mut(4) {
		pixel.copy_from_slice(&[21, 23, 28, 255]);
	}
	for (index, (label, rgba)) in tiles.iter().enumerate() {
		let (column, row) = (index as u32 % columns, index as u32 / columns);
		let left = GAP + column * (tile_width + GAP);
		let top = GAP + row * (tile_height + LABEL + GAP);
		fill(&mut sheet, sheet_width, left, top, tile_width, LABEL, [29, 32, 39]);
		draw_text(&mut sheet, sheet_width, left + 6, top + 5, label, [230, 230, 230]);
		// the clip's thumbnails are of the tiles' size already: taken as they are (downscaling one to its
		// own size changes only the colour of pixels of alpha 0, which the tile does not show)
		let thumbnail = if (width, height) == (tile_width, tile_height) { std::borrow::Cow::Borrowed(*rgba) } else { std::borrow::Cow::Owned(downscale(rgba, width, height, tile_width, tile_height)) };
		for y in 0..tile_height {
			for x in 0..tile_width {
				let source = ((y * tile_width + x) * 4) as usize;
				let pixel = &thumbnail[source..source + 4];
				// a transparent frame over a checkerboard
				let under = if transparent && ((x / 8 + y / 8) % 2 == 0) { [42u32, 45, 53] } else { [33, 36, 43] };
				let alpha = pixel[3] as u32;
				let target = (((top + LABEL + y) * sheet_width + left + x) * 4) as usize;
				for channel in 0..3 {
					sheet[target + channel] = ((pixel[channel] as u32 * alpha + under[channel] * (255 - alpha) + 127) / 255) as u8;
				}
				sheet[target + 3] = 255;
			}
		}
	}
	(sheet, sheet_width, sheet_height)
}

fn fill(image: &mut [u8], image_width: u32, left: u32, top: u32, width: u32, height: u32, color: [u8; 3]) {
	for y in top..top + height {
		for x in left..left + width {
			let index = ((y * image_width + x) * 4) as usize;
			if index + 3 < image.len() {
				image[index..index + 3].copy_from_slice(&color);
				image[index + 3] = 255;
			}
		}
	}
}

/// An RGBA8 image scaled down by averaging each target pixel's footprint (alpha-weighted).
fn downscale(rgba: &[u8], width: u32, height: u32, target_width: u32, target_height: u32) -> Vec<u8> {
	let mut out = vec![0u8; (target_width * target_height * 4) as usize];
	let (scale_x, scale_y) = (width as f64 / target_width as f64, height as f64 / target_height as f64);
	for y in 0..target_height {
		let (y0, y1) = ((y as f64 * scale_y) as u32, (((y + 1) as f64 * scale_y).ceil() as u32).min(height).max((y as f64 * scale_y) as u32 + 1));
		for x in 0..target_width {
			let (x0, x1) = ((x as f64 * scale_x) as u32, (((x + 1) as f64 * scale_x).ceil() as u32).min(width).max((x as f64 * scale_x) as u32 + 1));
			let mut sum = [0u64; 4];
			for sy in y0..y1 {
				for sx in x0..x1 {
					let index = ((sy * width + sx) * 4) as usize;
					let alpha = rgba[index + 3] as u64;
					for channel in 0..3 {
						sum[channel] += rgba[index + channel] as u64 * alpha;
					}
					sum[3] += alpha;
				}
			}
			let pixels = ((y1 - y0) * (x1 - x0)) as u64;
			let index = ((y * target_width + x) * 4) as usize;
			if sum[3] > 0 {
				for channel in 0..3 {
					out[index + channel] = (sum[channel] / sum[3]) as u8;
				}
			}
			out[index + 3] = (sum[3] / pixels.max(1)) as u8;
		}
	}
	out
}

/// A 5 x 7 pixel font for the sheet's labels: digits and the few letters they use.
fn glyph(character: char) -> Option<[&'static str; 7]> {
	Some(match character {
		'0' => [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
		'1' => ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
		'2' => [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
		'3' => ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
		'4' => ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
		'5' => ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
		'6' => ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
		'7' => ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
		'8' => [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
		'9' => [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
		'.' => [".....", ".....", ".....", ".....", ".....", ".##..", ".##.."],
		'·' => [".....", ".....", ".....", "..#..", ".....", ".....", "....."],
		's' => [".....", ".....", ".####", "#....", ".###.", "....#", "####."],
		't' => [".#...", ".#...", "####.", ".#...", ".#...", ".#..#", "..##."],
		'p' => [".....", ".....", "####.", "#...#", "####.", "#....", "#...."],
		'a' => [".....", ".....", ".###.", "....#", ".####", "#...#", ".####"],
		'r' => [".....", ".....", "#.##.", "##..#", "#....", "#....", "#...."],
		'i' => ["..#..", ".....", ".##..", "..#..", "..#..", "..#..", ".###."],
		'c' => [".....", ".....", ".###.", "#....", "#....", "#...#", ".###."],
		'l' => [".##..", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."],
		'e' => [".....", ".....", ".###.", "#...#", "#####", "#....", ".###."],
		' ' => [".....", ".....", ".....", ".....", ".....", ".....", "....."],
		_ => return None,
	})
}

/// Writes `text` with the sheet's font at twice its size, its top-left at (left, top).
fn draw_text(image: &mut [u8], image_width: u32, left: u32, top: u32, text: &str, color: [u8; 3]) {
	const SCALE: u32 = 2;
	let mut x = left;
	for character in text.chars() {
		if let Some(rows) = glyph(character) {
			for (row, line) in rows.iter().enumerate() {
				for (column, bit) in line.chars().enumerate() {
					if bit == '#' {
						fill(image, image_width, x + column as u32 * SCALE, top + row as u32 * SCALE, SCALE, SCALE, color);
					}
				}
			}
		}
		x += 6 * SCALE;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_clip_whose_lighting_alone_changes_still_emits() {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		let mut model = ModelBuilder::new("Glow");
		let part = model.part(&PartSpec::default());
		let attachment = model.child(part, InstanceBuilder::new("Attachment"));
		model.child(attachment, InstanceBuilder::new("ParticleEmitter").with_property("Rate", 30.0f32));
		let bytes = model.to_rbxm().unwrap();
		let playback = crate::scene::Playback { tweens: vec![crate::scene::playback::spec::tween("Lighting.ExposureCompensation=-2@0+1").unwrap()], ..Default::default() };
		let job = VideoJob {
			input: "glow.rbxm".into(),
			select: None,
			out: None,
			view: ViewRequest::default(),
			name: default_name(),
			playback: Some(playback),
			playback_file: None,
			start: 0.0,
			duration: 1.0,
			fps: 10.0,
			speed: 1.0,
			times: None,
			orbit: 0.0,
			format: VideoFormat::Png,
			sheet: 0,
			materials_from: None,
			lighting_from: None,
			map: None,
			frame: None,
			focus: None,
			ffmpeg: None,
			camera_track: None,
			sheet_first: false,
			copies: Vec::new(),
			relative_to: None,
			nvenc: None,
		};
		let scenes = SceneCache::default();
		let builder = Builder::new(|| Ok(bytes.clone()), &job, &scenes).unwrap();
		assert!(builder.lit.is_some(), "only the Lighting changes: one scene for the clip");
		let emits = |scene: &Scene| scene.particles[0].births.as_ref().is_some_and(|births| !births.is_empty());
		let early = builder.scene_at(0.5).unwrap();
		assert!(emits(&early) && early.environment.exposure_compensation < 0.0);
		// a thread's own document is a copy of the scene's, whose emitters it names (a file read
		// again names them anew, and its frames drew no particle)
		let held = builder.documents.lock().pop();
		let late = builder.scene_at(0.9).unwrap();
		assert!(held.is_some() && emits(&late));
		assert!(late.environment.exposure_compensation < early.environment.exposure_compensation);
	}

	/// A recording's writes to the Terrain's Clouds (target `Workspace.Terrain.Clouds`) play on the
	/// clouds ror draws, frame by frame: the stage's (`--lighting-from`) when there is one, else none
	/// until the playback switches them on. A playback that changes only the Lighting keeps the stage's.
	#[test]
	fn a_clip_plays_the_terrain_clouds_on_the_stages() {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::{InstanceBuilder, WeakDom, types::Color3};
		let folder = std::env::temp_dir().join(format!("ror-terrain-clouds-{}", std::process::id()));
		std::fs::create_dir_all(&folder).unwrap();
		let mut place = WeakDom::new(InstanceBuilder::new("DataModel"));
		let root = place.root_ref();
		let workspace = place.insert(root, InstanceBuilder::new("Workspace").with_name("Workspace"));
		let terrain = place.insert(workspace, InstanceBuilder::new("Terrain").with_name("Terrain"));
		let stage_clouds = InstanceBuilder::new("Clouds").with_name("Clouds").with_property("Cover", 0.84f32).with_property("Density", 0.03f32).with_property("Color", Color3::new(2.5, 3.49, 4.98)).with_property("Enabled", true);
		place.insert(terrain, stage_clouds);
		let lighting = place.insert(root, InstanceBuilder::new("Lighting").with_name("Lighting"));
		let stage = folder.join("stage.rbxl");
		let mut bytes = Vec::new();
		rbx_binary::to_writer(&mut bytes, &place, &[workspace, lighting]).unwrap();
		std::fs::write(&stage, bytes).unwrap();
		let stage = stage.display().to_string();
		let mut model = ModelBuilder::new("Model");
		model.part(&PartSpec::default());
		let bytes = model.to_rbxm().unwrap();
		let job = |playback: serde_json::Value, lighting_from: Option<&str>| VideoJob {
			input: "model.rbxm".into(),
			select: None,
			out: None,
			view: ViewRequest::default(),
			name: default_name(),
			playback: Some(serde_json::from_value(playback).unwrap()),
			playback_file: None,
			start: 0.0,
			duration: 1.0,
			fps: 10.0,
			speed: 1.0,
			times: None,
			orbit: 0.0,
			format: VideoFormat::Png,
			sheet: 0,
			materials_from: None,
			lighting_from: lighting_from.map(str::to_string),
			map: None,
			frame: None,
			focus: None,
			ffmpeg: None,
			camera_track: None,
			sheet_first: false,
			copies: Vec::new(),
			relative_to: None,
			nvenc: None,
		};
		let scenes = SceneCache::default();
		let clouds_at = |job: &VideoJob, times: &[f32]| -> Vec<Option<crate::scene::environment::Clouds>> {
			let builder = Builder::new(|| Ok(bytes.clone()), job, &scenes).unwrap();
			assert!(builder.lit.is_some(), "only the look changes: one scene for the clip");
			times.iter().map(|&time| builder.scene_at(time).unwrap().environment.clouds).collect()
		};
		let storm = serde_json::json!({
			"play": true,
			"actions": [{"at": 0, "target": "Workspace.Terrain.Clouds", "set": {"Color": [0.27451, 0.2902, 0.3451]}}],
			"tweens": [{"target": "Workspace.Terrain.Clouds", "property": "Cover", "keys": [{"at": 0, "value": 0.95}, {"at": 1, "value": 0.84}]}],
		});
		let storm = clouds_at(&job(storm, Some(&stage)), &[0.0, 0.5, 1.0]);
		let [start, middle, end] = [storm[0].unwrap(), storm[1].unwrap(), storm[2].unwrap()];
		assert!((start.cover - 0.95).abs() < 1e-4 && (end.cover - 0.84).abs() < 1e-4 && middle.cover < start.cover && middle.cover > end.cover, "{storm:?}");
		assert!(start.color.abs_diff_eq(Vec3::new(0.27451, 0.2902, 0.3451), 1e-4) && end.color == start.color, "{storm:?}");
		assert!((start.density - 0.03).abs() < 1e-6, "the stage's own Density stays: {storm:?}");
		// the Lighting alone: the stage's clouds as they are
		let dim = serde_json::json!({"tweens": [{"target": "Lighting", "property": "ExposureCompensation", "keys": [{"at": 0, "value": 0}, {"at": 1, "value": -1}]}]});
		let dim = clouds_at(&job(dim, Some(&stage)), &[0.5]);
		assert_eq!(dim[0].map(|clouds| (clouds.cover, clouds.density, clouds.color)), Some((0.84, 0.03, Vec3::new(2.5, 3.49, 4.98))));
		// no stage: ror's default environment draws none until the playback switches them on
		let own = serde_json::json!({"actions": [{"at": 0, "target": "Workspace.Terrain.Clouds", "set": {"Cover": 0.9}}, {"at": 0.5, "target": "Workspace.Terrain.Clouds", "set": {"Enabled": true}}]});
		let own = clouds_at(&job(own, None), &[0.2, 0.7]);
		assert!(own[0].is_none(), "{own:?}");
		assert_eq!(own[1].map(|clouds| (clouds.cover, clouds.density, clouds.color)), Some((0.9, 0.7, Vec3::ONE)));
		let _ = std::fs::remove_dir_all(&folder);
	}

	#[test]
	fn a_clip_counts_what_any_frame_holds() {
		// an effect the playback parents in after the first frame, as a game clones its effects in
		// (a mode's telegraph effect: the summary counted its first frame's 4 emitters and no beam, of
		// the 101 emitters and 15 beams the clip draws)
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		use rbx_dom_weak::types::Variant;
		let mut model = ModelBuilder::new("Link");
		let effect = model.child(model.model, InstanceBuilder::new("Folder").with_name("Effect"));
		let holder = model.child(effect, InstanceBuilder::new("Part").with_name("Holder").with_property("Transparency", 0.5f32));
		let a0 = model.child(holder, InstanceBuilder::new("Attachment").with_name("A0"));
		let a1 = model.child(holder, InstanceBuilder::new("Attachment").with_name("A1").with_property("CFrame", crate::scene::build::cframe(glam::Affine3A::from_translation(Vec3::new(6.0, 0.0, 0.0)))));
		model.child(holder, InstanceBuilder::new("Beam").with_name("Link").with_property("Attachment0", Variant::Ref(a0)).with_property("Attachment1", Variant::Ref(a1)));
		// two copies of one emitter (same-named siblings) and one more
		for name in ["Sparks", "Sparks", "Glow"] {
			model.child(a0, InstanceBuilder::new("ParticleEmitter").with_name(name).with_property("Rate", 20.0f32));
		}
		model.part(&PartSpec { name: "Floor".into(), ..Default::default() });
		let bytes = model.to_rbxm().unwrap();
		let playback = crate::scene::Playback {
			actions: vec![crate::scene::playback::spec::parent("Link.Effect=@0").unwrap(), crate::scene::playback::spec::parent("Link.Effect=Link@0.5").unwrap()],
			..Default::default()
		};
		let job = VideoJob {
			input: "link.rbxm".into(),
			select: None,
			out: None,
			view: ViewRequest::default(),
			name: default_name(),
			playback: Some(playback),
			playback_file: None,
			start: 0.0,
			duration: 1.0,
			fps: 10.0,
			speed: 1.0,
			times: None,
			orbit: 0.0,
			format: VideoFormat::Png,
			sheet: 0,
			materials_from: None,
			lighting_from: None,
			map: None,
			frame: None,
			focus: None,
			ffmpeg: None,
			camera_track: None,
			sheet_first: false,
			copies: Vec::new(),
			relative_to: None,
			nvenc: None,
		};
		let scenes = SceneCache::default();
		let builder = Builder::new(|| Ok(bytes.clone()), &job, &scenes).unwrap();
		let first = builder.scene_at(0.0).unwrap();
		assert_eq!((first.parts.len(), first.particles.len(), first.beams.len()), (1, 0, 0), "the effect is out of the world at first");
		let mut census = Census::default();
		// each frame after the last, as a clip counts them: the same counts
		let (mut after, mut last) = (Census::default(), None);
		let mut effects = crate::lint::Effects::default();
		for time in [0.0, 0.3, 0.6, 0.9, 0.95] {
			let scene = builder.scene_at(time).unwrap();
			census.add(&scene);
			after.add_after(&scene, last.as_ref());
			effects.add(&scene);
			last = Some(scene);
		}
		assert_eq!(census.counts(), (2, 3, 1, 0), "the floor and the holder, the three emitters (copies apart), the beam; each once");
		assert_eq!(after.counts(), census.counts());
		// the effect's own counts alike (the floor holds none)
		let counts = effects.counts(&builder.roots);
		assert_eq!(counts.effects.iter().map(|effect| (effect.root.as_str(), effect.emitters, effect.beams)).collect::<Vec<_>>(), [("Link.Effect", 3, 1)]);
	}

	#[test]
	fn a_sheet_first_clip_answers_with_its_sheet_and_draws_the_same_files() {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		let state = std::env::temp_dir().join(format!("ror-sheet-first-test-{}", std::process::id()));
		std::fs::create_dir_all(&state).unwrap();
		let gpu = Arc::new(crate::render::gpu::Gpu::new().expect("a GPU (or the software adapter)"));
		let assets = ror_assets::Assets::new(ror_assets::AssetConfig { offline: true, ..ror_assets::AssetConfig::new(&state) });
		let renderer = Renderer::new(gpu, assets).expect("a renderer");
		let mut model = ModelBuilder::new("Glow");
		let part = model.part(&PartSpec::default());
		let attachment = model.child(part, InstanceBuilder::new("Attachment"));
		model.child(attachment, InstanceBuilder::new("ParticleEmitter").with_property("Rate", 30.0f32));
		let input = state.join("glow.rbxm");
		std::fs::write(&input, model.to_rbxm().unwrap()).unwrap();
		let job = |out: &str, sheet_first: bool| VideoJob {
			input: input.display().to_string(),
			select: None,
			out: Some(state.join(out)),
			view: ViewRequest { width: 96, height: 64, ..Default::default() },
			name: default_name(),
			playback: None,
			playback_file: None,
			start: 0.0,
			duration: 1.0,
			fps: 8.0,
			speed: 1.0,
			times: None,
			orbit: 90.0,
			format: VideoFormat::Png,
			sheet: 4,
			materials_from: None,
			lighting_from: None,
			map: None,
			frame: None,
			focus: None,
			ffmpeg: None,
			camera_track: None,
			sheet_first,
			copies: Vec::new(),
			relative_to: None,
			nvenc: None,
		};
		let scenes = SceneCache::default();
		let plain = run(&renderer, &scenes, &job("plain", true), None).unwrap();
		assert!(plain.pending.is_none(), "without an early answer the job is plain");
		let mut answered = None;
		let mut early = |result: &VideoResult| {
			let pending = result.pending.clone().expect("the marker");
			assert!(pending.is_file() && result.sheet.as_ref().is_some_and(|sheet| sheet.is_file()) && result.frame_stats.is_empty());
			answered = Some(result.clone());
		};
		let first = run_with(&renderer, &scenes, &job("first", true), None, Some(&mut early)).unwrap();
		let answered = answered.expect("answered with the sheet");
		assert!(!answered.pending.as_ref().unwrap().exists(), "the marker goes with the clip");
		assert_eq!(answered.moments.iter().map(|moment| (moment.frame, moment.particles)).collect::<Vec<_>>(), first.moments.iter().map(|moment| (moment.frame, moment.particles)).collect::<Vec<_>>());
		let read = |dir: &str, name: &str| std::fs::read(state.join(dir).join(name)).unwrap();
		assert!(read("plain", "sheet.png") == read("first", "sheet.png"));
		for frame in 0..plain.frames {
			let name = format!("frames/{frame:04}.png");
			assert!(read("plain", &name) == read("first", &name), "{name}");
		}
		assert_eq!(serde_json::to_value(&plain.frame_stats).unwrap(), serde_json::to_value(&first.frame_stats).unwrap());
		drop(renderer);
		let _ = std::fs::remove_dir_all(&state);
	}

	#[test]
	fn a_copy_takes_every_frame_and_one_that_fails_leaves_the_clip_alone() {
		use crate::scene::build::{ModelBuilder, PartSpec};
		let Some(ffmpeg) = find_ffmpeg() else {
			eprintln!("skipped: no ffmpeg");
			return;
		};
		let state = std::env::temp_dir().join(format!("ror-copy-test-{}", std::process::id()));
		std::fs::create_dir_all(&state).unwrap();
		let gpu = Arc::new(crate::render::gpu::Gpu::new().expect("a GPU (or the software adapter)"));
		let assets = ror_assets::Assets::new(ror_assets::AssetConfig { offline: true, ..ror_assets::AssetConfig::new(&state) });
		let renderer = Renderer::new(gpu, assets).expect("a renderer");
		let mut model = ModelBuilder::new("Block");
		model.part(&PartSpec::default());
		let input = state.join("block.rbxm");
		std::fs::write(&input, model.to_rbxm().unwrap()).unwrap();
		let (good, bad) = (state.join("copy").join("small.mp4"), state.join("bad.mp4"));
		let job = VideoJob {
			input: input.display().to_string(),
			select: None,
			out: Some(state.join("clip")),
			view: ViewRequest { width: 96, height: 64, ..Default::default() },
			name: default_name(),
			playback: None,
			playback_file: None,
			start: 0.0,
			duration: 1.0,
			fps: 9.0,
			speed: 1.0,
			times: None,
			orbit: 90.0,
			format: VideoFormat::Mp4,
			sheet: 0,
			materials_from: None,
			lighting_from: None,
			map: None,
			frame: None,
			focus: None,
			ffmpeg: Some(ffmpeg.clone()),
			camera_track: None,
			sheet_first: false,
			copies: vec![
				VideoCopy { file: good.clone(), args: ["-vf", "scale=48:32", "-c:v", "mpeg4", "-q:v", "5"].map(String::from).to_vec() },
				VideoCopy { file: bad.clone(), args: ["-c:v", "no_such_encoder"].map(String::from).to_vec() },
			],
			relative_to: None,
			nvenc: None,
		};
		let result = run(&renderer, &SceneCache::default(), &job, None).unwrap();
		assert!(result.video.is_file(), "the clip is written whatever its copies do");
		assert_eq!(result.copies.len(), 2);
		assert!(result.copies[0].written && good.is_file(), "{:?}", result.copies[0]);
		assert!(!result.copies[1].written && !bad.exists() && result.copies[1].error.as_deref().is_some_and(|error| error.contains("no_such_encoder")), "{:?}", result.copies[1]);
		assert!(result.warnings.iter().any(|warning| warning.contains("bad.mp4 not written")));
		let leftovers: Vec<_> = std::fs::read_dir(&state).unwrap().chain(std::fs::read_dir(state.join("copy")).unwrap()).flatten().filter(|entry| entry.file_name().to_string_lossy().contains(".tmp")).collect();
		assert!(leftovers.is_empty(), "no partial file left: {leftovers:?}");
		let probe = ffmpeg.with_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" });
		if probe.is_file() {
			let output = Command::new(&probe).args(["-v", "error", "-count_frames", "-select_streams", "v", "-show_entries", "stream=nb_read_frames,width,height", "-of", "csv=p=0"]).arg(&good).output().unwrap();
			assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), format!("48,32,{}", result.frames));
		}
		drop(renderer);
		let _ = std::fs::remove_dir_all(&state);
	}

	#[test]
	fn a_still_frame_is_the_one_before_again_and_a_changing_one_is_drawn() {
		// a still part under a still camera: every frame the first's image (each after the first taken
		// again from the kept staging buffer, its rows packed at this width; sheet frames drawn); with an
		// emitter, each frame its own
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::{InstanceBuilder, types::Content};
		let state = std::env::temp_dir().join(format!("ror-frame-reuse-test-{}", std::process::id()));
		std::fs::create_dir_all(&state).unwrap();
		let gpu = Arc::new(crate::render::gpu::Gpu::new().expect("a GPU (or the software adapter)"));
		let assets = ror_assets::Assets::new(ror_assets::AssetConfig { offline: true, ..ror_assets::AssetConfig::new(&state) });
		let renderer = Renderer::new(gpu, assets).expect("a renderer");
		// the emitter's own texture: the default one is Studio's, which a machine without Studio lacks
		let sprite = state.join("sprite.png");
		crate::output::write_png(&sprite, 4, 4, &[255; 64], true).unwrap();
		let sprite = format!("file://{}", sprite.display());
		let write = |name: &str, emits: bool| {
			let mut model = ModelBuilder::new("Still");
			let part = model.part(&PartSpec::default());
			if emits {
				let attachment = model.child(part, InstanceBuilder::new("Attachment"));
				model.child(attachment, InstanceBuilder::new("ParticleEmitter").with_property("Rate", 30.0f32).with_property("TextureContent", Content::from_uri(sprite.as_str())));
			}
			let input = state.join(name);
			std::fs::write(&input, model.to_rbxm().unwrap()).unwrap();
			input
		};
		let job = |input: &Path, out: &str| VideoJob {
			input: input.display().to_string(),
			select: None,
			out: Some(state.join(out)),
			view: ViewRequest { width: 128, height: 64, ..Default::default() },
			name: default_name(),
			playback: None,
			playback_file: None,
			start: 0.0,
			duration: 1.0,
			fps: 12.0,
			speed: 1.0,
			times: None,
			orbit: 0.0,
			format: VideoFormat::Png,
			sheet: 3,
			materials_from: None,
			lighting_from: None,
			map: None,
			frame: None,
			focus: None,
			ffmpeg: None,
			camera_track: None,
			sheet_first: false,
			copies: Vec::new(),
			relative_to: None,
			nvenc: None,
		};
		let scenes = SceneCache::default();
		let read = |dir: &str, frame: usize| std::fs::read(state.join(dir).join(format!("frames/{frame:04}.png"))).unwrap();
		let still = run(&renderer, &scenes, &job(&write("still.rbxm", false), "still"), None).unwrap();
		assert_eq!(still.frames, 12);
		for frame in 1..still.frames {
			assert!(read("still", frame) == read("still", 0), "frame {frame} of a still clip");
		}
		assert!(still.frame_stats.windows(2).all(|pair| (pair[0].near_white, pair[0].bright, pair[0].luma) == (pair[1].near_white, pair[1].bright, pair[1].luma)));
		let glowing = run(&renderer, &scenes, &job(&write("glow.rbxm", true), "glow"), None).unwrap();
		let distinct: std::collections::HashSet<Vec<u8>> = (0..glowing.frames).map(|frame| read("glow", frame)).collect();
		assert!(distinct.len() > glowing.frames / 2, "an emitting clip's frames are drawn: {} distinct of {}", distinct.len(), glowing.frames);
		drop(renderer);
		let _ = std::fs::remove_dir_all(&state);
	}

	#[test]
	fn helpers_make_frames_ahead_each_once_and_in_order() {
		use std::sync::atomic::{AtomicUsize, Ordering};
		let count = 200;
		let ahead = super::Ahead::new(count);
		let built: Vec<AtomicUsize> = (0..count).map(|_| AtomicUsize::new(0)).collect();
		let build = |index: usize| {
			built[index].fetch_add(1, Ordering::Relaxed);
			std::thread::sleep(std::time::Duration::from_micros(200));
			index * 10
		};
		let furthest = AtomicUsize::new(0);
		let taken: Vec<usize> = std::thread::scope(|scope| {
			for _ in 0..3 {
				scope.spawn(|| ahead.help(&build));
			}
			let taken = (0..count)
				.map(|index| {
					let value = ahead.take(index, &build);
					ahead.sent(index);
					// helpers stay within twice the queue's length of what was sent
					furthest.fetch_max(ahead.state.lock().next, Ordering::Relaxed);
					assert!(ahead.state.lock().next <= index + 1 + 2 * super::SCENES_AHEAD);
					value
				})
				.collect();
			ahead.stop();
			taken
		});
		assert_eq!(taken, (0..count).map(|index| index * 10).collect::<Vec<_>>(), "every frame, in order");
		assert!(built.iter().all(|times| times.load(Ordering::Relaxed) == 1), "each made once");
		assert!(ahead.state.lock().made.is_empty());
	}

	#[test]
	fn a_stopped_producer_lets_its_helpers_go() {
		let ahead = super::Ahead::new(1000);
		std::thread::scope(|scope| {
			let helper = scope.spawn(|| ahead.help(&|index: usize| index));
			assert_eq!(ahead.take(0, &|index: usize| index), 0);
			ahead.stop();
			helper.join().unwrap();
		});
	}

	#[test]
	fn frames_spread_over_the_clip() {
		assert_eq!(spread(60, 12).first(), Some(&0));
		assert_eq!(spread(60, 12).last(), Some(&59));
		assert_eq!(spread(60, 12).len(), 12);
		assert_eq!(spread(3, 12), vec![0, 1, 2]);
		assert_eq!(spread(1, 12), vec![0]);
		assert!(spread(10, 0).is_empty());
	}

	#[test]
	fn a_camera_track_holds_each_key_until_the_next() {
		// a recording keys the game camera whenever it moves; ree writes them in time order but a
		// track read from anywhere may not be
		let key = |at: f32, x: f32| CameraKey { at, cframe: [x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], fov: 70.0 };
		let track = vec![key(0.5, 2.0), key(0.0, 1.0), key(1.0, 3.0)];
		let x = |time: f32| match track_camera(&track, time) {
			CameraSpec::CFrame { components, .. } => components[0],
			other => panic!("a track draws CFrame cameras, not {other:?}"),
		};
		assert_eq!([x(0.0), x(0.49), x(0.5), x(0.99), x(1.0), x(4.0)], [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
		let late = vec![key(0.3, 5.0)];
		assert!(matches!(track_camera(&late, 0.0), CameraSpec::CFrame { components, fov, near: None } if components == late[0].cframe && fov == 70.0), "before its first key a track holds that key");
		let parsed: Vec<CameraKey> = serde_json::from_str(r#"[{"at": 0.25, "cframe": [0,20,20,1,0,0,0,0.7,0.7,0,-0.7,0.7]}]"#).unwrap();
		assert_eq!(parsed[0].fov, 70.0, "fov defaults as a camera's does");
	}

	#[test]
	fn stray_particles_do_not_frame() {
		// a hundred particles round the origin and one spark 500 studs off
		let mut particles: Vec<(Vec3, f32)> = (0..100).map(|index| (Vec3::new((index % 10) as f32 - 4.5, (index / 10) as f32 - 4.5, 0.0), 1.0)).collect();
		particles.push((Vec3::new(500.0, 0.0, 0.0), 1.0));
		let (low, high) = robust_box(&particles);
		assert!(high.x < 10.0 && low.x > -10.0, "{low} {high}");
	}

	#[test]
	fn formats_and_encoders() {
		assert_eq!(VideoFormat::parse("MP4").unwrap(), VideoFormat::Mp4);
		assert!(VideoFormat::parse("avi").is_err());
		let listed = " V....D libx264              libx264 H.264\n V....D libvpx-vp9           libvpx VP9\n V....D gif                  GIF";
		assert_eq!(pick_encoder(VideoFormat::Mp4, listed), Some("libx264"));
		assert_eq!(pick_encoder(VideoFormat::Webm, listed), Some("libvpx-vp9"));
		assert_eq!(pick_encoder(VideoFormat::Gif, listed), Some("gif"));
		assert_eq!(pick_encoder(VideoFormat::Mp4, " V....D h264_nvenc  NVIDIA"), Some("h264_nvenc"));
	}

	#[test]
	fn unavailable_hardware_falls_back_before_the_clip_starts() {
		let listed = " V....D libx264 H.264\n V....D h264_nvenc NVIDIA";
		let missing = Path::new("missing-ffmpeg-for-video-output-test.exe");
		assert_eq!(video_encoder(VideoFormat::Mp4, missing, listed, 3840, 2160, 120, "h264_nvenc").unwrap(), "libx264");
		assert_eq!(video_encoder(VideoFormat::Mp4, missing, listed, 1920, 1080, 60, "auto").unwrap(), "libx264");
		assert_eq!(video_encoder(VideoFormat::Mp4, missing, listed, 3840, 2160, 240, "auto").unwrap(), "libx264");
		assert!(video_encoder(VideoFormat::Mp4, missing, listed, 1920, 1080, 60, "unknown").is_err());
	}

	#[test]
	fn x264_threads_follow_the_frame_size() {
		let threads = |width, height| x264_arguments(width, height).windows(2).find(|pair| pair[0] == "-threads").map(|pair| pair[1].clone());
		assert_eq!(threads(1920, 1080).as_deref(), Some("6"));
		assert_eq!(threads(2560, 1440).as_deref(), Some("6"));
		assert_eq!(threads(3840, 2160).as_deref(), Some("8"));
	}

	/// A frame with known near-white and bright pixels.
	fn known_frame() -> (Vec<u8>, u32, u32) {
		let (width, height) = (64u32, 32u32);
		let mut rgba = [100u8, 100, 100, 255].repeat((width * height) as usize);
		// blocks on even rows and columns, so every 2 x 2 chroma block is one colour
		let mut fill = |left: u32, top: u32, columns: u32, rows: u32, pixel: [u8; 4]| {
			for y in top..top + rows {
				for x in left..left + columns {
					let index = ((y * width + x) * 4) as usize;
					rgba[index..index + 4].copy_from_slice(&pixel);
				}
			}
		};
		fill(0, 0, 10, 8, [255, 255, 255, 255]); // 80 white
		fill(20, 0, 6, 4, [235, 240, 232, 255]); // 24 near white
		fill(30, 0, 4, 6, [200, 255, 255, 255]); // 24 bright, red too low for near white
		fill(40, 0, 8, 2, [255, 255, 0, 255]); // 16 bright yellow
		fill(0, 10, 8, 8, [255, 255, 255, 128]); // 64 white at half alpha: grey over black
		(rgba, width, height)
	}

	#[test]
	fn frame_brightness_counts_known_pixels() {
		let (rgba, width, height) = known_frame();
		let measured = rgba_brightness(&rgba);
		assert_eq!((measured.near_white, measured.bright), (80 + 24, 80 + 24 + 24 + 16));
		let luma = |[r, g, b]: [f64; 3]| (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
		let expected = (80.0 * luma([255.0; 3]) + 24.0 * luma([235.0, 240.0, 232.0]) + 24.0 * luma([200.0, 255.0, 255.0]) + 16.0 * luma([255.0, 255.0, 0.0]) + 64.0 * luma([128.0; 3]) + 1840.0 * luma([100.0; 3])) / 2048.0;
		assert!((measured.luma - expected).abs() < 1e-9, "{} {expected}", measured.luma);

		// the MP4's frame (ror's own conversion, a transparent background flattened over black)
		let nv12 = crate::render::nv12::from_rgba(&rgba, width, height, true);
		let decoded = nv12_brightness(&nv12, width as usize, height as usize);
		assert_eq!((decoded.near_white, decoded.bright), (measured.near_white, measured.bright));
		assert!((decoded.luma - measured.luma).abs() < 0.003, "{} {}", decoded.luma, measured.luma);

		// a level below near white on any channel is not near white (greys alike as NV12)
		for pixel in [[229, 255, 255, 255], [255, 229, 255, 255], [255, 255, 229, 255], [229, 229, 229, 255]] {
			let frame = pixel.repeat(16);
			assert_eq!(rgba_brightness(&frame).near_white, 0, "{pixel:?}");
		}
		let grey = |level: u8| nv12_brightness(&crate::render::nv12::from_rgba(&[level, level, level, 255].repeat(16), 4, 4, true), 4, 4).near_white;
		assert_eq!((grey(229), grey(230)), (0, 16));
		assert_eq!(nv12_brightness(&[0; 5], 2, 2), Brightness::default(), "a short buffer measures nothing");
		// sixteen at a time counts what one at a time does, at every level and with a row's odd end
		let mut state = 0x9e3779b97f4a7c15u64;
		let mut next = || {
			state ^= state << 13;
			state ^= state >> 7;
			state ^= state << 17;
			(state >> 32) as u8
		};
		for width in [2usize, 16, 30, 50, 258] {
			let line: Vec<u8> = (0..width).map(|index| if index % 7 == 0 { 255 - (next() % 48) } else { next() }).collect();
			let pairs: Vec<u8> = (0..width).map(|_| 96 + next() % 64).collect();
			assert_eq!(row_levels(&line, &pairs), row_levels_plain(&line, &pairs, 0), "a row {width} wide");
		}
	}

	#[test]
	fn the_frames_line_names_the_peaks() {
		let frame = |index: usize, near_white: u64, luma: f64| FrameStats { time: index as f64 / 30.0, frame: index, near_white, bright: near_white * 2, luma };
		let frames = vec![frame(0, 0, 0.2), frame(1, 50, 0.4), frame(2, 100, 0.35), frame(3, 0, 0.1)];
		assert_eq!(frames_line(&frames, 1000).unwrap(), "frames: near white peaks at 0.07 s (frame 2, 10.00 % of pixels), mean luma peaks at 0.03 s (frame 1, 0.400, 10.00 % bright)");
		let dark = vec![frame(0, 0, 0.2), frame(1, 0, 0.2)];
		assert_eq!(frames_line(&dark, 1000).unwrap(), "frames: no near-white pixel in any frame, mean luma peaks at 0.00 s (frame 0, 0.200, 0.00 % bright)");
		assert!(frames_line(&[], 1000).is_none());
	}

	#[test]
	fn sheet_lays_out_and_labels() {
		let frame = vec![255u8; 64 * 36 * 4];
		let tiles = vec![("t 0.00 s".to_string(), frame.as_slice()); 5];
		let (rgba, width, height) = contact_sheet(&tiles, 64, 36, false);
		assert_eq!(rgba.len(), (width * height * 4) as usize);
		assert_eq!(sheet_columns(5), 3);
		for character in "t 0.40 s · 128 particles".chars() {
			assert!(glyph(character).is_some(), "{character:?}");
		}
	}
}
