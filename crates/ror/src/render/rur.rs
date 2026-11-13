//! rur, the offline Roblox UI renderer, draws SurfaceGui and BillboardGui content for render::guis
//! (notes/render.md "GUIs"). The GUIs a render misses go into one atlas model per GUI scale: a
//! ScreenGui per GUI (keeping its ZIndexBehavior) holding a Frame the size of its canvas at whole
//! device pixels of the atlas, with the GUI's children in it. rur draws the atlas with a transparent
//! background, and it is cut back into one image per GUI.
//!
//! rur is found through `ROR_RUR` (its checkout), else beside this checkout
//! (`../roblox-ui-renderer`); Node through `ROR_NODE`, else `node` on the PATH. An atlas goes straight
//! to rur's render service (`rur-native serve`, the same engine arguments `rur render --model` sends
//! it) once a `rur render` through Node has started that service and it answers as this checkout's,
//! running the current engine build: Node's start-up and rur.mjs's checks cost 170-870 ms a call,
//! the service's draw of an atlas 30-650 ms (2026-10-10). Each call draws into one of a few work directories kept
//! between calls, so rur's service finds the images it decoded for the last atlas at the same paths
//! (its decoded images are keyed by file path).

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use parking_lot::Mutex;
use rbx_dom_weak::types::{Enum, UDim, UDim2};
use rbx_dom_weak::{InstanceBuilder, WeakDom};

/// A path without Windows' verbatim prefix ([`crate::output::plain_path`]): rur.mjs runs only when
/// Node's resolved argument equals its own path.
fn plain(path: &Path) -> PathBuf {
	crate::output::plain_path(path)
}

/// The rur checkout to use, if there is one.
pub fn locate() -> Option<PathBuf> {
	let candidate = match std::env::var_os("ROR_RUR") {
		Some(path) if !path.is_empty() => PathBuf::from(path),
		_ => crate::output::checkout_dir()?.join("..").join("roblox-ui-renderer"),
	};
	let candidate = plain(&candidate);
	candidate.join("cli").join("rur.mjs").is_file().then_some(candidate)
}

/// rur's state dir as its CLI resolves it (cli/paths.mjs): RUR_STATE_DIR, RUR_CACHE_DIR, else the
/// checkout's `.cache`.
fn state_dir(rur: &Path) -> PathBuf {
	let given = ["RUR_STATE_DIR", "RUR_CACHE_DIR"].into_iter().find_map(|name| std::env::var_os(name).filter(|value| !value.is_empty())).map(PathBuf::from);
	plain(&given.unwrap_or_else(|| rur.join(".cache")))
}

/// The native app's manifest (`native-app/manifest.json` in rur's state dir, RUR_STATE_DIR or its
/// checkout's `.cache`), when there is one.
fn native_manifest(rur: &Path) -> Option<serde_json::Value> {
	let state_dir = ["RUR_STATE_DIR", "RUR_CACHE_DIR"].into_iter().find_map(|name| std::env::var_os(name).filter(|value| !value.is_empty())).map(PathBuf::from);
	state_dir.into_iter().chain([rur.join(".cache")]).find_map(|directory| {
		let manifest = std::fs::read_to_string(directory.join("native-app").join("manifest.json")).ok()?;
		serde_json::from_str(&manifest).ok()
	})
}

/// Which rur draws: its native app's build fingerprint (`native-app/manifest.json` in rur's state dir,
/// RUR_STATE_DIR or its checkout's `.cache`), so images cached on disk follow rur's updates; "unbuilt"
/// before its first build.
pub fn revision(rur: &Path) -> String {
	match native_manifest(rur).and_then(|manifest| manifest["buildFingerprint"].as_str().map(str::to_string)) {
		Some(fingerprint) => fingerprint.chars().take(16).collect(),
		None => "unbuilt".into(),
	}
}

fn node() -> PathBuf {
	match std::env::var_os("ROR_NODE") {
		Some(path) if !path.is_empty() => PathBuf::from(path),
		_ => PathBuf::from("node"),
	}
}

/// A GUI to draw: its subtree (a binary model whose root holds the GUI instance) and canvas.
pub struct Request {
	pub content: Arc<Vec<u8>>,
	/// Canvas size in GUI pixels.
	pub canvas: [f32; 2],
	/// Device pixels kept around the canvas for content that overflows it (a GUI that does not clip).
	pub margin: [u32; 2],
	/// `Enum.ZIndexBehavior` of the GUI.
	pub z_index_behavior: u32,
}

/// A GUI's pixels: straight-alpha RGBA8 (display values), the canvas covering `canvas · scale` device
/// pixels from `margin`.
pub struct Image {
	pub width: u32,
	pub height: u32,
	pub rgba: Vec<u8>,
}

/// What rur reported besides the pixels (missing images and fonts, script errors).
#[derive(Default)]
pub struct Report {
	pub warnings: Vec<String>,
	pub milliseconds: f64,
}

/// Device pixels of a GUI's image: its canvas at `scale` and the margins on both sides.
pub fn image_size(canvas: [f32; 2], scale: f32, margin: [u32; 2]) -> (u32, u32) {
	let size = |value: f32, margin: u32| ((value * scale - 0.01).ceil() as u32).max(1) + 2 * margin;
	(size(canvas[0], margin[0]), size(canvas[1], margin[1]))
}

/// The children of a GUI root that only shape a GuiObject: under a LayerCollector they do nothing,
/// under the atlas's canvas Frame they would pad, scale, constrain or outline it.
const ROOT_ONLY: [&str; 9] = ["UIPadding", "UIScale", "UIAspectRatioConstraint", "UISizeConstraint", "UITextSizeConstraint", "UICorner", "UIStroke", "UIGradient", "UIFlexItem"];

/// Empty space between atlas cells (device pixels).
const GAP: u32 = 2;
/// The atlas's width limit (device pixels); wider canvases get a row of their own.
const ATLAS_WIDTH: u32 = 4096;
/// A taller atlas is split in two (rur draws its frame on the GPU in one go).
const ATLAS_HEIGHT: u32 = 4096;

/// A whole-and-fraction UDim for a logical length: the fraction as scale of `parent`.
fn udim(length: f64, parent: f64) -> UDim {
	let whole = length.floor();
	UDim::new(((length - whole) / parent) as f32, whole as i32)
}

/// Shelf packing, tallest first: each image's top-left corner and the atlas's size.
fn pack(sizes: &[(u32, u32)]) -> (Vec<(u32, u32)>, u32, u32) {
	let mut order: Vec<usize> = (0..sizes.len()).collect();
	order.sort_by_key(|&index| std::cmp::Reverse(sizes[index].1));
	let mut cells = vec![(0u32, 0u32); sizes.len()];
	let (mut x, mut y, mut row_height, mut width) = (GAP, GAP, 0u32, 0u32);
	for &index in &order {
		let (w, h) = sizes[index];
		if x > GAP && x + w + GAP > ATLAS_WIDTH {
			x = GAP;
			y += row_height + GAP;
			row_height = 0;
		}
		cells[index] = (x, y);
		x += w + GAP;
		width = width.max(x);
		row_height = row_height.max(h);
	}
	(cells, width, y + row_height + GAP)
}

/// Work directories kept between rur calls: `<base>/work/<pid>-<slot>`, one per call running at once.
/// rur's service keys the images it decoded by their path in the output directory, so an atlas drawn
/// in the same directory as the last finds them decoded (a game mode's telegraphs: 130 ms of a
/// 320 ms call went to decoding the same images again in a new directory).
static SLOTS: Mutex<Vec<bool>> = Mutex::new(Vec::new());

struct Slot(usize);

impl Slot {
	fn take() -> Self {
		let mut slots = SLOTS.lock();
		let index = slots.iter().position(|taken| !taken).unwrap_or_else(|| {
			slots.push(false);
			slots.len() - 1
		});
		slots[index] = true;
		Slot(index)
	}

	fn directory(&self, base: &Path) -> PathBuf {
		base.join("work").join(format!("{}-{}", std::process::id(), self.0))
	}
}

impl Drop for Slot {
	fn drop(&mut self) {
		SLOTS.lock()[self.0] = false;
	}
}

/// Draws `requests` at `scale` device pixels per GUI pixel in one rur render (or, when they do not
/// fit one atlas, in several one after another); one image per request. `base` holds the work
/// directories (`<base>/work`).
pub fn render(rur: &Path, requests: &[Request], scale: f32, base: &Path, offline: bool) -> Result<(Vec<Image>, Report)> {
	let sizes: Vec<(u32, u32)> = requests.iter().map(|request| image_size(request.canvas, scale, request.margin)).collect();
	let (cells, width, height) = pack(&sizes);
	if height > ATLAS_HEIGHT && requests.len() > 1 {
		let (first, second) = requests.split_at(requests.len() / 2);
		let (mut images, mut report) = render(rur, first, scale, base, offline)?;
		let (more, more_report) = render(rur, second, scale, base, offline)?;
		images.extend(more);
		report.warnings.extend(more_report.warnings);
		report.milliseconds += more_report.milliseconds;
		return Ok((images, report));
	}
	let slot = Slot::take();
	render_atlas(rur, requests, &sizes, &cells, (width, height), scale, &slot.directory(base), offline)
}

/// The atlas model: a ScreenGui per request at its cell, holding the GUI's children. rur draws an
/// atlas's GUIs a little differently by where they fall in it (an edge pixel 2-9 levels off here and
/// there: its sizes and places are fractions of the atlas, and its vertices are scaled into the
/// atlas's viewport), so a GUI's image depends on what shared its atlas.
fn atlas_model(requests: &[Request], cells: &[(u32, u32)], (width, height): (u32, u32), scale: f32) -> Result<Vec<u8>> {
	let (logical_width, logical_height) = (width as f64 / scale as f64, height as f64 / scale as f64);
	let mut atlas = WeakDom::new(InstanceBuilder::new("DataModel"));
	let root = atlas.root_ref();
	let mut roots = Vec::new();
	for (index, request) in requests.iter().enumerate() {
		let content = rbx_binary::from_reader(request.content.as_slice()).map_err(|error| anyhow!("GUI content: {error}"))?;
		let gui = content.root().children().first().copied().context("GUI content without the GUI")?;
		let screen = atlas.insert(
			root,
			InstanceBuilder::new("ScreenGui")
				.with_name(format!("RorGui{index}"))
				.with_property("IgnoreGuiInset", true)
				.with_property("ResetOnSpawn", false)
				.with_property("DisplayOrder", index as i32)
				.with_property("ZIndexBehavior", Enum::from_u32(request.z_index_behavior)),
		);
		roots.push(screen);
		let scale = scale as f64;
		let (cell_x, cell_y) = cells[index];
		let frame = |name: &str, position: UDim2, size: UDim2, clips: bool| {
			InstanceBuilder::new("Frame")
				.with_name(name)
				.with_property("Position", position)
				.with_property("Size", size)
				.with_property("BackgroundTransparency", 1.0f32)
				.with_property("BorderSizePixel", 0i32)
				.with_property("ClipsDescendants", clips)
		};
		let [canvas_width, canvas_height] = request.canvas.map(f64::from);
		let canvas = if request.margin == [0, 0] {
			// the canvas is the cell, clipping its content
			let position = UDim2::new(udim(cell_x as f64 / scale, logical_width), udim(cell_y as f64 / scale, logical_height));
			atlas.insert(screen, frame("Canvas", position, UDim2::new(udim(canvas_width, logical_width), udim(canvas_height, logical_height)), true))
		} else {
			// a clipping cell around a canvas that does not clip (margins on every side)
			let (margin_x, margin_y) = (request.margin[0] as f64 / scale, request.margin[1] as f64 / scale);
			let (cell_width, cell_height) = (canvas_width + 2.0 * margin_x, canvas_height + 2.0 * margin_y);
			let position = UDim2::new(udim(cell_x as f64 / scale, logical_width), udim(cell_y as f64 / scale, logical_height));
			let cell = atlas.insert(screen, frame("Cell", position, UDim2::new(udim(cell_width, logical_width), udim(cell_height, logical_height)), true));
			let inner = UDim2::new(udim(margin_x, cell_width), udim(margin_y, cell_height));
			atlas.insert(cell, frame("Canvas", inner, UDim2::new(udim(canvas_width, cell_width), udim(canvas_height, cell_height)), false))
		};
		let gui_instance = content.get_by_ref(gui).context("GUI content")?;
		for &child in gui_instance.children() {
			let Some(instance) = content.get_by_ref(child) else {
				continue;
			};
			if ROOT_ONLY.contains(&instance.class.as_str()) {
				continue;
			}
			let clone = content.clone_into_external(child, &mut atlas);
			atlas.transfer_within(clone, canvas);
		}
	}
	let mut model = Vec::new();
	rbx_binary::to_writer(&mut model, &atlas, &roots).map_err(|error| anyhow!("writing the GUI atlas: {error}"))?;
	Ok(model)
}

#[allow(clippy::too_many_arguments)]
fn render_atlas(rur: &Path, requests: &[Request], sizes: &[(u32, u32)], cells: &[(u32, u32)], (width, height): (u32, u32), scale: f32, work: &Path, offline: bool) -> Result<(Vec<Image>, Report)> {
	let started = Instant::now();
	let (logical_width, logical_height) = (width as f64 / scale as f64, height as f64 / scale as f64);
	let model = atlas_model(requests, cells, (width, height), scale)?;
	std::fs::create_dir_all(work).with_context(|| format!("creating {}", work.display()))?;
	let work = plain(work);
	let model_path = work.join("atlas.rbxm");
	std::fs::write(&model_path, &model)?;
	let out = work.join("out");
	// the directory is kept between calls: an earlier atlas's image must never be read as this one's
	for stale in ["render.png", "scene.json", "manifest.json"] {
		let _ = std::fs::remove_file(out.join(stale));
	}
	// an empty project (no story roots, no Studio snapshots), the same directory every time: rur keeps a
	// preloaded VM per startup configuration
	let empty = plain(&crate::output::state_dir().join("guis").join("project"));
	std::fs::create_dir_all(&empty)?;
	let profile = std::env::var_os("ROR_PROFILE").is_some();
	if profile {
		eprintln!("profile: rur atlas {width}x{height} of {} GUIs built in {:.1} ms", requests.len(), started.elapsed().as_secs_f64() * 1000.0);
	}
	let call = Instant::now();
	let atlas = Atlas { model: &model_path, viewport: format!("{logical_width}x{logical_height}"), scale: format!("{scale}"), empty: &empty, out: &out, offline };
	let (summary, pixels, how) = match direct::render(rur, &atlas) {
		Some((summary, pixels)) => (summary, pixels, "service"),
		None => {
			let summary = render_through_node(rur, &atlas)?;
			direct::found_after_node(rur);
			(summary, None, "node")
		}
	};
	if profile {
		eprintln!("profile: rur call {:.1} ms ({how}) seconds {} phases {}", call.elapsed().as_secs_f64() * 1000.0, summary["seconds"], summary["phases"]);
	}
	let mut report = Report::default();
	for (key, label) in [("missingImages", "missing image"), ("missingFonts", "missing font"), ("renderErrors", "render error")] {
		for entry in summary[key].as_array().into_iter().flatten() {
			report.warnings.push(format!("rur: {label}: {}", entry.as_str().map(str::to_string).unwrap_or_else(|| entry.to_string())));
		}
	}
	if summary["scriptErrors"].as_u64().unwrap_or(0) > 0 {
		report.warnings.push(format!("rur: {} script errors", summary["scriptErrors"]));
	}
	// the service's pixels as they came (rur 261164e on), else its render.png
	let (atlas_width, atlas_height, pixels) = match pixels {
		Some(pixels) => pixels,
		None => {
			let decode_start = Instant::now();
			let decoded = crate::output::decode_png(&out.join("render.png")).context("rur's render.png")?;
			if profile {
				eprintln!("profile: rur atlas decode {:.1} ms", decode_start.elapsed().as_secs_f64() * 1000.0);
			}
			decoded
		}
	};
	if (atlas_width, atlas_height) != (width, height) {
		bail!("rur drew the GUI atlas at {atlas_width}x{atlas_height}, expected {width}x{height}");
	}
	let mut images = Vec::with_capacity(requests.len());
	for (index, &(w, h)) in sizes.iter().enumerate() {
		let (cell_x, cell_y) = cells[index];
		let mut rgba = Vec::with_capacity((w * h * 4) as usize);
		for row in cell_y..cell_y + h {
			let start = ((row * width + cell_x) * 4) as usize;
			rgba.extend_from_slice(&pixels[start..start + (w * 4) as usize]);
		}
		images.push(Image { width: w, height: h, rgba });
	}
	report.milliseconds = started.elapsed().as_secs_f64() * 1000.0;
	Ok((images, report))
}

/// Keeps the rur services this process drew GUIs with from going idle while it runs (the render
/// service calls this every few minutes): rur's service exits after 15 minutes without a request,
/// and the GUIs of a render after that waited 4-5 s for a new one.
pub fn keep_alive() {
	direct::keep_alive();
}

/// One atlas for rur: the model, its logical viewport and scale, the empty project and the output
/// directory.
struct Atlas<'a> {
	model: &'a Path,
	viewport: String,
	scale: String,
	empty: &'a Path,
	out: &'a Path,
	offline: bool,
}

/// `rur render --model` through Node (which starts or replaces rur's render service as needed); the
/// summary it prints, after checking that it drew a transparent background.
fn render_through_node(rur: &Path, atlas: &Atlas) -> Result<serde_json::Value> {
	let mut command = Command::new(node());
	command
		.arg(rur.join("cli").join("rur.mjs"))
		.arg("render")
		.arg("--model")
		.arg(atlas.model)
		.arg("--viewport")
		.arg(&atlas.viewport)
		.arg("--scale")
		.arg(&atlas.scale)
		.arg("--background")
		.arg("transparent")
		.arg("--time")
		.arg("0")
		.arg("--roots")
		.arg(atlas.empty)
		.arg("--no-snapshots")
		.arg("--out")
		.arg(atlas.out)
		.arg("--json")
		.current_dir(rur)
		.stdin(Stdio::null())
		.stdout(Stdio::piped())
		.stderr(Stdio::piped());
	if atlas.offline {
		command.arg("--offline");
	}
	// the Vulkan drivers a render service was started with are its own (service::RememberedGpu): rur
	// chooses its GPU among all of them, as always
	if std::env::var_os("ROR_VULKAN_ADAPTER").is_some() {
		command.env_remove("VK_LOADER_DRIVERS_SELECT").env_remove("ROR_VULKAN_ADAPTER");
	}
	// and the layers ror turned off for its own device (render::gpu) are rur's to choose
	for name in super::gpu::own_loader_settings() {
		command.env_remove(name);
	}
	// rur reads the account cookie itself; only the path of ror's cookie file is passed on
	if std::env::var_os("RUR_COOKIE_ENV").is_none()
		&& let Some(path) = std::env::var_os("ROR_COOKIE_ENV")
	{
		command.env("RUR_COOKIE_ENV", path);
	}
	// the render service has no console: without this every Node gets a new one (400-500 ms a call)
	#[cfg(windows)]
	{
		use std::os::windows::process::CommandExt;
		const CREATE_NO_WINDOW: u32 = 0x0800_0000;
		command.creation_flags(CREATE_NO_WINDOW);
	}
	// rur's native helper outlives the call: it must not take this process's stdout along
	crate::private_std_handles();
	let mut child = command.spawn().with_context(|| format!("starting {} for rur", node().display()))?;
	// a first run after a rur update builds its native app (minutes)
	let deadline = Instant::now() + Duration::from_secs(600);
	let mut stdout = child.stdout.take().context("rur stdout")?;
	let mut stderr = child.stderr.take().context("rur stderr")?;
	let reader = std::thread::spawn(move || {
		let mut text = String::new();
		let _ = stdout.read_to_string(&mut text);
		text
	});
	let errors = std::thread::spawn(move || {
		let mut text = String::new();
		let _ = stderr.read_to_string(&mut text);
		text
	});
	let status = loop {
		if let Some(status) = child.try_wait()? {
			break status;
		}
		if Instant::now() > deadline {
			let _ = child.kill();
			bail!("rur did not finish within 10 minutes");
		}
		std::thread::sleep(Duration::from_millis(5));
	};
	let stdout = reader.join().unwrap_or_default();
	let stderr = errors.join().unwrap_or_default();
	let tail = |text: &str| text.lines().rev().take(6).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(" | ");
	if !status.success() {
		bail!("rur render failed ({status}): {}", tail(&stderr));
	}
	let Some(summary) = stdout.lines().rev().find_map(|line| serde_json::from_str::<serde_json::Value>(line).ok()) else {
		bail!("rur printed no summary: {} {}", tail(&stdout), tail(&stderr));
	};
	// a rur without transparent backgrounds paints its default colour behind everything (the scene
	// is written compact, keys sorted)
	let scene = std::fs::read_to_string(atlas.out.join("scene.json")).context("rur's scene.json")?;
	if !scene.contains("\"background\":null") {
		bail!("rur at {} has no transparent backgrounds (update it)", rur.display());
	}
	Ok(summary)
}

/// Atlases straight to rur's render service (native/src/serve.rs), as cli/client.mjs sends a
/// `rur render`: `POST /render {argv, out, png, offline, env}` with the engine arguments cli/rur.mjs
/// builds for `--model`, then the summary cli/rur.mjs would print, made from the scene and manifest
/// the service wrote. Used only once a Node call started or confirmed the service: it must answer
/// as this checkout's with rur's state dir, running the native app its manifest names (a rebuilt
/// engine goes back through Node, which replaces the service).
mod direct {
	use super::*;
	use crate::service::http;

	/// Environment variables rur's engine reads (cli/client.mjs ENGINE_ENV): sent as they are in this
	/// process, as Node sends its own.
	const ENGINE_ENV: [&str; 20] = [
		"RUR_CAMERA",
		"RUR_CLOCK",
		"RUR_ENGINE_DUMP",
		"RUR_ENGINE_TIMINGS",
		"RUR_EPOCH",
		"RUR_FONT_ROOT",
		"RUR_HINTED_MAX_SIZE",
		"RUR_LAYOUT_STATS",
		"RUR_LUAU_PROFILE",
		"RUR_NATIVE_ASSET_MB",
		"RUR_ORDER",
		"RUR_PLACE_ID",
		"RUR_PLAYER",
		"RUR_PLAYER_DISPLAY",
		"RUR_PLAYER_ID",
		"RUR_PROFILE_CALLBACKS",
		"RUR_REAL_CLOCK",
		"RUR_ROBLOX_CONTENT",
		"RUR_ROTATION_CLIPPING",
		"RUR_VERBOSE",
	];

	/// The service found for a checkout: its port and the native app build it runs.
	#[derive(Clone)]
	struct Found {
		port: u16,
		build: String,
	}

	static FOUND: Mutex<Option<HashMap<PathBuf, Found>>> = Mutex::new(None);

	fn disabled() -> bool {
		std::env::var("ROR_RUR_DIRECT").is_ok_and(|value| value == "0") || ["RUR_ISOLATED", "RUR_NO_SERVICE"].iter().any(|name| std::env::var(name).is_ok_and(|value| !value.is_empty() && value != "0"))
	}

	/// Node's `path.resolve` of a directory compared on Windows: case-insensitively.
	fn normalized(path: &Path) -> String {
		let text = plain(path).to_string_lossy().trim_end_matches(['\\', '/']).to_string();
		if cfg!(windows) { text.to_lowercase() } else { text }
	}

	/// The ports rur's client tries for a checkout and state dir (cli/client.mjs servicePorts): the
	/// first 8 hex digits of sha256("<root>\0<state dir>"), 4180 + (hash + i · 97) mod 400.
	pub(super) fn ports(rur: &Path) -> Vec<u16> {
		if let Some(port) = std::env::var("RUR_SERVE_PORT").ok().and_then(|value| value.trim().parse::<u16>().ok()).filter(|port| *port > 0) {
			return vec![port];
		}
		let digest = sha256(format!("{}\0{}", normalized(rur), normalized(&state_dir(rur))).as_bytes());
		let hash = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as u64;
		(0..4).map(|index| (4180 + (hash + index * 97) % 400) as u16).collect()
	}

	/// The native app's executable file name and build fingerprint, from its manifest.
	fn current_build(rur: &Path) -> Option<(String, String)> {
		let manifest = native_manifest(rur)?;
		Some((manifest["executable"].as_str()?.to_string(), manifest["buildFingerprint"].as_str()?.to_string()))
	}

	/// Whether a service's answer is this checkout's, with rur's state dir.
	fn owns(json: &serde_json::Value, rur: &Path) -> bool {
		let same = |key: &str, path: &Path| json[key].as_str().is_some_and(|text| normalized(Path::new(text)) == normalized(path));
		same("root", rur) && same("stateDir", &state_dir(rur))
	}

	/// Looks for this checkout's service on its ports: one answering as this checkout's and running
	/// the native app the manifest names.
	fn find(rur: &Path) -> Option<Found> {
		let (executable, build) = current_build(rur)?;
		for port in ports(rur) {
			let Ok((200, body)) = http::request_as(port, "GET", "/health", "X-RUR: 1", b"", Duration::from_millis(1500)) else {
				continue;
			};
			let Ok(json) = serde_json::from_slice::<serde_json::Value>(&body) else {
				continue;
			};
			let runs = json["executable"].as_str().and_then(|path| Path::new(path).file_name().map(|name| name.to_string_lossy().to_string()));
			if json["service"] == "rur-serve" && owns(&json, rur) && runs.is_some_and(|name| name.eq_ignore_ascii_case(&executable)) {
				return Some(Found { port, build });
			}
		}
		None
	}

	/// After a call through Node (which starts or replaces the service): finds the service for the
	/// calls after it.
	pub(super) fn found_after_node(rur: &Path) {
		if disabled() {
			return;
		}
		let found = find(rur);
		let mut known = FOUND.lock();
		let known = known.get_or_insert_with(HashMap::new);
		match found {
			Some(found) => {
				known.insert(rur.to_path_buf(), found);
			}
			None => {
				known.remove(rur);
			}
		}
	}

	/// Asks the services found so far how they are, which keeps them from going idle (rur's exits
	/// after 15 minutes without a request, and a new one takes 4-5 s to start).
	pub(super) fn keep_alive() {
		let found: Vec<Found> = FOUND.lock().as_ref().map(|known| known.values().cloned().collect()).unwrap_or_default();
		for service in found {
			let _ = http::request_as(service.port, "GET", "/health", "X-RUR: 1", b"", Duration::from_millis(1500));
		}
	}

	fn forget(rur: &Path) {
		if let Some(known) = FOUND.lock().as_mut() {
			known.remove(rur);
		}
	}

	/// The atlas drawn by the service, with the summary cli/rur.mjs would print and, from a service that
	/// answers with them (`"pixels": "rgba"`), its straight-alpha RGBA rows (width, height, bytes: no
	/// PNG encoded there and decoded here; an older service writes render.png); `None` when the service
	/// cannot be used for it (not found yet, a newer engine build, an error): the caller goes through
	/// Node, which also renders without a service.
	pub(super) fn render(rur: &Path, atlas: &Atlas) -> Option<(serde_json::Value, Option<(u32, u32, Vec<u8>)>)> {
		if disabled() {
			return None;
		}
		let found = FOUND.lock().as_ref().and_then(|known| known.get(rur).cloned())?;
		// the engine was rebuilt since: Node replaces the service
		if current_build(rur).is_none_or(|(_, build)| build != found.build) {
			forget(rur);
			return None;
		}
		let argv: Vec<String> = [
			"engine",
			"--repo",
			&rur.to_string_lossy(),
			"--roots",
			&atlas.empty.to_string_lossy(),
			"--model",
			&atlas.model.to_string_lossy(),
			"--viewport",
			&atlas.viewport,
			"--scale",
			&atlas.scale,
			"--time",
			"0",
			"--background",
			"transparent",
		]
		.iter()
		.map(|arg| arg.to_string())
		.collect();
		let env: serde_json::Map<String, serde_json::Value> = ENGINE_ENV.iter().filter_map(|name| std::env::var(name).ok().filter(|value| !value.is_empty()).map(|value| (name.to_string(), value.into()))).collect();
		let body = serde_json::json!({"argv": argv, "out": atlas.out.to_string_lossy(), "png": true, "pixels": "rgba", "offline": atlas.offline, "env": env});
		let started = Instant::now();
		let answer = http::request_typed(found.port, "POST", "/render", "X-RUR: 1", body.to_string().as_bytes(), Duration::from_secs(600));
		let (status, content_type, mut reply) = match answer {
			Ok(answer) => answer,
			Err(_) => {
				// gone (idle for 15 minutes): Node starts it again
				forget(rur);
				return None;
			}
		};
		// the JSON line, then the rows
		let pixels = content_type.as_deref() == Some(PIXELS);
		let rows = if pixels {
			let line = reply.iter().position(|&byte| byte == b'\n')?;
			let rows = reply.split_off(line + 1);
			reply.truncate(line);
			rows
		} else {
			Vec::new()
		};
		let json: serde_json::Value = serde_json::from_slice(&reply).ok()?;
		if status != 200 || !owns(&json, rur) {
			forget(rur);
			return None;
		}
		let result = &json["result"];
		let seconds = started.elapsed().as_secs_f64();
		if !pixels {
			return summary(rur, atlas.out, result, seconds).ok().map(|summary| (summary, None));
		}
		let (width, height) = (result["width"].as_u64()? as u32, result["height"].as_u64()? as u32);
		if rows.len() != width as usize * height as usize * 4 || result["summary"]["transparent"] != serde_json::Value::Bool(true) {
			return None;
		}
		let drawn = &result["summary"];
		let summary = serde_json::json!({
			"status": drawn["status"],
			"missingImages": drawn["missingImages"].as_array().cloned().unwrap_or_default(),
			"missingFonts": drawn["missingFonts"].as_array().cloned().unwrap_or_default(),
			"scriptErrors": drawn["scriptErrors"],
			"renderErrors": result["renderErrors"].as_array().cloned().unwrap_or_default(),
			"seconds": (seconds * 1000.0).round() / 1000.0,
			"phases": result["phases"],
		});
		Some((summary, Some((width, height, rows))))
	}

	/// The Content-Type of a render answered with its pixels.
	const PIXELS: &str = "application/x-rur-pixels";

	/// cli/rur.mjs's summary of a render from the service's result and the files it wrote: missing
	/// images (the manifest's entries with an error), missing fonts and script errors (the scene's
	/// diagnostics), render errors. Fails, as Node does, on a scene drawn over a background.
	fn summary(rur: &Path, out: &Path, result: &serde_json::Value, seconds: f64) -> Result<serde_json::Value> {
		let scene: serde_json::Value = serde_json::from_slice(&std::fs::read(out.join("scene.json")).context("rur's scene.json")?).context("rur's scene.json")?;
		if !scene.get("background").is_some_and(serde_json::Value::is_null) {
			bail!("rur at {} has no transparent backgrounds (update it)", rur.display());
		}
		let manifest: serde_json::Value = std::fs::read(out.join("manifest.json")).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default();
		// JavaScript's truthiness, as rur.mjs filters the entries
		let truthy = |value: &serde_json::Value| match value {
			serde_json::Value::Null => false,
			serde_json::Value::Bool(flag) => *flag,
			serde_json::Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
			serde_json::Value::String(text) => !text.is_empty(),
			_ => true,
		};
		let missing_images: Vec<String> = manifest["images"]
			.as_object()
			.into_iter()
			.flatten()
			.filter_map(|(uri, entry)| entry.get("error").filter(|error| truthy(error)).map(|error| format!("{uri}: {}", error.as_str().map(str::to_string).unwrap_or_else(|| error.to_string()))))
			.collect();
		let diagnostics = &scene["diagnostics"];
		let script_errors = diagnostics["output"].as_array().into_iter().flatten().filter(|entry| entry["kind"] == "error").count();
		Ok(serde_json::json!({
			"status": scene["status"],
			"missingImages": missing_images,
			"missingFonts": diagnostics["missingFontAssets"].as_array().cloned().unwrap_or_default(),
			"scriptErrors": script_errors,
			"renderErrors": result["renderErrors"].as_array().cloned().unwrap_or_default(),
			"seconds": (seconds * 1000.0).round() / 1000.0,
			"phases": result["phases"],
		}))
	}

	/// SHA-256 (FIPS 180-4), for the ports rur's client derives.
	pub(super) fn sha256(message: &[u8]) -> [u8; 32] {
		const K: [u32; 64] = [
			0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
			0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
			0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
			0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
		];
		let mut state: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
		let mut data = message.to_vec();
		data.push(0x80);
		while data.len() % 64 != 56 {
			data.push(0);
		}
		data.extend_from_slice(&((message.len() as u64) * 8).to_be_bytes());
		for block in data.chunks_exact(64) {
			let mut w = [0u32; 64];
			for (index, word) in block.chunks_exact(4).enumerate() {
				w[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
			}
			for index in 16..64 {
				let s0 = w[index - 15].rotate_right(7) ^ w[index - 15].rotate_right(18) ^ (w[index - 15] >> 3);
				let s1 = w[index - 2].rotate_right(17) ^ w[index - 2].rotate_right(19) ^ (w[index - 2] >> 10);
				w[index] = w[index - 16].wrapping_add(s0).wrapping_add(w[index - 7]).wrapping_add(s1);
			}
			let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
			for index in 0..64 {
				let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
				let choose = (e & f) ^ (!e & g);
				let t1 = h.wrapping_add(s1).wrapping_add(choose).wrapping_add(K[index]).wrapping_add(w[index]);
				let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
				let majority = (a & b) ^ (a & c) ^ (b & c);
				let t2 = s0.wrapping_add(majority);
				(h, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
			}
			for (value, add) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
				*value = value.wrapping_add(add);
			}
		}
		let mut digest = [0u8; 32];
		for (index, value) in state.iter().enumerate() {
			digest[index * 4..index * 4 + 4].copy_from_slice(&value.to_be_bytes());
		}
		digest
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn sha256_matches_the_standard() {
		let hex = |digest: [u8; 32]| digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
		assert_eq!(hex(direct::sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
		assert_eq!(hex(direct::sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
		// two blocks
		assert_eq!(hex(direct::sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
	}

	#[test]
	fn slots_are_reused_and_never_shared() {
		let first = Slot::take();
		let second = Slot::take();
		assert_ne!(first.0, second.0);
		let index = first.0;
		drop(first);
		assert_eq!(Slot::take().0, index, "a free slot is taken again");
	}
}
