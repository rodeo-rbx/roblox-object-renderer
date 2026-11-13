//! A render job: what to load, which views to draw, where to write them. The CLI runs jobs in
//! process; the render service runs many at once against one warm renderer.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime};

use anyhow::{Context, Result, bail};
use glam::Vec3;
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::lint::{Motion, Weightless};
use crate::output;
use crate::render::camera::{CameraSpec, default_fov};
use crate::render::{Background, RenderStats, Renderer, ViewRequest};
use crate::scene::{self, Scene};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Job {
	/// A model or place file (`.rbxm`, `.rbxmx`, `.rbxl`, `.rbxlx`), or an asset URI / id
	/// (`rbxassetid://123`, `123`).
	pub input: String,
	/// Draw only this instance (dotted path from the file's root).
	#[serde(default)]
	pub select: Option<String>,
	/// Output directory; `<state dir>/out/<input name>` when absent.
	#[serde(default)]
	pub out: Option<PathBuf>,
	#[serde(default = "default_views")]
	pub views: Vec<NamedView>,
	/// Skip PNG encoding (timings only, or a caller that reads `rgba` itself).
	#[serde(default)]
	pub no_png: bool,
	/// Smallest PNGs instead of the fastest encode.
	#[serde(default)]
	pub compact_png: bool,
	/// A place (or model) file whose MaterialService applies: its material generation and
	/// MaterialVariants (an object exported from a game has neither).
	#[serde(default)]
	pub materials_from: Option<String>,
	/// A place file whose Lighting applies (properties, sky, atmosphere, post effects) instead of the
	/// input's own.
	#[serde(default)]
	pub lighting_from: Option<String>,
	/// Draw the model at a moment of an effect's timeline: emits, enabled emitters, trails and beams,
	/// TimeScale, property sets and tweens, motion (scene::playback, docs/playback.md).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub playback: Option<scene::Playback>,
	/// Orbit views frame this box (min, max) rather than what the model draws: one camera for every
	/// moment of a timeline, or a region of a map.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub frame: Option<[[f32; 3]; 2]>,
	/// Orbit views frame what is at or under this instance ([`Scene::bounds_at`]), the rest drawn
	/// around it.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub focus: Option<String>,
	/// Draw what the parts collide with instead of what they show, those the filter keeps
	/// (scene::collision)
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub collision: Option<scene::collision::Filter>,
	/// Another file's world drawn around the model, as the setting it sits in (a game map).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub map: Option<MapSetting>,
	/// Also count what each view costs Roblox to draw: draw calls and triangles per render pass as
	/// its counters report them, the batches behind them and what would cut them most (crate::cost;
	/// cost.json, `<view>.draws.png` and `<view>.heat.png`)
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cost: Option<crate::cost::CostOptions>,
	/// The folder a relative local path in the input's content is read from (a MeshId
	/// `output/x.mesh` or `file://output/x.mesh`, as a game's asset table may name a file not uploaded yet): its
	/// caller's working directory, which the CLI gives, as the service runs in its state dir. None: the
	/// process's own.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub relative_to: Option<PathBuf>,
	/// With `ids`: how much of this instance (dotted path from the file's root) each view shows ([`Alone`]).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub alone: Option<String>,
}

/// How much of an instance a view shows (`--alone`): its parts' pixels in the view, and drawn with nothing else.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Alone {
	pub shown: u32,
	pub whole: u32,
}

/// A world drawn around the model (`--map`, [`scene::load::Document::add_setting`]): never framed, and
/// lighting the model with its Lighting and MaterialService unless `materials_from` / `lighting_from`
/// name others.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MapSetting {
	/// A place file (its Workspace is drawn) or a model file (its top-level instances).
	pub file: String,
	/// The point of the map that lands on the model's origin, and a yaw in degrees turning the map
	/// about it: x, y, z, yaw.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub origin: Option<[f32; 4]>,
}

impl MapSetting {
	/// `--map-origin`'s text: x,y,z or x,y,z,yaw.
	pub fn parse_origin(text: &str) -> Result<[f32; 4]> {
		let numbers: Vec<f32> = text.split(',').map(|word| word.trim().parse::<f32>().with_context(|| format!("--map-origin: {word:?} is not a number"))).collect::<Result<_>>()?;
		match numbers[..] {
			[x, y, z] => Ok([x, y, z, 0.0]),
			[x, y, z, yaw] => Ok([x, y, z, yaw]),
			_ => bail!("--map-origin takes x,y,z or x,y,z,yaw (got {} numbers)", numbers.len()),
		}
	}
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NamedView {
	pub name: String,
	#[serde(flatten)]
	pub view: ViewRequest,
}

fn default_views() -> Vec<NamedView> {
	vec![NamedView { name: "render".into(), view: ViewRequest::default() }]
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ViewResult {
	pub name: String,
	pub file: Option<PathBuf>,
	/// With `ids`: which part is in front at every pixel (`ids_png`), indices into `parts.json`.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub ids_file: Option<PathBuf>,
	pub width: u32,
	pub height: u32,
	/// The camera used: CFrame components (x y z R00 .. R22), vertical field of view and near plane
	/// (`--camera-from` draws another render from it).
	pub camera: [f32; 12],
	pub fov: f32,
	pub near: f32,
	/// With the view's `lod`: the levels its meshes drew at and the parts left out (render::LodView)
	#[serde(skip_serializing_if = "Option::is_none")]
	pub lod: Option<crate::render::LodView>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub alone: Option<Alone>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobResult {
	pub out: PathBuf,
	pub views: Vec<ViewResult>,
	pub parts: usize,
	pub lights: usize,
	pub bounds: Option<[[f32; 3]; 2]>,
	pub stats: RenderStats,
	/// The ParticleEmitters emitting at the moment drawn whose particles fly weightless (crate::lint).
	pub weightless: Weightless,
	/// With `cost`: what each view costs Roblox to draw (each view's costliest batches; cost.json
	/// holds every one)
	#[serde(skip_serializing_if = "Option::is_none")]
	pub cost: Option<crate::cost::CostReport>,
	pub warnings: Vec<String>,
	pub timings: Timings,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Timings {
	pub load_ms: f64,
	/// Getting GUI textures before the render (rur's renders on cache misses); part of `render_ms`.
	pub gui_ms: f64,
	/// Fetching the scene's files not yet on disk ahead of drawing, beside the GUI textures; part of
	/// `render_ms`.
	pub warm_ms: f64,
	pub render_ms: f64,
	pub encode_ms: f64,
	pub total_ms: f64,
	pub scene_cached: bool,
	/// Waiting for the GPU memory its draws' targets add (crate::budget::GpuGate); part of `render_ms`.
	pub gpu_wait_ms: f64,
}

/// Part ids as an opaque RGBA8 image: the id's low, middle and high bytes in red, green and blue.
fn ids_png(ids: &[u32]) -> Vec<u8> {
	ids.iter().flat_map(|id| [(id & 0xff) as u8, ((id >> 8) & 0xff) as u8, ((id >> 16) & 0xff) as u8, 255]).collect()
}

/// Named camera presets: `front`, `back`, `left`, `right`, `top`, `bottom`, `iso`, `default`,
/// and `turntable:N` (N views around the model).
pub fn preset_views(spec: &str, width: u32, height: u32, background: Background, fov: Option<f32>) -> Result<Vec<NamedView>> {
	let mut views = Vec::new();
	let orbit = |yaw: f32, pitch: f32| CameraSpec::Orbit { yaw, pitch, distance: None, padding: 0.08, target: None, fov: fov.unwrap_or_else(default_fov) };
	for name in spec.split(',').map(str::trim).filter(|name| !name.is_empty()) {
		let mut push = |name: String, camera: CameraSpec| {
			views.push(NamedView { name, view: ViewRequest { width, height, camera, background, ..Default::default() } });
		};
		match name {
			"front" => push(name.into(), orbit(0.0, 0.0)),
			"back" => push(name.into(), orbit(180.0, 0.0)),
			"right" => push(name.into(), orbit(90.0, 0.0)),
			"left" => push(name.into(), orbit(-90.0, 0.0)),
			"top" => push(name.into(), orbit(0.0, 89.9)),
			"bottom" => push(name.into(), orbit(0.0, -89.9)),
			"iso" => push(name.into(), orbit(45.0, 35.264)),
			"default" | "render" => push(name.into(), orbit(30.0, 20.0)),
			other => {
				if let Some(count) = other.strip_prefix("turntable:").or_else(|| other.strip_prefix("turntable")) {
					let count: u32 = if count.is_empty() { 8 } else { count.parse().context("turntable:N")? };
					for index in 0..count.max(1) {
						let yaw = index as f32 * 360.0 / count.max(1) as f32;
						push(format!("turntable-{index:02}"), orbit(yaw, 20.0));
					}
				} else {
					bail!("unknown view {other:?} (front, back, left, right, top, bottom, iso, default, turntable:N)");
				}
			}
		}
	}
	Ok(views)
}

/// What a place lends another scene: its MaterialService and its Lighting, as the environment it
/// makes and (for a playback that changes it) a copy of the instances.
struct PlaceSettings {
	materials: scene::Materials,
	environment: Option<scene::Environment>,
	lighting: Option<scene::load::LightingCopy>,
}

/// Parsed scenes, kept while their file is unchanged (the service reuses them across requests).
#[derive(Default)]
pub struct SceneCache {
	/// By file, selection, the folder its relative local paths are read from and whether it keeps the
	/// invisible parts (the collisions-only view's: [`scene::load::Document::keep_invisible`])
	entries: Mutex<HashMap<(PathBuf, Option<String>, Option<PathBuf>, bool), (SystemTime, u64, Arc<Scene>)>>,
	places: Mutex<HashMap<PathBuf, (SystemTime, u64, Arc<PlaceSettings>)>>,
	maps: Mutex<HashMap<PathBuf, (SystemTime, u64, Arc<scene::load::Document>)>>,
	/// Videos' inputs ([`crate::video::FixedInput`]) by [`SceneCache::video_key`], the last used last
	videos: Mutex<Vec<(String, Arc<crate::video::FixedInput>)>>,
	/// The file being loaded into each cache, so jobs missing the same one at once load it once
	loading: Loading,
	/// Per file: when the run of stills of a playback began, and how many there have been
	stills: Mutex<HashMap<PathBuf, (Instant, u32)>>,
}

/// Stills of one file's playback in [`STILL_WINDOW`] that make a timeline drawn frame by frame.
const FRAME_BY_FRAME: u32 = 8;
const STILL_WINDOW: std::time::Duration = std::time::Duration::from_secs(30);

/// One load per file at a time: a job that misses a file another job is loading waits for that load and
/// takes its result from the cache, instead of parsing the file again beside it (a batch of a hundred jobs on
/// one 20 MB place parsed it a hundred times at once, 2026-10-17).
#[derive(Default)]
struct Loading {
	files: Mutex<HashMap<(&'static str, PathBuf), Arc<Mutex<()>>>>,
}

impl Loading {
	/// Runs `load` holding the file's turn: `cached` is asked again once the turn is ours, since the job
	/// before may have loaded it.
	fn once<T>(&self, cache: &'static str, path: &Path, cached: impl Fn() -> Option<T>, load: impl FnOnce() -> Result<T>) -> Result<T> {
		let key = (cache, path.to_path_buf());
		let turn = self.files.lock().entry(key.clone()).or_default().clone();
		let result = {
			// a load this thread holds steps aside while it waits (crate::budget::park)
			let _turn = {
				let _parked = crate::budget::park();
				turn.lock()
			};
			match cached() {
				Some(hit) => Ok(hit),
				None => load(),
			}
		};
		let mut files = self.files.lock();
		// the map's and ours: no one else waits for this file
		if Arc::strong_count(&turn) == 2 {
			files.remove(&key);
		}
		result
	}
}

/// Video inputs kept ([`SceneCache::keep_video_input`])
const KEPT_VIDEO_INPUTS: usize = 2;

impl SceneCache {
	/// Counts a still of a playback of `path`; true once, when a run of them is a timeline drawn frame by
	/// frame. Each such still parses the file and extracts the whole scene again; `ror video --times` makes
	/// them from one load (a layer's moments 73 ms a frame by stills against 23, 2026-11-02).
	fn frame_by_frame(&self, path: &Path) -> bool {
		let mut stills = self.stills.lock();
		let now = Instant::now();
		let run = stills.entry(path.to_path_buf()).or_insert((now, 0));
		if now.duration_since(run.0) > STILL_WINDOW {
			*run = (now, 0);
		}
		run.1 += 1;
		run.1 == FRAME_BY_FRAME
	}

	/// What a video's input as it stands depends on: its file, the map drawn around it and the places
	/// lending MaterialService and Lighting (each with its modification time and size), what it
	/// selects and where its relative local paths are read from. `None` when the input is not a file
	/// (an asset).
	pub(crate) fn video_key(&self, input: &str, select: Option<&str>, map: Option<&MapSetting>, (materials, lighting): (Option<&str>, Option<&str>), relative_to: Option<&Path>) -> Option<String> {
		let stamp = |path: &str| -> Option<String> {
			let metadata = std::fs::metadata(path).ok().filter(|metadata| metadata.is_file())?;
			Some(format!("{path:?} {:?} {}", metadata.modified().ok(), metadata.len()))
		};
		let mut key = format!("{} select {select:?} relative to {relative_to:?}", stamp(input)?);
		if let Some(map) = map {
			key += &format!(" map {} at {:?}", stamp(&map.file)?, map.origin.map(|origin| origin.map(f32::to_bits)));
		}
		for (name, place) in [("materials", materials), ("lighting", lighting)] {
			if let Some(path) = place {
				key += &format!(" {name} {}", stamp(path)?);
			}
		}
		Some(key)
	}

	/// The video input kept under `key`.
	pub(crate) fn video_input(&self, key: &str) -> Option<Arc<crate::video::FixedInput>> {
		let mut videos = self.videos.lock();
		let index = videos.iter().position(|(kept, _)| kept == key)?;
		let entry = videos.remove(index);
		let input = entry.1.clone();
		videos.push(entry);
		Some(input)
	}

	/// Lets go of every video input kept.
	pub(crate) fn release_video_inputs(&self) {
		self.videos.lock().clear();
	}

	/// Lets go of every parsed scene, map, place and video input kept (the jobs holding one keep theirs):
	/// for a process short of memory, the next job reads its files again.
	pub fn release_all(&self) {
		self.release_video_inputs();
		self.entries.lock().clear();
		self.maps.lock().clear();
		self.places.lock().clear();
	}

	/// Keeps a video's input for the next job of the same, the least recently used going past
	/// [`KEPT_VIDEO_INPUTS`].
	pub(crate) fn keep_video_input(&self, key: String, input: Arc<crate::video::FixedInput>) {
		let mut videos = self.videos.lock();
		videos.retain(|(kept, _)| *kept != key);
		videos.push((key, input));
		while videos.len() > KEPT_VIDEO_INPUTS {
			videos.remove(0);
		}
	}

	/// A map file's document, parsed once while the file is unchanged (a game map is tens of
	/// thousands of instances).
	fn map_document(&self, path: &Path) -> Result<Arc<scene::load::Document>> {
		let metadata = std::fs::metadata(path).with_context(|| format!("--map: reading {}", path.display()))?;
		let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
		let cached = || {
			self.maps.lock().get(path).filter(|(time, size, _)| *time == modified && *size == metadata.len()).map(|(_, _, document)| document.clone())
		};
		if let Some(document) = cached() {
			return Ok(document);
		}
		self.loading.once("maps", path, cached, || {
			let document = Arc::new(scene::load::read_file(path).with_context(|| format!("--map {}", path.display()))?);
			let mut maps = self.maps.lock();
			if maps.len() > 4 {
				maps.clear();
			}
			maps.insert(path.to_path_buf(), (modified, metadata.len(), document.clone()));
			Ok(document)
		})
	}

	/// Draws `map`'s world around the document's ([`scene::load::Document::add_setting`]).
	pub(crate) fn add_map(&self, document: &mut scene::load::Document, map: &MapSetting) -> Result<()> {
		let world = self.map_document(Path::new(&map.file))?;
		document.add_setting(&world, map.origin);
		Ok(())
	}

	/// The places that lend the scene its MaterialService and Lighting: those named, else the map's
	/// own when it has one.
	pub(crate) fn lenders<'a>(&self, map: Option<&'a MapSetting>, materials_from: Option<&'a str>, lighting_from: Option<&'a str>) -> Result<(Option<&'a str>, Option<&'a str>)> {
		let Some(map) = map else {
			return Ok((materials_from, lighting_from));
		};
		let world = self.map_document(Path::new(&map.file))?;
		let materials = materials_from.or_else(|| world.material_service.map(|_| map.file.as_str()));
		let lighting = lighting_from.or_else(|| world.lighting.map(|_| map.file.as_str()));
		Ok((materials, lighting))
	}

	fn place(&self, path: &Path) -> Result<Arc<PlaceSettings>> {
		let metadata = std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
		let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
		let cached = || self.places.lock().get(path).filter(|(time, size, _)| *time == modified && *size == metadata.len()).map(|(_, _, place)| place.clone());
		if let Some(place) = cached() {
			return Ok(place);
		}
		self.loading.once("places", path, cached, || {
			let document = scene::load::read_file(path)?;
			let place = Arc::new(PlaceSettings { materials: scene::extract::materials(&document), environment: scene::extract::environment(&document), lighting: scene::load::LightingCopy::of(&document) });
			let mut places = self.places.lock();
			if places.len() > 16 {
				places.clear();
			}
			places.insert(path.to_path_buf(), (modified, metadata.len(), place.clone()));
			Ok(place)
		})
	}

	/// Gives `scene` another place's MaterialService (`materials_from`) and Lighting (`lighting_from`).
	pub(crate) fn lend(&self, scene: &mut Scene, materials_from: Option<&str>, lighting_from: Option<&str>) -> Result<()> {
		if let Some(path) = materials_from {
			scene.materials = self.place(Path::new(path))?.materials.clone();
		}
		if let Some(path) = lighting_from {
			scene.environment = self.place(Path::new(path))?.environment.clone().with_context(|| format!("{path} has no Lighting"))?;
		}
		Ok(())
	}

	/// Gives a playback's document the Lighting and the Terrain's Clouds it changes, when it names
	/// either: `lighting_from`'s (a copy), else the document's own, else ror's default
	/// ([`scene::load::Document::give_lighting`]). True when the document holds `lighting_from`'s: its
	/// scene then takes no Lighting from [`SceneCache::lend`].
	pub(crate) fn give_lighting(&self, document: &mut scene::load::Document, playback: &scene::Playback, lighting_from: Option<&str>) -> Result<bool> {
		if !playback.touches_lighting() {
			return Ok(false);
		}
		let Some(path) = lighting_from else {
			document.give_lighting(None);
			return Ok(false);
		};
		let place = self.place(Path::new(path))?;
		document.give_lighting(Some(place.lighting.as_ref().with_context(|| format!("{path} has no Lighting"))?));
		Ok(true)
	}

	/// A scene at a moment of a playback (not cached: every job has its own).
	fn load_playback(&self, document: scene::load::Document, playback: &scene::Playback) -> Result<Scene> {
		let mut document = document;
		let resolved = playback.resolve(&document)?;
		resolved.apply_state(&mut document)?;
		let mut scene = scene::extract::extract(&document);
		resolved.apply_history(&document, &mut scene);
		Ok(scene)
	}

	fn load(&self, path: &Path, select: Option<&str>, relative_to: Option<&Path>, keep_invisible: bool) -> Result<(Arc<Scene>, bool)> {
		let metadata = std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
		let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
		let key = (path.to_path_buf(), select.map(str::to_string), relative_to.map(Path::to_path_buf), keep_invisible);
		let cached = || self.entries.lock().get(&key).filter(|(time, size, _)| *time == modified && *size == metadata.len()).map(|(_, _, scene)| (scene.clone(), true));
		if let Some(hit) = cached() {
			return Ok(hit);
		}
		self.loading.once("scenes", path, cached, || {
			// the parse and the extraction within the machine's free memory, as one load
			let _budget = crate::budget::reserve(metadata.len());
			let mut document = scene::load::read_file(path)?;
			document.keep_invisible = keep_invisible;
			if let Some(select) = select {
				document.select(select)?;
			}
			if let Some(folder) = relative_to {
				document.read_relative_to(folder);
			}
			let scene = Arc::new(scene::extract::extract(&document));
			drop(document);
			let mut entries = self.entries.lock();
			if entries.len() > 256 {
				entries.clear();
			}
			entries.insert(key.clone(), (modified, metadata.len(), scene.clone()));
			Ok((scene, false))
		})
	}
}

/// An input's bytes: a model or place file, else an asset id or URI fetched.
pub(crate) fn input_bytes(renderer: &Renderer, input: &str) -> Result<Vec<u8>> {
	let path = Path::new(input);
	if path.is_file() {
		return std::fs::read(path).with_context(|| format!("reading {}", path.display()));
	}
	let uri = if input.chars().all(|c| c.is_ascii_digit()) { format!("rbxassetid://{input}") } else { input.to_string() };
	let fetched = renderer.assets.fetch(&uri).map_err(|error| anyhow::anyhow!("{input} is not a file, and as an asset: {error}"))?;
	Ok(fetched.bytes.to_vec())
}

/// An input's name as an output folder's (`<state dir>/out/<name>`).
pub fn input_name(input: &str) -> String {
	let path = Path::new(input);
	let stem = path.file_stem().map(|stem| stem.to_string_lossy().to_string()).unwrap_or_else(|| input.to_string());
	stem.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

/// Views drawn at once: a job of more (a map's sweep of views) draws them in groups, so their targets
/// never all take GPU memory together (72 views at 1920x1080 ran a 16 GB GPU out of memory).
const VIEWS_AT_ONCE: usize = 8;

/// [`Renderer::render`] of `requests`, [`VIEWS_AT_ONCE`] at a time, each group admitted within the
/// GPU's free memory by `gpu` ([`crate::budget::GpuGate`]); the stats added up over the groups.
fn render_in_groups(renderer: &Renderer, scene: &Scene, requests: &[ViewRequest], gpu: Option<&crate::budget::GpuGate>) -> Result<(Vec<crate::render::ViewImage>, RenderStats)> {
	let mut images = Vec::with_capacity(requests.len());
	let mut stats: Option<RenderStats> = None;
	for group in requests.chunks(VIEWS_AT_ONCE.max(1)) {
		let (drawn, more) = renderer.render_gated(scene, group, gpu)?;
		images.extend(drawn);
		match stats.as_mut() {
			None => stats = Some(more),
			Some(stats) => {
				stats.gpu_wait_ms += more.gpu_wait_ms;
				stats.prepare_ms += more.prepare_ms;
				stats.sun_shadow_ms += more.sun_shadow_ms;
				stats.light_shadow_ms += more.light_shadow_ms;
				stats.assets_ms += more.assets_ms;
				stats.light_grid_ms += more.light_grid_ms;
				stats.draw_ms += more.draw_ms;
				stats.readback_ms += more.readback_ms;
				stats.skinned += more.skinned;
				stats.skin_ms += more.skin_ms;
				stats.skin_upload_ms += more.skin_upload_ms;
				stats.skin_upload_bytes += more.skin_upload_bytes;
				stats.draws += more.draws;
				stats.triangles += more.triangles;
				stats.guis += more.guis;
				stats.gui_ms += more.gui_ms;
				stats.particles += more.particles;
				for missing in more.missing {
					if !stats.missing.contains(&missing) {
						stats.missing.push(missing);
					}
				}
				for warning in more.warnings {
					if !stats.warnings.contains(&warning) {
						stats.warnings.push(warning);
					}
				}
			}
		}
	}
	Ok((images, stats.unwrap_or_default()))
}

pub fn run(renderer: &Renderer, scenes: &SceneCache, job: &Job) -> Result<JobResult> {
	run_gated(renderer, scenes, job, None)
}

/// Each view's [`Alone`] for the instance at `path`: its parts' ids counted in `images`, and in the same views
/// drawn again from the same cameras with only those parts.
fn alone(renderer: &Renderer, scene: &Scene, path: &str, requests: &[ViewRequest], images: &[crate::render::ViewImage], gpu: Option<&crate::budget::GpuGate>) -> Result<Vec<Alone>> {
	let under = format!("{path}.");
	let mine: Vec<usize> = scene.parts.iter().enumerate().filter(|(_, part)| part.path == path || part.path.starts_with(&under)).map(|(index, _)| index).collect();
	if mine.is_empty() {
		bail!("--alone {path:?}: no part is at or under it");
	}
	let only = Scene { parts: mine.iter().map(|&index| scene.parts[index].clone()).collect(), environment: scene.environment.clone(), ..Default::default() };
	let again: Vec<ViewRequest> = requests
		.iter()
		.zip(images)
		.map(|(request, image)| ViewRequest { camera: CameraSpec::CFrame { components: image.camera.cframe_components(), fov: image.camera.fov, near: Some(image.camera.near) }, bloom: false, ..request.clone() })
		.collect();
	let (drawn, _) = render_in_groups(renderer, &only, &again, gpu)?;
	let mut is_mine = vec![false; scene.parts.len() + 1];
	for &index in &mine {
		is_mine[index + 1] = true;
	}
	let count = |ids: Option<&[u32]>, of: &dyn Fn(u32) -> bool| ids.unwrap_or(&[]).iter().filter(|&&id| of(id)).count() as u32;
	Ok(images.iter().zip(&drawn).map(|(image, alone)| Alone { shown: count(image.ids.as_deref(), &|id| is_mine[id as usize]), whole: count(alone.ids.as_deref(), &|id| id > 0) }).collect())
}

/// [`run`], each draw admitted within the GPU's free memory by `gpu` (loading, readbacks and PNG encoding
/// run outside it).
pub fn run_gated(renderer: &Renderer, scenes: &SceneCache, job: &Job, gpu: Option<&crate::budget::GpuGate>) -> Result<JobResult> {
	// within the memory the machine has free, beside the process's other jobs (crate::budget::admit)
	let pixels: u64 = job.views.iter().map(|named| named.view.width as u64 * named.view.height as u64 * (named.view.supersample.max(1) as u64).pow(2)).sum();
	let key = format!("render {} {:?} {pixels}", job.input, job.select);
	let input_bytes = crate::budget::file_bytes([Some(Path::new(&job.input)), job.map.as_ref().map(|map| Path::new(&map.file))].into_iter().flatten());
	let mut admitted = admit(renderer, scenes, &format!("render of {}", job.input), key, crate::budget::still_need(pixels, input_bytes))?;
	let result = run_admitted(renderer, scenes, job, gpu);
	if result.is_ok() {
		admitted.finished();
	}
	result
}

/// Admits a job ([`crate::budget::admit`]): short of memory, the renderer's and `scenes`' caches let go of
/// what they keep first.
pub(crate) fn admit(renderer: &Renderer, scenes: &SceneCache, what: &str, key: String, need: crate::budget::Need) -> Result<crate::budget::JobGuard> {
	crate::budget::admit(what, Some(key), need, &|goal| relieve(renderer, scenes, goal))
}

/// Has the renderer's caches and `scenes` let go of what they keep (up to `goal` bytes of the renderer's,
/// all of the parsed scenes) and the allocator give its free pages back; returns how far this process's
/// commit fell, or the renderer's bytes freed where that is not known.
pub fn relieve(renderer: &Renderer, scenes: &SceneCache, goal: u64) -> u64 {
	let before = crate::process_commit();
	let freed = renderer.relieve(goal);
	scenes.release_all();
	crate::release_free_memory();
	match (before, crate::process_commit()) {
		(Some(before), Some(after)) => before.saturating_sub(after),
		_ => freed,
	}
}

fn run_admitted(renderer: &Renderer, scenes: &SceneCache, job: &Job, gpu: Option<&crate::budget::GpuGate>) -> Result<JobResult> {
	let start = Instant::now();
	let mut timings = Timings::default();
	let path = Path::new(&job.input);
	// the places lending MaterialService and Lighting: those named, else a map's own
	let (materials_from, lighting_from) = scenes.lenders(job.map.as_ref(), job.materials_from.as_deref(), job.lighting_from.as_deref())?;
	// a playback that changes the Lighting or the Terrain's Clouds has `lighting_from`'s in its document already
	let mut lent_lighting = false;
	// the cost model reads the document the scene came from (its parts' referents point into it)
	let mut cost_document = None;
	let (scene, cached) = if job.playback.is_some() || job.map.is_some() || job.cost.is_some() {
		// every such job has a document of its own: its parse and scene within the machine's free memory, as one load
		let _budget = crate::budget::reserve(std::fs::metadata(path).map(|metadata| metadata.len()).unwrap_or(0));
		let mut document = if path.is_file() {
			scene::load::read_file(path)?
		} else {
			let uri = if job.input.chars().all(|c| c.is_ascii_digit()) { format!("rbxassetid://{}", job.input) } else { job.input.clone() };
			let fetched = renderer.assets.fetch(&uri).map_err(|error| anyhow::anyhow!("{} is not a file, and as an asset: {error}", job.input))?;
			scene::load::read_bytes(&fetched.bytes)?
		};
		document.keep_invisible = job.collision.is_some();
		if let Some(select) = &job.select {
			document.select(select)?;
		}
		if let Some(folder) = &job.relative_to {
			document.read_relative_to(folder);
		}
		if let Some(map) = &job.map {
			scenes.add_map(&mut document, map)?;
		}
		match &job.playback {
			Some(playback) => {
				if job.cost.is_none() && scenes.frame_by_frame(path) {
					crate::say(&format!("{FRAME_BY_FRAME} stills of {}'s playback in {} s: `ror video --times a,b,c --format png` draws moments from one load, about a third of the time each", path.display(), STILL_WINDOW.as_secs()));
				}
				document.play = playback.play;
				document.salt = playback.salt.clone();
				lent_lighting = scenes.give_lighting(&mut document, playback, lighting_from)?;
				let scene = if job.cost.is_some() {
					let resolved = playback.resolve(&document)?;
					resolved.apply_state(&mut document)?;
					let mut scene = scene::extract::extract(&document);
					resolved.apply_history(&document, &mut scene);
					cost_document = Some(document);
					scene
				} else {
					scenes.load_playback(document, playback)?
				};
				(Arc::new(scene), false)
			}
			None => {
				let scene = scene::extract::extract(&document);
				if job.cost.is_some() {
					cost_document = Some(document);
				}
				(Arc::new(scene), false)
			}
		}
	} else if path.is_file() {
		scenes.load(path, job.select.as_deref(), job.relative_to.as_deref(), job.collision.is_some())?
	} else {
		let uri = if job.input.chars().all(|c| c.is_ascii_digit()) { format!("rbxassetid://{}", job.input) } else { job.input.clone() };
		let fetched = renderer.assets.fetch(&uri).map_err(|error| anyhow::anyhow!("{} is not a file, and as an asset: {error}", job.input))?;
		let mut document = scene::load::read_bytes(&fetched.bytes)?;
		document.keep_invisible = job.collision.is_some();
		if let Some(select) = &job.select {
			document.select(select)?;
		}
		if let Some(folder) = &job.relative_to {
			document.read_relative_to(folder);
		}
		(Arc::new(scene::extract::extract(&document)), false)
	};
	// another place's MaterialService / Lighting
	let lighting_from = lighting_from.filter(|_| !lent_lighting);
	let scene = if materials_from.is_some() || lighting_from.is_some() {
		let mut scene = scene.as_ref().clone();
		scenes.lend(&mut scene, materials_from, lighting_from)?;
		Arc::new(scene)
	} else {
		scene
	};
	let scene = match job.collision {
		Some(filter) => {
			scene::collision::require()?;
			Arc::new(scene::collision::view(&scene, filter).0)
		}
		None => scene,
	};
	timings.load_ms = start.elapsed().as_secs_f64() * 1000.0;
	timings.scene_cached = cached;

	let render_start = Instant::now();
	let mut requests: Vec<ViewRequest> = job.views.iter().map(|view| view.view.clone()).collect();
	// orbit views on a given box, or on what is at an instance, rather than the whole model; with a
	// map, on the model alone
	let framed = match (&job.frame, &job.focus) {
		(Some([min, max]), _) => Some((Vec3::from(*min).min(Vec3::from(*max)), Vec3::from(*min).max(Vec3::from(*max)))),
		(None, Some(focus)) => Some(scene.bounds_at(focus).with_context(|| format!("focus {focus:?}: no part or effect is at or under it"))?),
		(None, None) => scene.framing_subject().and_then(|subject| crate::video::subject_box(renderer, &subject)),
	};
	if let Some(bounds) = framed {
		for view in &mut requests {
			view.camera = view.camera.framing(bounds, view.width.max(1), view.height.max(1));
		}
	}
	// a cost job without images needs only the views' cameras; a grid's too (its views' images are
	// never written: the grid's own image is drawn from above, cost::run)
	let cameras_only = job.cost.as_ref().is_some_and(|options| options.no_images || options.grid.is_some());
	// GUI content (rur's renders) before taking a GPU slot, and meanwhile the scene's downloads fetched
	// many at a time (Assets::warm: assetdelivery's batch endpoint) rather than a round trip per file and
	// transcode pack as drawing asks for each: the Lift an Egg map's 615 files (76 MB) from an empty
	// cache took 4-6 s so against 31-41 s. A scene the service has loaded before has its files already;
	// otherwise, all cached, the check takes ~20 ms.
	let mut warnings = scene.warnings.clone();
	if !cameras_only {
		let (gui_warnings, gui_ms, warm_ms) = std::thread::scope(|scope| {
			let warming = (!cached).then(|| {
				scope.spawn(|| {
					let start = Instant::now();
					renderer.assets.warm(&crate::render::downloads(&scene), &crate::render::transcode_packs);
					start.elapsed().as_secs_f64() * 1000.0
				})
			});
			let gui = renderer.prefetch_guis(&scene, &requests);
			let gui_ms = render_start.elapsed().as_secs_f64() * 1000.0;
			(gui, gui_ms, warming.map_or(0.0, |warming| warming.join().unwrap_or(0.0)))
		});
		warnings.extend(gui_warnings);
		timings.gui_ms = gui_ms;
		timings.warm_ms = warm_ms;
	}
	let (images, mut stats) = if cameras_only {
		(Vec::new(), RenderStats::default())
	} else {
		render_in_groups(renderer, &scene, &requests, gpu)?
	};
	let alone = match &job.alone {
		Some(path) if images.iter().any(|image| image.ids.is_some()) => alone(renderer, &scene, path, &requests, &images, gpu)?,
		_ => Vec::new(),
	};
	let bounds = if cameras_only {
		let subject = scene.framing_subject();
		renderer.framing_bounds(subject.as_ref().unwrap_or(&scene))
	} else {
		None
	};
	// each view's camera and size, as drawn (or as it would be)
	let shots: Vec<(crate::render::camera::Camera, u32, u32)> = if cameras_only {
		requests.iter().map(|request| (request.camera.place(bounds, request.width.max(1), request.height.max(1)), request.width, request.height)).collect()
	} else {
		images.iter().map(|image| (image.camera, image.width, image.height)).collect()
	};
	// with a map, the box the views framed is the model's, not the map's
	if job.frame.is_none() && job.focus.is_none() && scene.setting.is_some() {
		stats.framed = framed.map(|(min, max)| [min.into(), max.into()]);
	}
	for warning in &stats.warnings {
		if !warnings.contains(warning) {
			warnings.push(warning.clone());
		}
	}
	timings.render_ms = render_start.elapsed().as_secs_f64() * 1000.0;
	timings.gpu_wait_ms = stats.gpu_wait_ms;

	let out = job.out.clone().unwrap_or_else(|| output::state_dir().join("out").join(input_name(&job.input)));
	std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
	let encode_start = Instant::now();
	let fast = !job.compact_png;
	let files: Vec<Result<Option<PathBuf>>> = job
		.views
		.par_iter()
		.zip(images.par_iter())
		.map(|(view, image)| {
			if job.no_png {
				return Ok(None);
			}
			let file = out.join(format!("{}.png", view.name));
			output::write_png(&file, image.width, image.height, &image.rgba, fast)?;
			if let Some(ids) = &image.ids {
				output::write_png(&out.join(format!("{}.ids.png", view.name)), image.width, image.height, &ids_png(ids), fast)?;
			}
			Ok(Some(file))
		})
		.collect();
	if images.iter().any(|image| image.ids.is_some()) {
		let parts: Vec<serde_json::Value> = scene
			.parts
			.iter()
			.map(|part| serde_json::json!({"path": part.path, "class": part.class, "material": part.material, "transparency": part.transparency, "surfaceAppearance": part.surface_appearance.is_some(), "position": Vec3::from(part.cframe.translation).to_array()}))
			.collect();
		output::write_atomic(&out.join("parts.json"), serde_json::to_string_pretty(&parts)?.as_bytes())?;
		// visible.json: each view's parts and the pixels each shows (what --no-png still keeps of the ids:
		// visibility sweeps over thousands of views write no images)
		let visible: Vec<serde_json::Value> = job
			.views
			.par_iter()
			.zip(images.par_iter())
			.map(|(view, image)| {
				let mut counts: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
				for &id in image.ids.as_deref().unwrap_or(&[]) {
					if id > 0 {
						*counts.entry(id - 1).or_default() += 1;
					}
				}
				serde_json::json!({ "name": view.name, "parts": counts.into_iter().collect::<Vec<_>>() })
			})
			.collect();
		output::write_atomic(&out.join("visible.json"), serde_json::to_string(&serde_json::json!({ "views": visible }))?.as_bytes())?;
	}
	timings.encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;

	let mut views = Vec::with_capacity(shots.len());
	let mut files = files.into_iter();
	for (index, (view, (camera, width, height))) in job.views.iter().zip(&shots).enumerate() {
		let file = files.next().transpose()?.flatten();
		let ids = images.get(index).is_some_and(|image| image.ids.is_some());
		views.push(ViewResult {
			name: view.name.clone(),
			ids_file: file.as_ref().filter(|_| ids).map(|_| out.join(format!("{}.ids.png", view.name))),
			file,
			width: *width,
			height: *height,
			camera: camera.cframe_components(),
			fov: camera.fov,
			near: camera.near,
			lod: images.get(index).and_then(|image| image.lod.clone()),
			alone: alone.get(index).copied(),
		});
	}
	let cost = match (&job.cost, &cost_document) {
		(Some(options), Some(document)) => Some(crate::cost::run(renderer, &scene, document, options, &job.views, &shots, &out, gpu)?),
		_ => None,
	};
	// the authoring lint reads the model, not the map around it
	let mut motion = Motion::default();
	motion.add(scene.framing_subject().as_ref().unwrap_or(&scene));
	timings.total_ms = start.elapsed().as_secs_f64() * 1000.0;
	let result = JobResult {
		out: out.clone(),
		views,
		parts: scene.parts.len(),
		lights: scene.lights.len(),
		bounds: scene.bounds().map(|(min, max)| [min.into(), max.into()]),
		stats,
		weightless: motion.weightless(),
		cost,
		warnings,
		timings,
	};
	output::write_json_atomic(&out.join("summary.json"), &result, true)?;
	Ok(result)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A job's relative local paths are read from the folder it names (its caller's working directory,
	/// which the CLI sends to a service running in its state dir), whatever the process's own: a
	/// MeshPart naming its mesh by a repo-relative path or a relative file:// URI draws as one naming
	/// the absolute path, and without the folder it is missing, as the service drew it before.
	#[test]
	fn a_jobs_relative_local_paths_are_read_from_its_folder() {
		use rbx_dom_weak::types::{Content, Vector3};
		let state = std::env::temp_dir().join(format!("ror-relative-job-{}", std::process::id()));
		let project = state.join("project");
		std::fs::create_dir_all(project.join("output")).unwrap();
		// a 4 x 4 quad, both sides (FileMesh 1.01: position, normal and uvw a vertex, three vertices a face)
		let corner = |x: i32, y: i32| format!("[{x},{y},0][0,0,-1][{},{},0]", (x + 2) / 4, (y + 2) / 4);
		let faces = [(-2, -2), (2, -2), (2, 2), (-2, -2), (2, 2), (-2, 2), (-2, -2), (2, 2), (2, -2), (-2, -2), (-2, 2), (2, 2)].map(|(x, y)| corner(x, y)).concat();
		std::fs::write(project.join("output").join("Quad.mesh"), format!("version 1.01\n4\n{faces}\n")).unwrap();
		let absolute = project.join("output").join("Quad.mesh").display().to_string();
		let render = |name: &str, mesh: &str, relative_to: Option<&Path>| {
			let mut model = crate::scene::build::ModelBuilder::new("Model");
			let part = model.part(&crate::scene::build::PartSpec { class: "MeshPart", name: "Quad".into(), size: glam::Vec3::new(4.0, 4.0, 0.05), color: [220, 60, 40], ..Default::default() });
			model.set(part, "MeshContent", Content::from_uri(mesh));
			model.set(part, "InitialSize", Vector3::new(4.0, 4.0, 0.05));
			let input = state.join(format!("{name}.rbxm"));
			std::fs::write(&input, model.to_rbxm().unwrap()).unwrap();
			let camera = CameraSpec::LookAt { eye: [0.0, 0.0, -9.0], target: [0.0; 3], fov: 50.0 };
			Job {
				input: input.display().to_string(),
				select: None,
				out: Some(state.join(name)),
				views: vec![NamedView { name: "front".into(), view: ViewRequest { width: 96, height: 96, camera, ..Default::default() } }],
				no_png: false,
				compact_png: false,
				materials_from: None,
				lighting_from: None,
				playback: None,
				frame: None,
				focus: None,
				collision: None,
				map: None,
				cost: None,
				relative_to: relative_to.map(Path::to_path_buf),
				alone: None,
			}
		};
		let gpu = Arc::new(crate::render::gpu::Gpu::new().expect("a GPU (or the software adapter)"));
		let assets = ror_assets::Assets::new(ror_assets::AssetConfig { offline: true, ..ror_assets::AssetConfig::new(state.join("cache")) });
		let renderer = Renderer::new(gpu, assets).expect("a renderer");
		let scenes = SceneCache::default();
		let drawn = |job: &Job| {
			let result = run(&renderer, &scenes, job).unwrap();
			(result.stats.missing.clone(), std::fs::read(result.views[0].file.as_ref().unwrap()).unwrap())
		};
		let (missing, by_path) = drawn(&render("absolute", &format!("file://{absolute}"), None));
		// (without a Studio install its sky and sun are missing too, in every render here)
		let no_quad = |missing: &[String]| missing.iter().any(|line| line.contains("Quad.mesh"));
		assert!(!no_quad(&missing), "{missing:?}");
		for (name, mesh) in [("relative", "output/Quad.mesh"), ("uri", "file://output/Quad.mesh")] {
			let (missing, pixels) = drawn(&render(name, mesh, Some(&project)));
			assert!(!no_quad(&missing) && pixels == by_path, "{name}: {missing:?}");
			// the process's own folder (a service's state dir) has no output/Quad.mesh: the quad is not drawn
			let (missing, unmeshed) = drawn(&render(&format!("{name}-here"), mesh, None));
			assert!(no_quad(&missing) && unmeshed != by_path, "{name} read from here: {missing:?}");
		}
		let _ = std::fs::remove_dir_all(&state);
	}

	/// A still of a playback that sets the Terrain's Clouds draws the stage's clouds as it leaves them.
	#[test]
	fn a_still_draws_the_stages_clouds_as_the_playback_leaves_them() {
		use rbx_dom_weak::{InstanceBuilder, WeakDom, types::Color3};
		let folder = std::env::temp_dir().join(format!("ror-still-clouds-{}", std::process::id()));
		std::fs::create_dir_all(&folder).unwrap();
		let mut place = WeakDom::new(InstanceBuilder::new("DataModel"));
		let root = place.root_ref();
		let workspace = place.insert(root, InstanceBuilder::new("Workspace").with_name("Workspace"));
		let terrain = place.insert(workspace, InstanceBuilder::new("Terrain").with_name("Terrain"));
		place.insert(terrain, InstanceBuilder::new("Clouds").with_name("Clouds").with_property("Cover", 0.84f32).with_property("Density", 0.03f32).with_property("Color", Color3::new(2.5, 3.49, 4.98)));
		let lighting = place.insert(root, InstanceBuilder::new("Lighting").with_name("Lighting"));
		let stage = folder.join("stage.rbxl");
		let mut bytes = Vec::new();
		rbx_binary::to_writer(&mut bytes, &place, &[workspace, lighting]).unwrap();
		std::fs::write(&stage, bytes).unwrap();
		let mut model = crate::scene::build::ModelBuilder::new("Model");
		model.part(&crate::scene::build::PartSpec::default());
		let playback: scene::Playback = serde_json::from_value(serde_json::json!({
			"time": 0.5,
			"actions": [{"at": 0.2, "target": "Workspace.Terrain.Clouds", "set": {"Cover": 0.95, "Color": [0.27451, 0.2902, 0.3451]}}],
		}))
		.unwrap();
		let scenes = SceneCache::default();
		let mut document = model.into_document();
		assert!(scenes.give_lighting(&mut document, &playback, Some(&stage.display().to_string())).unwrap(), "the stage's look is in the document");
		let clouds = scenes.load_playback(document, &playback).unwrap().environment.clouds.expect("the stage's clouds");
		assert!((clouds.cover - 0.95).abs() < 1e-6 && (clouds.density - 0.03).abs() < 1e-6 && clouds.color.abs_diff_eq(glam::Vec3::new(0.27451, 0.2902, 0.3451), 1e-5), "{clouds:?}");
		let _ = std::fs::remove_dir_all(&folder);
	}

	/// The collisions-only view draws an invisible wall that collides as it draws the same wall seen, and
	/// leaves out one that does not collide; a render without the view still draws nothing of either.
	#[test]
	fn the_collision_view_draws_invisible_parts_that_collide() {
		if !scene::collision::AVAILABLE {
			return;
		}
		use crate::scene::build::{ModelBuilder, PartSpec};
		use crate::scene::collision::Filter;
		let state = std::env::temp_dir().join(format!("ror-collision-invisible-{}", std::process::id()));
		std::fs::create_dir_all(&state).unwrap();
		// a floor and, in front of the camera, a wall (its Transparency and CanCollide) or none
		let model = |name: &str, wall: Option<(f32, bool)>| {
			let mut model = ModelBuilder::new("Model");
			model.part(&PartSpec { name: "Floor".into(), cframe: glam::Affine3A::from_translation(Vec3::new(0.0, -3.0, 0.0)), size: Vec3::new(12.0, 1.0, 12.0), ..Default::default() });
			if let Some((transparency, can_collide)) = wall {
				let part = model.part(&PartSpec { name: "Wall".into(), size: Vec3::new(6.0, 4.0, 1.0), color: [40, 120, 220], transparency, ..Default::default() });
				model.set(part, "CanCollide", can_collide);
			}
			let input = state.join(format!("{name}.rbxm"));
			std::fs::write(&input, model.to_rbxm().unwrap()).unwrap();
			input
		};
		let inputs = [("none", None), ("seen", Some((0.0, true))), ("invisible", Some((1.0, true))), ("ghost", Some((1.0, false)))].map(|(name, wall)| (name, model(name, wall)));
		let gpu = Arc::new(crate::render::gpu::Gpu::new().expect("a GPU (or the software adapter)"));
		let assets = ror_assets::Assets::new(ror_assets::AssetConfig { offline: true, ..ror_assets::AssetConfig::new(state.join("cache")) });
		let renderer = Renderer::new(gpu, assets).expect("a renderer");
		let scenes = SceneCache::default();
		let drawn = |name: &str, collision: Option<Filter>| {
			let input = &inputs.iter().find(|(wall, _)| *wall == name).unwrap().1;
			let camera = CameraSpec::LookAt { eye: [0.0, 1.0, -10.0], target: [0.0; 3], fov: 50.0 };
			let job = Job {
				input: input.display().to_string(),
				select: None,
				out: Some(state.join(format!("{name}-{}", if collision.is_some() { "collision" } else { "plain" }))),
				views: vec![NamedView { name: "front".into(), view: ViewRequest { width: 64, height: 64, camera, ..Default::default() } }],
				no_png: false,
				compact_png: false,
				materials_from: None,
				lighting_from: None,
				playback: None,
				frame: None,
				focus: None,
				collision,
				map: None,
				cost: None,
				relative_to: None,
				alone: None,
			};
			let result = run(&renderer, &scenes, &job).unwrap();
			(result.parts, std::fs::read(result.views[0].file.as_ref().unwrap()).unwrap())
		};
		let plain = |name: &str| drawn(name, None);
		let collision = |name: &str| drawn(name, Some(Filter::Collide));
		assert!(plain("seen") != plain("none"), "the camera sees the wall");
		assert!(plain("invisible") == plain("none") && plain("ghost") == plain("none"), "an invisible wall draws nothing");
		assert!(collision("seen") != collision("none"), "the view sees the wall");
		let (invisible, seen) = (collision("invisible"), collision("seen"));
		assert!(invisible == seen, "an invisible wall collides as the wall seen ({} parts drawn, {} with the wall seen)", invisible.0, seen.0);
		assert!(collision("ghost") == collision("none"), "a wall that does not collide is left out");
		let _ = std::fs::remove_dir_all(&state);
	}

	#[test]
	fn alone_counts_what_hides_an_instance_in_each_view() {
		use crate::scene::build::{ModelBuilder, PartSpec};
		let state = std::env::temp_dir().join(format!("ror-alone-{}", std::process::id()));
		std::fs::create_dir_all(&state).unwrap();
		// a block, and between it and the front camera a wall over its left half
		let mut model = ModelBuilder::new("Model");
		model.part(&PartSpec { name: "Subject".into(), size: Vec3::new(2.0, 2.0, 2.0), ..Default::default() });
		model.part(&PartSpec { name: "Wall".into(), cframe: glam::Affine3A::from_translation(Vec3::new(3.0, 0.0, -4.0)), size: Vec3::new(6.0, 6.0, 0.5), ..Default::default() });
		let input = state.join("model.rbxm");
		std::fs::write(&input, model.to_rbxm().unwrap()).unwrap();
		let gpu = Arc::new(crate::render::gpu::Gpu::new().expect("a GPU (or the software adapter)"));
		let assets = ror_assets::Assets::new(ror_assets::AssetConfig { offline: true, ..ror_assets::AssetConfig::new(state.join("cache")) });
		let renderer = Renderer::new(gpu, assets).expect("a renderer");
		let view = |name: &str, z: f32| NamedView { name: name.into(), view: ViewRequest { width: 96, height: 96, ids: true, camera: CameraSpec::LookAt { eye: [0.0, 0.0, z], target: [0.0; 3], fov: 40.0 }, ..Default::default() } };
		let job = |alone: &str| Job {
			input: input.display().to_string(),
			select: None,
			out: Some(state.join("out")),
			views: vec![view("front", -12.0), view("back", 12.0)],
			no_png: true,
			compact_png: false,
			materials_from: None,
			lighting_from: None,
			playback: None,
			frame: None,
			focus: None,
			collision: None,
			map: None,
			cost: None,
			relative_to: None,
			alone: Some(alone.into()),
		};
		let result = run(&renderer, &SceneCache::default(), &job("Model.Subject")).unwrap();
		let [front, back] = [result.views[0].alone.unwrap(), result.views[1].alone.unwrap()];
		assert!(front.whole > 100 && back.whole == front.whole, "the block alone fills the same pixels from both sides: {front:?} {back:?}");
		assert_eq!(back.shown, back.whole, "nothing hides it from behind");
		assert!(front.shown * 3 > front.whole && front.shown * 3 < front.whole * 2, "the wall hides about half of it from the front: {front:?}");
		assert!(run(&renderer, &SceneCache::default(), &job("Model.Nothing")).is_err(), "an instance with no part is an error");
		let _ = std::fs::remove_dir_all(&state);
	}
}
