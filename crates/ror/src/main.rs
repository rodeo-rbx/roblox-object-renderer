#[cfg(all(windows, not(feature = "dhat-heap")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Heap profiling (`--features dhat-heap`, a command run with --no-service): dhat-heap.json on exit
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static GLOBAL: dhat::Alloc = dhat::Alloc;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use ror::job::{self, Job, SceneCache};
use ror::output;
use ror::render::camera::CameraSpec;
use ror::render::gpu::Gpu;
use ror::render::{Background, Renderer};

#[derive(Parser)]
#[command(name = "ror", version, about = "Roblox object renderer")]
struct Cli {
	#[command(subcommand)]
	command: Command,
}

#[derive(Subcommand)]
enum Command {
	/// Render a model, place or asset to PNGs (through the warm render service unless --no-service)
	Render(RenderArgs),
	/// Render a model or an effect over a stretch of its timeline to a video (MP4 through ffmpeg),
	/// with a contact sheet of moments (through the warm render service unless --no-service)
	Video(VideoArgs),
	/// Run the render service in the foreground (`ror render` starts one automatically)
	Serve(ServeArgs),
	/// The render service of this state dir
	#[command(subcommand)]
	Service(ServiceCommand),
	/// The dedicated capture Studio (parity measurements)
	#[command(subcommand)]
	Studio(StudioCommand),
	/// Render a model in the capture Studio and in ror with the same camera, and compare
	Compare(CompareArgs),
	/// Disk use: identical files linked into one (nothing lost); with --older-than, old derived images removed
	#[command(long_about = "Disk use. Identical files become one file under all their names (hard links: every name reads the same bytes, nothing is lost), each kind among its own: out/'s PNG files, the state dir's downloads (assets/, with those of ROR_SHARED_STATE_DIRS), its Studio captures (captures/) and decoded material packs (material-packs/). Captures never link to out/. Names written in the last 10 minutes are left alone.\n\nror never writes into an existing file, only replaces it by a rename, which leaves a link's other names as they were. Copy over such a file only by replacing it (`cp --remove-destination`, delete first): writing into it changes every name.")]
	Clean(CleanArgs),
	/// What a render shows, as text: each view's parts by the share of the image they cover, or the
	/// part at a pixel (an output folder of `ror render --ids`)
	Inspect(InspectArgs),
	/// Two renders of the same views compared: the pixels that changed and where, and (both rendered
	/// with --ids) the parts that gained or lost pixels; writes <view>.diff.png
	Diff(DiffArgs),
	/// What a model or map costs Roblox to draw from each view: draw calls and triangles per render
	/// pass as Roblox's own counters report them, the batches behind them and what would cut them
	/// most; <view>.draws.png gives each draw call a colour, <view>.heat.png shows what costs a draw
	/// call of its own (through the warm render service unless --no-service)
	Cost(CostArgs),
}

#[derive(clap::Args)]
struct CostArgs {
	/// Model or place file (.rbxm, .rbxmx, .rbxl, .rbxlx), an asset id / rbxassetid:// URI, or with
	/// --studio an instance path in that Studio
	input: String,
	/// Export `input` (a dotted instance path) from the open Studio whose name contains this
	/// (read-only) and count it
	#[arg(long)]
	studio: Option<String>,
	/// Count only this instance (dotted path from the file's root, e.g. Workspace.Map)
	#[arg(long)]
	select: Option<String>,
	/// Output directory (default <state dir>/out/<input name>-cost)
	#[arg(long)]
	out: Option<PathBuf>,
	/// View size, WIDTHxHEIGHT: its shape decides what is in view (a player's window is about 16:9)
	#[arg(long, default_value = "1920x1080")]
	size: String,
	/// Views: front, back, left, right, top, bottom, iso, default, turntable:N (comma-separated)
	#[arg(long, default_value = "default")]
	views: String,
	/// Orbit camera yaw in degrees (overrides --views)
	#[arg(long)]
	yaw: Option<f32>,
	/// Orbit camera pitch in degrees (with --yaw)
	#[arg(long, default_value_t = 20.0)]
	pitch: f32,
	/// Vertical field of view in degrees (Roblox's default camera: 70)
	#[arg(long)]
	fov: Option<f32>,
	/// Camera JSON (see CameraSpec), e.g. '{"mode":"lookAt","eye":[0,5,-10],"target":[0,2,0]}'
	#[arg(long)]
	camera: Option<String>,
	/// Named cameras from a JSON file: [{"name", "eye": [x,y,z], "target": [x,y,z], "fov"}, ...] or
	/// {"name", "cframe": [x,y,z,R00..R22], "fov"} (a game camera's CFrame)
	#[arg(long, value_name = "FILE")]
	cameras: Option<PathBuf>,
	/// Views where the map is densest, as a player stands there: the N densest 64-stud cells, each
	/// looked at four ways (N, E, S, W) from eye height over its floor
	#[arg(long, value_name = "N")]
	sweep: Option<usize>,
	/// Views as a match plays: N points spread along the map's enemy paths (parts under its outermost
	/// folders named Path(s), Node(s) or Waypoint(s)), each seen from a near and a far gameplay camera,
	/// four ways
	#[arg(long, value_name = "N", conflicts_with_all = ["cameras", "sweep", "camera_from"])]
	paths: Option<usize>,
	/// The cost by where the camera stands: a far gameplay camera every STUDS studs (20 suits a match
	/// map) over the track's placement area, looking four ways; each cell painted by its views' draws,
	/// triangles and occlusion work over the map seen from above (grid.draws.png, grid.triangles.png,
	/// grid.occlusion.png, grid.json)
	#[arg(long, value_name = "STUDS", conflicts_with_all = ["cameras", "sweep", "paths", "camera_from", "check"])]
	grid: Option<f32>,
	/// The grid's area, X0,Z0,X1,Z1 in studs (default: the track's placement area, the box of the
	/// enemy path's nodes and 40 studs around it, where a match lets units stand; without a path,
	/// everything the scene draws), and with a fifth number the floor height every camera stands over
	/// (default: the nearest path node's height, else the highest large part under each cell)
	#[arg(long, value_name = "X0,Z0,X1,Z1[,Y]", requires = "grid", allow_hyphen_values = true)]
	region: Option<String>,
	/// Each view takes the camera of the view of its name in this earlier render or cost run; with
	/// no other view option, every view of that run as it was (a before / after pair)
	#[arg(long, value_name = "DIR", conflicts_with_all = ["camera", "cameras", "sweep"])]
	camera_from: Option<PathBuf>,
	/// Orbit views frame what is at or under this instance
	#[arg(long, value_name = "PATH")]
	focus: Option<String>,
	/// Take MaterialService (MaterialVariants) from this place or model file
	#[arg(long)]
	materials_from: Option<PathBuf>,
	/// Take Lighting (Technology, sky, clouds) from this place file
	#[arg(long)]
	lighting_from: Option<PathBuf>,
	/// Count for this Lighting.Technology instead of the scene's (transparent Glass depends on it)
	#[arg(long, value_parser = ["legacy", "voxel", "compatibility", "shadowmap", "future", "unified"])]
	technology: Option<String>,
	/// Graphics quality level 1-21 (default 21)
	#[arg(long)]
	quality: Option<u32>,
	/// Frustum culling only (no occlusion culling)
	#[arg(long)]
	no_occlusion: bool,
	/// Every mesh at full detail, as Studio draws by default
	#[arg(long)]
	full_detail: bool,
	/// Count at a graphics quality as a game draws (roblox or roblox:<1-21>, as render --lod): sets
	/// --quality, and with --check has the capture Studio count at the levels of detail too (its
	/// MeshPartDetailLevel DistanceBased and edit quality set while it measures), not at full detail
	#[arg(long, value_name = "roblox[:QUALITY]", conflicts_with_all = ["full_detail", "quality"])]
	lod: Option<String>,
	/// Numbers only: no draws.png and heat.png
	#[arg(long)]
	no_images: bool,
	/// Also write items.json: every part, decal and effect the scene draws (path, mesh, pass,
	/// triangles per level, box, CastShadow) and, per view, the ones it draws (their level and
	/// triangles) and the ones occlusion culling hides: what each copy costs, view by view
	#[arg(long)]
	items: bool,
	/// Every emitter in the view's `fullest` list, not the fullest few
	#[arg(long)]
	all_emitters: bool,
	/// Batches, occluders used, missed occluders and tested groups kept per view in summary.json and
	/// --json (cost.json keeps every one)
	#[arg(long)]
	top: Option<usize>,
	/// Also measure the views in the capture Studio (ror studio open) with Roblox's own counters and
	/// print them beside the model's (the views take the capture viewport's shape)
	#[arg(long)]
	check: bool,
	/// With --check: also read the render job's CPU time per view from Studio's MicroProfiler (LibMP):
	/// occlusion culling (queryOcclusion on the culling jobs), the render thread's wait for it and its
	/// own work, into cost-check.json
	#[arg(long, requires = "check")]
	profile: bool,
	/// Compare with an earlier `ror cost` run (its output folder, views matched by name): each pass's
	/// draws and triangles and the occlusion work per view, before and after (pair with --camera-from)
	#[arg(long, value_name = "DIR")]
	compare: Option<PathBuf>,
	/// No network access for assets
	#[arg(long)]
	offline: bool,
	/// Print the cost report as JSON
	#[arg(long)]
	json: bool,
	/// Count in this process instead of the warm render service
	#[arg(long)]
	no_service: bool,
	/// Count a moment of an effect's timeline: a playback as JSON, inline ('{...}') or a file
	#[arg(long)]
	playback: Option<String>,
	/// The moment drawn, seconds after the playback's start
	#[arg(long)]
	time: Option<f32>,
	/// Emit at a time: <path>[=<count>|=attributes|=immediate|=cutscene|=delayed][@<seconds>] (as render)
	#[arg(long)]
	emit: Vec<String>,
}

#[derive(clap::Args)]
struct InspectArgs {
	/// A render's output folder
	dir: PathBuf,
	/// Only this view (default every view)
	#[arg(long)]
	view: Option<String>,
	/// Part paths listed per view, most covered first
	#[arg(long, default_value_t = 12)]
	top: usize,
	/// The part at this pixel, X,Y from the top left (of --view, default the first view)
	#[arg(long, value_name = "X,Y")]
	at: Option<String>,
	/// JSON instead of text
	#[arg(long)]
	json: bool,
}

#[derive(clap::Args)]
struct DiffArgs {
	/// The render before (an output folder)
	before: PathBuf,
	/// The render after
	after: PathBuf,
	/// Where the diff images go (default <after>/diff)
	#[arg(long)]
	out: Option<PathBuf>,
	/// A channel differing by more than this (of 255) counts as changed
	#[arg(long, default_value_t = 8)]
	threshold: u8,
	/// Part paths listed per view, most changed first
	#[arg(long, default_value_t = 10)]
	top: usize,
	/// JSON instead of text
	#[arg(long)]
	json: bool,
}

#[derive(clap::Args)]
struct CleanArgs {
	/// Report what would be linked and removed, change nothing
	#[arg(long)]
	dry_run: bool,
	/// Also remove out/'s diff.png and side-by-side.png last written more than this many days ago
	/// (made from their run's studio.png and ror.png; tools/analysis/side_by_side.py draws them again)
	#[arg(long, value_name = "DAYS")]
	older_than: Option<f64>,
	/// Output folder (default: checkout's out/, or <state dir>/out for a standalone binary)
	#[arg(long)]
	out: Option<PathBuf>,
}

#[derive(clap::Args)]
struct CompareArgs {
	/// Model or place file, or an asset id
	input: PathBuf,
	#[arg(long)]
	select: Option<String>,
	/// Views (presets as for render)
	#[arg(long, default_value = "default")]
	views: String,
	/// Vertical field of view in degrees
	#[arg(long)]
	fov: Option<f32>,
	/// Camera JSON (see CameraSpec) instead of the views, e.g. '{"mode":"lookAt","eye":[0,5,-10],"target":[0,2,0]}'
	#[arg(long)]
	camera: Option<String>,
	/// The views of an earlier render or cost run (its output folder): every one, or those --views names
	#[arg(long, value_name = "DIR", conflicts_with = "camera")]
	camera_from: Option<PathBuf>,
	/// Output directory (default out/compare/<input name>)
	#[arg(long)]
	out: Option<PathBuf>,
	/// MSAA samples for ror's side
	#[arg(long, default_value_t = 4)]
	msaa: u32,
	#[arg(long)]
	offline: bool,
	/// Light both sides with this place's Lighting (properties, sky, atmosphere, post effects)
	/// instead of the input's own (or the default lighting)
	#[arg(long)]
	lighting_from: Option<PathBuf>,
	/// Both sides as a game draws them at a graphics quality: the capture Studio's MeshPartDetailLevel
	/// DistanceBased and edit quality set while it captures (put back after), ror's side with its
	/// levels of detail and culling (render --lod): roblox or roblox:<1-21>
	#[arg(long, value_name = "roblox[:QUALITY]|levels")]
	lod: Option<String>,
}

#[derive(Subcommand)]
enum StudioCommand {
	/// Write the capture place and open it in a new Studio
	Open {
		/// Use the pre-2022 materials (places saved without MaterialService.Use2022Materials)
		#[arg(long)]
		pre2022: bool,
		/// Close an open capture Studio first (its place is regenerated; nothing is saved)
		#[arg(long)]
		restart: bool,
		/// Lighting.Technology of the capture place (Unified by default): scripts cannot set it, so a
		/// measurement under another technology needs a place written with it
		#[arg(long, value_parser = ["legacy", "voxel", "compatibility", "shadowmap", "future", "unified"])]
		technology: Option<String>,
		/// Workspace.MeshStreamingAndImprovedLods of the capture place (left out by default; scripts
		/// cannot set it): a game's places may have it `disabled` (notes/cost.md "Levels of detail")
		#[arg(long, value_parser = ["default", "disabled", "enabled"])]
		mesh_lods: Option<String>,
	},
	/// Close the capture Studio (ends its process; its place is regenerated by `open`, nothing is saved)
	Close,
	/// Show the capture Studio's windows and viewport
	Status,
	/// Run a Luau file (or `-e code`) in the capture Studio and print its result
	Luau {
		file: Option<PathBuf>,
		#[arg(short = 'e', long)]
		code: Option<String>,
	},
	/// Run a calibration experiment (out/calibrate/<name>/samples.json)
	Calibrate { name: String },
	/// Export an instance from any open Studio as a .rbxm (read-only: SerializationService)
	Export {
		/// Part of the Studio's name, e.g. "My Place"
		studio: String,
		/// Dotted instance path, e.g. ReplicatedStorage.Models.Character
		path: String,
		#[arg(long)]
		out: PathBuf,
		/// Leave out the place's MaterialVariants (exported by default so the model's parts keep them)
		#[arg(long)]
		no_materials: bool,
	},
	/// Capture the capture Studio's viewport to a PNG
	Capture {
		/// Output PNG (default out/studio-capture.png, under the state dir for a standalone binary)
		#[arg(long)]
		out: Option<PathBuf>,
	},
}

#[derive(clap::Args)]
struct ServeArgs {
	/// State dir (default ROR_STATE_DIR or <checkout>/.cache)
	#[arg(long)]
	state_dir: Option<PathBuf>,
	/// Exit after this many seconds without jobs (a client starting one passes
	/// `ROR_SERVICE_IDLE`). 10 s after its last job it hands over to a dormant copy of itself, which
	/// holds tens of MB (the GPU's adapter, no device) and wakes for a job in ~0.15 s; a job after it
	/// exits starts a new service, ~0.5-0.9 s
	#[arg(long, default_value_t = 600)]
	idle: u64,
	/// Start dormant: the GPU's adapter found, no device made until a job comes (the copy a service
	/// hands over to as it goes quiet)
	#[arg(long, hide = true)]
	dormant: bool,
	/// No longer used: draws start within the GPU's free memory (ror::budget::GpuGate), as many at once
	/// as fit; still accepted from older scripts
	#[arg(long, hide = true)]
	gpu_slots: Option<usize>,
	/// Build id to report (set by the client that started this copy of the binary)
	#[arg(long, hide = true)]
	build: Option<String>,
	/// The executable of the client that started this copy of the binary
	#[arg(long, hide = true)]
	source: Option<String>,
	#[arg(long)]
	offline: bool,
}

#[derive(Subcommand)]
enum ServiceCommand {
	/// The running services (one per build) and their health
	Status,
	/// Ask every service of this state dir to exit once its jobs finish
	Stop,
	/// Start this build's service if it is not running (and print its port)
	Start {
		/// The offline service: a job sent with --offline only ever runs on one
		#[arg(long)]
		offline: bool,
	},
	/// Start this build's service if it is not running and bring the GPU up to its working clocks
	/// for a job about to come: one burst of at most 0.5 s, ended when a job starts. A GPU idle a
	/// few seconds runs a clip's first ~0.5 s at idle clocks; a caller that knows a clip is coming
	/// (before an export or a recording) runs this first, without waiting for it.
	Warm {
		/// The offline service: warm the one the coming job runs on
		#[arg(long)]
		offline: bool,
	},
}

#[derive(clap::Args)]
struct RenderArgs {
	/// Model or place file (.rbxm, .rbxmx, .rbxl, .rbxlx), an asset id / rbxassetid:// URI, a .luau
	/// script that builds into the Workspace (docs/scripts.md), or with --studio an instance path in
	/// that Studio
	input: String,
	/// Export `input` (a dotted instance path) from the open Studio whose name contains this
	/// (read-only) and render it
	#[arg(long)]
	studio: Option<String>,
	/// Draw only this instance (dotted path from the file's root, e.g. Workspace.Map.House)
	#[arg(long)]
	select: Option<String>,
	/// Output directory (default <state dir>/out/<input name>)
	#[arg(long)]
	out: Option<PathBuf>,
	/// Image size, WIDTHxHEIGHT
	#[arg(long, default_value = "1536x1536")]
	size: String,
	/// Views: front, back, left, right, top, bottom, iso, default, turntable:N (comma-separated)
	#[arg(long, default_value = "default")]
	views: String,
	/// Orbit camera yaw in degrees (overrides --views)
	#[arg(long)]
	yaw: Option<f32>,
	/// Orbit camera pitch in degrees (with --yaw)
	#[arg(long, default_value_t = 20.0)]
	pitch: f32,
	/// Vertical field of view in degrees
	#[arg(long)]
	fov: Option<f32>,
	/// Camera JSON (see CameraSpec), e.g. '{"mode":"lookAt","eye":[0,5,-10],"target":[0,2,0]}'; a JSON
	/// list of them draws a view each (camera1, camera2, ...) from one load
	#[arg(long)]
	camera: Option<String>,
	/// Orbit views frame this box instead of the model: minX,minY,minZ,maxX,maxY,maxZ, or as
	/// summary.json's stats.framed prints it ([[x,y,z],[x,y,z]]); one camera for a timeline's moments
	#[arg(long, value_name = "BOX", conflicts_with = "focus")]
	frame: Option<String>,
	/// Orbit views frame what is at or under this instance (a dotted path from the file's root, or its
	/// last names: Head, Armor.Torso), the rest drawn around it
	#[arg(long, value_name = "PATH")]
	focus: Option<String>,
	/// Each view takes the camera of the view of its name in this earlier render or cost run (its
	/// output folder): a before / after pair from one camera, whatever changed between them; --views
	/// may name that run's own views, and left at its default takes every one of them
	#[arg(long, value_name = "DIR", conflicts_with_all = ["frame", "focus", "camera"])]
	camera_from: Option<PathBuf>,
	/// sky, transparent, or a colour (#rrggbb or r,g,b)
	#[arg(long, default_value = "sky")]
	background: String,
	/// MSAA samples (1 or 4)
	#[arg(long, default_value_t = 4)]
	msaa: u32,
	/// Render at N times the size and average down
	#[arg(long, default_value_t = 1)]
	supersample: u32,
	/// No network access for assets
	#[arg(long)]
	offline: bool,
	/// Print the result JSON on one line
	#[arg(long)]
	json: bool,
	/// Render in this process instead of the warm render service
	#[arg(long)]
	no_service: bool,
	/// Skip PNG encoding (timing runs)
	#[arg(long)]
	no_png: bool,
	/// Disable Studio's built-in bloom
	#[arg(long)]
	no_bloom: bool,
	/// Also write which part is in front at every pixel: <view>.ids.png (the part's index + 1 in
	/// red, green and blue, low byte first; 0 for none) and parts.json (each part's path, class,
	/// material, transparency)
	#[arg(long)]
	ids: bool,
	/// With --ids: how much of this instance (dotted path from the file's root) each view shows: the
	/// view's `alone` in the result gives its parts' pixels in the view (`shown`) and drawn from the
	/// same camera with nothing else (`whole`)
	#[arg(long, value_name = "PATH", requires = "ids")]
	alone: Option<String>,
	/// Draw what Roblox draws at a graphics quality level: each MeshPart at the level of detail its
	/// distance and RenderFidelity give from each view's camera, and none of the parts the quality's
	/// view distance and size cull leave out: roblox (quality 21) or roblox:<1-21> (notes/cost.md
	/// "Levels of detail"); levels: the meshes' levels alone, nothing left out (an A/B against a
	/// render at full detail). Without it every mesh draws at full detail and every part is drawn
	#[arg(long, value_name = "roblox[:QUALITY]|levels")]
	lod: Option<String>,
	/// Draw what the parts collide with, not what they show: each MeshPart and union as its saved
	/// collision data (every hull its own colour; Box fidelity a warm grey box), Parts as their shape,
	/// invisible parts too. Which parts: collide (CanCollide, the default), query, touch or all
	#[arg(long, value_name = "FILTER", num_args = 0..=1, default_missing_value = "collide")]
	collision: Option<String>,
	/// Take MaterialService (material generation, MaterialVariants) from this place or model file
	#[arg(long)]
	materials_from: Option<PathBuf>,
	/// Take Lighting (properties, sky, atmosphere, effects) from this place file
	#[arg(long)]
	lighting_from: Option<PathBuf>,
	/// Draw the input in this place's world (a game map): its Workspace (a model file's top-level
	/// instances) drawn around the input and never framed, its Lighting and MaterialService used
	/// unless --lighting-from / --materials-from name others
	#[arg(long, value_name = "FILE")]
	map: Option<PathBuf>,
	/// The point of the map that lands on the input's origin, x,y,z, and a yaw in degrees turning the
	/// map about it (x,y,z,yaw)
	#[arg(long, value_name = "X,Y,Z[,YAW]", requires = "map")]
	map_origin: Option<String>,
	/// Draw a moment of an effect's timeline: a playback as JSON, inline ('{...}') or a file
	/// (docs/playback.md)
	#[arg(long)]
	playback: Option<String>,
	/// The moment drawn, seconds after the playback's start (time 0: the effect placed in the world)
	#[arg(long)]
	time: Option<f32>,
	/// Emit at a time: <path>[=<count>|=attributes|=immediate|=cutscene|=delayed][@<seconds>]
	/// (repeatable); the path is dotted from the file's root, empty for everything drawn; "attributes"
	/// (the default) plays each emitter by its EmitCount, EmitDelay and EmitDuration attributes (a
	/// burst EmitDelay after the call, enabled for EmitDuration), "immediate" the same without the
	/// delay, "cutscene" with cutscene timing (docs/playback.md)
	#[arg(long)]
	emit: Vec<String>,
	/// Enable emitters, trails, beams and lights at and under a path from a time on, whatever the
	/// playback switches, sets, tweens or parents (a post effect, Sky or Atmosphere only when the path
	/// names it): <path>[@<seconds>] (repeatable)
	#[arg(long)]
	enable: Vec<String>,
	/// Disable emitters, trails, beams and lights at and under a path from a time on (no more emits),
	/// whatever the playback switches, sets, tweens or parents (a post effect, Sky or Atmosphere only
	/// when the path names it): <path>[@<seconds>] (repeatable)
	#[arg(long)]
	disable: Vec<String>,
	#[command(flatten)]
	changes: ChangeArgs,
	#[command(flatten)]
	script: ScriptArgs,
}

/// A .luau input's run in the engine emulator (`render` and `video`; docs/scripts.md).
#[derive(clap::Args)]
struct ScriptArgs {
	/// A .luau input's Context fields, as JSON: '{"Color":"red"}' arrives as Context.Color
	#[arg(long, value_name = "JSON")]
	context: Option<String>,
	/// Mount a directory or model file at an instance path for a .luau input's requires:
	/// Service/Name=<dir|file> (repeatable; ree's --mount)
	#[arg(long, value_name = "SERVICE/NAME=PATH")]
	mount: Vec<String>,
}

impl ScriptArgs {
	fn options(&self, offline: bool) -> ror::script::Options<'_> {
		ror::script::Options { context: self.context.as_deref(), mounts: &self.mount, offline }
	}
}

/// Records a .luau input (`ror::script`) into `<out>/record`, for `seconds` of its run. Returns the
/// output directory (`--out`, else `<state dir>/out/<script name>[-video]`) and the recording.
fn record_script(input: &str, out: Option<PathBuf>, suffix: &str, seconds: f32, options: &ror::script::Options<'_>) -> Result<(PathBuf, ror::script::Recording)> {
	let script = std::path::Path::new(input);
	let name = script.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_else(|| "script".into());
	let out = match out {
		Some(out) => out,
		None => output::state_dir().join("out").join(format!("{name}{suffix}")),
	};
	let recording = ror::script::record(script, seconds, options, &out.join("record"))?;
	for error in &recording.errors {
		eprintln!("ror: script error: {error}");
	}
	Ok((out, recording))
}

/// Property sets, tweens and parenting over a playback (`render` and `video`).
#[derive(clap::Args)]
struct ChangeArgs {
	/// Set a property at a time: <path>.<Property>[+|-]=<value>[@<seconds>] (repeatable). Values:
	/// numbers, true/false, #rrggbb or r,g,b (0-1), JSON arrays ([0.3,0,0.3,0]: a UDim2), text,
	/// "initial" (as in the file), each as the property's type takes it (else an error); += / -= add
	/// to its value then. Lighting.<Property> is the Lighting (a model's: --lighting-from's, else ror's)
	#[arg(long, value_name = "SPEC")]
	set: Vec<String>,
	/// Tween a property as TweenService does, from its value at the start (repeatable):
	/// <path>.<Property>[+|-]=<value>[@<start>[+<duration>]][:<EasingStyle>][:<In|Out|InOut>]
	/// [:reverses][:repeat=<n>][:delay=<s>][:class=<Class>] (1 s, Quad, Out by default)
	#[arg(long, value_name = "SPEC")]
	tween: Vec<String>,
	/// Parent an instance at a time: <path>=<new parent path>[@<seconds>] (repeatable; "" or nil:
	/// out of the world), as an attack parents its ColorCorrection or Sky to Lighting
	#[arg(long, value_name = "SPEC")]
	parent: Vec<String>,
}

#[derive(clap::Args)]
struct VideoArgs {
	/// Model or place file (.rbxm, .rbxmx, .rbxl, .rbxlx), an asset id / rbxassetid:// URI, or a
	/// .luau script that builds into the Workspace (docs/scripts.md)
	input: String,
	/// Draw only this instance (dotted path from the file's root, e.g. Workspace.Map.House)
	#[arg(long)]
	select: Option<String>,
	/// Output directory (default <state dir>/out/<input name>-video)
	#[arg(long)]
	out: Option<PathBuf>,
	/// Seconds of the effect's timeline drawn, from --start
	#[arg(long, default_value_t = 2.0)]
	duration: f32,
	/// Frames per second of video
	#[arg(long, default_value_t = 30.0)]
	fps: f32,
	/// The first frame's moment, seconds after the playback's start (0: the effect placed in the world)
	#[arg(long, default_value_t = 0.0)]
	start: f32,
	/// The speed the video plays the effect at: 0.25 shows --duration four times slower
	#[arg(long, default_value_t = 1.0)]
	speed: f32,
	/// Draw exactly these moments, one frame each, in place of --start/--duration: seconds after the
	/// playback's start, comma-separated, or a JSON file holding an array of them (stills at their own times)
	#[arg(long)]
	times: Option<String>,
	/// Image size, WIDTHxHEIGHT (even sizes for mp4 and webm)
	#[arg(long, default_value = "1920x1080")]
	size: String,
	/// One view: front, back, left, right, top, bottom, iso, default
	#[arg(long, default_value = "default")]
	view: String,
	/// Orbit camera yaw in degrees (overrides --view)
	#[arg(long)]
	yaw: Option<f32>,
	/// Orbit camera pitch in degrees (with --yaw)
	#[arg(long, default_value_t = 20.0)]
	pitch: f32,
	/// Vertical field of view in degrees
	#[arg(long)]
	fov: Option<f32>,
	/// Camera JSON (see CameraSpec), e.g. '{"mode":"lookAt","eye":[0,5,-10],"target":[0,2,0]}'
	#[arg(long)]
	camera: Option<String>,
	/// The orbit camera turns this many degrees round what it frames over the clip (360: once round)
	#[arg(long, default_value_t = 0.0)]
	orbit: f32,
	/// The orbit frames this box instead of what the clip draws: minX,minY,minZ,maxX,maxY,maxZ
	#[arg(long, value_name = "BOX", conflicts_with = "focus")]
	frame: Option<String>,
	/// The orbit frames what is at or under this instance over the clip (a dotted path, or its last names)
	#[arg(long, value_name = "PATH")]
	focus: Option<String>,
	/// The camera of an earlier video or render (its folder), for a before / after pair: a video's
	/// framed box, which the same --view and --orbit frame alike, else the render's view of this name
	#[arg(long, value_name = "DIR", conflicts_with_all = ["frame", "focus", "camera"])]
	camera_from: Option<PathBuf>,
	/// A moving camera in place of the view: a JSON file of keys [{"at": s, "cframe": [12 numbers],
	/// "fov": deg}], each held from its moment until the next (a recording's camera.json)
	#[arg(long, value_name = "FILE", conflicts_with_all = ["camera", "yaw", "camera_from", "frame", "focus"])]
	camera_track: Option<PathBuf>,
	/// sky, transparent, or a colour (#rrggbb or r,g,b)
	#[arg(long, default_value = "sky")]
	background: String,
	/// mp4 (H.264), webm (VP9, keeps a transparent background's alpha), gif, apng (no ffmpeg needed),
	/// png (one PNG per frame). Default mp4, webm over a transparent background; apng without ffmpeg
	#[arg(long)]
	format: Option<String>,
	/// Moments on the contact sheet, sheet.png (0: none)
	#[arg(long, default_value_t = 12)]
	sheet: usize,
	/// Through the service: return once sheet.png is written (its moments drawn first), the clip and
	/// summary.json following in the background while clip.pending stands in --out
	#[arg(long)]
	sheet_first: bool,
	/// Another file made from the clip's frames as they are drawn (an mp4 clip's): '{"file": "copy.mp4",
	/// "args": [ffmpeg's output options, e.g. "-c:v", "h264_nvenc", "-b:v", "8M"]}', the container
	/// named by the extension; repeat for more. The summary's copies say which were written
	#[arg(long, value_name = "JSON")]
	copy: Vec<String>,
	/// Encode the mp4 on the GPU's hardware encoder (NVIDIA's NVENC, in the process) instead of x264:
	/// constant quality QUALITY (1-51, lower is better), spending at most KBPS kbit/s, so a clip
	/// of a known length fits a size. The frames cost the CPU no encoding; the clip fails where the
	/// machine has no such encoder
	#[arg(long, value_name = "QUALITY[:KBPS]")]
	nvenc: Option<String>,
	/// Draw what Roblox draws at a graphics quality level: each MeshPart at the level of detail its
	/// distance and RenderFidelity give from the camera in each frame, and none of the parts the quality's
	/// view distance and size cull leave out: roblox (quality 21) or roblox:<1-21> (notes/cost.md
	/// "Levels of detail"); levels: the meshes' levels alone, nothing left out (an A/B against a
	/// render at full detail). Without it every mesh draws at full detail and every part is drawn
	#[arg(long, value_name = "roblox[:QUALITY]|levels")]
	lod: Option<String>,
	/// MSAA samples (1 or 4)
	#[arg(long, default_value_t = 4)]
	msaa: u32,
	/// Disable Studio's built-in bloom
	#[arg(long)]
	no_bloom: bool,
	/// No network access for assets
	#[arg(long)]
	offline: bool,
	/// Print the result JSON on one line
	#[arg(long)]
	json: bool,
	/// Render in this process instead of the warm render service
	#[arg(long)]
	no_service: bool,
	/// Take MaterialService (material generation, MaterialVariants) from this place or model file
	#[arg(long)]
	materials_from: Option<PathBuf>,
	/// Take Lighting (properties, sky, atmosphere, effects) from this place file
	#[arg(long)]
	lighting_from: Option<PathBuf>,
	/// Draw the input in this place's world (a game map), as `render --map`: never framed, its Lighting
	/// and MaterialService used unless --lighting-from / --materials-from name others
	#[arg(long, value_name = "FILE")]
	map: Option<PathBuf>,
	/// The point of the map that lands on the input's origin: x,y,z or x,y,z,yaw
	#[arg(long, value_name = "X,Y,Z[,YAW]", requires = "map")]
	map_origin: Option<String>,
	/// The effect's timeline: a playback as JSON, inline ('{...}') or a file (docs/playback.md)
	#[arg(long)]
	playback: Option<String>,
	/// Emit at a time: <path>[=<count>|=attributes|=immediate|=cutscene|=delayed][@<seconds>]
	/// (repeatable), as `render --emit`. Without --emit, --enable, --disable or --playback every
	/// emitter plays by its EmitCount, EmitDelay and EmitDuration attributes at 0 (--emit =attributes;
	/// --emit =cutscene with cutscene timing)
	#[arg(long)]
	emit: Vec<String>,
	/// Enable emitters, trails, beams and lights at and under a path from a time on, whatever the
	/// playback switches, sets, tweens or parents (a post effect, Sky or Atmosphere only when the path
	/// names it): <path>[@<seconds>] (repeatable)
	#[arg(long)]
	enable: Vec<String>,
	/// Disable emitters, trails, beams and lights at and under a path from a time on (no more emits),
	/// whatever the playback switches, sets, tweens or parents (a post effect, Sky or Atmosphere only
	/// when the path names it): <path>[@<seconds>] (repeatable)
	#[arg(long)]
	disable: Vec<String>,
	/// No emission unless asked: what is enabled in the file runs, nothing is emitted
	#[arg(long)]
	no_emit: bool,
	#[command(flatten)]
	changes: ChangeArgs,
	#[command(flatten)]
	script: ScriptArgs,
}

/// `--frame`'s box: six numbers, or two [x, y, z] as summary.json prints them.
fn parse_box(text: &str) -> Result<[[f32; 3]; 2]> {
	let numbers: Vec<f32> = text.split(|c: char| c == ',' || c == '[' || c == ']' || c.is_whitespace()).filter(|word| !word.is_empty()).map(|word| word.parse::<f32>().with_context(|| format!("--frame: {word:?} is not a number"))).collect::<Result<_>>()?;
	let [x0, y0, z0, x1, y1, z1] = numbers[..] else {
		bail!("--frame takes six numbers, minX,minY,minZ,maxX,maxY,maxZ (got {})", numbers.len());
	};
	Ok([[x0.min(x1), y0.min(y1), z0.min(z1)], [x0.max(x1), y0.max(y1), z0.max(z1)]])
}

/// The cameras an earlier render's views used, by view name (its summary.json).
fn earlier_cameras(dir: &std::path::Path) -> Result<std::collections::HashMap<String, CameraSpec>> {
	Ok(earlier_views(dir)?.into_iter().collect())
}

/// An earlier run's views in their order: names and cameras (its summary.json).
fn earlier_views(dir: &std::path::Path) -> Result<Vec<(String, CameraSpec)>> {
	let file = dir.join("summary.json");
	let summary: serde_json::Value = serde_json::from_slice(&std::fs::read(&file).with_context(|| format!("--camera-from: reading {}", file.display()))?)?;
	let mut cameras = Vec::new();
	for view in summary["views"].as_array().into_iter().flatten() {
		let (Some(name), Some(components), Some(fov)) = (view["name"].as_str(), view["camera"].as_array(), view["fov"].as_f64()) else {
			continue;
		};
		let components: Vec<f32> = components.iter().filter_map(|value| value.as_f64()).map(|value| value as f32).collect();
		let Ok(components) = <[f32; 12]>::try_from(components) else {
			continue;
		};
		cameras.push((name.to_string(), CameraSpec::CFrame { components, fov: fov as f32, near: view["near"].as_f64().map(|near| near as f32) }));
	}
	if cameras.is_empty() {
		bail!("--camera-from: {} lists no views", file.display());
	}
	Ok(cameras)
}

/// The box an earlier video's orbit framed (its summary.json), if it was one.
fn earlier_framed(dir: &std::path::Path) -> Result<Option<[[f32; 3]; 2]>> {
	let file = dir.join("summary.json");
	let summary: serde_json::Value = serde_json::from_slice(&std::fs::read(&file).with_context(|| format!("--camera-from: reading {}", file.display()))?)?;
	match summary.get("framed") {
		Some(framed) if framed.is_array() => Ok(Some(parse_box(&framed.to_string()).with_context(|| format!("--camera-from: {}'s framed box", file.display()))?)),
		_ => Ok(None),
	}
}

/// The timeline options `render` and `video` share.
struct TimelineArgs<'a> {
	playback: Option<&'a str>,
	time: Option<f32>,
	emit: &'a [String],
	enable: &'a [String],
	disable: &'a [String],
	changes: Option<&'a ChangeArgs>,
}

/// The playback of `--playback`, `--time`, `--emit`, `--enable`, `--disable`, `--set`, `--tween`
/// and `--parent`, if any is given.
fn playback_args(args: &TimelineArgs) -> Result<Option<ror::scene::Playback>> {
	use ror::scene::playback::{Action, Emit, spec};
	let changed = args.changes.is_some_and(|changes| !changes.set.is_empty() || !changes.tween.is_empty() || !changes.parent.is_empty());
	// the command's own setting goes in its job: a running service's environment is not the caller's
	let salt = std::env::var("ROR_PARTICLE_SALT").unwrap_or_default();
	if args.playback.is_none() && args.time.is_none() && args.emit.is_empty() && args.enable.is_empty() && args.disable.is_empty() && !changed && salt.is_empty() {
		return Ok(None);
	}
	let mut playback: ror::scene::Playback = match args.playback {
		Some(text) if text.trim_start().starts_with('{') => serde_json::from_str(text).context("--playback JSON")?,
		Some(path) => serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?).with_context(|| format!("--playback {path}"))?,
		None => ror::scene::Playback::default(),
	};
	if let Some(time) = args.time {
		playback.time = time;
	}
	if !salt.is_empty() {
		playback.salt = salt;
	}
	// <path>[=value][@seconds]; a quoted segment (["a=b@c"]) may hold either sign
	let split = |spec: &str| -> Result<(String, Option<String>, f32)> {
		let mut quote: Option<char> = None;
		let mut end = spec.len();
		for (index, c) in spec.char_indices() {
			match quote {
				Some(open) if c == open => quote = None,
				Some(_) => {}
				None if (c == '"' || c == '\'') && spec[..index].ends_with('[') => quote = Some(c),
				None if c == '=' || c == '@' => {
					end = index;
					break;
				}
				None => {}
			}
		}
		let (path, tail) = spec.split_at(end);
		let (value, at) = match tail.strip_prefix('=') {
			Some(rest) => match rest.rsplit_once('@') {
				Some((value, at)) => (Some(value.trim().to_string()), Some(at)),
				None => (Some(rest.trim().to_string()), None),
			},
			None => (None, tail.strip_prefix('@')),
		};
		let at = match at {
			Some(at) => at.trim().parse::<f32>().with_context(|| format!("{spec}: the time after @"))?,
			None => 0.0,
		};
		Ok((path.trim().to_string(), value, at))
	};
	for spec in args.emit {
		let (target, value, at) = split(spec)?;
		let emit = match value.as_deref() {
			None => Emit::Mode("attributes".into()),
			Some(mode) if ror::scene::playback::ByAttributes::parse(mode).is_ok() => Emit::Mode(mode.into()),
			Some(count) => Emit::Count(count.parse().with_context(|| format!("--emit {spec}: a count, attributes, immediate, cutscene or delayed"))?),
		};
		playback.actions.push(Action { at, target, emit: Some(emit), ..Default::default() });
	}
	for (specs, enabled) in [(args.enable, true), (args.disable, false)] {
		for spec in specs {
			let (target, _, at) = split(spec)?;
			// the command line's switches win over the playback's (bisecting a recording)
			playback.actions.push(Action { at, target, enabled: Some(enabled), force: true, ..Default::default() });
		}
	}
	if let Some(changes) = args.changes {
		for text in &changes.set {
			playback.actions.push(spec::set(text).with_context(|| format!("--set {text}"))?);
		}
		for text in &changes.tween {
			playback.tweens.push(spec::tween(text).with_context(|| format!("--tween {text}"))?);
		}
		for text in &changes.parent {
			playback.actions.push(spec::parent(text).with_context(|| format!("--parent {text}"))?);
		}
	}
	Ok(Some(playback))
}

fn parse_size(size: &str) -> Result<(u32, u32)> {
	let (width, height) = size.split_once(['x', 'X']).context("--size WIDTHxHEIGHT")?;
	let width: u32 = width.trim().parse().context("--size width")?;
	let height: u32 = height.trim().parse().context("--size height")?;
	if width == 0 || height == 0 || width > 16384 || height > 16384 {
		bail!("--size must be between 1x1 and 16384x16384");
	}
	Ok((width, height))
}

fn parse_background(value: &str) -> Result<Background> {
	match value {
		"sky" => Ok(Background::Sky),
		"transparent" | "none" => Ok(Background::Transparent),
		other => {
			let channels: Vec<f32> = if let Some(hex) = other.strip_prefix('#') {
				if hex.len() != 6 {
					bail!("--background #rrggbb");
				}
				(0..3).map(|index| u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).map(|v| v as f32 / 255.0)).collect::<Result<_, _>>()?
			} else {
				other.split(',').map(|part| part.trim().parse::<f32>().map(|v| v / 255.0)).collect::<Result<_, _>>().context("--background r,g,b")?
			};
			if channels.len() != 3 {
				bail!("--background needs three channels");
			}
			Ok(Background::Color([channels[0], channels[1], channels[2]]))
		}
	}
}

fn assets(offline: bool) -> Result<ror_assets::Assets> {
	Ok(ror_assets::Assets::new(output::asset_config(output::state_dir(), offline)))
}

/// An absolute path string (the render service may run in another working directory).
fn absolute(path: &std::path::Path) -> String {
	std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()).display().to_string()
}

/// `--map` and `--map-origin` as a job's map.
fn map_setting(map: Option<&std::path::Path>, origin: Option<&str>) -> Result<Option<job::MapSetting>> {
	let Some(map) = map else {
		return Ok(None);
	};
	if !map.is_file() {
		bail!("--map {}: no such file", map.display());
	}
	Ok(Some(job::MapSetting { file: absolute(map), origin: origin.map(job::MapSetting::parse_origin).transpose()? }))
}

fn render(mut args: RenderArgs) -> Result<()> {
	if let Some(studio) = &args.studio {
		let bytes = ror::studio::export::export(studio, &args.input, true)?;
		let name: String = args.input.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
		let path = output::state_dir().join("exports").join(format!("{name}.rbxm"));
		output::write_atomic(&path, &bytes)?;
		if args.out.is_none() {
			args.out = Some(output::state_dir().join("out").join(&name));
		}
		args.input = path.display().to_string();
	}
	// a script: its recording drawn at --time (default 0), one frame past it recorded
	let mut script = None;
	if ror::script::is_script(&args.input) {
		if args.playback.is_some() {
			bail!("--playback: a .luau input's playback is its own recording");
		}
		let time = args.time.unwrap_or(0.0);
		if !time.is_finite() || time < 0.0 {
			bail!("--time: a script's time must be finite and non-negative");
		}
		let (out, recording) = record_script(&args.input, args.out.take(), "", time + 1.0 / 60.0, &args.script.options(args.offline))?;
		args.input = recording.scene.display().to_string();
		args.playback = Some(recording.playback.display().to_string());
		args.time = Some(time);
		args.out = Some(out);
		script = Some(recording.report);
	}
	let (width, height) = parse_size(&args.size)?;
	let background = parse_background(&args.background)?;
	let mut views = if let Some(camera) = &args.camera {
		// a list of cameras: a view each, camera1 on
		let named = |name: String, camera: CameraSpec| job::NamedView { name, view: ror::render::ViewRequest { width, height, camera, background, ..Default::default() } };
		if camera.trim_start().starts_with('[') {
			let cameras: Vec<CameraSpec> = serde_json::from_str(camera).context("--camera JSON")?;
			cameras.into_iter().enumerate().map(|(index, camera)| named(format!("camera{}", index + 1), camera)).collect()
		} else {
			vec![named("render".into(), serde_json::from_str(camera).context("--camera JSON")?)]
		}
	} else if let Some(yaw) = args.yaw {
		let camera = CameraSpec::Orbit { yaw, pitch: args.pitch, distance: None, padding: 0.08, target: None, fov: args.fov.unwrap_or(70.0) };
		vec![job::NamedView { name: "render".into(), view: ror::render::ViewRequest { width, height, camera, background, ..Default::default() } }]
	} else if let Some(dir) = &args.camera_from {
		// an earlier run's own views by name (a cost run's gameplay views), else the presets; with
		// --views left at its default, every view of that run
		let cameras = earlier_views(dir)?;
		let names: Vec<String> = if args.views == "default" && !cameras.iter().any(|(name, _)| name == "default") {
			cameras.iter().map(|(name, _)| name.clone()).collect()
		} else {
			args.views.split(',').map(str::trim).filter(|name| !name.is_empty()).map(str::to_string).collect()
		};
		let mut views = Vec::new();
		for name in names {
			match cameras.iter().find(|(earlier, _)| *earlier == name) {
				Some((_, camera)) => views.push(job::NamedView { name, view: ror::render::ViewRequest { width, height, camera: camera.clone(), background, ..Default::default() } }),
				None => views.extend(job::preset_views(&name, width, height, background, args.fov)?),
			}
		}
		views
	} else {
		job::preset_views(&args.views, width, height, background, args.fov)?
	};
	if let Some(dir) = &args.camera_from {
		let cameras = earlier_cameras(dir)?;
		for view in &mut views {
			let names = || cameras.keys().cloned().collect::<Vec<_>>().join(", ");
			view.view.camera = cameras.get(&view.name).cloned().with_context(|| format!("--camera-from {}: it has no view {:?} (it has {})", dir.display(), view.name, names()))?;
		}
	}
	let lod = args.lod.as_deref().map(ror::lod::Lod::parse).transpose()?;
	for view in &mut views {
		view.view.msaa = args.msaa;
		view.view.supersample = args.supersample;
		view.view.bloom = !args.no_bloom;
		view.view.ids = args.ids;
		view.view.lod = lod;
	}
	let playback = playback_args(&TimelineArgs { playback: args.playback.as_deref(), time: args.time, emit: &args.emit, enable: &args.enable, disable: &args.disable, changes: Some(&args.changes) })?;
	let job = Job {
		input: args.input,
		select: args.select,
		out: args.out,
		views,
		no_png: args.no_png,
		compact_png: false,
		materials_from: args.materials_from.map(|path| absolute(&path)),
		lighting_from: args.lighting_from.map(|path| absolute(&path)),
		playback,
		frame: args.frame.as_deref().map(parse_box).transpose()?,
		focus: args.focus.clone(),
		collision: args
			.collision
			.as_deref()
			.map(|text| ror::scene::collision::require().and_then(|()| ror::scene::collision::Filter::parse(text).with_context(|| format!("--collision takes collide, query, touch or all, not {text:?}"))))
			.transpose()?,
		map: map_setting(args.map.as_deref(), args.map_origin.as_deref())?,
		cost: None,
		relative_to: std::env::current_dir().ok(),
		alone: args.alone.filter(|_| args.ids),
	};
	let mut result = if args.no_service || std::env::var_os("ROR_NO_SERVICE").is_some() {
		ror::memory_checkpoint("start");
		let gpu = Arc::new(Gpu::new()?);
		ror::memory_checkpoint("GPU device");
		let renderer = Renderer::new(gpu, assets(args.offline)?)?;
		ror::memory_checkpoint("renderer");
		let value = serde_json::to_value(job::run(&renderer, &SceneCache::default(), &job)?)?;
		ror::memory_checkpoint("job done");
		report_held(&renderer);
		value
	} else {
		ror::service::client::render(&output::state_dir(), &job, args.offline)?
	};
	if let Some(missing) = result.pointer("/stats/missing").and_then(|value| value.as_array()) {
		for missing in missing {
			eprintln!("ror: missing: {}", missing.as_str().unwrap_or_default());
		}
	}
	// the authoring lints (crate::lint)
	if let Some(line) = result.pointer("/weightless/line").and_then(|value| value.as_str()) {
		eprintln!("ror: {line}");
	}
	if let Some(script) = script {
		result["script"] = script;
	}
	output::print_json(&result, !args.json)?;
	Ok(())
}

/// `--camera-track`'s keys, in time order.
fn read_camera_track(path: &std::path::Path) -> Result<Vec<ror::video::CameraKey>> {
	let text = std::fs::read_to_string(path).with_context(|| format!("--camera-track {}", path.display()))?;
	let mut keys: Vec<ror::video::CameraKey> = serde_json::from_str(&text).with_context(|| format!("--camera-track {}: a list of {{\"at\", \"cframe\", \"fov\"}} keys", path.display()))?;
	if keys.is_empty() {
		bail!("--camera-track {}: no keys", path.display());
	}
	keys.sort_by(|a, b| a.at.total_cmp(&b.at));
	Ok(keys)
}

fn read_times(text: &str) -> Result<Vec<f32>> {
	let path = std::path::Path::new(text);
	if path.is_file() {
		let file = std::fs::read_to_string(path).with_context(|| format!("--times {text}"))?;
		return serde_json::from_str(&file).with_context(|| format!("--times {text}: a JSON array of seconds"));
	}
	text.split(',').map(|part| part.trim().parse::<f32>().with_context(|| format!("--times: {part:?} is not a number of seconds"))).collect()
}

fn video(mut args: VideoArgs) -> Result<()> {
	use ror::video::{VideoFormat, VideoJob};
	// a script: recorded to the clip's last moment and one frame past it, its recording the playback
	// (so nothing else is emitted)
	let mut script = None;
	if ror::script::is_script(&args.input) {
		if args.playback.is_some() {
			bail!("--playback: a .luau input's playback is its own recording");
		}
		let end = match args.times.as_deref() {
			Some(times) => {
				let times = read_times(times)?;
				if times.is_empty() || times.iter().any(|time| !time.is_finite() || *time < 0.0) {
					bail!("--times: a script's times must be finite and non-negative");
				}
				times.into_iter().fold(0.0, f32::max)
			}
			None => {
				if !args.start.is_finite() || args.start < 0.0 || !args.duration.is_finite() || args.duration <= 0.0 {
					bail!("a script's --start must be finite and non-negative and --duration must be finite and positive");
				}
				args.start + args.duration
			}
		};
		let (out, recording) = record_script(&args.input, args.out.take(), "-video", end + 1.0 / 60.0, &args.script.options(args.offline))?;
		args.input = recording.scene.display().to_string();
		args.playback = Some(recording.playback.display().to_string());
		args.out = Some(out);
		script = Some(recording.report);
	}
	let (width, height) = parse_size(&args.size)?;
	let background = parse_background(&args.background)?;
	// one view: a camera given, an orbit at a yaw, or a preset (named as `render --views` takes it, so
	// `render --camera-from <out> --views <name>` draws a still from the video's camera)
	let (name, mut camera) = if let Some(camera) = &args.camera {
		("default".to_string(), serde_json::from_str::<CameraSpec>(camera).context("--camera JSON")?)
	} else if let Some(yaw) = args.yaw {
		("default".to_string(), CameraSpec::Orbit { yaw, pitch: args.pitch, distance: None, padding: 0.08, target: None, fov: args.fov.unwrap_or(70.0) })
	} else {
		if args.view.contains(',') || args.view.starts_with("turntable") {
			bail!("--view takes one view (front, back, left, right, top, bottom, iso, default); --orbit 360 turns round the model");
		}
		let preset = job::preset_views(&args.view, width, height, background, args.fov)?.into_iter().next().context("--view")?;
		(preset.name, preset.view.camera)
	};
	let mut frame = args.frame.as_deref().map(parse_box).transpose()?;
	if let Some(dir) = &args.camera_from {
		match earlier_framed(dir)? {
			Some(framed) if matches!(camera, CameraSpec::Orbit { .. }) => frame = Some(framed),
			_ => {
				let mut cameras = earlier_cameras(dir)?;
				let names = cameras.keys().cloned().collect::<Vec<_>>().join(", ");
				camera = cameras.remove(&name).with_context(|| format!("--camera-from {}: it has no view {name:?} (it has {names})", dir.display()))?;
			}
		}
	}
	if args.orbit != 0.0 && !matches!(camera, CameraSpec::Orbit { .. }) {
		bail!("--orbit turns an orbit camera (--view, --yaw); this one is fixed (--camera, or --camera-from a render)");
	}
	let lod = args.lod.as_deref().map(ror::lod::Lod::parse).transpose()?;
	let view = ror::render::ViewRequest { width, height, camera, background, msaa: args.msaa, bloom: !args.no_bloom, lod, ..Default::default() };
	let mut playback = playback_args(&TimelineArgs { playback: args.playback.as_deref(), time: None, emit: &args.emit, enable: &args.enable, disable: &args.disable, changes: Some(&args.changes) })?;
	// an effect plays by its emitters' attributes at 0 (`--emit =attributes`),
	// unless told what it emits (its Lighting's changes or a property's say nothing of that)
	let emissions = args.playback.is_some() || !args.emit.is_empty() || !args.enable.is_empty() || !args.disable.is_empty();
	if !emissions && !args.no_emit {
		let effect = playback_args(&TimelineArgs { playback: None, time: None, emit: &["=attributes".to_string()], enable: &[], disable: &[], changes: None })?.unwrap_or_default();
		match &mut playback {
			Some(playback) => playback.actions.extend(effect.actions),
			None => playback = Some(effect),
		}
	}
	let format = match &args.format {
		Some(format) => VideoFormat::parse(format)?,
		None if background == Background::Transparent => VideoFormat::Webm,
		None => VideoFormat::Mp4,
	};
	let job = VideoJob {
		input: args.input,
		select: args.select,
		out: args.out,
		view,
		name,
		playback,
		playback_file: args.playback.as_deref().filter(|text| !text.trim_start().starts_with('{')).map(|path| absolute(std::path::Path::new(path)).into()),
		start: args.start,
		duration: args.duration,
		fps: args.fps,
		speed: args.speed,
		times: args.times.as_deref().map(read_times).transpose()?,
		orbit: args.orbit,
		format,
		sheet: args.sheet,
		materials_from: args.materials_from.map(|path| absolute(&path)),
		lighting_from: args.lighting_from.map(|path| absolute(&path)),
		map: map_setting(args.map.as_deref(), args.map_origin.as_deref())?,
		frame,
		focus: args.focus,
		ffmpeg: ror::video::find_ffmpeg(),
		camera_track: args.camera_track.as_deref().map(read_camera_track).transpose()?,
		sheet_first: args.sheet_first,
		copies: args
			.copy
			.iter()
			.map(|text| {
				let copy: ror::video::VideoCopy = serde_json::from_str(text).with_context(|| format!("--copy {text}: {{\"file\": ..., \"args\": [...]}}"))?;
				Ok(ror::video::VideoCopy { file: absolute(&copy.file).into(), ..copy })
			})
			.collect::<Result<_>>()?,
		relative_to: std::env::current_dir().ok(),
		nvenc: args.nvenc.as_deref().map(ror::nvenc::Rate::parse).transpose()?,
	};
	let mut result = if args.no_service || std::env::var_os("ROR_NO_SERVICE").is_some() {
		let gpu = Arc::new(Gpu::new()?);
		let renderer = Renderer::new(gpu, assets(args.offline)?)?;
		let value = serde_json::to_value(ror::video::run(&renderer, &SceneCache::default(), &job, None)?)?;
		report_held(&renderer);
		value
	} else {
		ror::service::client::video(&output::state_dir(), &job, args.offline)?
	};
	if let Some(missing) = result.pointer("/stats/missing").and_then(|value| value.as_array()) {
		for missing in missing {
			eprintln!("ror: missing: {}", missing.as_str().unwrap_or_default());
		}
	}
	// the authoring lints (crate::lint), the frames' brightness peaks and the effects' counts
	if let Some(line) = result.pointer("/weightless/line").and_then(|value| value.as_str()) {
		eprintln!("ror: {line}");
	}
	for line in result.pointer("/lines").and_then(|value| value.as_array()).into_iter().flatten() {
		eprintln!("ror: {}", line.as_str().unwrap_or_default());
	}
	if let Some(script) = script {
		result["script"] = script;
	}
	if args.json {
		output::print_json(&result, false)?;
	} else {
		// a line per frame is for summary.json and --json, not a reader
		let mut shown = result;
		if let Some(frames) = shown.as_object_mut().and_then(|summary| summary.remove("frameStats")) {
			shown["frameStats"] = format!("{} frames, in summary.json", frames.as_array().map_or(0, Vec::len)).into();
		}
		output::print_json(&shown, true)?;
	}
	Ok(())
}

#[cfg(windows)]
use ror::studio::capture::CAPTURE_WINDOW;

fn studio(command: StudioCommand) -> Result<()> {
	use ror::studio::{CaptureStudio, place};
	match command {
		StudioCommand::Open { pre2022, restart, technology, mesh_lods } => {
			let dir = output::state_dir().join("studio");
			std::fs::create_dir_all(&dir)?;
			let path = dir.join(place::place_file_name());
			if let Ok(mut studio) = CaptureStudio::connect() {
				if !restart {
					println!("the capture Studio is already open ({}); --restart reopens it", studio.name);
					let _ = studio.viewport();
					return Ok(());
				}
				#[cfg(windows)]
				{
					let window = studio.window()?;
					let pid = ror::studio::capture::window_process(window.hwnd);
					println!("closing the capture Studio (process {pid})");
					ror::studio::capture::terminate(pid)?;
				}
			}
			let mut environment = ror::scene::Environment::default();
			if let Some(name) = technology {
				environment.technology = ["legacy", "voxel", "compatibility", "shadowmap", "future", "unified"].iter().position(|item| *item == name).unwrap_or(5) as u32;
			}
			let mesh_lods = mesh_lods.map(|name| ["default", "disabled", "enabled"].iter().position(|item| *item == name).unwrap_or(0) as u32);
			output::write_atomic(&path, &place::build_with(&environment, !pre2022, mesh_lods)?)?;
			let exe = ror::studio::studio_executable().context("Roblox Studio is not installed")?;
			// Studio must not inherit this process's pipes (a caller reading them would wait for Studio)
			std::process::Command::new(exe)
				.arg(&path)
				.stdin(std::process::Stdio::null())
				.stdout(std::process::Stdio::null())
				.stderr(std::process::Stdio::null())
				.spawn()
				.context("starting Studio")?;
			println!("opening {} ({} materials)", path.display(), if pre2022 { "pre-2022" } else { "2022" });
			for _ in 0..90 {
				std::thread::sleep(std::time::Duration::from_secs(2));
				if let Ok(mut studio) = CaptureStudio::connect() {
					#[cfg(windows)]
					if let Ok(window) = studio.window() {
						// whole logical pixels for the viewport at 150 % scaling (1644 x 1116)
						ror::studio::capture::resize(window.hwnd, CAPTURE_WINDOW.0, CAPTURE_WINDOW.1)?;
					}
					std::thread::sleep(std::time::Duration::from_secs(1));
					let viewport = studio.viewport().unwrap_or_default();
					println!("capture Studio ready: {} (viewport {}x{} logical)", studio.name, viewport.0, viewport.1);
					return Ok(());
				}
			}
			bail!("the capture Studio did not appear in the MCP's Studio list within 3 minutes")
		}
		StudioCommand::Close => {
			#[cfg(windows)]
			{
				let studio = CaptureStudio::connect()?;
				let window = studio.window()?;
				let pid = ror::studio::capture::window_process(window.hwnd);
				println!("closing {} (process {pid})", studio.name);
				ror::studio::capture::terminate(pid)?;
			}
			Ok(())
		}
		StudioCommand::Status => {
			let mut studio = CaptureStudio::connect()?;
			let viewport = studio.viewport()?;
			println!("studio {} ({}) viewport {}x{} logical", studio.name, studio.id, viewport.0, viewport.1);
			#[cfg(windows)]
			{
				let window = studio.window()?;
				println!("window {}x{} at {},{} dpi {}", window.width, window.height, window.x, window.y, window.dpi);
				for child in ror::studio::capture::children(window.hwnd).iter().filter(|child| child.visible && child.width >= 200 && child.height >= 150) {
					println!("  child {} {}x{} at {},{} {:?}", child.class, child.width, child.height, child.x, child.y, child.title);
				}
			}
			Ok(())
		}
		StudioCommand::Luau { file, code } => {
			let code = match (file, code) {
				(_, Some(code)) => code,
				(Some(file), None) => std::fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?,
				(None, None) => bail!("give a file or -e code"),
			};
			let mut studio = CaptureStudio::connect()?;
			println!("{}", studio.luau(&code)?);
			Ok(())
		}
		StudioCommand::Calibrate { name } => {
			#[cfg(windows)]
			{
				let experiment = ror::studio::calibrate::experiment(&name)?;
				let out = output::repo_dir().join("out").join("calibrate").join(&experiment.name);
				let gpu = Arc::new(Gpu::new()?);
				let renderer = Renderer::new(gpu, assets(false)?)?;
				ror::studio::calibrate::run(&renderer, experiment, &out)?;
				println!("{}", out.join("samples.json").display());
				Ok(())
			}
			#[cfg(not(windows))]
			{
				let _ = name;
				bail!("calibration needs Windows (Studio capture)")
			}
		}
		StudioCommand::Export { studio, path, out, no_materials } => {
			let bytes = ror::studio::export::export(&studio, &path, !no_materials)?;
			output::write_atomic(&out, &bytes)?;
			println!("{} ({} bytes)", out.display(), bytes.len());
			Ok(())
		}
		StudioCommand::Capture { out } => {
			#[cfg(windows)]
			{
				let out = output::studio_capture_path(out)?;
				let mut studio = CaptureStudio::connect()?;
				let (width, height, rgba) = studio.capture_viewport()?;
				output::write_png(&out, width, height, &rgba, true)?;
				println!("{} ({width}x{height})", out.display());
				Ok(())
			}
			#[cfg(not(windows))]
			{
				let _ = out;
				bail!("viewport capture needs Windows")
			}
		}
	}
}

fn compare(args: CompareArgs) -> Result<()> {
	#[cfg(windows)]
	{
		let name = args.input.file_stem().map(|stem| stem.to_string_lossy().to_string()).unwrap_or_else(|| "compare".into());
		let out = match args.out {
			Some(out) => out,
			None => output::repo_dir().join("out").join("compare").join(name),
		};
		let mut views = match (&args.camera, &args.camera_from) {
			(Some(camera), _) => {
				let camera: CameraSpec = serde_json::from_str(camera).context("--camera JSON")?;
				vec![job::NamedView { name: "camera".into(), view: ror::render::ViewRequest { camera, ..Default::default() } }]
			}
			(None, Some(dir)) => {
				let wanted: Vec<&str> = args.views.split(',').map(str::trim).filter(|name| !name.is_empty() && *name != "default").collect();
				let views: Vec<job::NamedView> = earlier_views(dir)?
					.into_iter()
					.filter(|(name, _)| wanted.is_empty() || wanted.contains(&name.as_str()))
					.map(|(name, camera)| job::NamedView { name, view: ror::render::ViewRequest { camera, ..Default::default() } })
					.collect();
				if views.is_empty() {
					bail!("--camera-from {}: none of --views {}", dir.display(), args.views);
				}
				views
			}
			(None, None) => job::preset_views(&args.views, 1, 1, Background::Sky, args.fov)?,
		};
		for view in &mut views {
			view.view.msaa = args.msaa;
		}
		let environment = match &args.lighting_from {
			Some(place) => {
				let document = ror::scene::load::read_file(place)?;
				Some(ror::scene::extract::environment(&document).with_context(|| format!("{} has no Lighting", place.display()))?)
			}
			None => None,
		};
		let gpu = Arc::new(Gpu::new()?);
		let renderer = Renderer::new(gpu, assets(args.offline)?)?;
		let lod = args.lod.as_deref().map(ror::lod::Lod::parse).transpose()?;
		let options = ror::studio::compare::CompareOptions { input: args.input, select: args.select, views, out, environment, lod };
		let report = ror::studio::compare::run(&renderer, &options)?;
		println!("{}", serde_json::to_string_pretty(&report)?);
		Ok(())
	}
	#[cfg(not(windows))]
	{
		let _ = args;
		bail!("compare needs Windows (Studio capture)")
	}
}

/// `--cameras`: named cameras, each an eye and a target or a CFrame.
#[derive(serde::Deserialize)]
struct NamedCamera {
	name: String,
	#[serde(default)]
	eye: Option<[f32; 3]>,
	#[serde(default)]
	target: Option<[f32; 3]>,
	#[serde(default)]
	cframe: Option<[f32; 12]>,
	#[serde(default)]
	fov: Option<f32>,
}

/// `--sweep`: where a map is densest, as a player stands there. The parts' 64-stud cells by how much
/// they hold (a mesh three times a part, as Roblox's scene-analysis sweep scores them), the densest
/// `count`; in each the camera stands 5 studs over the floor (the top of the widest part under the
/// cell's middle) and looks north, east, south and west, a little down.
fn sweep_views(document: &ror::scene::load::Document, count: usize, fov: f32) -> Vec<(String, CameraSpec)> {
	use std::collections::HashMap;
	let scene = ror::scene::extract::extract(document);
	let Some((low, high)) = scene.framing_bounds() else { return Vec::new() };
	struct Cell {
		score: u32,
		sum: glam::Vec2,
		parts: Vec<usize>,
	}
	let mut cells: HashMap<(i32, i32), Cell> = HashMap::new();
	for (index, part) in scene.parts.iter().enumerate() {
		let center = glam::Vec3::from(part.cframe.translation);
		if part.transparency >= 1.0 || center.cmplt(low).any() || center.cmpgt(high).any() {
			continue;
		}
		let cell = cells.entry(((center.x / 64.0).floor() as i32, (center.z / 64.0).floor() as i32)).or_insert(Cell { score: 0, sum: glam::Vec2::ZERO, parts: Vec::new() });
		cell.score += if matches!(part.geometry, ror::scene::Geometry::MeshPart { .. }) { 3 } else { 1 };
		cell.sum += glam::Vec2::new(center.x, center.z);
		cell.parts.push(index);
	}
	let mut ranked: Vec<((i32, i32), Cell)> = cells.into_iter().collect();
	ranked.sort_by(|a, b| b.1.score.cmp(&a.1.score).then(a.0.cmp(&b.0)));
	let mut views = Vec::new();
	for (rank, (_, cell)) in ranked.into_iter().take(count).enumerate() {
		let middle = cell.sum / cell.parts.len() as f32;
		let boxes: Vec<(glam::Vec3, glam::Vec3)> = cell.parts.iter().map(|&index| ror::cost::cull::world_box(&scene.parts[index].cframe, scene.parts[index].size)).collect();
		let area = |(min, max): &(glam::Vec3, glam::Vec3)| (max.x - min.x) * (max.z - min.z);
		let under = boxes.iter().filter(|(min, max)| min.x <= middle.x && max.x >= middle.x && min.z <= middle.y && max.z >= middle.y).max_by(|a, b| area(a).total_cmp(&area(b)));
		let floor = under.map(|(_, max)| max.y).unwrap_or_else(|| boxes.iter().map(|(_, max)| max.y).fold(f32::INFINITY, f32::min));
		let eye = glam::Vec3::new(middle.x, floor + 5.0, middle.y);
		for (name, direction) in [("N", glam::Vec3::NEG_Z), ("E", glam::Vec3::X), ("S", glam::Vec3::Z), ("W", glam::Vec3::NEG_X)] {
			let target = eye + direction * 50.0 - glam::Vec3::Y * 4.0;
			views.push((format!("dense{}-{name}", rank + 1), CameraSpec::LookAt { eye: eye.into(), target: target.into(), fov }));
		}
	}
	views
}

/// How far the placement area reaches around the enemy path's nodes (40 studs by default): the box of
/// the nodes and this much around it is where a match's camera works, and where its costliest views
/// are.
const TRACK_PADDING: f32 = 40.0;

/// The enemy path: parts under a folder named Path(s), Node(s) or Waypoint(s), in their order; of
/// those folders only the outermost (a map's own Paths and Nodes, not scenery models named "path"
/// inside its assets, as some maps have). Invisible ones too: some maps' nodes are Transparency 1.
fn track_nodes(document: &ror::scene::load::Document) -> Vec<glam::Vec3> {
	let track = |name: &str| matches!(name.to_lowercase().as_str(), "path" | "paths" | "node" | "nodes" | "waypoint" | "waypoints");
	// depth first in the file's order: (instance, its depth, the depth of the track folder above it)
	let mut found: Vec<(usize, glam::Vec3)> = Vec::new();
	let mut stack: Vec<(rbx_dom_weak::types::Ref, usize, Option<usize>)> = document.roots.iter().rev().map(|root| (*root, 0, None)).collect();
	while let Some((referent, depth, folder)) = stack.pop() {
		let Some(instance) = document.dom.get_by_ref(referent) else { continue };
		if let Some(folder) = folder
			&& ror::scene::props::class_is_a(instance.class.as_str(), "BasePart")
			&& let Some(cframe) = ror::scene::props::cframe(instance, "CFrame")
		{
			found.push((folder, glam::Vec3::from(cframe.translation)));
		}
		let below = folder.or_else(|| track(instance.name.as_str()).then_some(depth));
		stack.extend(instance.children().iter().rev().map(|child| (*child, depth + 1, below)));
	}
	let Some(outermost) = found.iter().map(|(folder, _)| *folder).min() else { return Vec::new() };
	found.into_iter().filter(|(folder, _)| *folder == outermost).map(|(_, node)| node).collect()
}

/// `--grid STUDS`: a cell every STUDS studs where some part lies under the cell's centre, each seen
/// four ways from a far gameplay camera; over `region`, else the track's placement area (the path
/// nodes' box and TRACK_PADDING around it, the camera over the nearest node's height), else the whole
/// map (the floor: the top of the widest part under the cell). Also what area that was, for the report.
fn grid_views(document: &ror::scene::load::Document, spacing: f32, fov: f32, region: Option<(glam::Vec2, glam::Vec2, Option<f32>)>) -> (Vec<(String, CameraSpec)>, String) {
	let scene = ror::scene::extract::extract(document);
	let Some((mut low, mut high)) = scene.framing_bounds() else { return (Vec::new(), String::new()) };
	let mut floor = None;
	let nodes = track_nodes(document);
	let area = if let Some((min, max, height)) = region {
		(low.x, low.z, high.x, high.z) = (min.x, min.y, max.x, max.y);
		floor = height;
		format!("x {:.0}..{:.0}, z {:.0}..{:.0}", min.x, max.x, min.y, max.y)
	} else if !nodes.is_empty() {
		let (min, max) = nodes.iter().fold((glam::Vec3::splat(f32::INFINITY), glam::Vec3::splat(f32::NEG_INFINITY)), |(min, max), node| (min.min(*node), max.max(*node)));
		(low.x, low.z, high.x, high.z) = (min.x - TRACK_PADDING, min.z - TRACK_PADDING, max.x + TRACK_PADDING, max.z + TRACK_PADDING);
		format!("the track's placement area (its path nodes and {TRACK_PADDING:.0} studs around them)")
	} else {
		"the whole map".to_string()
	};
	let track_height = |point: glam::Vec2| -> Option<f32> {
		nodes.iter().min_by(|a, b| (glam::Vec2::new(a.x, a.z) - point).length_squared().total_cmp(&(glam::Vec2::new(b.x, b.z) - point).length_squared())).map(|node| node.y)
	};
	let spacing = spacing.max(8.0);
	let boxes: Vec<(glam::Vec3, glam::Vec3)> = scene.parts.iter().filter(|part| part.transparency < 1.0).map(|part| ror::cost::cull::world_box(&part.cframe, part.size)).collect();
	let (columns, rows) = (((high.x - low.x) / spacing).ceil().max(1.0) as i32, ((high.z - low.z) / spacing).ceil().max(1.0) as i32);
	let (back, up, pitch) = ror::cost::grid::FAR;
	let mut views = Vec::new();
	for row in 0..rows {
		for column in 0..columns {
			let middle = glam::Vec2::new(low.x + (column as f32 + 0.5) * spacing, low.z + (row as f32 + 0.5) * spacing);
			// the floor: the highest part at least 16 studs square under the cell's centre (a big base
			// plane far below the playable floors is not it), else any part under it
			let under: Vec<&(glam::Vec3, glam::Vec3)> = boxes.iter().filter(|(min, max)| min.x <= middle.x && max.x >= middle.x && min.z <= middle.y && max.z >= middle.y).collect();
			let large = under.iter().filter(|(min, max)| max.x - min.x >= 16.0 && max.z - min.z >= 16.0).map(|(_, max)| max.y).fold(f32::NEG_INFINITY, f32::max);
			let top = if large.is_finite() { large } else { under.iter().map(|(_, max)| max.y).fold(f32::NEG_INFINITY, f32::max) };
			if !top.is_finite() {
				continue;
			}
			let height = floor.or_else(|| if region.is_none() { track_height(middle) } else { None }).unwrap_or(top);
			let point = glam::Vec3::new(middle.x, height, middle.y);
			for yaw_degrees in [0u32, 90, 180, 270] {
				let yaw = (yaw_degrees as f32).to_radians();
				let eye = glam::Vec3::new(point.x - yaw.sin() * back, point.y + 4.0 + up, point.z - yaw.cos() * back);
				let forward = glam::Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos());
				views.push((format!("grid{column}_{row}-{yaw_degrees}"), CameraSpec::LookAt { eye: eye.into(), target: (eye + forward * 50.0).into(), fov }));
			}
		}
	}
	(views, area)
}

/// The near and far gameplay cameras of a match (behind and above the action, looking down): studs
/// back, studs up and pitch in radians.
const PATH_ZOOMS: [(&str, f32, f32, f32); 2] = [("near", 25.0, 14.0, -0.45), ("far", 70.0, 55.0, -0.65)];

/// `--paths N`: N points spread along the map's enemy paths (the parts under a folder named like
/// "Paths", "Nodes" or "Waypoints", in their order), each seen four ways from a near and a far camera.
fn path_views(document: &ror::scene::load::Document, count: usize, fov: f32) -> Vec<(String, CameraSpec)> {
	let nodes = track_nodes(document);
	if nodes.is_empty() || count == 0 {
		return Vec::new();
	}
	let picks: Vec<glam::Vec3> = if nodes.len() <= count { nodes } else { (0..count).map(|index| nodes[(index as f32 * (nodes.len() - 1) as f32 / (count - 1).max(1) as f32).round() as usize]).collect() };
	let mut views = Vec::new();
	for (index, point) in picks.iter().enumerate() {
		for (zoom, back, up, pitch) in PATH_ZOOMS {
			for yaw_degrees in [0u32, 90, 180, 270] {
				let yaw = (yaw_degrees as f32).to_radians();
				let eye = glam::Vec3::new(point.x - yaw.sin() * back, point.y + 4.0 + up, point.z - yaw.cos() * back);
				let forward = glam::Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos());
				views.push((format!("path{}-{yaw_degrees}-{zoom}", index + 1), CameraSpec::LookAt { eye: eye.into(), target: (eye + forward * 50.0).into(), fov }));
			}
		}
	}
	views
}

fn cost(mut args: CostArgs) -> Result<()> {
	if let Some(studio) = &args.studio {
		let bytes = ror::studio::export::export(studio, &args.input, true)?;
		let name: String = args.input.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
		let path = output::state_dir().join("exports").join(format!("{name}.rbxm"));
		output::write_atomic(&path, &bytes)?;
		args.input = path.display().to_string();
	}
	#[allow(unused_mut)]
	let (mut width, mut height) = parse_size(&args.size)?;
	// --check: Studio's viewport decides the shape of the views (what is in view)
	#[cfg(windows)]
	if args.check {
		(width, height) = ror::studio::capture::CAPTURE_VIEWPORT;
	}
	let fov = args.fov.unwrap_or(70.0);
	let input_path = std::path::Path::new(&args.input).to_path_buf();
	let read_document = || -> Result<ror::scene::load::Document> {
		let mut document = if input_path.is_file() {
			ror::scene::load::read_file(&input_path)?
		} else {
			let uri = if args.input.chars().all(|c| c.is_ascii_digit()) { format!("rbxassetid://{}", args.input) } else { args.input.clone() };
			let fetched = assets(args.offline)?.fetch(&uri).map_err(|error| anyhow::anyhow!("{} is not a file, and as an asset: {error}", args.input))?;
			ror::scene::load::read_bytes(&fetched.bytes)?
		};
		if let Some(select) = &args.select {
			document.select(select)?;
		}
		Ok(document)
	};
	let named = |name: String, camera: CameraSpec| job::NamedView { name, view: ror::render::ViewRequest { width, height, camera, background: Background::Sky, ..Default::default() } };
	// a grid run's area, for its report
	let mut grid_area = None;
	let mut views = if let Some(file) = &args.cameras {
		let text = std::fs::read(file).with_context(|| format!("--cameras {}", file.display()))?;
		let cameras: Vec<NamedCamera> = serde_json::from_slice(&text).with_context(|| format!("--cameras {}: a list of {{name, eye, target, fov}} or {{name, cframe, fov}}", file.display()))?;
		let mut views = Vec::new();
		for camera in cameras {
			let spec = match (camera.cframe, camera.eye, camera.target) {
				(Some(components), _, _) => CameraSpec::CFrame { components, fov: camera.fov.unwrap_or(fov), near: None },
				(None, Some(eye), Some(target)) => CameraSpec::LookAt { eye, target, fov: camera.fov.unwrap_or(fov) },
				_ => bail!("--cameras: {} needs a cframe, or an eye and a target", camera.name),
			};
			views.push(named(camera.name, spec));
		}
		views
	} else if let Some(count) = args.sweep {
		let views: Vec<_> = sweep_views(&read_document()?, count, fov).into_iter().map(|(name, camera)| named(name, camera)).collect();
		if views.is_empty() {
			bail!("--sweep: the scene draws no part");
		}
		views
	} else if let (Some(dir), None, None) = (&args.camera_from, &args.camera, args.yaw) {
		// --camera-from alone: every view of the earlier run, as it was
		earlier_views(dir)?.into_iter().map(|(name, camera)| named(name, camera)).collect()
	} else if let Some(spacing) = args.grid {
		let region = match &args.region {
			Some(text) => {
				let values: Vec<f32> = text.split(',').map(|value| value.trim().parse::<f32>()).collect::<Result<_, _>>().context("--region: X0,Z0,X1,Z1[,Y]")?;
				if values.len() != 4 && values.len() != 5 {
					bail!("--region: four or five numbers, X0,Z0,X1,Z1[,Y]");
				}
				let (x0, z0, x1, z1) = (values[0], values[1], values[2], values[3]);
				Some((glam::Vec2::new(x0.min(x1), z0.min(z1)), glam::Vec2::new(x0.max(x1), z0.max(z1)), values.get(4).copied()))
			}
			None => None,
		};
		let (views, area) = grid_views(&read_document()?, spacing, fov, region);
		if views.is_empty() {
			bail!("--grid: the scene draws no part to stand on");
		}
		grid_area = Some(area);
		views.into_iter().map(|(name, camera)| named(name, camera)).collect()
	} else if let Some(count) = args.paths {
		let views: Vec<_> = path_views(&read_document()?, count, fov).into_iter().map(|(name, camera)| named(name, camera)).collect();
		if views.is_empty() {
			bail!("--paths: the scene has no parts under a folder named Path(s), Node(s) or Waypoint(s)");
		}
		views
	} else if let Some(camera) = &args.camera {
		vec![named("render".into(), serde_json::from_str(camera).context("--camera JSON")?)]
	} else if let Some(yaw) = args.yaw {
		vec![named("render".into(), CameraSpec::Orbit { yaw, pitch: args.pitch, distance: None, padding: 0.08, target: None, fov })]
	} else {
		job::preset_views(&args.views, width, height, Background::Sky, args.fov)?
	};
	if let Some(dir) = &args.camera_from {
		let cameras = earlier_cameras(dir)?;
		for view in &mut views {
			let names = || cameras.keys().cloned().collect::<Vec<_>>().join(", ");
			view.view.camera = cameras.get(&view.name).cloned().with_context(|| format!("--camera-from {}: it has no view {:?} (it has {})", dir.display(), view.name, names()))?;
		}
	}
	let technology = args.technology.as_deref().map(|name| ["legacy", "voxel", "compatibility", "shadowmap", "future", "unified"].iter().position(|item| *item == name).unwrap_or(5) as u32);
	let lod = args.lod.as_deref().map(ror::lod::Lod::parse).transpose()?;
	let playback = playback_args(&TimelineArgs { playback: args.playback.as_deref(), time: args.time, emit: &args.emit, enable: &[], disable: &[], changes: None })?;
	let out = match args.out.clone() {
		Some(out) => out,
		None => output::state_dir().join("out").join(format!("{}-cost", job::input_name(&args.input))),
	};
	let input = if input_path.is_file() { absolute(&input_path) } else { args.input.clone() };
	let job = Job {
		input,
		select: args.select.clone(),
		out: Some(out.clone()),
		views,
		no_png: args.grid.is_some() || args.no_images,
		compact_png: false,
		materials_from: args.materials_from.as_deref().map(absolute),
		lighting_from: args.lighting_from.as_deref().map(absolute),
		playback,
		frame: None,
		focus: args.focus.clone(),
		collision: None,
		map: None,
		cost: Some(ror::cost::CostOptions {
			technology,
			quality: lod.map(|lod| lod.quality).or(args.quality),
			no_occlusion: args.no_occlusion,
			// the capture Studio draws every mesh at full detail (its MeshPartDetailLevel setting)
			// unless told to count as a game does
			full_detail: args.full_detail || (args.check && lod.is_none()),
			studio_lod: args.check && lod.is_some(),
			top: args.top,
			no_images: args.no_images,
			grid: args.grid,
			grid_area,
			items: args.items,
			all_emitters: args.all_emitters,
		}),
		relative_to: std::env::current_dir().ok(),
		alone: None,
	};
	let result = if args.no_service || std::env::var_os("ROR_NO_SERVICE").is_some() {
		let gpu = Arc::new(Gpu::new()?);
		let renderer = Renderer::new(gpu, assets(args.offline)?)?;
		let value = serde_json::to_value(job::run(&renderer, &SceneCache::default(), &job)?)?;
		report_held(&renderer);
		value
	} else {
		ror::service::client::render(&output::state_dir(), &job, args.offline)?
	};
	// the report taken out of the result, unless --check reads both (a grid's is ~240 MB as a Value)
	let mut result = result;
	let report = if args.check { result.get("cost").cloned() } else { result.as_object_mut().and_then(|result| result.remove("cost")) };
	let report = report.context("the render service returned no cost (an older build: rebuild ror)")?;
	if args.json {
		output::print_json(&report, false)?;
	} else {
		for line in report.get("lines").and_then(|lines| lines.as_array()).into_iter().flatten() {
			println!("{}", line.as_str().unwrap_or_default());
		}
		println!("cost.json, <view>.draws.png and <view>.heat.png in {}", out.display());
	}
	if let Some(dir) = &args.compare {
		for line in cost_compare(&report, dir)? {
			println!("{line}");
		}
	}
	if args.check {
		#[cfg(windows)]
		cost_check(&args, &read_document()?, &result, &report, &out, width, height)?;
		#[cfg(not(windows))]
		bail!("--check needs Windows (the capture Studio)");
	}
	Ok(())
}

/// `ror cost --compare`: this run's views against an earlier run's of the same names.
fn cost_compare(report: &serde_json::Value, dir: &std::path::Path) -> Result<Vec<String>> {
	let file = dir.join("cost.json");
	let earlier: serde_json::Value = serde_json::from_slice(&std::fs::read(&file).with_context(|| format!("--compare: reading {}", file.display()))?)?;
	let views = |value: &serde_json::Value| -> std::collections::BTreeMap<String, serde_json::Value> {
		value["views"].as_array().into_iter().flatten().filter_map(|view| Some((view["name"].as_str()?.to_string(), view.clone()))).collect()
	};
	let (before, after) = (views(&earlier), views(report));
	let names: Vec<&String> = after.keys().filter(|name| before.contains_key(*name)).collect();
	if names.is_empty() {
		bail!("--compare: {} has none of these views (use --camera-from {} for the same cameras)", file.display(), dir.display());
	}
	let number = |view: &serde_json::Value, pointer: &str| view.pointer(pointer).and_then(|value| value.as_f64()).unwrap_or(0.0);
	let mean = |side: &std::collections::BTreeMap<String, serde_json::Value>, pointer: &str| names.iter().map(|name| number(&side[*name], pointer)).sum::<f64>() / names.len() as f64;
	let change = |pointer: &str, label: &str, digits: usize| {
		let (a, b) = (mean(&before, pointer), mean(&after, pointer));
		let percent = if a.abs() > 1e-9 { format!(" ({:+.1}%)", 100.0 * (b - a) / a) } else { String::new() };
		format!("{label} {a:.digits$} -> {b:.digits$}{percent}")
	};
	let mut lines = vec![format!("compared with {} ({} views in both, per view on average):", dir.display(), names.len())];
	lines.push(format!("  {}, {}", change("/total/draws", "draws", 1), change("/total/triangles", "triangles", 0)));
	let mut passes: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
	for name in &names {
		for side in [&before, &after] {
			passes.extend(side[*name]["passes"].as_object().into_iter().flatten().map(|(pass, _)| pass.clone()));
		}
	}
	let per_pass: Vec<String> = passes.iter().filter(|pass| pass.as_str() != "Shadows").map(|pass| change(&format!("/passes/{pass}/draws"), pass, 1)).collect();
	lines.push(format!("  draws by pass: {}", per_pass.join(", ")));
	lines.push(format!(
		"  effects: {}, {}, {}, {}, {}",
		change("/visible/emitters", "emitters", 1),
		change("/visible/particles", "particles", 0),
		change("/visible/particleFill", "particle fill (views)", 2),
		change("/visible/beams", "beams", 1),
		change("/visible/transparentParts", "transparent parts", 1)
	));
	lines.push(format!(
		"  occlusion: {}, {}, {}, {}",
		change("/occlusion/tested", "tested", 0),
		change("/occlusion/hidden", "hidden", 0),
		change("/occlusion/occluders", "occluders", 0),
		change("/occlusion/estimateMs", "~ms", 2)
	));
	let mut moved: Vec<(f64, String)> = names
		.iter()
		.map(|name| {
			let (a, b) = (number(&before[*name], "/total/draws"), number(&after[*name], "/total/draws"));
			(b - a, format!("{name} {a:.0} -> {b:.0} ({:+.0})", b - a))
		})
		.collect();
	moved.sort_by(|a, b| a.0.total_cmp(&b.0));
	let listed: Vec<String> = moved.iter().filter(|(delta, _)| *delta < 0.0).take(6).map(|(_, text)| text.clone()).collect();
	if !listed.is_empty() {
		lines.push(format!("  most fewer draws: {}", listed.join(", ")));
	}
	let worse: Vec<String> = moved.iter().rev().filter(|(delta, _)| *delta > 0.0).take(4).map(|(_, text)| text.clone()).collect();
	if !worse.is_empty() {
		lines.push(format!("  more draws: {}", worse.join(", ")));
	}
	Ok(lines)
}

/// `ror cost --check`: the views measured in the capture Studio, beside the model's counts.
#[cfg(windows)]
fn cost_check(args: &CostArgs, document: &ror::scene::load::Document, result: &serde_json::Value, report: &serde_json::Value, out: &std::path::Path, width: u32, height: u32) -> Result<()> {
	use ror::studio::compare::{CaptureSession, material_bytes, model_bytes};
	let mut scene = ror::scene::extract::extract(document);
	if let Some(place) = &args.lighting_from {
		let place = ror::scene::load::read_file(place)?;
		scene.environment = ror::scene::extract::environment(&place).context("--lighting-from has no Lighting")?;
	}
	let mut cameras = Vec::new();
	for view in result.get("views").and_then(|views| views.as_array()).into_iter().flatten() {
		let name = view.get("name").and_then(|name| name.as_str()).unwrap_or_default().to_string();
		let components: [f32; 12] = serde_json::from_value(view.get("camera").cloned().unwrap_or_default())?;
		let fov = view.get("fov").and_then(|fov| fov.as_f64()).unwrap_or(70.0) as f32;
		let near = view.get("near").and_then(|near| near.as_f64()).map(|near| near as f32);
		cameras.push((name, CameraSpec::CFrame { components, fov, near }.place(None, width, height)));
	}
	let mut studio = ror::studio::CaptureStudio::connect()?;
	let model = model_bytes(document)?;
	ror::cost::studio::load(&mut studio, &model)?;
	let mut session = CaptureSession::attach(&mut studio, &model)?;
	session.set_materials(material_bytes(document, &scene.materials)?.as_ref())?;
	session.set_environment(&scene.environment)?;
	// counted as a game draws (--lod): Studio's levels of detail and quality while it measures
	let lod = args.lod.as_deref().map(ror::lod::Lod::parse).transpose()?;
	if let (Some(lod), Some(studio)) = (lod, session.studio.as_deref_mut()) {
		studio.luau(&ror::studio::compare::game_lod_on(lod.quality)).context("setting the capture Studio's levels of detail")?;
	}
	let measured = ror::cost::studio::measure(&mut session, &cameras, args.profile);
	if let (Some(_), Some(studio)) = (lod, session.studio.as_deref_mut()) {
		studio.luau(ror::studio::compare::GAME_LOD_OFF).context("putting back the capture Studio's levels of detail")?;
	}
	let measured = measured?;
	let views = report.get("views").and_then(|views| views.as_array()).cloned().unwrap_or_default();
	let mut rows = Vec::new();
	for (name, studio) in &measured {
		let model = views.iter().find(|view| view.get("name").and_then(|value| value.as_str()) == Some(name.as_str()));
		let model_draws = |pass: &str| model.and_then(|view| view.pointer(&format!("/passes/{pass}/draws"))).and_then(|value| value.as_u64()).unwrap_or(0);
		let mut names: std::collections::BTreeSet<String> = studio.passes.keys().cloned().collect();
		if let Some(passes) = model.and_then(|view| view.get("passes")).and_then(|passes| passes.as_object()) {
			names.extend(passes.keys().cloned());
		}
		let passes: Vec<String> = names.iter().map(|pass| format!("{pass} {}/{}", model_draws(pass), studio.passes.get(pass).map(|count| count.draws).unwrap_or(0))).collect();
		let model_total = model.and_then(|view| view.pointer("/total/draws")).and_then(|value| value.as_u64()).unwrap_or(0);
		let model_triangles = model.and_then(|view| view.pointer("/total/triangles")).and_then(|value| value.as_u64()).unwrap_or(0);
		let total = studio.total();
		println!(
			"check {name}: ror {model_total} draws, Studio {} ({:+}); triangles ror {model_triangles}, Studio {} | per pass ror/Studio: {}",
			total.draws,
			model_total as i64 - total.draws as i64,
			total.triangles,
			passes.join(", ")
		);
		if let Some(profile) = &studio.profile {
			let scope = |name: &str| profile.scopes.get(name).copied().unwrap_or(0.0);
			let calls = |name: &str| profile.calls.get(name).copied().unwrap_or(0.0);
			println!(
				"  profile {name}: occlusion {:.2} ms in {:.1} culling jobs (render thread waits {:.2}); render thread {:.2} ms (scene {:.2}, render queue {:.2}, particles {:.2}, lighting {:.2}); {} frames",
				scope("queryOcclusion"),
				calls("CullJob"),
				scope("WaitOnRenderNodeQueryFence"),
				scope("Perform"),
				scope("Scene"),
				scope("updateRenderQueue"),
				scope("updateParticles"),
				scope("SceneUpdater::computeLightingPerform"),
				profile.renders
			);
		}
		rows.push(serde_json::json!({"view": name, "studio": studio, "ror": model.and_then(|view| view.get("passes")).cloned()}));
	}
	output::write_atomic(&out.join("cost-check.json"), serde_json::to_string_pretty(&rows)?.as_bytes())?;
	Ok(())
}

fn service(command: ServiceCommand) -> Result<()> {
	let state_dir = output::state_dir();
	match command {
		ServiceCommand::Status => {
			let services = ror::service::client::status(&state_dir);
			if services.is_empty() {
				println!("no render service running for {}", state_dir.display());
				return Ok(());
			}
			let services: Vec<_> = services.into_iter().map(|(info, health)| serde_json::json!({"port": info.port, "pid": info.pid, "build": info.build, "source": info.source, "health": health})).collect();
			println!("{}", serde_json::to_string_pretty(&services)?);
			Ok(())
		}
		ServiceCommand::Stop => {
			let stopped = ror::service::client::stop(&state_dir)?;
			println!("{}", if stopped > 0 { format!("stopping {stopped}") } else { "not running".into() });
			Ok(())
		}
		ServiceCommand::Start { offline } => {
			let port = ror::service::client::ensure(&state_dir, offline)?;
			println!("render service on 127.0.0.1:{port}");
			Ok(())
		}
		ServiceCommand::Warm { offline } => {
			let (port, warming) = ror::service::client::warm(&state_dir, offline)?;
			println!("render service on 127.0.0.1:{port}{}", if warming { ": warming the GPU" } else { " (busy or already warming)" });
			Ok(())
		}
	}
}

fn clean(args: CleanArgs) -> Result<()> {
	use std::path::Path;
	let out = match args.out {
		Some(out) => out,
		None => output::repo_dir().join("out"),
	};
	let state = output::state_dir();
	let gb = |bytes: u64| bytes as f64 / 1e9;
	let (freed, skipped) = (if args.dry_run { "to free" } else { "freed" }, |count: usize| if count > 0 { format!(", {count} left as they were") } else { String::new() });
	if let Some(days) = args.older_than {
		if days.is_nan() || days < 0.0 {
			bail!("--older-than takes a number of days");
		}
		let derived = |path: &Path| path.file_name().is_some_and(|name| name == "diff.png" || name == "side-by-side.png");
		let removed = ror::clean::remove_older(std::slice::from_ref(&out), &derived, std::time::Duration::from_secs_f64(days * 86400.0), args.dry_run);
		let failed = if removed.failed > 0 { format!(", {} could not be removed", removed.failed) } else { String::new() };
		println!("{}: diff.png and side-by-side.png older than {days} days: {} files, {:.2} GB {freed}{failed}", out.display(), removed.files, gb(removed.freed));
	}
	let mut downloads = vec![state.join("assets")];
	downloads.extend(output::shared_state_dirs().iter().chain(output::store_dir().as_ref()).map(|dir| dir.join("assets")));
	for (what, roots, extension) in [("PNG files", vec![out], "png"), ("downloads", downloads, "bin"), ("Studio captures", vec![state.join("captures")], "png"), ("material packs", vec![state.join("material-packs")], "png")] {
		let accept = |path: &Path| path.extension().is_some_and(|found| found.eq_ignore_ascii_case(extension));
		let linked = ror::clean::link_duplicates(&roots, &accept, args.dry_run);
		let places: Vec<String> = roots.iter().map(|root| root.display().to_string()).collect();
		println!(
			"{}: {what}: {} files, {:.2} GB ({:.2} GB of it shared already); {} contents held more than once, {} names linked, {:.2} GB {freed}{}",
			places.join(" + "),
			linked.files,
			gb(linked.bytes),
			gb(linked.shared),
			linked.groups,
			linked.links,
			gb(linked.freed),
			skipped(linked.skipped)
		);
	}
	Ok(())
}

fn percent(part: u64, whole: u64) -> String {
	format!("{:.1} %", part as f64 * 100.0 / whole.max(1) as f64)
}

fn inspect(args: InspectArgs) -> Result<()> {
	use ror::inspect;
	let rendered = inspect::Rendered::open(&args.dir)?;
	if let Some(at) = &args.at {
		let (x, y) = at.split_once(',').and_then(|(x, y)| Some((x.trim().parse::<u32>().ok()?, y.trim().parse::<u32>().ok()?))).context("--at takes X,Y")?;
		let view = rendered.view(args.view.as_deref())?;
		let (part, pixel) = inspect::at(&rendered, view, x, y)?;
		if args.json {
			let part = part.map(|(index, part)| serde_json::json!({"index": index, "path": part.path, "class": part.class, "material": inspect::material_name(part.material).map_or_else(|| part.material.to_string(), str::to_string), "transparency": part.transparency, "surfaceAppearance": part.surface_appearance}));
			println!("{}", serde_json::json!({"view": view.name, "x": x, "y": y, "rgba": pixel, "part": part}));
		} else {
			let what = match (&part, &view.ids) {
				(Some((index, part)), _) => format!("{} (part {}) · {}", part.path, index + 1, inspect::describe(&part.class, part.material, part.transparency, part.surface_appearance)),
				(None, Some(_)) => "no part (the background, or effects: particles, beams and trails have no ids)".to_string(),
				(None, None) => "unknown: the render has no ids (render with --ids)".to_string(),
			};
			println!("{} at {x},{y}: {what}; pixel rgba({}, {}, {}, {})", view.name, pixel[0], pixel[1], pixel[2], pixel[3]);
		}
		return Ok(());
	}
	let views: Vec<&inspect::RenderedView> = match &args.view {
		Some(name) => vec![rendered.view(Some(name))?],
		None => rendered.views.iter().collect(),
	};
	let mut shown = Vec::new();
	for view in views {
		let coverage = inspect::coverage(&rendered, view)?;
		let total = coverage.width as u64 * coverage.height as u64;
		if args.json {
			let paths: Vec<_> = coverage
				.paths
				.iter()
				.take(args.top)
				.map(|path| serde_json::json!({"path": path.path, "class": path.class, "material": inspect::material_name(path.material).map_or_else(|| path.material.to_string(), str::to_string), "transparency": path.transparency, "surfaceAppearance": path.surface_appearance, "parts": path.parts, "pixels": path.pixels, "share": path.pixels as f64 / total.max(1) as f64}))
				.collect();
			shown.push(serde_json::json!({"view": view.name, "width": coverage.width, "height": coverage.height, "drawn": coverage.drawn, "drawnShare": coverage.drawn as f64 / total.max(1) as f64, "pathsSeen": coverage.paths.len(), "paths": paths}));
			continue;
		}
		println!("{} ({}x{}): parts cover {} of the image · {} part paths seen", view.name, coverage.width, coverage.height, percent(coverage.drawn, total), coverage.paths.len());
		if coverage.paths.is_empty() {
			println!("  (what is drawn is effects or the sky: particles, beams and trails have no ids)");
		}
		for path in coverage.paths.iter().take(args.top) {
			let parts = if path.parts > 1 { format!(" ({} parts)", path.parts) } else { String::new() };
			println!("  {:>7}  {}{parts} · {}", percent(path.pixels, total), path.path, inspect::describe(&path.class, path.material, path.transparency, path.surface_appearance));
		}
		if coverage.paths.len() > args.top {
			let rest: u64 = coverage.paths.iter().skip(args.top).map(|path| path.pixels).sum();
			println!("  {:>7}  … {} more part paths", percent(rest, total), coverage.paths.len() - args.top);
		}
	}
	if args.json {
		println!("{}", serde_json::to_string_pretty(&shown)?);
	}
	Ok(())
}

fn diff(args: DiffArgs) -> Result<()> {
	use ror::inspect;
	let before = inspect::Rendered::open(&args.before)?;
	let after = inspect::Rendered::open(&args.after)?;
	let out = args.out.unwrap_or_else(|| args.after.join("diff"));
	let diffs = inspect::diff(&before, &after, args.threshold, &out)?;
	if args.json {
		let views: Vec<_> = diffs
			.iter()
			.map(|view| {
				let parts: Vec<_> = view.parts.iter().take(args.top).map(|(path, old, new)| serde_json::json!({"path": path, "before": old, "after": new})).collect();
				serde_json::json!({"view": view.name, "width": view.width, "height": view.height, "changed": view.changed, "changedShare": view.changed as f64 / (view.width as f64 * view.height as f64).max(1.0), "largest": view.largest, "mean": view.mean, "region": view.region, "parts": parts, "image": view.image, "skipped": view.skipped})
			})
			.collect();
		println!("{}", serde_json::to_string_pretty(&views)?);
		return Ok(());
	}
	for view in &diffs {
		if let Some(reason) = &view.skipped {
			println!("{}: not compared ({reason})", view.name);
			continue;
		}
		let total = view.width as u64 * view.height as u64;
		if view.changed == 0 {
			println!("{}: the same (no channel differs by more than {}/255)", view.name, args.threshold);
			continue;
		}
		let region = view.region.map_or_else(String::new, |(left, top, right, bottom)| format!(", within x {left}..{right}, y {top}..{bottom}"));
		println!("{}: {} of the pixels changed ({} of {total}, by more than {}/255; mean {:.0}, largest {}){region}", view.name, percent(view.changed, total), view.changed, args.threshold, view.mean, view.largest);
		for (path, old, new) in view.parts.iter().take(args.top) {
			let change = *new as i64 - *old as i64;
			println!("  {change:+8} px  {path} ({old} → {new})");
		}
		if view.parts.len() > args.top {
			println!("  … {} more part paths changed", view.parts.len() - args.top);
		}
		if let Some(image) = &view.image {
			println!("  diff image  {}", image.display());
		}
	}
	Ok(())
}

fn main() {
	#[cfg(feature = "dhat-heap")]
	let _profiler = dhat::Profiler::new_heap();
	ror::private_std_handles();
	ror::report_crashes();
	ror::cap_memory();
	ror::init_threads();
	let cli = Cli::parse();
	let result = run(cli.command);
	if std::env::var_os("ROR_MEMORY_REPORT").is_some() {
		memory_report();
	}
	if let Err(error) = result {
		eprintln!("ror: {error:#}");
		std::process::exit(1);
	}
}

/// Whether `command` works only on the paths it was given, so it needs no checkout, state dir or
/// store. Without a per-user cache folder these commands still run.
fn uses_no_binary_defaults(command: &Command) -> bool {
	matches!(command, Command::Inspect(_) | Command::Diff(_) | Command::Studio(StudioCommand::Export { .. }))
}

/// Chooses this binary's checkout, state dir and store before a command reads them, which is what
/// separates ror's binary from a host that links the library (`output::init_binary`). Commands that
/// only use the paths they were given skip it ([`uses_no_binary_defaults`]).
///
/// A service runs a copy of ror from its state dir, so it takes the checkout, or the lack of one, of
/// the ror that started it (`--source`) and the state dir it was given (`--state-dir`). Every other
/// command goes by its own executable.
fn select_binary_defaults(command: &Command) -> Result<()> {
	if uses_no_binary_defaults(command) {
		return Ok(());
	}
	let exe = std::env::current_exe().context("locating the executable")?;
	match command {
		Command::Serve(args) => {
			let starter = args.source.as_deref().map(std::path::Path::new).unwrap_or(&exe);
			output::init_binary(starter, args.state_dir.clone())
		}
		_ => output::init_binary(&exe, None),
	}
}

/// Runs one command. `main` prints its error, so a missing state dir ends as an ordinary error message.
fn run(command: Command) -> Result<()> {
	select_binary_defaults(&command)?;
	match command {
		Command::Render(args) => render(args),
		Command::Video(args) => video(args),
		Command::Serve(args) => {
			if args.gpu_slots.is_some() {
				eprintln!("ror serve: --gpu-slots does nothing now: draws start within the GPU's free memory, as many at once as fit");
			}
			ror::service::serve(ror::service::ServeOptions {
				// `--state-dir`, or the default; `select_binary_defaults` has already chosen it
				state_dir: output::state_dir(),
				build: args.build,
				source: args.source,
				idle: std::time::Duration::from_secs(args.idle),
				offline: args.offline,
				dormant: args.dormant,
			})
		}
		Command::Service(command) => service(command),
		Command::Studio(command) => studio(command),
		Command::Compare(args) => compare(args),
		Command::Clean(args) => clean(args),
		Command::Inspect(args) => inspect(args),
		Command::Diff(args) => diff(args),
		Command::Cost(args) => cost(args),
	}
}

/// `ROR_MEMORY_REPORT`: what the renderer still holds after a job run in this process.
fn report_held(renderer: &Renderer) {
	if std::env::var_os("ROR_MEMORY_REPORT").is_some() {
		eprintln!("ror memory held: {}", renderer.memory_breakdown());
	}
}

/// `ROR_MEMORY_REPORT`: the process's peak commit and working set, on stderr at exit. On Windows the
/// GPU's allocations are charged to the commit too (a render whose textures overflowed the GPU took
/// 47.6 GB of it), so the peak commit is the whole of what a command cost the machine.
fn memory_report() {
	#[cfg(windows)]
	// SAFETY: this process's own counters, read into a zeroed struct of the size passed
	unsafe {
		use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
		use windows_sys::Win32::System::Threading::GetCurrentProcess;
		let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
		counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
		if GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0 {
			eprintln!("ror memory: peak commit {} MB, peak working set {} MB ({:.0} ms)", counters.PeakPagefileUsage >> 20, counters.PeakWorkingSetSize >> 20, ror::since_first_checkpoint_ms());
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The `--out` that `ror studio capture <args>` parses to.
	fn capture_out(args: &[&str]) -> Option<PathBuf> {
		let argv = ["ror", "studio", "capture"].into_iter().chain(args.iter().copied());
		match Cli::try_parse_from(argv).unwrap().command {
			Command::Studio(StudioCommand::Capture { out }) => out,
			_ => panic!("expected studio capture"),
		}
	}

	#[test]
	fn studio_capture_distinguishes_an_omitted_output_from_an_explicit_one() {
		assert_eq!(capture_out(&[]), None);
		assert_eq!(capture_out(&["--out", "out/studio-capture.png"]), Some(PathBuf::from("out/studio-capture.png")));
	}
}
