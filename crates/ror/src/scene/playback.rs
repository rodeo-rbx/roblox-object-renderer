//! Playback: an effect at a moment of its timeline (notes/render.md "Playback"). Time 0 is when the
//! effect is placed in the world (an attack clones its effects then); `time` is the moment drawn.
//! Actions emit particles, enable and disable emitters, trails, beams and lights, change TimeScale
//! and set properties at given times; tweens interpolate properties; motion moves parts, models and
//! attachments along keyframed CFrames. The state at `time` is applied to the model before its scene
//! is extracted ([`Playback::apply_state`]); then trails get the points their attachments left and
//! emitters the particles they emitted, when and where ([`Playback::apply_history`]).

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use glam::{Affine3A, Quat, Vec3};
use rayon::prelude::*;
use rbx_dom_weak::Instance;
use rbx_dom_weak::types::{Ref, Variant};
use serde::{Deserialize, Serialize};

use super::load::Document;
use super::trails::TrailPoint;
use super::{Scene, props};

/// Studio's rendered frames come at 60 a second: trails take a point and the world updates once a
/// frame (the capture Studio's RenderStepped, 58-60 a second in edit mode).
pub const FRAME_RATE: f32 = 60.0;

/// The most of a frame's step that counts towards Rate emission in that frame (the rest counts with
/// the next one; particle-rate-start).
const EMISSION_STEP: f32 = 1.0 / 60.0;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Playback {
	/// The moment drawn, seconds after the start.
	#[serde(default)]
	pub time: f32,
	/// What happens when (any order).
	#[serde(default)]
	pub actions: Vec<Action>,
	/// Properties interpolated over time.
	#[serde(default)]
	pub tweens: Vec<Tween>,
	/// Parts, models and attachments moving along keyframed CFrames.
	#[serde(default)]
	pub motion: Vec<Motion>,
	/// When the world updates, seconds after the start (Studio's rendered frames, as a capture
	/// recorded them): trails take a point at each, and every event happens at the first of them at
	/// or after its time. Default: every 1/60 s from 0.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub frames: Option<Vec<f32>>,
	/// Emitters and trails enabled in the file have been running this many seconds at time 0
	/// (default 0: they start at 0, as an effect cloned into the world does).
	#[serde(default)]
	pub prewarm: f32,
	/// A recording of play (an engine emulator's `ree record`): drawn as a running game draws it
	/// ([`super::Scene::play`]).
	#[serde(default, skip_serializing_if = "std::ops::Not::not")]
	pub play: bool,
	/// The camera the playback was watched through (a recording's camera track: `ror video
	/// --camera-track` gives it), each key from its moment until the next: Studio updates only the
	/// emitters it could draw, so one out of that camera's view neither emits nor ages ([`ViewerKey`]).
	/// Empty: every emitter runs all along.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub viewer: Vec<ViewerKey>,
	/// Draws every emitter's random stream anew, as another cast of the same effect would show (how far
	/// one draw strays from another: notes/render.md "Particles"); empty: the streams of every other render.
	/// In the job, not the environment, so it reaches whichever service runs it.
	#[serde(default, skip_serializing_if = "String::is_empty")]
	pub salt: String,
}

/// One key of the camera a playback was watched through: from `at` the camera stands at this Roblox
/// CFrame (x, y, z and the rotation's rows), with this vertical field of view (degrees) and the view's
/// width over its height.
///
/// Studio does not update an emitter it does not draw (particle-offscreen: Rate columns and a burst
/// started while the camera looked the other way for 3 s show, once it is back, a column grown from
/// the emitter since then and the burst still a heap at the emitter, at 0.1, 0.5 and 1.5 s; one watched
/// for 1.5 s, unseen for 1 s and frozen as the camera came back holds its 1.5 s column, no gap at its
/// foot; an emitter 2 studs past the view's edge sending its particles across it runs). What counts is
/// how far its particles can reach ([`reach`]; particle-offscreen-reach: emitters 1, 5 and 10 studs past
/// the edge whose particles fly 15 studs in their life run, one 20 studs past does not, and one 20
/// studs past whose particles fly 30 does). A cutscene
/// that builds its scene away from the camera and runs its emitters while it loads (one ran six seconds
/// so in Studio) shows them starting as its camera arrives.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ViewerKey {
	pub at: f32,
	pub cframe: [f32; 12],
	pub fov: f32,
	pub aspect: f32,
}

impl ViewerKey {
	/// Whether a ball reaches into this camera's view (its four sides and its near plane).
	fn sees(&self, center: Vec3, radius: f32) -> bool {
		let c = &self.cframe;
		let eye = Vec3::new(c[0], c[1], c[2]);
		let right = Vec3::new(c[3], c[6], c[9]);
		let up = Vec3::new(c[4], c[7], c[10]);
		let back = Vec3::new(c[5], c[8], c[11]);
		let offset = center - eye;
		let (x, y, depth) = (offset.dot(right), offset.dot(up), -offset.dot(back));
		if depth < -radius {
			return false;
		}
		let half_height = (self.fov.clamp(1.0, 179.0).to_radians() * 0.5).tan();
		let half_width = half_height * self.aspect.max(1e-3);
		// a side plane through the eye with the inward normal (-1, 0, slope) normalised, in the camera's
		// right / depth plane (and its up / depth plane)
		let inside = |across: f32, slope: f32| (slope * depth - across.abs()) / (1.0 + slope * slope).sqrt() > -radius;
		inside(x, half_width) && inside(y, half_height)
	}
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Action {
	/// Seconds after the start.
	#[serde(default)]
	pub at: f32,
	/// A dotted instance path from the file's root (as `select`; every instance of that path; `*` any
	/// child). Emit, enabled, timeScale and clear apply to it and to the ParticleEmitters, Trails,
	/// Beams, lights, Highlights and post effects under it; `set` and `parent` to it alone.
	pub target: String,
	/// ParticleEmitter:Emit(n), or a mode reading each emitter's EmitCount, EmitDelay and EmitDuration
	/// attributes ([`ByAttributes`]): "attributes" (EmitCount after EmitDelay, enabled from the call
	/// for EmitDuration), "immediate" (the same without the delay), "cutscene" (cutscene timing) or
	/// "delayed" (the calibration runs').
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub emit: Option<Emit>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub enabled: Option<bool>,
	/// With `enabled`: a switch that wins, as the command line's `--enable` / `--disable` are. From
	/// `at` on, what it reaches stays switched so whatever else the playback does (other switches,
	/// sets and tweens of Enabled), wherever the playback parents it, and what the playback parents
	/// under it is reached too; switched off, it emits nothing more. A later forced switch takes over.
	#[serde(default, skip_serializing_if = "std::ops::Not::not")]
	pub force: bool,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub time_scale: Option<f32>,
	/// ParticleEmitter:Clear(), Trail:Clear().
	#[serde(default, skip_serializing_if = "std::ops::Not::not")]
	pub clear: bool,
	/// Emit, enabled, timeScale and clear apply only to instances of this class (or one inheriting it)
	/// under the target (e.g. "Trail": a projectile's trails, not its emitters); `set` and `parent`
	/// then apply to every such instance at or under it.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub class: Option<String>,
	/// Properties of the target from this time on (numbers, booleans, [x, y, z] vectors and colours,
	/// 12-number CFrames, strings and content URIs, sequences and ranges, an instance reference as the
	/// dotted path of the instance it names or null for none), or relative to its value then (`{"add":
	/// 4}`, numbers and vectors), or as in the file (`{"initial": true}`).
	#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
	pub set: BTreeMap<String, serde_json::Value>,
	/// The target's parent from this time on: a dotted path (`Lighting`, where a post effect an effect
	/// carries takes effect, as code parents it there), or "" for none (out of the world, as `Parent =
	/// nil`).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub parent: Option<String>,
	/// With an `emit` count: where the burst leaves from, the world CFrame (12 numbers) of the emitter's
	/// part or attachment at the call, instead of where the playback has that part or attachment at
	/// the frame. Roblox's `Emit` spawns its particles in the call (not measured here), so a script that
	/// moves an attachment between calls in one frame leaves a burst at each place, where a frame's
	/// motion holds one (a recording's emits carry it: ree; one without it is placed by the motion).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cframe: Option<Vec<f32>>,
	/// With `cframe`, an emitter on a part: the part's Size at the call, which its particles spawn over.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub part_size: Option<[f32; 3]>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Emit {
	Count(u32),
	/// "attributes", "immediate", "cutscene" or "delayed" ([`ByAttributes`])
	Mode(String),
}

/// A property of one target over time (numbers, vectors, colours, CFrames), one of two ways:
/// - `keys`: between keyframes, each segment eased like TweenService (`easing` / `direction` of the
///   key it ends at; Linear when absent), the first key's value before it and the last's after it:
///   the property at every moment.
/// - `to`: as `TweenService:Create(target, TweenInfo.new(duration, easing, direction, repeatCount,
///   reverses, delay), {[property] = to}):Play()` at `at`: from the value the property has when it
///   starts to `to` (TweenInfo's defaults: 1 s, Quad, Out), back again each time with `reverses`; a
///   later tween of the property takes over from where this one is, and a set while it runs is lost.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Tween {
	pub target: String,
	pub property: String,
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub keys: Vec<TweenKey>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub to: Option<serde_json::Value>,
	/// When `Play` is called (seconds after the start); the tween starts `delay` later.
	#[serde(default)]
	pub at: f32,
	#[serde(default = "one_second")]
	pub duration: f32,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub easing: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub direction: Option<String>,
	/// Times played again after the first (-1: for ever).
	#[serde(default)]
	pub repeat_count: i32,
	#[serde(default)]
	pub reverses: bool,
	#[serde(default)]
	pub delay: f32,
	/// The tween applies to every instance of this class (or one inheriting it) at or under the
	/// target instead: "Light" fades every light of an effect, as effect scripts often do.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub class: Option<String>,
}

fn one_second() -> f32 {
	1.0
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TweenKey {
	pub at: f32,
	pub value: serde_json::Value,
	/// `Enum.EasingStyle` name (Linear when absent) and `Enum.EasingDirection` name (Out).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub easing: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub direction: Option<String>,
}

/// A target's world CFrame over time (a part's CFrame, a model's pivot, an attachment's world
/// CFrame): everything under it moves with it.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Motion {
	pub target: String,
	pub keys: Vec<MotionKey>,
	/// The keys are in the target's own frame at rest (its pose is its rest CFrame times the key):
	/// [0, 0, -20] is 20 studs ahead of where it stands.
	#[serde(default, skip_serializing_if = "std::ops::Not::not")]
	pub relative: bool,
	/// Only the instances at the path holding this CollectionService tag move: a path names every
	/// same-named sibling (a map's "Models.Part" is one CameraFollow mist host and 49 floors).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub tagged: Option<String>,
	/// From this moment physics moves it (a recording's free fall): its LockedToPart particles stay
	/// where it left them (a capture Studio's playtest client: four locked particles, their emitters on or off,
	/// TimeScale 1 or 0.125, stayed put while their parts fell 27 studs in a playtest; a CFrame write
	/// carries them).
	#[serde(default, rename = "physicsFrom", skip_serializing_if = "Option::is_none")]
	pub physics_from: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MotionKey {
	pub at: f32,
	/// `CFrame.new(x, y, z, R00, R01, R02, R10, R11, R12, R20, R21, R22)` components; a position alone
	/// ([x, y, z]) keeps the target's rotation at rest.
	pub cframe: Vec<f32>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub easing: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub direction: Option<String>,
}

/// One particle emission of an emitter's playback: the particle's age at the moment drawn (in the
/// emitter's particle time), where the emitter was then, its velocity (VelocityInheritance) and the
/// particle's random stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Birth {
	pub age: f32,
	/// Its age by the spawning clock (its death test): as `age` but for the frame a TimeScale changes
	/// in, which still spawns and kills at the old TimeScale (particle-rate-edges front).
	pub lived: f32,
	pub cframe: Affine3A,
	pub velocity: Vec3,
	pub stream: u64,
	/// The emitting part's size then (its shape's volume; a tweened Size changes it).
	pub size: Vec3,
	/// A LockedToPart particle's frame now when not the emitter's: where the emitter was as physics
	/// took its part over ([`Motion::physics_from`]), or where it was born after that.
	pub follow: Option<Affine3A>,
}

/// TweenService's easing (`Enum.EasingStyle`, `Enum.EasingDirection`) of a 0..1 fraction.
pub fn ease(alpha: f32, style: Option<&str>, direction: Option<&str>) -> f32 {
	let t = alpha.clamp(0.0, 1.0);
	let ease_in = |t: f32| -> f32 {
		match style.unwrap_or("Linear") {
			"Sine" => 1.0 - (t * std::f32::consts::FRAC_PI_2).cos(),
			"Quad" => t * t,
			"Cubic" => t * t * t,
			"Quart" => t.powi(4),
			"Quint" => t.powi(5),
			"Exponential" => {
				if t <= 0.0 {
					0.0
				} else {
					2f32.powf(10.0 * t - 10.0)
				}
			}
			"Circular" => 1.0 - (1.0 - t * t).max(0.0).sqrt(),
			"Back" => 2.70158 * t * t * t - 1.70158 * t * t,
			"Elastic" => {
				if t <= 0.0 || t >= 1.0 {
					t
				} else {
					-(2f32.powf(10.0 * t - 10.0)) * ((t * 10.0 - 10.75) * std::f32::consts::TAU / 3.0).sin()
				}
			}
			"Bounce" => 1.0 - bounce_out(1.0 - t),
			_ => t,
		}
	};
	match direction.unwrap_or("Out") {
		"In" => ease_in(t),
		"InOut" => {
			if t < 0.5 {
				ease_in(2.0 * t) * 0.5
			} else {
				1.0 - ease_in(2.0 - 2.0 * t) * 0.5
			}
		}
		_ => 1.0 - ease_in(1.0 - t),
	}
}

fn bounce_out(t: f32) -> f32 {
	let (n, d) = (7.5625, 2.75);
	if t < 1.0 / d {
		n * t * t
	} else if t < 2.0 / d {
		let t = t - 1.5 / d;
		n * t * t + 0.75
	} else if t < 2.5 / d {
		let t = t - 2.25 / d;
		n * t * t + 0.9375
	} else {
		let t = t - 2.625 / d;
		n * t * t + 0.984375
	}
}

/// Every instance at a dotted path from the file's root (siblings may share a name; `*` is every
/// child: `Lighting.*`, all that is in Lighting); the empty path is everything drawn (the document's
/// roots).
pub fn resolve(document: &Document, path: &str) -> Result<Vec<Ref>> {
	if path.trim().is_empty() {
		return Ok(document.roots.clone());
	}
	let segments = super::load::split_path(path)?;
	let mut current = vec![document.dom.root_ref()];
	for segment in &segments {
		let mut next = Vec::new();
		for referent in &current {
			let Some(instance) = document.dom.get_by_ref(*referent) else {
				continue;
			};
			next.extend(instance.children().iter().copied().filter(|&child| segment == "*" || document.dom.get_by_ref(child).is_some_and(|child| &child.name == segment)));
		}
		// a name that is not there is a mistake; every child of what has none is nothing
		if next.is_empty() && segment != "*" {
			bail!("playback: no instance {path:?} ({segment:?} not found)");
		}
		current = next;
	}
	Ok(current)
}

/// [`resolve`] for many paths of one document: each parent's children by name indexed at its first
/// lookup, each path found once (a cutscene's 17,000 actions name parts in a folder of 3,000: a scan
/// of the folder per action took a second a document).
struct Paths<'d> {
	document: &'d Document,
	children: std::cell::RefCell<std::collections::HashMap<Ref, std::collections::HashMap<String, Vec<Ref>>>>,
	found: std::cell::RefCell<std::collections::HashMap<String, Vec<Ref>>>,
}

impl<'d> Paths<'d> {
	fn new(document: &'d Document) -> Self {
		Self { document, children: Default::default(), found: Default::default() }
	}

	/// What `path` names, as [`resolve`] finds it.
	fn get(&self, path: &str) -> Result<Vec<Ref>> {
		if let Some(found) = self.found.borrow().get(path) {
			return Ok(found.clone());
		}
		let document = self.document;
		if path.trim().is_empty() {
			return Ok(document.roots.clone());
		}
		let segments = super::load::split_path(path)?;
		let mut current = vec![document.dom.root_ref()];
		for segment in &segments {
			let mut next = Vec::new();
			for &referent in &current {
				let Some(instance) = document.dom.get_by_ref(referent) else {
					continue;
				};
				if segment == "*" {
					next.extend(instance.children().iter().copied());
					continue;
				}
				let mut children = self.children.borrow_mut();
				let by_name = children.entry(referent).or_insert_with(|| {
					let mut by_name: std::collections::HashMap<String, Vec<Ref>> = std::collections::HashMap::new();
					for &child in instance.children() {
						if let Some(child_instance) = document.dom.get_by_ref(child) {
							by_name.entry(child_instance.name.clone()).or_default().push(child);
						}
					}
					by_name
				});
				if let Some(named) = by_name.get(segment.as_str()) {
					next.extend(named.iter().copied());
				}
			}
			// a name that is not there is a mistake; every child of what has none is nothing
			if next.is_empty() && segment != "*" {
				bail!("playback: no instance {path:?} ({segment:?} not found)");
			}
			current = next;
		}
		self.found.borrow_mut().insert(path.to_string(), current.clone());
		Ok(current)
	}

	/// An action's new parent: the first instance at `path`, or none for "" and "nil".
	fn parent(&self, path: &str) -> Result<Ref> {
		let path = path.trim();
		if path.is_empty() || path == "nil" {
			return Ok(Ref::none());
		}
		self.get(path)?.first().copied().with_context(|| format!("playback: no parent {path:?}"))
	}
}

fn descendants(document: &Document, referent: Ref) -> Vec<Ref> {
	let mut list = Vec::new();
	let mut stack = vec![referent];
	while let Some(current) = stack.pop() {
		list.push(current);
		if let Some(instance) = document.dom.get_by_ref(current) {
			stack.extend(instance.children().iter().copied());
		}
	}
	list
}

fn affine(components: &[f32], rest: Affine3A) -> Result<Affine3A> {
	match components.len() {
		3 => Ok(Affine3A::from_mat3_translation(rest.matrix3.into(), Vec3::new(components[0], components[1], components[2]))),
		12 => {
			let c = components;
			let matrix = glam::Mat3::from_cols(Vec3::new(c[3], c[6], c[9]), Vec3::new(c[4], c[7], c[10]), Vec3::new(c[5], c[8], c[11]));
			Ok(Affine3A::from_mat3_translation(matrix, Vec3::new(c[0], c[1], c[2])))
		}
		count => bail!("playback: a CFrame has 3 or 12 numbers, not {count}"),
	}
}

/// A rotation without scale (an odd CFrame may carry some), for slerping.
fn rotation(transform: &Affine3A) -> Quat {
	let m = transform.matrix3;
	let x = Vec3::from(m.x_axis).normalize_or(Vec3::X);
	let y = Vec3::from(m.y_axis);
	let z = x.cross(y).normalize_or(Vec3::Z);
	Quat::from_mat3(&glam::Mat3::from_cols(x, z.cross(x), z))
}

/// A target's resolved motion: its keys as world transforms and its pose at rest.
#[derive(Clone, Debug, PartialEq)]
struct Track {
	referent: Ref,
	rest: Affine3A,
	keys: Vec<(f32, Affine3A, Option<String>, Option<String>)>,
	/// [`Motion::physics_from`]
	physics_from: Option<f32>,
}

impl Track {
	/// The target's world CFrame at `t`: its first key before the first, its last after the last,
	/// position lerped and rotation slerped between, eased by the later key; between equal keys
	/// exactly that key.
	fn pose(&self, t: f32) -> Affine3A {
		let Some(first) = self.keys.first() else {
			return self.rest;
		};
		if t <= first.0 {
			return first.1;
		}
		for pair in self.keys.windows(2) {
			let ((t0, a, ..), (t1, b, easing, direction)) = (&pair[0], &pair[1]);
			if t <= *t1 {
				// a key held (a recording leaves out the frames a part stood still in) stays put: a lerp of
				// a key with itself lands a last bit off it at some moments, and the rotation rebuilt from
				// its quaternion a little off the key's own (rock unions held still for a second shaded a
				// few hundred pixels 1-6 levels differently from frame to frame, an edge pixel 171)
				if a == b {
					return *a;
				}
				let alpha = if t1 - t0 > 1e-6 { ease((t - t0) / (t1 - t0), easing.as_deref(), direction.as_deref()) } else { 1.0 };
				let position = Vec3::from(a.translation).lerp(Vec3::from(b.translation), alpha);
				return Affine3A::from_rotation_translation(rotation(a).slerp(rotation(b), alpha), position);
			}
		}
		self.keys[self.keys.len() - 1].1
	}

	/// What the motion does to everything under the target at `t` (its pose over its rest).
	fn delta(&self, t: f32) -> Affine3A {
		self.pose(t) * self.rest.inverse()
	}
}

/// An instance's world CFrame in the file: a part's CFrame, an attachment's (its parent's times its
/// own), a model's pivot (WorldPivot, else its PrimaryPart's CFrame, else the centre of its parts).
pub fn rest_cframe(document: &Document, referent: Ref) -> Option<Affine3A> {
	let instance = document.dom.get_by_ref(referent)?;
	if props::is_a(instance, "Attachment") {
		return super::beams::world_cframe(document, referent);
	}
	if props::is_a(instance, "BasePart") {
		return props::cframe(instance, "CFrame");
	}
	if props::is_a(instance, "PVInstance") || instance.class == "Folder" {
		if let Some(pivot) = props::cframe(instance, "WorldPivotData") {
			return Some(pivot);
		}
		if let Some(primary) = props::reference(instance, "PrimaryPart").and_then(|part| document.dom.get_by_ref(part)).and_then(|part| props::cframe(part, "CFrame")) {
			return Some(primary);
		}
		let centers: Vec<Vec3> = descendants(document, referent)
			.into_iter()
			.filter_map(|child| document.dom.get_by_ref(child))
			.filter(|child| props::is_a(child, "BasePart"))
			.filter_map(|part| props::cframe(part, "CFrame").map(|cframe| Vec3::from(cframe.translation)))
			.collect();
		if centers.is_empty() {
			return Some(Affine3A::IDENTITY);
		}
		let (min, max) = centers.iter().fold((Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)), |(min, max), center| (min.min(*center), max.max(*center)));
		return Some(Affine3A::from_translation((min + max) * 0.5));
	}
	None
}

/// The world updates of a playback up to `end`: its frame times, or every 1/60 s from 0, and every
/// 1/60 s of the prewarm before 0.
fn frame_times(frames: Option<&[f32]>, end: f32, prewarm: f32) -> Vec<f32> {
	let mut times: Vec<f32> = (1..=(prewarm.max(0.0) * FRAME_RATE).ceil() as usize).rev().map(|index| -(index as f32) / FRAME_RATE).collect();
	match frames {
		Some(frames) => times.extend(frames.iter().copied().filter(|&time| time >= 0.0 && time <= end + 1e-4)),
		None => times.extend((0..=(end.max(0.0) * FRAME_RATE + 1e-3).floor() as usize).map(|index| index as f32 / FRAME_RATE)),
	}
	times
}

/// The time an event happens: the first frame at or after it (events between frames wait for the
/// next one, as a script's do).
fn quantize(frames: &[f32], time: f32) -> f32 {
	frames.iter().copied().find(|&frame| frame >= time - 1e-5).unwrap_or(time)
}

/// When a TimeScale set at `time` starts to count: the frame before the one it is set in, whose
/// step it already governs (particle-emission emit-times, timescale-change: frozen particles are a
/// frame younger than the frame they were frozen in).
fn quantize_step(frames: &[f32], time: f32) -> f32 {
	match frames.iter().position(|&frame| frame >= time - 1e-5) {
		Some(index) if index > 0 => frames[index - 1],
		Some(index) => frames[index],
		None => time,
	}
}

/// A JSON value as the Variant of the type `existing` has (the property's current value or default):
/// UDim2s are `[xScale, xOffset, yScale, yOffset]`, UDims `[scale, offset]` and Rects `[minX, minY,
/// maxX, maxY]` (a GUI's sizes, placements and image rects, as a SurfaceGui's content animates them;
/// offsets round to whole pixels, as Roblox stores them). A property keeps its type, as in Roblox,
/// where `Size = "big"` is an error: a value its type cannot take is one here too, never stored (ror's
/// readers would draw the property's default, and rbx_binary refuses a GUI holding it: guis::content).
fn variant(value: &serde_json::Value, existing: Option<&Variant>) -> Result<Variant> {
	use rbx_dom_weak::types::{Color3, Color3uint8, ColorSequence, ColorSequenceKeypoint, Content, ContentId, EnumItem, NumberRange, NumberSequence, NumberSequenceKeypoint, Rect, UDim, UDim2, Vector2, Vector3};
	let numbers = |value: &serde_json::Value| -> Option<Vec<f32>> { value.as_array().map(|array| array.iter().filter_map(|item| item.as_f64()).map(|item| item as f32).collect()) };
	let converted = match (value, existing) {
		(serde_json::Value::Bool(flag), _) => Variant::Bool(*flag),
		(serde_json::Value::Number(number), existing) => {
			let number = number.as_f64().unwrap_or_default();
			match existing {
				Some(Variant::Float64(_)) => Variant::Float64(number),
				Some(Variant::Int32(_)) => Variant::Int32(number as i32),
				Some(Variant::Int64(_)) => Variant::Int64(number as i64),
				Some(Variant::NumberSequence(_)) => NumberSequence { keypoints: vec![NumberSequenceKeypoint::new(0.0, number as f32, 0.0), NumberSequenceKeypoint::new(1.0, number as f32, 0.0)] }.into(),
				Some(Variant::NumberRange(_)) => NumberRange::new(number as f32, number as f32).into(),
				Some(Variant::Enum(_)) => rbx_dom_weak::types::Enum::from_u32(number as u32).into(),
				Some(Variant::EnumItem(item)) => EnumItem { ty: item.ty.clone(), value: number as u32 }.into(),
				// text, as Luau writes a number given to a string property (`Text = 5` shows "5")
				Some(Variant::String(_)) => Variant::String(number.to_string()),
				_ => Variant::Float32(number as f32),
			}
		}
		(serde_json::Value::String(text), existing) => match existing {
			// "" clears it, as `Image = ""` leaves ImageContent none
			Some(Variant::Content(_)) if text.trim().is_empty() => Content::none().into(),
			Some(Variant::Content(_)) => Content::from_uri(text.as_str()).into(),
			Some(Variant::ContentId(_)) => ContentId::from(text.as_str()).into(),
			_ => Variant::String(text.clone()),
		},
		// sequences: [[time, value, envelope], ...] or [[time, r, g, b], ...]
		(serde_json::Value::Array(items), existing) if items.first().is_some_and(|item| item.is_array()) => {
			let rows: Vec<Vec<f32>> = items.iter().map(|item| numbers(item).unwrap_or_default()).collect();
			match existing {
				Some(Variant::ColorSequence(_)) => {
					if rows.iter().any(|row| row.len() < 4) {
						bail!("playback: a ColorSequence key is [time, r, g, b]");
					}
					ColorSequence { keypoints: rows.iter().map(|row| ColorSequenceKeypoint::new(row[0], Color3::new(row[1], row[2], row[3]))).collect() }.into()
				}
				_ => {
					if rows.iter().any(|row| row.len() < 2) {
						bail!("playback: a NumberSequence key is [time, value, envelope]");
					}
					NumberSequence { keypoints: rows.iter().map(|row| NumberSequenceKeypoint::new(row[0], row[1], row.get(2).copied().unwrap_or(0.0))).collect() }.into()
				}
			}
		}
		(serde_json::Value::Array(_), existing) => {
			let numbers = numbers(value).unwrap_or_default();
			match (numbers.len(), existing) {
				(2, Some(Variant::NumberRange(_))) => NumberRange::new(numbers[0], numbers[1]).into(),
				(2, Some(Variant::UDim(_))) => UDim::new(numbers[0], numbers[1].round() as i32).into(),
				(4, Some(Variant::UDim2(_))) => UDim2::new(UDim::new(numbers[0], numbers[1].round() as i32), UDim::new(numbers[2], numbers[3].round() as i32)).into(),
				(4, Some(Variant::Rect(_))) => Rect::new(Vector2::new(numbers[0], numbers[1]), Vector2::new(numbers[2], numbers[3])).into(),
				(2, Some(Variant::Vector2(_))) => Vector2::new(numbers[0], numbers[1]).into(),
				(3, Some(Variant::ColorSequence(_))) => {
					let color = Color3::new(numbers[0], numbers[1], numbers[2]);
					ColorSequence { keypoints: vec![ColorSequenceKeypoint::new(0.0, color), ColorSequenceKeypoint::new(1.0, color)] }.into()
				}
				(3, Some(Variant::Color3(_))) => Color3::new(numbers[0], numbers[1], numbers[2]).into(),
				(3, Some(Variant::Color3uint8(_))) => Color3uint8::new((numbers[0] * 255.0).round() as u8, (numbers[1] * 255.0).round() as u8, (numbers[2] * 255.0).round() as u8).into(),
				(3, _) => Vector3::new(numbers[0], numbers[1], numbers[2]).into(),
				// a model's pivot (WorldPivotData) holds an optional one
				(12, Some(Variant::OptionalCFrame(_))) => Variant::OptionalCFrame(Some(crate::scene::build::cframe(affine(&numbers, Affine3A::IDENTITY)?))),
				(12, _) => crate::scene::build::cframe(affine(&numbers, Affine3A::IDENTITY)?).into(),
				(_, Some(existing)) => bail!("{} is not {}", shown(value), takes(existing)),
				(count, None) => bail!("playback: a value of {count} numbers"),
			}
		}
		(other, _) => bail!("playback: cannot set {other}"),
	};
	if let Some(existing) = existing
		&& converted.ty() != existing.ty()
	{
		bail!("{} is not {}", shown(value), takes(existing));
	}
	Ok(converted)
}

/// A playback value as messages quote it: text in quotes, anything else as JSON.
fn shown(value: &serde_json::Value) -> String {
	match value {
		serde_json::Value::String(text) => format!("{text:?}"),
		other => other.to_string(),
	}
}

/// What a property of `existing`'s type takes in a playback, for messages.
fn takes(existing: &Variant) -> String {
	match existing {
		Variant::UDim2(_) => "a UDim2 ([xScale, xOffset, yScale, yOffset])".into(),
		Variant::UDim(_) => "a UDim ([scale, offset])".into(),
		Variant::Rect(_) => "a Rect ([minX, minY, maxX, maxY])".into(),
		Variant::Vector2(_) => "a Vector2 ([x, y])".into(),
		Variant::Vector3(_) => "a Vector3 ([x, y, z])".into(),
		Variant::Color3(_) | Variant::Color3uint8(_) => "a Color3 ([r, g, b] from 0 to 1, or #rrggbb)".into(),
		Variant::CFrame(_) | Variant::OptionalCFrame(_) => "a CFrame (its 12 numbers, x, y, z, R00 ... R22)".into(),
		Variant::NumberSequence(_) => "a NumberSequence (a number, or [[time, value, envelope], ...])".into(),
		Variant::ColorSequence(_) => "a ColorSequence ([r, g, b], or [[time, r, g, b], ...])".into(),
		Variant::NumberRange(_) => "a NumberRange (a number, or [min, max])".into(),
		Variant::Bool(_) => "a bool (true or false)".into(),
		Variant::Float32(_) | Variant::Float64(_) | Variant::Int32(_) | Variant::Int64(_) => "a number".into(),
		Variant::Enum(_) | Variant::EnumItem(_) => "an enum (its item's number)".into(),
		Variant::String(_) => "text".into(),
		Variant::Content(_) | Variant::ContentId(_) => "content (a URI, \"\" for none)".into(),
		other => format!("a {:?}", other.ty()),
	}
}

/// A keyframe tween's value at `time`: its first key's before it, its last key's after it, eased
/// between; keys relative to the property's value in the file (`initial`) taken from it.
fn tween_at(tween: &Tween, time: f32, initial: Option<&serde_json::Value>) -> Option<serde_json::Value> {
	let mut keys: Vec<&TweenKey> = tween.keys.iter().collect();
	keys.sort_by(|a, b| a.at.total_cmp(&b.at));
	let value = |key: &TweenKey| resolved_value(&key.value, initial, initial).unwrap_or_else(|| key.value.clone());
	let first = keys.first()?;
	if time <= first.at {
		return Some(value(first));
	}
	Some(match keys.windows(2).find(|pair| time <= pair[1].at) {
		Some(pair) => {
			let span = pair[1].at - pair[0].at;
			let alpha = if span > 1e-6 { ease((time - pair[0].at) / span, pair[1].easing.as_deref(), pair[1].direction.as_deref()) } else { 1.0 };
			lerp_value(&value(pair[0]), &value(pair[1]), alpha)
		}
		None => value(keys[keys.len() - 1]),
	})
}

/// A property value between two keys (numbers, vectors, colours, sequences key by key; a CFrame's
/// position lerped and its rotation slerped, as CFrame:Lerp), `alpha` of the way; anything else
/// changes at the end.
fn lerp_value(a: &serde_json::Value, b: &serde_json::Value, alpha: f32) -> serde_json::Value {
	let cframe = |value: &serde_json::Value| -> Option<Affine3A> {
		let numbers: Vec<f32> = value.as_array().filter(|array| array.len() == 12)?.iter().map(|item| item.as_f64().map(|item| item as f32)).collect::<Option<_>>()?;
		affine(&numbers, Affine3A::IDENTITY).ok()
	};
	match (a, b) {
		(serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
			let (a, b) = (a.as_f64().unwrap_or_default(), b.as_f64().unwrap_or_default());
			serde_json::json!(a + (b - a) * alpha as f64)
		}
		// a skewed CFrame (a window camera's: scene::guis) has no rotation to slerp: its numbers are
		// lerped one by one below, so it stays as skewed as its keys
		(serde_json::Value::Array(_), serde_json::Value::Array(_)) if cframe(a).is_some_and(|from| !skewed(&from)) && cframe(b).is_some_and(|to| !skewed(&to)) => {
			let (from, to) = (cframe(a).unwrap_or_default(), cframe(b).unwrap_or_default());
			let position = Vec3::from(from.translation).lerp(Vec3::from(to.translation), alpha);
			cframe_json(Affine3A::from_rotation_translation(rotation(&from).slerp(rotation(&to), alpha), position))
		}
		(serde_json::Value::Array(a), serde_json::Value::Array(b)) if a.len() == b.len() => {
			serde_json::Value::Array(a.iter().zip(b).map(|(a, b)| lerp_value(a, b, alpha)).collect())
		}
		_ => if alpha < 1.0 { a.clone() } else { b.clone() },
	}
}

/// Whether a CFrame's axes are off unit length or square (a window camera's skewed CFrame).
pub(crate) fn skewed(cframe: &Affine3A) -> bool {
	let m = cframe.matrix3;
	let (x, y, z) = (Vec3::from(m.x_axis), Vec3::from(m.y_axis), Vec3::from(m.z_axis));
	[x, y, z].iter().any(|axis| (axis.length() - 1.0).abs() > 0.01) || x.dot(y).abs() > 0.01 || y.dot(z).abs() > 0.01 || z.dot(x).abs() > 0.01
}

/// A CFrame as a playback value: `CFrame.new(x, y, z, R00, R01, R02, R10, R11, R12, R20, R21, R22)`'s numbers.
fn cframe_json(cframe: Affine3A) -> serde_json::Value {
	let (m, t) = (cframe.matrix3, cframe.translation);
	serde_json::json!([t.x, t.y, t.z, m.x_axis.x, m.y_axis.x, m.z_axis.x, m.x_axis.y, m.y_axis.y, m.z_axis.y, m.x_axis.z, m.y_axis.z, m.z_axis.z])
}

/// A property's value as the file holds it (or its class's default), as a playback value: what a
/// tween starts from when nothing changed the property before, and `{"initial": true}`. A content
/// value is the URI drawn from either of its names ("" for none).
fn authored(instance: &Instance, property: &str) -> Option<serde_json::Value> {
	use serde_json::json;
	if let Some(names) = super::load::content_names(instance.class.as_str(), property) {
		return Some(json!(props::content(instance, &[names.content, names.legacy]).unwrap_or_default()));
	}
	Some(match props::get(instance, property)? {
		Variant::Bool(value) => json!(value),
		Variant::Float32(value) => json!(value),
		Variant::Float64(value) => json!(value),
		Variant::Int32(value) => json!(value),
		Variant::Int64(value) => json!(value),
		Variant::Enum(value) => json!(value.to_u32()),
		Variant::Color3(color) => json!([color.r, color.g, color.b]),
		Variant::Color3uint8(color) => json!([color.r as f32 / 255.0, color.g as f32 / 255.0, color.b as f32 / 255.0]),
		Variant::Vector3(vector) => json!([vector.x, vector.y, vector.z]),
		Variant::Vector2(vector) => json!([vector.x, vector.y]),
		Variant::UDim(udim) => json!([udim.scale, udim.offset]),
		Variant::UDim2(udim) => json!([udim.x.scale, udim.x.offset, udim.y.scale, udim.y.offset]),
		Variant::Rect(rect) => json!([rect.min.x, rect.min.y, rect.max.x, rect.max.y]),
		Variant::NumberRange(range) => json!([range.min, range.max]),
		Variant::NumberSequence(sequence) => json!(sequence.keypoints.iter().map(|key| [key.time, key.value, key.envelope]).collect::<Vec<_>>()),
		Variant::ColorSequence(sequence) => json!(sequence.keypoints.iter().map(|key| [key.time, key.color.r, key.color.g, key.color.b]).collect::<Vec<_>>()),
		Variant::CFrame(_) | Variant::OptionalCFrame(_) => cframe_json(props::cframe(instance, property)?),
		Variant::String(text) => json!(text),
		Variant::ContentId(id) => json!(id.as_str()),
		Variant::Content(content) => json!(content.as_uri().unwrap_or_default()),
		_ => return None,
	})
}

/// A playback value as the property then takes it: `{"add": x}` added to `current` (a number, or a
/// vector or colour component by component; a number adds to every component), `{"initial": true}`
/// its value in the file, anything else as it is. None when there is nothing to add to.
fn resolved_value(value: &serde_json::Value, current: Option<&serde_json::Value>, initial: Option<&serde_json::Value>) -> Option<serde_json::Value> {
	fn added(base: &serde_json::Value, add: &serde_json::Value) -> serde_json::Value {
		match (base, add) {
			(serde_json::Value::Number(a), serde_json::Value::Number(b)) => serde_json::json!(a.as_f64().unwrap_or_default() + b.as_f64().unwrap_or_default()),
			(serde_json::Value::Array(a), serde_json::Value::Array(b)) if a.len() == b.len() => serde_json::Value::Array(a.iter().zip(b).map(|(a, b)| added(a, b)).collect()),
			(serde_json::Value::Array(a), serde_json::Value::Number(_)) => serde_json::Value::Array(a.iter().map(|a| added(a, add)).collect()),
			_ => base.clone(),
		}
	}
	match value {
		serde_json::Value::Object(map) if map.get("initial").and_then(|flag| flag.as_bool()) == Some(true) => initial.cloned(),
		serde_json::Value::Object(map) if map.contains_key("add") => Some(added(current.or(initial)?, &map["add"])),
		other => Some(other.clone()),
	}
}

/// A TweenService tween started at `start` from `from`: its value at `t` (its end once it is done:
/// `to`, or `from` again when it reverses).
fn tweened(tween: &Tween, start: f32, from: &serde_json::Value, t: f32, initial: Option<&serde_json::Value>) -> serde_json::Value {
	let to = tween.to.as_ref().and_then(|to| resolved_value(to, Some(from), initial)).unwrap_or_else(|| from.clone());
	let duration = tween.duration.max(0.0);
	if duration <= 0.0 || tween_done(tween, start, t) {
		return if tween.reverses { from.clone() } else { to };
	}
	let cycle = if tween.reverses { 2.0 * duration } else { duration };
	let local = (t - start).max(0.0) % cycle;
	let progress = if local <= duration { local / duration } else { (cycle - local) / duration };
	lerp_value(from, &to, ease(progress, Some(tween.easing.as_deref().unwrap_or("Quad")), Some(tween.direction.as_deref().unwrap_or("Out"))))
}

/// Whether a TweenService tween started at `start` is done by `t` (one repeating for ever never is).
fn tween_done(tween: &Tween, start: f32, t: f32) -> bool {
	let cycle = tween.duration.max(0.0) * if tween.reverses { 2.0 } else { 1.0 };
	if tween.repeat_count < 0 {
		return cycle <= 0.0;
	}
	t - start >= cycle * (tween.repeat_count + 1) as f32 - 1e-6
}

/// One property of one instance over a playback: what sets and tweens it, in time order, or a
/// keyframe tween holding it at every moment.
struct Channel<'a> {
	referent: Ref,
	/// A content value's Content name (`ImageContent` for `Image` too: one value, which ror draws).
	property: &'a str,
	/// That value's content ID name, set alike where the instance holds it (a model built in code).
	alias: Option<&'static str>,
	/// The value in the file (or the class's default).
	initial: Option<serde_json::Value>,
	events: Vec<(f32, Event<'a>)>,
	keyframes: Option<&'a Tween>,
}

enum Event<'a> {
	/// The value from this time on (an action's `set` or `enabled`).
	Set(serde_json::Value),
	/// A TweenService tween starts.
	Tween(&'a Tween),
}

impl Channel<'_> {
	/// The property at `t`; None while nothing has changed it.
	fn at(&self, t: f32) -> Option<serde_json::Value> {
		let initial = self.initial.as_ref();
		if let Some(tween) = self.keyframes {
			return tween_at(tween, t, initial);
		}
		let mut value: Option<serde_json::Value> = None;
		let mut running: Option<(f32, serde_json::Value, &Tween)> = None;
		for (time, event) in &self.events {
			if *time > t + 1e-6 {
				break;
			}
			let now = match &running {
				Some((start, from, tween)) => Some(tweened(tween, *start, from, *time, initial)),
				None => value.clone().or_else(|| self.initial.clone()),
			};
			match event {
				// a tween still running writes the property again the next frame: the set is lost
				Event::Set(_) if running.as_ref().is_some_and(|(start, _, tween)| !tween_done(tween, *start, *time)) => {}
				Event::Set(set) => {
					running = None;
					value = resolved_value(set, now.as_ref(), initial).or(value);
				}
				// the tween before stops where it is, and this one goes on from there
				Event::Tween(tween) => match now {
					Some(from) => running = Some((*time, from, tween)),
					None => value = tween.to.as_ref().and_then(|to| resolved_value(to, None, initial)),
				},
			}
		}
		match running {
			Some((start, from, tween)) => Some(tweened(tween, start, &from, t, initial)),
			None => value,
		}
	}
}

/// What an action or tween with `class` applies to: every instance of that class (or one inheriting
/// it) at or under each target; without one, the targets.
fn of_class(document: &Document, targets: Vec<Ref>, class: Option<&str>) -> Vec<Ref> {
	match class {
		None => targets,
		Some(class) => targets.into_iter().flat_map(|target| descendants(document, target)).filter(|&referent| document.dom.get_by_ref(referent).is_some_and(|instance| props::is_a(instance, class))).collect(),
	}
}


/// What draws nothing in the world but changes the Lighting it is in: post effects, a Sky, an
/// Atmosphere.
fn environment_class(class: &str) -> bool {
	matches!(class, "Sky" | "Atmosphere" | "BloomEffect" | "BlurEffect" | "ColorCorrectionEffect" | "ColorGradingEffect" | "DepthOfFieldEffect" | "SunRaysEffect")
}

/// An instance's dotted path from the file's root, for messages.
fn path_of(document: &Document, referent: Ref) -> String {
	let mut names = Vec::new();
	let mut current = referent;
	while let Some(instance) = document.dom.get_by_ref(current) {
		if current == document.dom.root_ref() {
			break;
		}
		names.push(instance.name.clone());
		current = instance.parent();
	}
	names.reverse();
	names.join(".")
}

/// The classes `enabled` turns on and off: effects, lights, Highlights, and post effects (with a Sky
/// and an Atmosphere, which a playback keeps out of the Lighting while their Enabled is false).
fn switchable(class: &str) -> bool {
	matches!(class, "ParticleEmitter" | "Trail" | "Beam" | "PointLight" | "SpotLight" | "SurfaceLight" | "Highlight" | "Fire" | "Smoke" | "Sparkles") || environment_class(class)
}

/// Whether an `enabled` action switches `instance` (`referent`, at or under one of its `targets`): what
/// is switchable, of its class if it names one. A Sky, an Atmosphere or a post effect only when its path
/// or class is named: under a broader path (`--disable ""`, bisecting a frame) it is the look the effects
/// play in, not one of them.
fn switches(action: &Action, instance: &Instance, referent: Ref, targets: &[Ref]) -> bool {
	let class = instance.class.as_str();
	switchable(class) && action.class.as_deref().is_none_or(|named| props::is_a(instance, named)) && (!environment_class(class) || action.class.is_some() || targets.contains(&referent))
}

/// The classes with a TimeScale (`timeScale` sets it).
fn timed(class: &str) -> bool {
	matches!(class, "ParticleEmitter" | "Fire" | "Smoke" | "Sparkles" | "Explosion")
}

fn attribute_number(instance: &Instance, name: &str) -> Option<f32> {
	match instance.properties.get(&rbx_dom_weak::ustr("Attributes")) {
		Some(Variant::Attributes(attributes)) => match attributes.get(name)? {
			Variant::Float64(value) => Some(*value as f32),
			Variant::Float32(value) => Some(*value),
			Variant::Int32(value) => Some(*value as f32),
			Variant::Int64(value) => Some(*value as f32),
			_ => None,
		},
		_ => None,
	}
}

/// How an `"emit"` mode plays a ParticleEmitter by its EmitCount, EmitDelay and EmitDuration
/// attributes, a convention effect scripts use (ror's own reading of it, not measured in Studio).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByAttributes {
	/// `"attributes"`: EmitCount (5 without one) EmitDelay after the call (at once without one), and
	/// Enabled from the call until EmitDuration after it, 0 without one (an emitter enabled in the file
	/// is switched off at once too).
	Attributes,
	/// `"immediate"`: the same without the delay: EmitCount at once whatever its EmitDelay.
	Immediate,
	/// `"cutscene"`: cutscene timing: after EmitDelay, enabled for EmitDuration when it is above 0 (no burst), else EmitCount emitted
	/// (`ParticleEmitter:Emit`'s own 16 without one).
	Cutscene,
	/// `"delayed"`: ror's rule before 2026-10-10, which the Studio timeline runs were captured with
	/// and still play (`studio::calibrate::timeline`'s runner): EmitCount (5 without one) after
	/// EmitDelay, and enabled from then for EmitDuration when it is above 0.
	Delayed,
}

impl ByAttributes {
	pub fn parse(mode: &str) -> Result<Self> {
		Ok(match mode {
			"attributes" => Self::Attributes,
			"immediate" => Self::Immediate,
			"cutscene" => Self::Cutscene,
			"delayed" => Self::Delayed,
			other => bail!("playback: emit {other:?} (a count, \"attributes\", \"immediate\", \"cutscene\" or \"delayed\")"),
		})
	}

	/// An emitter's events for an emit called at `at` (each at the first frame at or after its
	/// time, as a script's waits end).
	fn schedule(self, instance: &Instance, grid: &[f32], at: f32, timeline: &mut Timeline) {
		let delay = attribute_number(instance, "EmitDelay").unwrap_or(0.0).max(0.0);
		let duration = attribute_number(instance, "EmitDuration").unwrap_or(0.0).max(0.0);
		let count = |default: f32| attribute_number(instance, "EmitCount").unwrap_or(default).max(0.0).round() as u32;
		match self {
			// Enabled = true, then Enabled = false Duration later, both from the call
			Self::Attributes | Self::Immediate => {
				let burst = if self == Self::Attributes { at + delay } else { at };
				timeline.emits.push(Burst::at(quantize(grid, burst), count(5.0)));
				timeline.enabled.push((quantize(grid, at), true));
				timeline.enabled.push((quantize(grid, at + duration), false));
			}
			Self::Cutscene if duration > 0.0 => {
				timeline.enabled.push((quantize(grid, at + delay), true));
				timeline.enabled.push((quantize(grid, at + delay + duration), false));
			}
			Self::Cutscene => timeline.emits.push(Burst::at(quantize(grid, at + delay), count(16.0))),
			Self::Delayed => {
				let start = quantize(grid, at + delay);
				timeline.emits.push(Burst::at(start, count(5.0)));
				if duration > 0.0 {
					timeline.enabled.push((start, true));
					timeline.enabled.push((quantize(grid, at + delay + duration), false));
				}
			}
		}
	}
}

/// One burst of an emitter's playback: when, how many, and where it left from when the action says
/// ([`Action::cframe`]; else where the emitter's part or attachment is at that time).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Burst {
	at: f32,
	count: u32,
	from: Option<Affine3A>,
	/// the emitting part's size then ([`Action::part_size`])
	part_size: Option<Vec3>,
}

impl Burst {
	fn at(at: f32, count: u32) -> Self {
		Self { at, count, from: None, part_size: None }
	}
}

/// What happens to one emitter or trail over the playback.
#[derive(Clone, Debug, Default, PartialEq)]
struct Timeline {
	/// (time, enabled)
	enabled: Vec<(f32, bool)>,
	/// (time, TimeScale) for ageing: from the step of the frame it is set in
	time_scale: Vec<(f32, f32)>,
	/// (time, TimeScale) for spawning and dying: from the step after the frame it is set in
	spawn_scale: Vec<(f32, f32)>,
	emits: Vec<Burst>,
	clears: Vec<f32>,
}

impl Timeline {
	fn sorted(mut self) -> Self {
		self.enabled.sort_by(|a, b| a.0.total_cmp(&b.0));
		self.time_scale.sort_by(|a, b| a.0.total_cmp(&b.0));
		self.spawn_scale.sort_by(|a, b| a.0.total_cmp(&b.0));
		self.emits.sort_by(|a, b| a.at.total_cmp(&b.at));
		self.clears.sort_by(f32::total_cmp);
		self
	}

	fn enabled_at(&self, t: f32) -> bool {
		self.enabled.iter().take_while(|(time, _)| *time <= t + 1e-6).last().is_some_and(|(_, enabled)| *enabled)
	}

	/// Forced switches ([`Action::force`], in time order) win: from the first on only they switch it,
	/// and its emits while forced off are dropped.
	fn force(&mut self, switches: &[(f32, bool)]) {
		let Some(&(first, _)) = switches.first() else {
			return;
		};
		self.enabled.retain(|(time, _)| *time < first - 1e-6);
		self.enabled.extend_from_slice(switches);
		let off = |t: f32| switches.iter().take_while(|(time, _)| *time <= t + 1e-6).last().is_some_and(|(_, on)| !on);
		self.emits.retain(|burst| !off(burst.at));
	}

	fn last_clear(&self, t: f32) -> Option<f32> {
		self.clears.iter().copied().take_while(|time| *time <= t + 1e-6).last()
	}

	/// Particle time at `t`: the integral of TimeScale from the start (TimeScale slows ageing and
	/// emission alike: notes/render.md "Particles").
	fn particle_time(&self, start: f32, t: f32) -> f32 {
		Self::integral(&self.time_scale, start, t)
	}

	/// The spawning clock at `t`: as particle time, but a TimeScale set in a frame counts from the
	/// frame's next step.
	fn spawn_time(&self, start: f32, t: f32) -> f32 {
		Self::integral(&self.spawn_scale, start, t)
	}

	fn integral(scales: &[(f32, f32)], start: f32, t: f32) -> f32 {
		let mut total = 0.0;
		let mut previous = start;
		let mut scale = scales.first().map(|(_, scale)| *scale).unwrap_or(1.0);
		for &(time, next) in scales {
			if time <= start {
				scale = next;
				continue;
			}
			if time >= t {
				break;
			}
			total += (time - previous) * scale;
			previous = time;
			scale = next;
		}
		total + (t - previous).max(0.0) * scale
	}
}

/// How far an emitter's particles can reach from where it emits: half its part's diagonal, its
/// largest particle's corner and the farthest one flies in its life (the view decides whether Studio
/// updates it: [`ViewerKey`]).
fn reach(emitter: &super::particles::ParticleEmitter, size: Vec3) -> f32 {
	let life = emitter.lifetime.0.max(emitter.lifetime.1).max(0.0);
	let speed = emitter.speed.0.abs().max(emitter.speed.1.abs());
	let flight = if emitter.drag > 0.0 { (speed / (emitter.drag * std::f32::consts::LN_2)).min(speed * life) } else { speed * life };
	let pull = 0.5 * emitter.acceleration.length() * life * life;
	let largest = emitter.size_keys.iter().map(|&(_, value, envelope)| value + envelope.abs()).fold(0.0f32, f32::max);
	let stretch = 1.0 + emitter.squash.iter().map(|&(_, value, envelope)| value.abs() + envelope.abs()).fold(0.0f32, f32::max);
	size.length() * 0.5 + largest * stretch * std::f32::consts::SQRT_2 + flight + pull
}

/// Two step functions of time multiplied (each as [`Timeline::integral`] reads one: a value from its
/// time on, the first one before it too).
fn masked(scales: &[(f32, f32)], view: &[(f32, f32)]) -> Vec<(f32, f32)> {
	let value = |list: &[(f32, f32)], t: f32| -> f32 {
		match list.partition_point(|(time, _)| *time <= t) {
			0 => list.first().map_or(1.0, |&(_, value)| value),
			index => list[index - 1].1,
		}
	};
	let mut times: Vec<f32> = scales.iter().chain(view).map(|&(time, _)| time).collect();
	times.sort_by(f32::total_cmp);
	times.dedup();
	times.into_iter().map(|t| (t, value(scales, t) * value(view, t))).collect()
}

/// What [`Resolved::apply_state`] changed in a document, to put back with [`Undo::restore`]: the
/// document then draws another moment without being read again (a video's frames).
#[derive(Default)]
pub struct Undo(Vec<Was>);

/// One change [`Undo`] puts back.
enum Was {
	/// A property and what it held (None: unset).
	Property(Ref, rbx_dom_weak::Ustr, Option<Variant>),
	/// An instance's parent, and that parent's children in their order.
	Parent(Ref, Ref, Vec<Ref>),
}

/// One change a moment's state made to the file's ([`Undo::changes`]).
pub enum Change {
	/// A property of an instance
	Property(Ref, rbx_dom_weak::Ustr),
	/// An instance's parent, and the parent it had
	Parent(Ref, Ref),
}

impl Undo {
	/// What the state changed, in the order it changed it.
	pub fn changes(&self) -> impl Iterator<Item = Change> + '_ {
		self.0.iter().map(|was| match was {
			Was::Property(referent, name, _) => Change::Property(*referent, *name),
			Was::Parent(referent, parent, _) => Change::Parent(*referent, *parent),
		})
	}

	/// Puts every change back as it was, the last first.
	pub fn restore(self, document: &mut Document) {
		for was in self.0.into_iter().rev() {
			match was {
				Was::Property(referent, name, old) => {
					if let Some(instance) = document.dom.get_by_ref_mut(referent) {
						match old {
							Some(value) => {
								instance.properties.insert(name, value);
							}
							None => {
								instance.properties.remove(&name);
							}
						}
					}
				}
				Was::Parent(referent, parent, order) => {
					document.dom.transfer_within(referent, parent);
					// the parent's children in their order again, each moved to the end in turn
					if parent.is_some() {
						for child in order {
							if document.dom.get_by_ref(child).is_some_and(|instance| instance.parent() == parent) {
								document.dom.transfer_within(child, parent);
							}
						}
					}
				}
			}
		}
	}

	/// Sets a property of `referent`, keeping what it held.
	fn set(&mut self, document: &mut Document, referent: Ref, name: &str, value: Variant) {
		// a part moved or resized keeps the random draws of where it was loaded (Part::seed)
		if matches!(name, "CFrame" | "Size") {
			document.keep_seed(referent);
		}
		if let Some(instance) = document.dom.get_by_ref_mut(referent) {
			let name = rbx_dom_weak::ustr(name);
			let old = instance.properties.insert(name, value);
			self.0.push(Was::Property(referent, name, old));
		}
	}

	/// Parents `referent` to `parent` (none: out of the world), keeping where it was.
	fn reparent(&mut self, document: &mut Document, referent: Ref, parent: Ref) {
		let Some(old) = document.dom.get_by_ref(referent).map(|instance| instance.parent()) else {
			return;
		};
		if old == parent || (parent.is_some() && document.dom.get_by_ref(parent).is_none()) {
			return;
		}
		let order = document.dom.get_by_ref(old).map(|instance| instance.children().to_vec()).unwrap_or_default();
		document.dom.transfer_within(referent, parent);
		self.0.push(Was::Parent(referent, old, order));
	}
}

/// A playback ready to apply to one document: targets resolved, events at their frames. Only the
/// moment drawn and the frames up to it depend on the time: [`Resolved::at`] moves it to another
/// moment of the same document (a clip's frames), which resolving again gave exactly (a barrage
/// recording's 6,021 actions took 42 ms a frame to resolve).
pub struct Resolved<'a> {
	playback: &'a Playback,
	/// The moment drawn, seconds after the start
	time: f32,
	/// Every frame up to `latest`: the times events wait for
	grid: Vec<f32>,
	/// The last time `grid` covers: ten seconds past the last event, or the moment drawn after that
	latest: f32,
	tracks: Vec<Track>,
	/// Per instance, the nearest moving target at or above it (index into `tracks`).
	moving: parking_lot::RwLock<std::collections::HashMap<Ref, Option<usize>>>,
	/// What the tracks move, found at the first moment's state (every moment starts from the file's:
	/// [`Undo::restore`]), so later moments need not walk the document again, and `moving` as that walk
	/// left it (what a moment's history then reads, as when each moment walked)
	movers: std::sync::OnceLock<(Vec<Mover>, std::collections::HashMap<Ref, Option<usize>>)>,
	timelines: std::collections::HashMap<Ref, Timeline>,
	frames: Vec<f32>,
	/// Every property set, switched or tweened.
	channels: Vec<Channel<'a>>,
	/// Every instance parented elsewhere: (time, parent; none out of the world), in time order.
	parents: Vec<(Ref, Vec<(f32, Ref)>)>,
	/// Every instance reference set (a beam given its attachments after it appeared): (instance,
	/// property, (time, the instance it names; none for null) in time order).
	refs: Vec<(Ref, &'a str, Vec<(f32, Ref)>)>,
	/// The forced switches of every instance they reach ([`Action::force`]): (time, on), in time order.
	forced: std::collections::HashMap<Ref, Vec<(f32, bool)>>,
	/// Each Rate emitter's scan as it stood before the first frame whose births could still live at
	/// the moment drawn ([`Resolved::apply_history`]), for a later moment to go on from
	scans: parking_lot::Mutex<std::collections::HashMap<Ref, RateScan>>,
	/// Each emitter's view by the playback's viewer over the grid ([`Resolved::watched`]): from each
	/// time on, 1 while that camera could draw it, else 0
	views: parking_lot::Mutex<std::collections::HashMap<Ref, std::sync::Arc<Vec<(f32, f32)>>>>,
}

/// An instance a track moves ([`Resolved::apply_state`]): a part (its CFrame), a model (its pivot) or a
/// moving attachment (placed under its parent), with its nearest moving target.
#[derive(Clone, Copy)]
enum Mover {
	Part(Ref, usize),
	Model(Ref, usize),
	Attachment(Ref, usize),
}

/// An emitter's Rate accumulator before frame `index` of the grid, made at the moment `time` for this
/// `rate` and longest lifetime. Every frame before it had births already gone at that moment, so at any
/// later one too (particle time only grows): a later moment scans on from here and makes the births it
/// would from the start. A clip from 10 s replayed its emitters' 600 frames before each frame drawn.
#[derive(Clone, Copy)]
struct RateScan {
	time: f32,
	rate: f32,
	longest: f32,
	index: usize,
	accumulated: f32,
	deferred: f32,
	on: bool,
	previous: Option<f32>,
	stream: u64,
}

impl Playback {
	/// No motion, tweens, property sets, switching or parenting at any moment: only emits (and
	/// Clear, TimeScale), which the extracted scene's history carries ([`Resolved::apply_history`]).
	pub fn changes_nothing(&self) -> bool {
		self.motion.is_empty() && self.tweens.is_empty() && self.actions.iter().all(|action| action.set.is_empty() && action.enabled.is_none() && action.parent.is_none())
	}

	/// Whether it names the Lighting or the Terrain's Clouds (a target or a new parent under either): a
	/// model file is then given them for it to change ([`super::load::Document::give_lighting`]).
	pub fn touches_lighting(&self) -> bool {
		let clouds = |segments: &[String]| segments.len() >= super::load::CLOUDS_PATH.len() && segments.iter().zip(super::load::CLOUDS_PATH).all(|(segment, name)| segment == name);
		let lighting = |path: &str| super::load::split_path(path).is_ok_and(|segments| segments.first().is_some_and(|first| first == "Lighting") || clouds(&segments));
		self.actions.iter().any(|action| lighting(&action.target) || action.parent.as_deref().is_some_and(lighting)) || self.tweens.iter().any(|tween| lighting(&tween.target))
	}

	/// Resolves targets and events against `document` (before [`Playback::apply_state`]).
	pub fn resolve<'a>(&'a self, document: &Document) -> Result<Resolved<'a>> {
		self.resolve_at(document, self.time)
	}

	/// The last time a moment at `time` waits for frames up to: events up to some time after the
	/// moment drawn wait for frames too.
	fn latest(&self, time: f32) -> f32 {
		self.actions.iter().map(|action| action.at).chain(self.tweens.iter().map(|tween| tween.at)).map(|at| at + 10.0).fold(time, f32::max)
	}

	/// [`Playback::resolve`] for the moment `time` (whatever `self.time` holds).
	pub fn resolve_at<'a>(&'a self, document: &Document, time: f32) -> Result<Resolved<'a>> {
		let latest = self.latest(time);
		let paths = Paths::new(document);
		let grid = frame_times(self.frames.as_deref(), latest, self.prewarm);
		let mut tracks = Vec::new();
		for motion in &self.motion {
			for referent in paths.get(&motion.target)? {
				if let Some(tag) = &motion.tagged {
					if !document.dom.get_by_ref(referent).is_some_and(|instance| super::follow::has_tag(instance, tag)) {
						continue;
					}
				}
				let rest = rest_cframe(document, referent).with_context(|| format!("playback: {} has no CFrame to move", motion.target))?;
				let mut keys = Vec::new();
				for key in &motion.keys {
					let pose = if motion.relative { rest * affine(&key.cframe, Affine3A::IDENTITY)? } else { affine(&key.cframe, rest)? };
					keys.push((key.at, pose, key.easing.clone(), key.direction.clone()));
				}
				keys.sort_by(|a, b| a.0.total_cmp(&b.0));
				tracks.push(Track { referent, rest, keys, physics_from: motion.physics_from });
			}
		}
		// every emitter, trail and legacy effect's events, from the file's state and the actions
		let mut timelines: std::collections::HashMap<Ref, Timeline> = std::collections::HashMap::new();
		for instance in document.dom.descendants() {
			let class = instance.class.as_str();
			if matches!(class, "ParticleEmitter" | "Trail" | "Beam" | "Fire" | "Smoke" | "Sparkles" | "ForceField" | "Explosion") {
				let timeline = timelines.entry(instance.referent()).or_default();
				// an Explosion is not switched on: it goes off (below); a ForceField shows while Visible
				let on = match class {
					"Explosion" => false,
					"ForceField" => props::bool(instance, "Visible", true),
					_ => props::bool(instance, "Enabled", true),
				};
				if on {
					timeline.enabled.push((-self.prewarm, true));
				}
				if timed(class) {
					timeline.time_scale.push((-self.prewarm, props::f32(instance, "TimeScale", 1.0)));
					timeline.spawn_scale.push((-self.prewarm, props::f32(instance, "TimeScale", 1.0)));
				}
			}
		}
		for action in &self.actions {
			let at = quantize(&grid, action.at);
			let by_attributes = match &action.emit {
				Some(Emit::Mode(mode)) => Some(ByAttributes::parse(mode)?),
				_ => None,
			};
			// where a count's burst leaves from, when the action says
			let from = match &action.cframe {
				Some(_) if !matches!(action.emit, Some(Emit::Count(_))) => bail!("playback: {}: a cframe goes with an emit count", action.target),
				Some(components) if components.len() != 12 => bail!("playback: {}: an emit's cframe has 12 numbers, not {}", action.target, components.len()),
				Some(components) => Some(affine(components, Affine3A::IDENTITY)?),
				None => None,
			};
			let part_size = action.part_size.map(Vec3::from);
			for target in paths.get(&action.target)? {
				for referent in descendants(document, target) {
					let Some(instance) = document.dom.get_by_ref(referent) else {
						continue;
					};
					if action.class.as_deref().is_some_and(|class| !props::is_a(instance, class)) {
						continue;
					}
					let Some(timeline) = timelines.get_mut(&referent) else {
						continue;
					};
					if let Some(enabled) = action.enabled
						&& switchable(instance.class.as_str())
					{
						timeline.enabled.push((at, enabled));
					}
					if let Some(scale) = action.time_scale
						&& timed(instance.class.as_str())
					{
						timeline.time_scale.push((quantize_step(&grid, action.at), scale));
						timeline.spawn_scale.push((at, scale));
					}
					// an Explosion goes off at each emit (a copy of it parented then)
					if instance.class == "Explosion" && action.emit.is_some() {
						timeline.emits.push(Burst::at(at, 1));
					}
					if instance.class == "ParticleEmitter" {
						match (&action.emit, by_attributes) {
							(Some(Emit::Count(count)), _) => timeline.emits.push(Burst { at, count: *count, from, part_size }),
							(_, Some(by_attributes)) => by_attributes.schedule(instance, &grid, action.at, timeline),
							_ => {}
						}
					}
					if action.clear {
						timeline.clears.push(at);
					}
				}
			}
		}
		// an Explosion nothing emits goes off when the effect is placed in the world, at 0
		for instance in document.dom.descendants() {
			if instance.class == "Explosion"
				&& let Some(timeline) = timelines.get_mut(&instance.referent())
				&& timeline.emits.is_empty()
			{
				timeline.emits.push(Burst::at(quantize(&grid, 0.0), 1));
			}
		}
		// forced switches win over the playback's own
		let forced = self.forced_switches(document, &paths, &grid)?;
		for (referent, switches) in &forced {
			if let Some(timeline) = timelines.get_mut(referent) {
				timeline.force(switches);
			}
		}
		let timelines = timelines.into_iter().map(|(referent, timeline)| (referent, timeline.sorted())).collect();
		// every property set, switched or tweened, and every instance parented elsewhere, their events
		// in time order (an action's before a tween's at the same time)
		let mut channels: Vec<Channel<'a>> = Vec::new();
		let mut known: std::collections::HashMap<(Ref, &'a str), usize> = std::collections::HashMap::new();
		let mut channel = |channels: &mut Vec<Channel<'a>>, referent: Ref, property: &'a str| -> usize {
			// a content ID and the Content property that replaced it are one value in Roblox (`Image`
			// sets ImageContent): one channel under the Content name, which files hold and ror draws
			let names = document.dom.get_by_ref(referent).and_then(|instance| super::load::content_names(instance.class.as_str(), property));
			let property = names.map_or(property, |names| names.content);
			*known.entry((referent, property)).or_insert_with(|| {
				let initial = document.dom.get_by_ref(referent).and_then(|instance| authored(instance, property));
				channels.push(Channel { referent, property, alias: names.map(|names| names.legacy), initial, events: Vec::new(), keyframes: None });
				channels.len() - 1
			})
		};
		let mut parents: Vec<(Ref, Vec<(f32, Ref)>)> = Vec::new();
		let mut refs: Vec<(Ref, &'a str, Vec<(f32, Ref)>)> = Vec::new();
		for action in &self.actions {
			let at = quantize(&grid, action.at);
			let targets = paths.get(&action.target)?;
			let own = if action.set.is_empty() && action.parent.is_none() { Vec::new() } else { of_class(document, targets.clone(), action.class.as_deref()) };
			for &referent in &own {
				for (name, value) in &action.set {
					// an instance reference names an instance by its path (null: none), as a beam is given
					// its attachments after it appeared
					if document.dom.get_by_ref(referent).is_some_and(|instance| props::is_reference(instance, name)) {
						let named = match value {
							serde_json::Value::Null => Ref::none(),
							serde_json::Value::String(path) if path.trim().is_empty() || path.trim() == "nil" => Ref::none(),
							serde_json::Value::String(path) => paths.get(path)?.first().copied().with_context(|| format!("playback: {}.{name}: no instance {path:?}", path_of(document, referent)))?,
							other => bail!("playback: {}.{name} is an instance reference: the path of the instance it names, or null for none, not {}", path_of(document, referent), shown(other)),
						};
						match refs.iter_mut().find(|(set, property, _)| *set == referent && *property == name.as_str()) {
							Some((_, _, events)) => events.push((at, named)),
							None => refs.push((referent, name.as_str(), vec![(at, named)])),
						}
						continue;
					}
					let index = channel(&mut channels, referent, name);
					channels[index].events.push((at, Event::Set(value.clone())));
				}
			}
			if let Some(path) = &action.parent {
				let parent = paths.parent(path)?;
				for &referent in &own {
					if parent.is_some() && descendants(document, referent).contains(&parent) {
						bail!("playback: {} cannot go under {path:?}, which is in it", path_of(document, referent));
					}
					match parents.iter_mut().find(|(moved, _)| *moved == referent) {
						Some((_, events)) => events.push((at, parent)),
						None => parents.push((referent, vec![(at, parent)])),
					}
				}
			}
			if let Some(enabled) = action.enabled {
				for &target in &targets {
					for referent in descendants(document, target) {
						if document.dom.get_by_ref(referent).is_some_and(|instance| switches(action, instance, referent, &targets)) {
							let index = channel(&mut channels, referent, "Enabled");
							channels[index].events.push((at, Event::Set(serde_json::Value::Bool(enabled))));
						}
					}
				}
			}
		}
		for tween in &self.tweens {
			if tween.keys.is_empty() && tween.to.is_none() {
				bail!("playback: tween {}.{} has neither keys nor to", tween.target, tween.property);
			}
			let start = quantize(&grid, tween.at) + tween.delay.max(0.0);
			for referent in of_class(document, paths.get(&tween.target)?, tween.class.as_deref()) {
				let index = channel(&mut channels, referent, &tween.property);
				if tween.keys.is_empty() {
					channels[index].events.push((start, Event::Tween(tween)));
				} else {
					channels[index].keyframes = Some(tween);
				}
			}
		}
		for channel in &mut channels {
			channel.events.sort_by(|a, b| a.0.total_cmp(&b.0));
		}
		for (_, events) in &mut parents {
			events.sort_by(|a, b| a.0.total_cmp(&b.0));
		}
		for (_, _, events) in &mut refs {
			events.sort_by(|a, b| a.0.total_cmp(&b.0));
		}
		let frames = grid.iter().copied().filter(|&frame| frame <= time + 1e-4).collect();
		Ok(Resolved { playback: self, time, grid, latest, tracks, moving: Default::default(), movers: Default::default(), timelines, frames, channels, parents, refs, forced, scans: Default::default(), views: Default::default() })
	}

	/// The forced switches ([`Action::force`]) of every switchable instance they reach: at or under
	/// the target in the file, and at or under whatever the playback parents under what they reach.
	fn forced_switches(&self, document: &Document, paths: &Paths<'_>, grid: &[f32]) -> Result<std::collections::HashMap<Ref, Vec<(f32, bool)>>> {
		let mut forced: std::collections::HashMap<Ref, Vec<(f32, bool)>> = std::collections::HashMap::new();
		if !self.actions.iter().any(|action| action.force && action.enabled.is_some()) {
			return Ok(forced);
		}
		// (instance, new parent) of every parenting into the world
		let mut moves: Vec<(Ref, Ref)> = Vec::new();
		for action in &self.actions {
			let Some(path) = &action.parent else {
				continue;
			};
			let parent = paths.parent(path)?;
			if parent.is_some() {
				moves.extend(of_class(document, paths.get(&action.target)?, action.class.as_deref()).into_iter().map(|referent| (referent, parent)));
			}
		}
		for action in self.actions.iter().filter(|action| action.force) {
			let Some(on) = action.enabled else {
				continue;
			};
			let at = quantize(grid, action.at);
			let targets = paths.get(&action.target)?;
			let mut reached: std::collections::HashSet<Ref> = std::collections::HashSet::new();
			for &target in &targets {
				reached.extend(descendants(document, target));
			}
			loop {
				let before = reached.len();
				for &(moved, parent) in &moves {
					if reached.contains(&parent) && !reached.contains(&moved) {
						reached.extend(descendants(document, moved));
					}
				}
				if reached.len() == before {
					break;
				}
			}
			for referent in reached {
				if document.dom.get_by_ref(referent).is_some_and(|instance| switches(action, instance, referent, &targets)) {
					forced.entry(referent).or_default().push((at, on));
				}
			}
		}
		for switches in forced.values_mut() {
			switches.sort_by(|a, b| a.0.total_cmp(&b.0));
		}
		Ok(forced)
	}
}

impl Resolved<'_> {
	/// Moves to the moment `time` of the same document, as resolving it there gives, when its frames
	/// are those this one waits for (false: resolve again).
	pub fn at(&mut self, time: f32) -> bool {
		let latest = self.playback.latest(time);
		if latest.to_bits() != self.latest.to_bits() {
			// with no action or tween nothing is timed by the grid: resolving again gives this but for
			// the grid's end (a map clip without a playback resolved its whole document every frame)
			if !self.playback.actions.is_empty() || !self.playback.tweens.is_empty() {
				return false;
			}
			self.grid = frame_times(self.playback.frames.as_deref(), latest, self.playback.prewarm);
			self.latest = latest;
			self.views.get_mut().clear();
		}
		self.time = time;
		self.frames = self.grid.iter().copied().filter(|&frame| frame <= time + 1e-4).collect();
		// what moves each instance follows the parents of the moment drawn
		self.moving.get_mut().clear();
		true
	}

	/// The nearest moving target at or above an instance.
	fn track_of(&self, document: &Document, referent: Ref) -> Option<usize> {
		if let Some(known) = self.moving.read().get(&referent) {
			return *known;
		}
		let found = match self.tracks.iter().position(|track| track.referent == referent) {
			Some(index) => Some(index),
			None => {
				let parent = document.dom.get_by_ref(referent).map(|instance| instance.parent()).filter(|parent| parent.is_some());
				parent.and_then(|parent| self.track_of(document, parent))
			}
		};
		self.moving.write().insert(referent, found);
		found
	}

	/// An emitter's view by the playback's viewer ([`Playback::viewer`]), none without one: from each
	/// frame of the grid's previous one on, 1 where the viewer's camera at that frame could draw the
	/// emitter's particles (a ball about where it emits as far as they reach), else 0. Worked out once
	/// per grid.
	fn watched(&self, referent: Ref, emitter: &super::particles::ParticleEmitter, frame_at: &dyn Fn(f32) -> Affine3A, size_at: &dyn Fn(f32) -> Vec3) -> Option<std::sync::Arc<Vec<(f32, f32)>>> {
		let viewer = &self.playback.viewer;
		if viewer.is_empty() {
			return None;
		}
		if let Some(view) = self.views.lock().get(&referent) {
			return Some(view.clone());
		}
		let mut view: Vec<(f32, f32)> = Vec::new();
		for (index, &frame) in self.grid.iter().enumerate() {
			let key = &viewer[viewer.partition_point(|key| key.at <= frame + 1e-4).saturating_sub(1)];
			let seen = if key.sees(Vec3::from(frame_at(frame).translation), reach(emitter, size_at(frame))) { 1.0 } else { 0.0 };
			let from = if index == 0 { f32::NEG_INFINITY } else { self.grid[index - 1] };
			if view.last().is_none_or(|&(_, last)| last != seen) {
				view.push((from, seen));
			}
		}
		let view = std::sync::Arc::new(view);
		self.views.lock().insert(referent, view.clone());
		Some(view)
	}

	/// An instance's world CFrame at `t`, from `now`, its world CFrame at the moment drawn: moved as its
	/// nearest moving target moves between then and `t`.
	pub fn world_at(&self, document: &Document, referent: Ref, now: Affine3A, t: f32) -> Affine3A {
		self.moved(self.track_of(document, referent), now, t)
	}

	/// [`Resolved::world_at`] of an instance whose nearest moving target is `track`.
	fn moved(&self, track: Option<usize>, now: Affine3A, t: f32) -> Affine3A {
		match track {
			Some(index) => {
				let track = &self.tracks[index];
				track.pose(t) * track.pose(self.time).inverse() * now
			}
			None => now,
		}
	}

	/// Sets the model's state at the moment drawn: every moving part's CFrame and every moving
	/// attachment's (relative to its parent), the actions' `set`s and Enabled, and the tweens. What it
	/// changed comes back to undo.
	pub fn apply_state(&self, document: &mut Document) -> Result<Undo> {
		let mut undo = Undo::default();
		let time = self.time;
		// motion: parts follow their nearest moving target, moving attachments are placed, and models'
		// pivots go along as PivotTo carries them (an attachment under a model is relative to its pivot:
		// an effect's pieces fly with the attachments under their models)
		let mut moves: Vec<(Ref, Affine3A)> = Vec::new();
		let mut pivots: Vec<(Ref, Affine3A)> = Vec::new();
		if !self.tracks.is_empty() {
			let mut first = false;
			let (movers, at_rest) = self.movers.get_or_init(|| {
				first = true;
				let mut movers = Vec::new();
				for instance in document.dom.descendants() {
					let referent = instance.referent();
					if props::is_a(instance, "BasePart") {
						movers.extend(self.track_of(document, referent).map(|index| Mover::Part(referent, index)));
					} else if props::is_a(instance, "Model") {
						movers.extend(self.track_of(document, referent).map(|index| Mover::Model(referent, index)));
					} else if props::is_a(instance, "Attachment") && self.tracks.iter().any(|track| track.referent == referent) {
						movers.push(Mover::Attachment(referent, self.track_of(document, referent).unwrap_or_default()));
					}
				}
				(movers, self.moving.read().clone())
			});
			// (a moment moved to with `at` starts from what the walk left, as the walk itself did)
			if !first {
				let mut moving = self.moving.write();
				for (&referent, &found) in at_rest {
					moving.insert(referent, found);
				}
			}
			for &mover in movers {
				// a track's own target goes where its keys put it; what hangs under it follows by the delta
				// (pose times rest inverse times rest: exact in theory, but a part parked far away, as the
				// game's part pools park theirs 9e9 studs out, loses its place to f32 rounding that way)
				match mover {
					Mover::Part(referent, index) => {
						if let Some(rest) = document.dom.get_by_ref(referent).and_then(|instance| props::cframe(instance, "CFrame")) {
							let track = &self.tracks[index];
							moves.push((referent, if track.referent == referent { track.pose(time) } else { track.delta(time) * rest }));
						}
					}
					Mover::Model(referent, index) => {
						if let Some(rest) = document.dom.get_by_ref(referent).and_then(|instance| props::cframe(instance, "WorldPivotData")) {
							let track = &self.tracks[index];
							pivots.push((referent, if track.referent == referent { track.pose(time) } else { track.delta(time) * rest }));
						}
					}
					Mover::Attachment(referent, index) => {
						// placed relative to its parent where the parent is then
						let parent = document.dom.get_by_ref(referent).map(|instance| instance.parent()).unwrap_or_else(Ref::none);
						let parent_now = match (rest_cframe(document, parent), self.track_of(document, parent)) {
							(Some(rest), Some(moving)) => self.tracks[moving].delta(time) * rest,
							(Some(rest), None) => rest,
							_ => Affine3A::IDENTITY,
						};
						moves.push((referent, parent_now.inverse() * self.tracks[index].pose(time)));
					}
				}
			}
		}
		for (referent, cframe) in moves {
			undo.set(document, referent, "CFrame", crate::scene::build::cframe(cframe).into());
		}
		for (referent, pivot) in pivots {
			undo.set(document, referent, "WorldPivotData", Variant::OptionalCFrame(Some(crate::scene::build::cframe(pivot))));
		}
		// every property as its sets, switches and tweens leave it by then
		for channel in &self.channels {
			let Some(value) = channel.at(time) else {
				continue;
			};
			let existing = document.dom.get_by_ref(channel.referent).and_then(|instance| props::get(instance, channel.property).cloned());
			let converted = variant(&value, existing.as_ref()).with_context(|| format!("playback: {}.{}", path_of(document, channel.referent), channel.property))?;
			// (a relative local path set is read where the document's are)
			let converted = document.local_content(converted);
			undo.set(document, channel.referent, channel.property, converted);
			// the content ID's own name too where the instance holds it, which would draw otherwise
			if let Some(alias) = channel.alias
				&& let Some(old) = document.dom.get_by_ref(channel.referent).and_then(|instance| instance.properties.get(&rbx_dom_weak::ustr(alias)).cloned())
			{
				let converted = variant(&value, Some(&old)).with_context(|| format!("playback: {}.{alias}", path_of(document, channel.referent)))?;
				let converted = document.local_content(converted);
				undo.set(document, channel.referent, alias, converted);
			}
		}
		// and every instance under the parent it has by then (a post effect in the Lighting)
		for (referent, events) in &self.parents {
			if let Some(&(_, parent)) = events.iter().take_while(|(at, _)| *at <= time + 1e-6).last() {
				undo.reparent(document, *referent, parent);
			}
		}
		// every instance reference as its sets leave it (a beam's attachments)
		for (referent, property, events) in &self.refs {
			if let Some(&(_, named)) = events.iter().take_while(|(at, _)| *at <= time + 1e-6).last() {
				undo.set(document, *referent, property, Variant::Ref(named));
			}
		}
		// forced switches over whatever switched, set or tweened Enabled
		for (referent, switches) in &self.forced {
			if let Some(&(_, on)) = switches.iter().take_while(|(at, _)| *at <= time + 1e-6).last() {
				undo.set(document, *referent, "Enabled", Variant::Bool(on));
			}
		}
		Ok(undo)
	}

	/// Every instance whose state it changes at some moment: the moving targets (what hangs under
	/// them moves with them), every property's instance, every instance parented elsewhere and each
	/// of its new parents, every forced switch's instance, every instance whose references it sets.
	/// What it leaves alone is as in the file at every moment (video.rs: a root none of these is at,
	/// under or above is extracted once).
	pub fn touched(&self) -> Vec<Ref> {
		let mut touched: Vec<Ref> = self.tracks.iter().map(|track| track.referent).collect();
		touched.extend(self.channels.iter().map(|channel| channel.referent));
		for (referent, events) in &self.parents {
			touched.push(*referent);
			touched.extend(events.iter().map(|(_, parent)| *parent).filter(|parent| parent.is_some()));
		}
		touched.extend(self.forced.keys().copied());
		touched.extend(self.refs.iter().map(|(referent, ..)| *referent));
		touched
	}

	/// The channel of one property of one instance, if anything changes it.
	fn channel(&self, referent: Ref, property: &str) -> Option<&Channel<'_>> {
		self.channels.iter().find(|channel| channel.referent == referent && channel.property == property)
	}

	/// Whether all it changes over time is the Lighting, what is in it, the Terrain's Clouds, and the
	/// post effects, Skies and Atmospheres it moves (which draw nothing in the world): what is drawn stays
	/// as in the file, and each moment needs only its Lighting read again (video.rs).
	pub fn changes_only_lighting(&self, document: &Document) -> bool {
		let clouds = document.clouds();
		let lit = |referent: Ref| -> bool {
			if document.dom.get_by_ref(referent).is_some_and(|instance| environment_class(instance.class.as_str())) {
				return true;
			}
			let mut current = referent;
			while let Some(instance) = document.dom.get_by_ref(current) {
				if Some(current) == document.lighting || Some(current) == clouds {
					return true;
				}
				current = instance.parent();
			}
			false
		};
		self.tracks.is_empty() && self.channels.iter().all(|channel| lit(channel.referent)) && self.parents.iter().all(|(referent, _)| lit(*referent)) && self.refs.iter().all(|(referent, ..)| lit(*referent))
	}

	/// Fills the extracted scene's history: each trail's points and each emitter's particles (when
	/// and where it emitted them). `document` is the model at the moment drawn (after `apply_state`),
	/// the scene extracted from it.
	pub fn apply_history(&self, document: &Document, scene: &mut Scene) {
		let time = self.time;
		// the model is placed at the start, its lights with it
		for light in &mut scene.lights {
			light.placed = true;
		}
		for trail in &mut scene.trails {
			let timeline = self.timelines.get(&trail.referent).cloned().unwrap_or_default();
			let sample = |t: f32| -> TrailPoint {
				TrailPoint {
					time: t,
					position0: Vec3::from(self.world_at(document, trail.attachment0, trail.cframe0, t).translation),
					position1: Vec3::from(self.world_at(document, trail.attachment1, trail.cframe1, t).translation),
					starts_strip: false,
				}
			};
			let middle = |point: &TrailPoint| (point.position0 + point.position1) * 0.5;
			let mut points: Vec<TrailPoint> = Vec::new();
			let mut previous = f32::NEG_INFINITY;
			// the next point starts a strip of its own (after the trail was disabled: trail-shape gap)
			let mut fresh = true;
			for &frame in &self.frames {
				// Clear() drops what the trail left so far
				if timeline.clears.iter().any(|&clear| clear > previous && clear <= frame + 1e-6) {
					points.clear();
				}
				previous = frame;
				if !timeline.enabled_at(frame) {
					fresh = true;
					continue;
				}
				let mut point = sample(frame);
				// a new point once the attachments moved MinLength from the last one (their midpoint;
				// trail-shape minlength: the circle's corners match)
				if !fresh
					&& let Some(last) = points.last()
					&& middle(&point).distance(middle(last)) < trail.min_length.max(1e-4)
				{
					continue;
				}
				point.starts_strip = fresh;
				fresh = false;
				points.push(point);
			}
			// the head follows the attachments while enabled
			if timeline.enabled_at(time)
				&& let Some(last) = points.last()
				&& last.time < time - 1e-6
			{
				let head = sample(time);
				if middle(&head).distance(middle(last)) > 1e-4 {
					points.push(head);
				}
			}
			// points older than Lifetime are gone
			let lifetime = trail.lifetime.max(0.0);
			let mut dropped = 0.0;
			while points.len() > 1 && time - points[0].time > lifetime {
				let gone = points.remove(0);
				if !points[0].starts_strip {
					dropped += middle(&gone).distance(middle(&points[0]));
				}
				points[0].starts_strip = true;
			}
			if points.len() == 1 && time - points[0].time > lifetime {
				points.clear();
			}
			// MaxLength keeps the newest points within that many studs of the head, whole points
			// (trail-shape maxlength: 3 studs of 0.267-stud steps leave 11 of them, 2.935 studs)
			if trail.max_length > 0.0 {
				let mut length = 0.0;
				for index in (1..points.len()).rev() {
					if points[index].starts_strip {
						continue;
					}
					let step = middle(&points[index]).distance(middle(&points[index - 1]));
					if length + step > trail.max_length + 1e-4 {
						dropped += points[..index].windows(2).filter(|pair| !pair[1].starts_strip).map(|pair| middle(&pair[0]).distance(middle(&pair[1]))).sum::<f32>() + step;
						points.drain(..index);
						points[0].starts_strip = true;
						break;
					}
					length += step;
				}
			}
			trail.points = points;
			trail.time = time;
			trail.dropped_length = dropped;
		}
		// each emitter's particles on every core: what one emits depends on no other
		let play = |emitter: &mut super::particles::ParticleEmitter| {
			let Some(referent) = emitter.referent else {
				return;
			};
			let timeline = self.timelines.get(&referent).cloned().unwrap_or_default();
			// a legacy effect's system follows its mount (an Explosion's stays where it went off)
			let parent = emitter.mount.unwrap_or_else(|| document.dom.get_by_ref(referent).map(|instance| instance.parent()).unwrap_or_else(Ref::none));
			// an emitter outside a part or an attachment stays at the world's origin whatever moves
			// (vfx-model-emitter)
			let orphan = emitter.mount.is_none() && document.dom.get_by_ref(parent).is_none_or(|instance| !props::is_a(instance, "BasePart") && !props::is_a(instance, "Attachment"));
			let fixed = emitter.mount.is_some_and(|mount| mount.is_none()) || orphan;
			let parent_now = if fixed {
				emitter.cframe
			} else {
				let Some(frame) = rest_cframe(document, parent) else {
					return;
				};
				frame
			};
			let track = if fixed { None } else { self.track_of(document, parent) };
			let frame_at = |t: f32| self.moved(track, parent_now, t);
			// LockedToPart particles stay behind once physics moves the part (Motion::physics_from)
			let physics = track.and_then(|index| self.tracks[index].physics_from).filter(|&from| emitter.locked_to_part && time > from);
			let follow_of = |born: f32| physics.map(|from| frame_at(born.max(from)));
			// a part emitter spawns over the part's size at the time (an attack run: a burst at 0 from
			// parts tweened from 0.001 to 20 studs leaves from their centre); a legacy effect's own size
			let size_channel = if emitter.mount.is_none() { self.channel(parent, "Size") } else { None };
			let size_at = |t: f32| -> Vec3 {
				match size_channel.and_then(|channel| channel.at(t)) {
					Some(serde_json::Value::Array(values)) if values.len() == 3 => Vec3::new(values[0].as_f64().unwrap_or_default() as f32, values[1].as_f64().unwrap_or_default() as f32, values[2].as_f64().unwrap_or_default() as f32),
					_ => emitter.size,
				}
			};
			let velocity_at = |t: f32| (Vec3::from(frame_at(t + 1.0 / 120.0).translation) - Vec3::from(frame_at(t - 1.0 / 120.0).translation)) * 60.0;
			// out of the viewer's view the emitter's clock stands still: it neither emits nor ages
			let timeline = match self.watched(referent, emitter, &frame_at, &size_at) {
				Some(view) => Timeline { time_scale: masked(&timeline.time_scale, &view), spawn_scale: masked(&timeline.spawn_scale, &view), ..timeline },
				None => timeline,
			};
			let start = -self.playback.prewarm;
			let now = timeline.particle_time(start, time);
			let spawn_now = timeline.spawn_time(start, time);
			let cleared = timeline.last_clear(time).unwrap_or(f32::NEG_INFINITY);
			let mut births = Vec::new();
			for (burst, emitted) in timeline.emits.iter().enumerate() {
				let (at, count) = (emitted.at, emitted.count);
				if at > time + 1e-6 || at < cleared {
					continue;
				}
				let age = now - timeline.particle_time(start, at);
				let lived = spawn_now - timeline.spawn_time(start, at);
				// where the emitter was at the call when the playback says (several bursts of one frame from
				// an attachment moved between them), else where its part or attachment is then
				let cframe = emitted.from.unwrap_or_else(|| frame_at(at));
				let (velocity, size) = (velocity_at(at), emitted.part_size.unwrap_or_else(|| size_at(at)));
				for index in 0..count * emitter.emit_count.unwrap_or(1) {
					births.push(Birth { age, lived, cframe, velocity, stream: (1 << 40) + ((burst as u64) << 20) + index as u64, size, follow: follow_of(at) });
				}
			}
			// Rate: Studio's per-frame accumulator, in step across emitters (particle-emission rate,
			// particle-moving stream: the particles sit where the emitter was at frames). Each frame's
			// step spawns with the state of the frame before: an emitter enabled in a frame adds Rate times
			// each later step's particle time and emits a particle at that frame for every whole one, the
			// frame it is disabled in still counts, and a TimeScale set in a frame spawns at the old one
			// for that step (particle-rate-edges: Rate 4 enabled at 0.37 s emits in the frame of its
			// disabling at 1.62 s; frozen emitters still spawn in the freezing frame). A step counts up to
			// 1/60 s in its own frame and the rest with the next frame (particle-rate-start: Rates 2-18
			// from the first frame and from 0.3 s, 22 ms frames: Studio's particles a frame younger where a
			// long frame crossed a whole one by less than its excess over 1/60 s; 155 of 166 ages, 109
			// before). That rest is the frame's, not the emitter's: an emitter enabled in a long frame
			// counts the frame's excess with the next one too (particle-rate-fast: Rates 4-65 enabled at 0.2
			// s in three captures; their particles' ages fit a head start of the enabling frame's excess
			// over 1/60 s, 0 / 5.1 / 7.0 ms there, 102 of 112 against 90 without it). The first frame's
			// own step is not recorded, so emitters enabled in it (from the file, or by an action at 0)
			// start without one: Studio's there fit head starts of 8, 0 and 21 ms in the three captures
			if emitter.rate > 0.0 {
				let (mut accumulated, mut deferred, mut on, mut previous, mut stream) = (0.0f32, 0.0f32, false, None, 0u64);
				// births already dead at the moment drawn take no room under the cap: counted from 0, Rate
				// 400 filled it with particles long gone by 10.2 s and the emitter stopped (map
				// clips from 10 s: a map's Specks thinned to none, 4,718 particles to 1,464)
				let longest = emitter.lifetime.0.max(emitter.lifetime.1);
				// on from where an earlier moment's scan stood, when that holds for this one (RateScan)
				let resume = self.scans.lock().get(&referent).copied().filter(|scan| scan.time <= time && scan.rate.to_bits() == emitter.rate.to_bits() && scan.longest.to_bits() == longest.to_bits() && scan.index <= self.frames.len());
				let first = match resume {
					Some(scan) => {
						(accumulated, deferred, on, previous, stream) = (scan.accumulated, scan.deferred, scan.on, scan.previous, scan.stream);
						scan.index
					}
					None => 0,
				};
				let mut saved = None;
				for (index, &frame) in self.frames.iter().enumerate().skip(first) {
					if saved.is_none() && now - timeline.particle_time(start, frame) <= longest {
						saved = Some(RateScan { time, rate: emitter.rate, longest, index, accumulated, deferred, on, previous, stream });
					}
					let enabled = timeline.enabled_at(frame);
					if on && let Some(previous) = previous {
						let step = timeline.spawn_time(start, frame) - timeline.spawn_time(start, previous);
						let counted = step * (EMISSION_STEP / (frame - previous).max(1e-6)).min(1.0);
						accumulated += emitter.rate * (deferred + counted);
						deferred = step - counted;
						while accumulated >= 1.0 {
							accumulated -= 1.0;
							stream += 1;
							if frame >= cleared && now - timeline.particle_time(start, frame) <= longest && births.len() < 4096 {
								births.push(Birth {
									age: now - timeline.particle_time(start, frame),
									lived: spawn_now - timeline.spawn_time(start, frame),
									cframe: frame_at(frame),
									velocity: velocity_at(frame),
									stream,
									size: size_at(frame),
									follow: follow_of(frame),
								});
							}
						}
					}
					if enabled && !on {
						accumulated = 0.0;
						// the enabling frame's own excess over 1/60 s, counted with the next frame
						deferred = match previous {
							Some(previous) => {
								let step = timeline.spawn_time(start, frame) - timeline.spawn_time(start, previous);
								step - step * (EMISSION_STEP / (frame - previous).max(1e-6)).min(1.0)
							}
							None => 0.0,
						};
					}
					on = enabled;
					previous = Some(frame);
				}
				let saved = saved.unwrap_or(RateScan { time, rate: emitter.rate, longest, index: self.frames.len(), accumulated, deferred, on, previous, stream });
				self.scans.lock().insert(referent, saved);
			}
			births.retain(|birth| birth.age >= 0.0);
			emitter.cframe = parent_now;
			emitter.births = Some(births);
		};
		if crate::par::spread(scene.particles.len(), crate::par::CLIP_EMITTERS) {
			scene.particles.par_iter_mut().for_each(play);
		} else {
			scene.particles.iter_mut().for_each(play);
		}
		// beams: TextureSpeed scrolls the texture frame by frame, each frame by its step times the speed
		// set in that frame, from the frame the beam is enabled in; a disabled beam starts again at 0
		// (beam-scroll: within 0.003 of a repeat in every mode, late, toggle and tween)
		for beam in &mut scene.beams {
			let timeline = self.timelines.get(&beam.referent).cloned().unwrap_or_default();
			let speed_channel = self.channel(beam.referent, "TextureSpeed");
			let speed = beam.texture_speed;
			let speed_at = |t: f32| -> f32 {
				match speed_channel.and_then(|channel| channel.at(t)) {
					Some(serde_json::Value::Number(value)) => value.as_f64().unwrap_or_default() as f32,
					_ => speed,
				}
			};
			let mut offset = 0.0f64;
			let mut previous: Option<f32> = None;
			for &frame in self.frames.iter().chain(std::iter::once(&time)) {
				if !timeline.enabled_at(frame) {
					offset = 0.0;
					previous = None;
					continue;
				}
				if let Some(previous) = previous
					&& frame > previous
				{
					offset += (speed_at(frame) * (frame - previous)) as f64;
				}
				previous = Some(frame);
			}
			beam.texture_offset = offset.rem_euclid(1.0) as f32;
		}
	}
}

/// The command line's forms of playback events (`ror render` and `ror video` --set, --tween,
/// --parent). `<path>.<Property>` names a property: the instance path dotted from the file's root
/// (`Lighting.Grade`, `Lighting.*`, `Effect["A.B"]`) and the property last.
pub mod spec {
	use anyhow::{Context, Result, bail};

	use super::{Action, Tween};

	/// A value: true / false, a number, `#rrggbb` (a colour), numbers with commas (a vector, or a
	/// colour 0-1), a JSON array as a playback file writes one (`[0.3, 0, 0.3, 0]` a UDim2, `[[0, 1, 0],
	/// [1, 0, 0]]` a NumberSequence), `initial` (the property's value in the file), else text.
	pub fn value(text: &str) -> serde_json::Value {
		let text = text.trim();
		match text {
			"true" => return serde_json::json!(true),
			"false" => return serde_json::json!(false),
			"initial" => return serde_json::json!({"initial": true}),
			_ => {}
		}
		if let Ok(number) = text.parse::<f64>() {
			return serde_json::json!(number);
		}
		// a playback file's arrays, which the commas below would leave as text ("[0.3" is no number)
		if text.starts_with('[')
			&& let Ok(array @ serde_json::Value::Array(_)) = serde_json::from_str(text)
		{
			return array;
		}
		if let Some(hex) = text.strip_prefix('#')
			&& hex.len() == 6
			&& let Ok(rgb) = u32::from_str_radix(hex, 16)
		{
			return serde_json::json!([((rgb >> 16) & 255) as f64 / 255.0, ((rgb >> 8) & 255) as f64 / 255.0, (rgb & 255) as f64 / 255.0]);
		}
		match text.split(',').map(|part| part.trim().parse::<f64>().ok()).collect::<Option<Vec<f64>>>() {
			Some(numbers) if numbers.len() > 1 => serde_json::json!(numbers),
			_ => serde_json::json!(text),
		}
	}

	/// An instance path from its segments, quoting those that hold a dot.
	fn join_path(segments: &[String]) -> String {
		let mut path = String::new();
		for segment in segments {
			if segment.contains(['.', '[', ']']) {
				let quote = if segment.contains('"') { '\'' } else { '"' };
				path.push_str(&format!("[{quote}{segment}{quote}]"));
			} else {
				if !path.is_empty() {
					path.push('.');
				}
				path.push_str(segment);
			}
		}
		path
	}

	/// `<path>.<Property>[+|-]=<value>`: the instance path, the property and the value (with `+=` or
	/// `-=`, added to the property's value then).
	fn assignment(text: &str) -> Result<(String, String, serde_json::Value)> {
		let equals = text.find('=').with_context(|| format!("{text:?}: <path>.<Property>=<value>"))?;
		let (left, right) = (&text[..equals], &text[equals + 1..]);
		let (left, sign) = match left.strip_suffix('+') {
			Some(left) => (left, Some(1.0)),
			None => match left.strip_suffix('-') {
				Some(left) => (left, Some(-1.0)),
				None => (left, None),
			},
		};
		let mut segments = crate::scene::load::split_path(left)?;
		if segments.len() < 2 {
			bail!("{text:?}: name the instance and the property, <path>.<Property>");
		}
		let property = segments.pop().unwrap_or_default();
		let value = match sign {
			Some(sign) => serde_json::json!({ "add": scaled(&value(right), sign) }),
			None => value(right),
		};
		Ok((join_path(&segments), property, value))
	}

	fn scaled(value: &serde_json::Value, by: f64) -> serde_json::Value {
		match value {
			serde_json::Value::Number(number) => serde_json::json!(number.as_f64().unwrap_or_default() * by),
			serde_json::Value::Array(items) => serde_json::Value::Array(items.iter().map(|item| scaled(item, by)).collect()),
			other => other.clone(),
		}
	}

	/// `text`'s `@<seconds>` at its end, if it has one.
	fn at(text: &str) -> Result<(&str, f32)> {
		match text.rsplit_once('@') {
			Some((rest, seconds)) => Ok((rest, seconds.trim().parse().with_context(|| format!("{text:?}: the seconds after @"))?)),
			None => Ok((text, 0.0)),
		}
	}

	/// `--set <path>.<Property>[+|-]=<value>[@<seconds>]`.
	pub fn set(text: &str) -> Result<Action> {
		let (rest, seconds) = at(text)?;
		let (target, property, value) = assignment(rest)?;
		Ok(Action { at: seconds, target, set: [(property, value)].into(), ..Default::default() })
	}

	/// `--tween <path>.<Property>[+|-]=<value>[@<start>[+<duration>]][:<EasingStyle>][:<EasingDirection>]
	/// [:reverses][:repeat=<n>][:delay=<s>][:class=<Class>]`: TweenService from the property's value
	/// at the start (TweenInfo's 1 s, Quad, Out unless given).
	pub fn tween(text: &str) -> Result<Tween> {
		let mut tokens = text.split(':');
		let head = tokens.next().unwrap_or_default();
		let (rest, timing) = head.rsplit_once('@').unwrap_or((head, ""));
		let (target, property, to) = assignment(rest)?;
		let mut tween = Tween { target, property, to: Some(to), duration: 1.0, ..Default::default() };
		if !timing.is_empty() {
			let (start, duration) = timing.split_once('+').unwrap_or((timing, ""));
			tween.at = start.trim().parse().with_context(|| format!("{text:?}: the start after @"))?;
			if !duration.is_empty() {
				tween.duration = duration.trim().parse().with_context(|| format!("{text:?}: the duration after +"))?;
			}
		}
		const STYLES: [&str; 11] = ["Linear", "Sine", "Quad", "Cubic", "Quart", "Quint", "Exponential", "Circular", "Back", "Elastic", "Bounce"];
		for token in tokens.map(str::trim).filter(|token| !token.is_empty()) {
			let (key, setting) = token.split_once('=').unwrap_or((token, ""));
			match (key.to_ascii_lowercase().as_str(), setting) {
				("in", "") => tween.direction = Some("In".into()),
				("out", "") => tween.direction = Some("Out".into()),
				("inout", "") => tween.direction = Some("InOut".into()),
				("reverses", "") => tween.reverses = true,
				("repeat", count) => tween.repeat_count = count.trim().parse().with_context(|| format!("{text:?}: repeat=<times again, -1 for ever>"))?,
				("delay", seconds) => tween.delay = seconds.trim().parse().with_context(|| format!("{text:?}: delay=<seconds>"))?,
				("class", class) if !class.is_empty() => tween.class = Some(class.trim().to_string()),
				(style, "") if STYLES.iter().any(|name| name.eq_ignore_ascii_case(style)) => tween.easing = STYLES.iter().find(|name| name.eq_ignore_ascii_case(style)).map(|name| name.to_string()),
				_ => bail!("{text:?}: {token:?} is not an easing style ({}), a direction (In, Out, InOut), reverses, repeat=<n>, delay=<s> or class=<Class>", STYLES.join(", ")),
			}
		}
		Ok(tween)
	}

	/// `--parent <path>=<new parent path>[@<seconds>]` ("" or nil: out of the world).
	pub fn parent(text: &str) -> Result<Action> {
		let (rest, seconds) = at(text)?;
		let (target, parent) = rest.split_once('=').with_context(|| format!("{text:?}: <path>=<new parent path>[@<seconds>]"))?;
		Ok(Action { at: seconds, target: target.trim().to_string(), parent: Some(parent.trim().to_string()), ..Default::default() })
	}
}

#[cfg(test)]
mod moments {
	use super::*;

	/// A clip's frames drawn from one resolution moved from moment to moment ([`Resolved::at`]) are
	/// the scenes resolving each moment gave: motion, emits, switches, sets, tweens and parenting,
	/// past the last event's ten seconds too.
	#[test]
	fn moving_a_resolution_matches_resolving_again() {
		use crate::scene::build::{ModelBuilder, PartSpec, cframe};
		use rbx_dom_weak::InstanceBuilder;
		let mut model = ModelBuilder::new("Sim");
		let streak = model.part(&PartSpec { name: "Streak".into(), transparency: 1.0, ..PartSpec::default() });
		let a0 = model.child(streak, InstanceBuilder::new("Attachment").with_name("A0"));
		let a1 = model.child(streak, InstanceBuilder::new("Attachment").with_name("A1").with_property("CFrame", cframe(Affine3A::from_translation(Vec3::new(4.0, 0.0, 0.0)))));
		model.child(streak, InstanceBuilder::new("Beam").with_name("Ray").with_property("Enabled", false).with_property("Attachment0", Variant::Ref(a0)).with_property("Attachment1", Variant::Ref(a1)));
		model.child(a0, InstanceBuilder::new("ParticleEmitter").with_name("Sparks").with_property("Enabled", false).with_property("Rate", 20.0f32));
		let blade = model.part(&PartSpec { name: "Blade".into(), transparency: 0.5, ..PartSpec::default() });
		model.child(blade, InstanceBuilder::new("ParticleEmitter").with_name("Glint").with_property("Rate", 20.0f32));
		model.child(blade, InstanceBuilder::new("Trail").with_name("Wake"));
		let mut document = model.into_document();
		let moved = |x: f32| cframe_json(Affine3A::from_translation(Vec3::new(x, 1.0, 0.0))).as_array().unwrap().iter().map(|value| value.as_f64().unwrap() as f32).collect::<Vec<f32>>();
		let playback = Playback {
			actions: vec![
				spec::parent("Sim.Streak=nil@0").unwrap(),
				spec::parent("Sim.Streak=Sim@0.5").unwrap(),
				Action { at: 0.5, target: "Sim.Streak.Ray".into(), enabled: Some(true), ..Default::default() },
				Action { at: 0.6, target: "Sim.Streak.A0.Sparks".into(), enabled: Some(true), ..Default::default() },
				Action { at: 0.8, target: "Sim.Streak.A0.Sparks".into(), emit: Some(Emit::Count(12)), ..Default::default() },
				spec::set("Sim.Blade.Transparency=0.2@1").unwrap(),
				spec::parent("Sim.Blade=Sim.Streak@1.2").unwrap(),
				Action { at: 1.5, target: "Sim.Streak.A0.Sparks".into(), enabled: Some(false), ..Default::default() },
			],
			tweens: vec![spec::tween("Sim.Streak.Ray.Width0=3@0.5+1").unwrap()],
			motion: vec![Motion { target: "Sim.Streak".into(), keys: vec![MotionKey { at: 0.0, cframe: moved(0.0), easing: None, direction: None }, MotionKey { at: 2.0, cframe: moved(6.0), easing: None, direction: None }], relative: false, tagged: None, physics_from: None }],
			..Default::default()
		};
		let draw = |resolved: &Resolved, document: &mut Document| {
			let undo = resolved.apply_state(document).unwrap();
			let mut scene = crate::scene::extract::extract(document);
			resolved.apply_history(document, &mut scene);
			undo.restore(document);
			// in name order: a HashMap's own differs from map to map
			let lineage: std::collections::BTreeMap<_, _> = std::mem::take(&mut scene.lineage).into_iter().collect();
			format!("{scene:?} {lineage:?}")
		};
		let mut kept = playback.resolve_at(&document, 0.0).unwrap();
		let (mut moves, mut fresh_resolves) = (0, 0);
		for step in 0..=150 {
			let time = step as f32 / 12.0;
			let fresh = playback.resolve_at(&document, time).unwrap();
			let expected = draw(&fresh, &mut document);
			if kept.at(time) {
				moves += 1;
				assert_eq!(draw(&kept, &mut document), expected, "at {time}");
			} else {
				fresh_resolves += 1;
				kept = fresh;
			}
		}
		// up to 11.5 s (the last event at 1.5 s and ten seconds) one resolution serves; after, each
		// moment its own
		assert!(moves > 130 && fresh_resolves > 0, "{moves} moves, {fresh_resolves} resolutions");
	}
}

#[cfg(test)]
mod skewed_cframes {
	use super::*;

	/// A window camera's skewed CFrame tweened from key to key stays skewed: its numbers are lerped,
	/// where an upright CFrame's rotation is slerped (which squares the axes).
	#[test]
	fn a_skewed_cframe_tween_keeps_its_skew() {
		let a = serde_json::json!([1.0, 2.0, 3.0, -0.85, 0.0, 0.0, 0.0, 0.75, 0.0, -0.014, -0.11, -1.0]);
		let b = serde_json::json!([3.0, 2.0, 3.0, -0.71, 0.0, 0.0, 0.0, 0.65, 0.0, -0.005, -0.09, -1.0]);
		let numbers = |value: &serde_json::Value| -> Vec<f64> { value.as_array().unwrap().iter().map(|n| n.as_f64().unwrap()).collect() };
		assert_eq!(numbers(&lerp_value(&a, &b, 0.0)), numbers(&a));
		let half = numbers(&lerp_value(&a, &b, 0.5));
		assert!((half[3] + 0.78).abs() < 1e-6 && (half[7] - 0.7).abs() < 1e-6 && (half[0] - 2.0).abs() < 1e-6, "{half:?}");
		let upright = serde_json::json!([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
		assert!(!skewed(&affine(&numbers(&upright).iter().map(|&n| n as f32).collect::<Vec<_>>(), Affine3A::IDENTITY).unwrap()));
		assert!(skewed(&affine(&numbers(&a).iter().map(|&n| n as f32).collect::<Vec<_>>(), Affine3A::IDENTITY).unwrap()));
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn easing_matches_tweenservice() {
		assert_eq!(ease(0.5, None, None), 0.5);
		assert!((ease(0.5, Some("Quad"), Some("In")) - 0.25).abs() < 1e-6);
		assert!((ease(0.5, Some("Quad"), Some("Out")) - 0.75).abs() < 1e-6);
		assert!((ease(0.25, Some("Cubic"), Some("InOut")) - 0.0625).abs() < 1e-6);
		assert!((ease(1.0, Some("Elastic"), Some("Out")) - 1.0).abs() < 1e-6);
	}

	#[test]
	fn particle_time_follows_timescale() {
		let timeline = Timeline { time_scale: vec![(0.0, 1.0), (1.0, 0.5), (2.0, 0.0)], ..Default::default() };
		assert!((timeline.particle_time(0.0, 0.5) - 0.5).abs() < 1e-6);
		assert!((timeline.particle_time(0.0, 1.5) - 1.25).abs() < 1e-6);
		assert!((timeline.particle_time(0.0, 5.0) - 1.5).abs() < 1e-6);
	}

	#[test]
	fn sequences_and_ranges_play_back() {
		use rbx_dom_weak::types::{ColorSequence, NumberRange, NumberSequence};
		let existing: Variant = NumberSequence { keypoints: vec![] }.into();
		let Variant::NumberSequence(sequence) = variant(&serde_json::json!([[0, 1, 0], [1, 2, 0.5]]), Some(&existing)).unwrap() else {
			panic!("not a NumberSequence");
		};
		assert_eq!(sequence.keypoints.len(), 2);
		assert_eq!(sequence.keypoints[1].envelope, 0.5);
		let existing: Variant = ColorSequence { keypoints: vec![] }.into();
		let Variant::ColorSequence(colors) = variant(&serde_json::json!([[0, 1, 0, 0], [1, 0, 0, 1]]), Some(&existing)).unwrap() else {
			panic!("not a ColorSequence");
		};
		assert_eq!(colors.keypoints[1].color.b, 1.0);
		let existing: Variant = NumberRange::new(0.0, 0.0).into();
		assert_eq!(variant(&serde_json::json!([2, 3]), Some(&existing)).unwrap(), NumberRange::new(2.0, 3.0).into());
		// sequences of one shape tween key by key
		assert_eq!(lerp_value(&serde_json::json!([[0, 1, 0]]), &serde_json::json!([[0, 3, 0]]), 0.5), serde_json::json!([[0.0, 2.0, 0.0]]));
	}

	#[test]
	fn gui_values_play_back() {
		use rbx_dom_weak::types::{Rect, UDim, UDim2, Vector2};
		let existing: Variant = UDim2::new(UDim::new(0.0, 0), UDim::new(0.0, 0)).into();
		assert_eq!(variant(&serde_json::json!([0.5, 10, 1, -4.6]), Some(&existing)).unwrap(), UDim2::new(UDim::new(0.5, 10), UDim::new(1.0, -5)).into());
		let existing: Variant = UDim::new(0.0, 0).into();
		assert_eq!(variant(&serde_json::json!([1, 0]), Some(&existing)).unwrap(), UDim::new(1.0, 0).into());
		let existing: Variant = Rect::new(Vector2::new(0.0, 0.0), Vector2::new(0.0, 0.0)).into();
		assert_eq!(variant(&serde_json::json!([1, 2, 3, 4]), Some(&existing)).unwrap(), Rect::new(Vector2::new(1.0, 2.0), Vector2::new(3.0, 4.0)).into());
		// a UDim2 tweens component by component from its value in the file
		let dom = rbx_dom_weak::WeakDom::new(rbx_dom_weak::InstanceBuilder::new("ImageLabel").with_property("Size", UDim2::new(UDim::new(0.2, 4), UDim::new(0.2, 4))));
		assert_eq!(authored(dom.root(), "Size"), Some(serde_json::json!([0.2f32, 4, 0.2f32, 4])));
		assert_eq!(lerp_value(&serde_json::json!([0.0, 0, 0.0, 0]), &serde_json::json!([1.0, 10, 1.0, 10]), 0.5), serde_json::json!([0.5, 5.0, 0.5, 5.0]));
	}

	#[test]
	fn a_property_keeps_its_type() {
		use rbx_dom_weak::types::{CFrame, EnumItem, UDim, UDim2, Vector3};
		// text, a number, three numbers or a bool is no UDim2: an error naming the forms it takes, where
		// the text was stored before and the SurfaceGui holding it could not be written for rur
		let size: Variant = UDim2::new(UDim::new(0.0, 0), UDim::new(0.0, 0)).into();
		let error = |value: serde_json::Value, existing: &Variant| variant(&value, Some(existing)).err().map(|error| error.to_string()).unwrap_or_default();
		assert_eq!(error(serde_json::json!("[0.3,0,0.3"), &size), "\"[0.3,0,0.3\" is not a UDim2 ([xScale, xOffset, yScale, yOffset])");
		assert_eq!(error(serde_json::json!(0.3), &size), "0.3 is not a UDim2 ([xScale, xOffset, yScale, yOffset])");
		assert_eq!(error(serde_json::json!([0.3, 0, 0.3]), &size), "[0.3,0,0.3] is not a UDim2 ([xScale, xOffset, yScale, yOffset])");
		assert_eq!(error(serde_json::json!(true), &Variant::Float32(0.0)), "true is not a number");
		assert_eq!(error(serde_json::json!("Neon"), &rbx_dom_weak::types::Enum::from_u32(256).into()), "\"Neon\" is not an enum (its item's number)");
		assert_eq!(error(serde_json::json!(1), &Vector3::new(0.0, 0.0, 0.0).into()), "1 is not a Vector3 ([x, y, z])");
		assert_eq!(error(serde_json::json!([1, 2, 3, 4]), &Vector3::new(0.0, 0.0, 0.0).into()), "[1,2,3,4] is not a Vector3 ([x, y, z])");
		assert_eq!(error(serde_json::json!([1]), &Variant::String(String::new())), "[1] is not text");
		// what each type takes still converts: a number given to text is its digits, as Luau writes
		// it; an enum item keeps its enum; a model's pivot is an optional CFrame
		assert_eq!(variant(&serde_json::json!(5.0), Some(&Variant::String(String::new()))).unwrap(), Variant::String("5".into()));
		assert_eq!(variant(&serde_json::json!(0.25), Some(&Variant::String(String::new()))).unwrap(), Variant::String("0.25".into()));
		let item: Variant = EnumItem { ty: "Material".into(), value: 256 }.into();
		assert_eq!(variant(&serde_json::json!(288), Some(&item)).unwrap(), EnumItem { ty: "Material".into(), value: 288 }.into());
		let pivot = variant(&cframe_json(Affine3A::from_translation(Vec3::new(1.0, 2.0, 3.0))), Some(&Variant::OptionalCFrame(None))).unwrap();
		assert!(matches!(&pivot, Variant::OptionalCFrame(Some(CFrame { position, .. })) if *position == Vector3::new(1.0, 2.0, 3.0)), "{pivot:?}");
		// with nothing to go by (a property no class gives), as before
		assert_eq!(variant(&serde_json::json!("big"), None).unwrap(), Variant::String("big".into()));
	}

	/// An effect as effect bundles carry one: a part with a PointLight and an emitter, and a
	/// ColorCorrection its code parents into Lighting; given ror's default Lighting.
	fn effect() -> Document {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		let mut model = ModelBuilder::new("Hit");
		let core = model.part(&PartSpec { name: "Core".into(), ..PartSpec::default() });
		model.child(core, InstanceBuilder::new("PointLight").with_property("Brightness", 6.0f32));
		let attachment = model.child(core, InstanceBuilder::new("Attachment"));
		model.child(attachment, InstanceBuilder::new("ParticleEmitter").with_name("Burst").with_property("Rate", 20.0f32));
		let root = model.model;
		model.child(root, InstanceBuilder::new("ColorCorrectionEffect").with_name("Impact").with_property("Contrast", 1.0f32));
		let mut document = model.into_document();
		document.give_lighting(None);
		document
	}

	fn number(document: &Document, path: &str, property: &str) -> f32 {
		let referent = resolve(document, path).unwrap()[0];
		props::f32(document.dom.get_by_ref(referent).unwrap(), property, f32::NAN)
	}

	/// `playback`'s state at `time` applied to `document`.
	fn state_at(playback: &Playback, document: &mut Document, time: f32) -> Undo {
		let playback = Playback { time, ..playback.clone() };
		playback.resolve(document).unwrap().apply_state(document).unwrap()
	}

	#[test]
	fn tweens_go_from_where_the_property_is() {
		let mut document = effect();
		// an attack's flash: exposure += 4 at 0.2 s, tweened back to what it was over 0.5 s (Quad, Out)
		let playback = Playback {
			actions: vec![spec::set("Lighting.ExposureCompensation+=4@0.2").unwrap()],
			tweens: vec![spec::tween("Lighting.ExposureCompensation=initial@0.2+0.5").unwrap()],
			..Default::default()
		};
		for (time, expected) in [(0.1, 0.0), (0.2, 4.0), (0.45, 4.0 * (1.0 - ease(0.5, Some("Quad"), Some("Out")))), (1.0, 0.0)] {
			let undo = state_at(&playback, &mut document, time);
			let value = number(&document, "Lighting", "ExposureCompensation");
			assert!((value - expected).abs() < 1e-4, "at {time} s: {value}, not {expected}");
			undo.restore(&mut document);
		}
		assert_eq!(number(&document, "Lighting", "ExposureCompensation"), 0.0, "put back");
		// a later tween takes over from where the one before is; a set while one runs is lost
		let playback = Playback {
			actions: vec![spec::set("Lighting.Brightness=5@0.25").unwrap()],
			tweens: vec![spec::tween("Lighting.Brightness=0@0+1:Linear").unwrap(), spec::tween("Lighting.Brightness=4@0.5+0.5:Linear").unwrap()],
			..Default::default()
		};
		for (time, expected) in [(0.25, 1.5), (0.5, 1.0), (0.75, 2.5), (2.0, 4.0)] {
			let undo = state_at(&playback, &mut document, time);
			assert!((number(&document, "Lighting", "Brightness") - expected).abs() < 1e-4, "at {time} s");
			undo.restore(&mut document);
		}
		// there and back, twice
		let playback = Playback { tweens: vec![spec::tween("Lighting.Brightness=4@0+0.5:Linear:reverses:repeat=1").unwrap()], ..Default::default() };
		for (time, expected) in [(0.25, 3.0), (0.5, 4.0), (0.75, 3.0), (1.25, 3.0), (2.5, 2.0)] {
			let undo = state_at(&playback, &mut document, time);
			assert!((number(&document, "Lighting", "Brightness") - expected).abs() < 1e-4, "at {time} s");
			undo.restore(&mut document);
		}
	}

	#[test]
	fn post_effects_go_into_the_lighting_and_out() {
		let mut document = effect();
		let lighting = document.lighting.unwrap();
		let children = |document: &Document| document.dom.get_by_ref(document.roots[0]).unwrap().children().to_vec();
		let before = children(&document);
		// an impact frame in the Lighting from 0.3 s, destroyed at 0.5 s; every light fading meanwhile
		let playback = Playback {
			actions: vec![spec::parent("Hit.Impact=Lighting@0.3").unwrap(), spec::parent("Hit.Impact=nil@0.5").unwrap()],
			tweens: vec![spec::tween("Hit.Brightness=0@0+1:Linear:class=Light").unwrap()],
			..Default::default()
		};
		let undo = state_at(&playback, &mut document, 0.4);
		let impact = resolve(&document, "Lighting.Impact").unwrap()[0];
		assert_eq!(document.dom.get_by_ref(impact).unwrap().parent(), lighting);
		let environment = crate::scene::extract::environment(&document).unwrap();
		assert!(environment.effects.iter().any(|effect| matches!(effect, crate::scene::environment::PostEffect::ColorCorrection { contrast, .. } if *contrast == 1.0)));
		assert!((number(&document, "Hit.Core.PointLight", "Brightness") - 3.6).abs() < 1e-4);
		undo.restore(&mut document);
		assert_eq!(children(&document), before, "back in its place among its siblings");
		let undo = state_at(&playback, &mut document, 0.6);
		assert!(resolve(&document, "Hit.Impact").is_err() && resolve(&document, "Lighting.Impact").is_err(), "out of the world");
		undo.restore(&mut document);
		assert_eq!(children(&document), before);
		// every child of a Lighting with none is nothing, not a mistake; a name not there is one
		assert!(resolve(&document, "Lighting.*").unwrap().is_empty());
		assert!(resolve(&document, "Lighting.Sky").is_err());
		// only the Lighting and what moves into it changes: a clip reads its Lighting alone each frame
		let resolved = playback.resolve(&document).unwrap();
		assert!(!resolved.changes_only_lighting(&document), "the light fades too");
		let lighting_only = Playback { actions: playback.actions.clone(), tweens: vec![spec::tween("Lighting.Ambient=1,0,0").unwrap()], ..Default::default() };
		assert!(lighting_only.touches_lighting() && lighting_only.resolve(&document).unwrap().changes_only_lighting(&document));
	}

	/// `--disable ""` takes every effect out, not the look they play in: a post effect, Sky or Atmosphere
	/// is switched only when its path or class is named (a cutscene's own Sky drew ror's default one)
	#[test]
	fn a_broad_disable_leaves_the_lighting_alone() {
		let mut document = effect();
		let corrected = |playback: &Playback, document: &mut Document| {
			let undo = state_at(playback, document, 0.4);
			let environment = crate::scene::extract::environment(document).unwrap();
			undo.restore(document);
			environment.effects.iter().any(|effect| matches!(effect, crate::scene::environment::PostEffect::ColorCorrection { .. }))
		};
		let disabled = |target: &str, class: Option<&str>| Playback {
			actions: vec![spec::parent("Hit.Impact=Lighting@0.3").unwrap(), Action { target: target.into(), enabled: Some(false), force: true, class: class.map(str::to_string), ..Default::default() }],
			..Default::default()
		};
		assert!(corrected(&disabled("", None), &mut document), "everything disabled: the impact frame still grades the frame");
		assert!(corrected(&disabled("Hit", None), &mut document), "its effect disabled: the same");
		assert!(!corrected(&disabled("Hit.Impact", None), &mut document), "named, it is switched off");
		assert!(!corrected(&disabled("Hit", Some("ColorCorrectionEffect")), &mut document), "its class named, the same");
	}

	#[test]
	fn command_line_events() {
		let action = spec::set("Lighting.Grade.TintColor=#ff8000@0.5").unwrap();
		assert_eq!((action.at, action.target.as_str()), (0.5, "Lighting.Grade"));
		assert_eq!(action.set["TintColor"], serde_json::json!([1.0, 128.0 / 255.0, 0.0]));
		assert_eq!(spec::set("Lighting.ExposureCompensation-=2").unwrap().set["ExposureCompensation"], serde_json::json!({"add": -2.0}));
		assert_eq!(spec::set("Effect[\"A.B\"].Enabled=false").unwrap().target, "Effect[\"A.B\"]");
		assert!(spec::set("Brightness=2").is_err(), "an instance and its property");
		// JSON arrays as a playback file writes them: a UDim2, a NumberSequence; bare commas as before,
		// and what is no JSON stays text
		assert_eq!(spec::set("Live.FloorRings.Ring1.Size=[0.3,0,0.3,0]").unwrap().set["Size"], serde_json::json!([0.3, 0, 0.3, 0]));
		assert_eq!(spec::value("[[0, 1, 0], [1, 0, 0]]"), serde_json::json!([[0, 1, 0], [1, 0, 0]]));
		assert_eq!(spec::value("0.3,0,0.3,0"), serde_json::json!([0.3, 0.0, 0.3, 0.0]));
		assert_eq!(spec::value("[0.3,0,0.3"), serde_json::json!("[0.3,0,0.3"));
		assert_eq!(spec::set("Ring.Size+=[0.1,0,0.1,0]").unwrap().set["Size"], serde_json::json!({"add": [0.1, 0.0, 0.1, 0.0]}));
		let tween = spec::tween("Lighting.Ambient=0.2,0,0@1+0.25:Exponential:In:reverses:repeat=2:delay=0.1:class=Light").unwrap();
		assert_eq!((tween.at, tween.duration, tween.delay, tween.repeat_count, tween.reverses), (1.0, 0.25, 0.1, 2, true));
		assert_eq!((tween.easing.as_deref(), tween.direction.as_deref(), tween.class.as_deref()), (Some("Exponential"), Some("In"), Some("Light")));
		assert_eq!(tween.to, Some(serde_json::json!([0.2, 0.0, 0.0])));
		let plain = spec::tween("Lighting.Brightness=0").unwrap();
		assert_eq!((plain.at, plain.duration, plain.easing.as_deref()), (0.0, 1.0, None));
		assert!(spec::tween("Lighting.Brightness=0:Wobbly").is_err());
		let parent = spec::parent("Hit.Impact=Lighting@0.3").unwrap();
		assert_eq!((parent.target.as_str(), parent.parent.as_deref(), parent.at), ("Hit.Impact", Some("Lighting"), 0.3));
	}

	/// The command line's --disable / --enable (forced switches) win over what a recording switches,
	/// sets and parents for everything they reach, what it parents under them included; a later
	/// forced switch takes over
	#[test]
	fn forced_switches_win_over_the_playback() {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		let mut model = ModelBuilder::new("Sim");
		let streak = model.part(&PartSpec { name: "Streak".into(), transparency: 1.0, ..PartSpec::default() });
		let a0 = model.child(streak, InstanceBuilder::new("Attachment").with_name("A0"));
		let a1 = model.child(streak, InstanceBuilder::new("Attachment").with_name("A1").with_property("CFrame", crate::scene::build::cframe(Affine3A::from_translation(Vec3::new(4.0, 0.0, 0.0)))));
		model.child(streak, InstanceBuilder::new("Beam").with_name("Ray").with_property("Enabled", false).with_property("Attachment0", Variant::Ref(a0)).with_property("Attachment1", Variant::Ref(a1)));
		model.child(a0, InstanceBuilder::new("ParticleEmitter").with_name("Sparks").with_property("Enabled", false).with_property("Rate", 20.0f32));
		let blade = model.part(&PartSpec { name: "Blade".into(), transparency: 1.0, ..PartSpec::default() });
		model.child(blade, InstanceBuilder::new("ParticleEmitter").with_name("Glint").with_property("Rate", 20.0f32));
		let mut document = model.into_document();
		let sparks = resolve(&document, "Sim.Streak.A0.Sparks").unwrap()[0];
		let glint = resolve(&document, "Sim.Blade.Glint").unwrap()[0];
		let switch = |at: f32, target: &str, on: bool, force: bool| Action { at, target: target.into(), enabled: Some(on), force, ..Default::default() };
		// as ree records an effect: out of the world at 0, parented in and switched on when it appears
		let recording = Playback {
			actions: vec![
				spec::parent("Sim.Streak=nil@0").unwrap(),
				spec::parent("Sim.Streak=Sim@1").unwrap(),
				switch(1.0, "Sim.Streak.Ray", true, false),
				switch(1.0, "Sim.Streak.A0.Sparks", true, false),
				spec::set("Sim.Streak.Ray.Enabled=true@1.1").unwrap(),
				Action { at: 1.2, target: "Sim.Streak.A0.Sparks".into(), emit: Some(Emit::Count(10)), ..Default::default() },
				spec::parent("Sim.Blade=Sim.Streak@1").unwrap(),
			],
			..Default::default()
		};
		let enabled = |document: &Document, path: &str| props::bool(document.dom.get_by_ref(resolve(document, path).unwrap()[0]).unwrap(), "Enabled", true);
		let at = |playback: &Playback, document: &mut Document, time: f32| -> (bool, bool, bool, usize, usize) {
			let playback = Playback { time, ..playback.clone() };
			let resolved = playback.resolve(document).unwrap();
			let undo = resolved.apply_state(document).unwrap();
			let state = (enabled(document, "Sim.Streak.Ray"), enabled(document, "Sim.Streak.A0.Sparks"), resolved.timelines[&sparks].enabled_at(time) || resolved.timelines[&glint].enabled_at(time), resolved.timelines[&sparks].emits.len(), crate::scene::extract::extract(document).beams.len());
			undo.restore(document);
			state
		};
		assert_eq!(at(&recording, &mut document, 1.5), (true, true, true, 1, 1), "the recording alone");

		let mut disabled = recording.clone();
		disabled.actions.push(switch(0.0, "Sim.Streak", false, true));
		assert_eq!(at(&disabled, &mut document, 1.5), (false, false, false, 0, 0), "disabled from 0: nothing switches it on, sets it or emits");
		assert!(!enabled(&{
			let playback = Playback { time: 1.5, ..disabled.clone() };
			playback.resolve(&document).unwrap().apply_state(&mut document).unwrap();
			document
		}, "Sim.Streak.Blade.Glint"), "what the recording parents under it is disabled too");

		let mut document = {
			let mut model = ModelBuilder::new("Sim");
			let streak = model.part(&PartSpec { name: "Streak".into(), transparency: 1.0, ..PartSpec::default() });
			let a0 = model.child(streak, InstanceBuilder::new("Attachment").with_name("A0"));
			let a1 = model.child(streak, InstanceBuilder::new("Attachment").with_name("A1").with_property("CFrame", crate::scene::build::cframe(Affine3A::from_translation(Vec3::new(4.0, 0.0, 0.0)))));
			model.child(streak, InstanceBuilder::new("Beam").with_name("Ray").with_property("Enabled", false).with_property("Attachment0", Variant::Ref(a0)).with_property("Attachment1", Variant::Ref(a1)));
			model.into_document()
		};
		let mut bisected = Playback { actions: vec![switch(1.0, "Sim.Streak.Ray", true, false), switch(0.0, "Sim.Streak", false, true), switch(1.3, "Sim.Streak.Ray", true, true), switch(2.0, "Sim.Streak.Ray", false, false)], ..Default::default() };
		let ray = |playback: &Playback, document: &mut Document, time: f32| {
			let undo = state_at(playback, document, time);
			let on = crate::scene::extract::extract(document).beams.len() == 1;
			undo.restore(document);
			on
		};
		assert!(!ray(&bisected, &mut document, 1.2), "still forced off");
		assert!(ray(&bisected, &mut document, 1.5) && ray(&bisected, &mut document, 2.5), "forced on from 1.3, whatever the playback switches after");
		bisected.actions.iter_mut().for_each(|action| action.force = false);
		assert!(!ray(&bisected, &mut document, 2.5), "unforced, the last switch wins");
	}

	#[test]
	fn a_cframe_turns_on_the_way() {
		let quarter = cframe_json(Affine3A::from_rotation_y(std::f32::consts::FRAC_PI_2) * Affine3A::from_translation(Vec3::ZERO));
		let halfway = lerp_value(&cframe_json(Affine3A::IDENTITY), &quarter, 0.5);
		let numbers: Vec<f32> = halfway.as_array().unwrap().iter().map(|value| value.as_f64().unwrap() as f32).collect();
		let turned = affine(&numbers, Affine3A::IDENTITY).unwrap();
		// an eighth of a turn, still a rotation (lerping its numbers would shrink it)
		assert!((Vec3::from(turned.matrix3.x_axis).length() - 1.0).abs() < 1e-4);
		assert!((turned.matrix3.x_axis.x - std::f32::consts::FRAC_PI_4.cos()).abs() < 1e-4);
	}

	#[test]
	fn a_held_key_stays_put() {
		// one of the rock clip's recorded keys (its rotation as the recording rounds it), held for a
		// second between equal keys, then a rise of a stud
		let key = [198.40267944335938, -3.4036407470703125, 202.14102172851562, -0.9160308837890625, -0.1452178955078125, 0.373870849609375, -0.0574798583984375, 0.970062255859375, 0.2359619140625, -0.3969573974609375, 0.1946563720703125, -0.8969573974609375];
		let key = affine(&key, Affine3A::IDENTITY).unwrap();
		let risen = Affine3A::from_translation(Vec3::Y) * key;
		let track = Track { referent: Ref::none(), rest: key, keys: vec![(0.0, key, None, None), (1.0, key, None, None), (1.5, risen, None, None)], physics_from: None };
		for frame in 0..=60 {
			assert_eq!(track.pose(frame as f32 / 60.0), key, "frame {frame}");
		}
		let halfway = Vec3::from(track.pose(1.25).translation) - Vec3::from(key.translation);
		assert!((halfway - Vec3::Y * 0.5).length() < 1e-4, "rises from it: {halfway}");
	}

	#[test]
	fn a_moving_part_keeps_its_random_draws() {
		// Studio draws a part's material offsets once per part and load, not again as it moves; ror
		// draws them from the part's seed, which stays the one of where it was loaded whatever a
		// moment's motion, resizing or rounding makes of it. CrackedLava rocks held still between equal
		// keys changed their cracks from frame to frame: the lerp between the keys lands a bit off them
		// at some moments, and the offsets were drawn from that position.
		use crate::scene::build::{ModelBuilder, PartSpec};
		use crate::scene::extract::extract;
		let held = Vec3::new(198.40268, -3.4036407, 202.14102);
		let mut model = ModelBuilder::new("Rocks");
		model.part(&PartSpec { name: "Rock".into(), cframe: Affine3A::from_translation(held), material: 804, ..PartSpec::default() });
		let mut document = model.into_document();
		let at_rest = extract(&document).parts[0].clone();
		assert_eq!(at_rest.seed, crate::scene::part_seed(held, at_rest.size), "a still part's from where it is");
		// it rises 10 studs, growing, then stays between equal keys
		let key = |at: f32, position: Vec3| MotionKey { at, cframe: position.to_array().to_vec(), easing: None, direction: None };
		let playback = Playback {
			motion: vec![Motion { target: "Rocks.Rock".into(), keys: vec![key(0.0, held - Vec3::Y * 10.0), key(0.5, held), key(2.0, held)], relative: false, tagged: None, physics_from: None }],
			tweens: vec![spec::tween("Rocks.Rock.Size=4,2,2@0+0.5:Linear").unwrap()],
			..Default::default()
		};
		let mut moved = 0;
		for frame in 0..=120 {
			let time = frame as f32 / 60.0;
			let undo = state_at(&playback, &mut document, time);
			let part = extract(&document).parts[0].clone();
			assert_eq!(part.seed, at_rest.seed, "at {time} s");
			moved += (part.cframe != at_rest.cframe || part.size != at_rest.size) as usize;
			undo.restore(&mut document);
		}
		assert!(moved > 30, "it moved ({moved} frames)");
		// and in copies of the document (a clip's threads each change one), made before or while it moves
		let mut before = document.copy();
		let _ = state_at(&playback, &mut before, 0.25);
		assert_eq!(extract(&before).parts[0].seed, at_rest.seed);
		let undo = state_at(&playback, &mut document, 0.25);
		let during = document.copy();
		undo.restore(&mut document);
		assert_eq!(extract(&during).parts[0].seed, at_rest.seed);
	}

	#[test]
	fn a_tagged_motion_moves_only_the_tagged_sibling() {
		// a path names every same-named sibling: a map's "Models.Part" is one CameraFollow
		// mist host and 49 floors, and a clip's follower motion moved the floors to the camera too
		use crate::scene::build::{ModelBuilder, PartSpec};
		let mut model = ModelBuilder::new("Map");
		let host = model.part(&PartSpec { name: "Part".into(), transparency: 1.0, ..PartSpec::default() });
		let floor = model.part(&PartSpec { name: "Part".into(), cframe: Affine3A::from_translation(Vec3::new(50.0, 0.0, 0.0)), ..PartSpec::default() });
		let mut document = model.into_document();
		document.dom.get_by_ref_mut(host).unwrap().properties.insert(rbx_dom_weak::ustr("Tags"), Variant::Tags(rbx_dom_weak::types::Tags::from(vec!["CameraFollow".to_string()])));
		let key = |at: f32| MotionKey { at, cframe: vec![0.0, 100.0, 0.0], easing: None, direction: None };
		let playback = Playback { motion: vec![Motion { target: "Map.Part".into(), keys: vec![key(0.0), key(1.0)], relative: false, tagged: Some(super::super::follow::TAG.to_string()), physics_from: None }], ..Default::default() };
		let undo = state_at(&playback, &mut document, 0.5);
		assert_eq!(rest_cframe(&document, host).unwrap().translation.y, 100.0, "the tagged host moves");
		assert_eq!(Vec3::from(rest_cframe(&document, floor).unwrap().translation), Vec3::new(50.0, 0.0, 0.0), "its untagged sibling stays");
		undo.restore(&mut document);
	}

	/// What every reader of a content value draws (`props::content` over both its names).
	fn drawn(document: &Document, path: &str, property: &str) -> Option<String> {
		let instance = document.dom.get_by_ref(resolve(document, path).unwrap()[0]).unwrap();
		let names = crate::scene::load::content_names(instance.class.as_str(), property).unwrap();
		props::content(instance, &[names.content, names.legacy])
	}

	/// An effect as a file brings it: saved and read back, its images and meshes under their Content
	/// names alone (TextureContent, ImageContent, MeshContent).
	fn saved_effect() -> Document {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		use rbx_dom_weak::types::Content;
		let uri = |id: u32| Content::from_uri(format!("rbxassetid://{id}"));
		let mut model = ModelBuilder::new("Crater");
		let plane = model.part(&PartSpec { name: "Plane".into(), ..PartSpec::default() });
		model.child(plane, InstanceBuilder::new("Decal").with_name("Veins").with_property("TextureContent", uri(1)));
		model.child(plane, InstanceBuilder::new("Texture").with_name("Cracks").with_property("TextureContent", uri(2)));
		let a0 = model.child(plane, InstanceBuilder::new("Attachment").with_name("A0"));
		let a1 = model.child(plane, InstanceBuilder::new("Attachment").with_name("A1"));
		model.child(a0, InstanceBuilder::new("ParticleEmitter").with_name("Embers").with_property("TextureContent", uri(3)));
		model.child(plane, InstanceBuilder::new("Beam").with_name("Arc").with_property("Attachment0", a0).with_property("Attachment1", a1).with_property("TextureContent", uri(4)));
		model.child(plane, InstanceBuilder::new("Trail").with_name("Streak").with_property("Attachment0", a0).with_property("Attachment1", a1).with_property("TextureContent", uri(5)));
		let rock = model.part(&PartSpec { class: "MeshPart", name: "Rock".into(), ..PartSpec::default() });
		model.set(rock, "MeshContent", uri(6));
		model.set(rock, "TextureContent", uri(7));
		let gui = model.child(plane, InstanceBuilder::new("SurfaceGui").with_name("Sign"));
		model.child(gui, InstanceBuilder::new("ImageLabel").with_name("Icon").with_property("ImageContent", uri(8)));
		crate::scene::load::read_bytes(&model.to_rbxm().unwrap()).unwrap()
	}

	#[test]
	fn a_content_id_sets_the_content_drawn() {
		// a reused crater plane's Decal swapped to ember veins with `set {"Texture": ...}` kept its
		// old image: the set landed on a Texture property beside the TextureContent ror draws from
		let mut document = saved_effect();
		let swaps = [
			("Crater.Plane.Veins", "Texture", 11),
			("Crater.Plane.Cracks", "Texture", 12),
			("Crater.Plane.A0.Embers", "Texture", 13),
			("Crater.Plane.Arc", "Texture", 14),
			("Crater.Plane.Streak", "Texture", 15),
			("Crater.Rock", "MeshId", 16),
			("Crater.Rock", "TextureID", 17),
			("Crater.Plane.Sign.Icon", "Image", 18),
		];
		let before: Vec<Option<String>> = swaps.iter().map(|(path, property, _)| drawn(&document, path, property)).collect();
		let playback = Playback {
			actions: swaps.iter().map(|(path, property, id)| Action { at: 0.2, target: path.to_string(), set: [(property.to_string(), serde_json::json!(format!("rbxassetid://{id}")))].into(), ..Default::default() }).collect(),
			..Default::default()
		};
		let undo = state_at(&playback, &mut document, 0.3);
		for (path, property, id) in swaps {
			assert_eq!(drawn(&document, path, property), Some(format!("rbxassetid://{id}")), "{path}.{property}");
		}
		// and what the scene takes from them: the emitter's, beam's and trail's textures, the GUI rur draws
		let scene = crate::scene::extract::extract(&document);
		assert_eq!(scene.particles.iter().find(|emitter| emitter.path.ends_with("Embers")).unwrap().texture.as_deref(), Some("rbxassetid://13"));
		assert_eq!(scene.beams[0].texture.as_deref(), Some("rbxassetid://14"));
		assert_eq!(scene.trails[0].texture.as_deref(), Some("rbxassetid://15"));
		let gui = crate::scene::load::read_bytes(&scene.guis[0].content).unwrap();
		assert_eq!(drawn(&gui, "Sign.Icon", "Image").as_deref(), Some("rbxassetid://18"));
		// the old names stay out of the file's instances, and everything goes back
		let legacy = ["Texture", "TextureID", "MeshId", "Image"].map(rbx_dom_weak::ustr);
		assert!(document.dom.descendants().all(|instance| legacy.iter().all(|name| !instance.properties.contains_key(name))));
		undo.restore(&mut document);
		let after: Vec<Option<String>> = swaps.iter().map(|(path, property, _)| drawn(&document, path, property)).collect();
		assert_eq!(after, before);
	}

	#[test]
	fn either_name_of_a_content_value_plays_back() {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		use rbx_dom_weak::types::ContentId;
		// the Content name on a file's instance; the old name and the Content name are one timeline
		let mut document = saved_effect();
		let set = |at: f32, property: &str, value: serde_json::Value| Action { at, target: "Crater.Plane.Sign.Icon".into(), set: [(property.to_string(), value)].into(), ..Default::default() };
		let playback = Playback {
			actions: vec![
				set(0.2, "ImageContent", serde_json::json!("rbxassetid://21")),
				set(0.4, "Image", serde_json::json!("rbxassetid://22")),
				set(0.6, "ImageContent", serde_json::json!({"initial": true})),
				set(0.8, "Image", serde_json::json!("")),
			],
			..Default::default()
		};
		for (time, expected) in [(0.1, Some("rbxassetid://8")), (0.3, Some("rbxassetid://21")), (0.5, Some("rbxassetid://22")), (0.7, Some("rbxassetid://8")), (0.9, None)] {
			let undo = state_at(&playback, &mut document, time);
			assert_eq!(drawn(&document, "Crater.Plane.Sign.Icon", "Image").as_deref(), expected, "at {time} s");
			undo.restore(&mut document);
		}
		// a flipbook: keys under the old name switch the image at each key
		let flipbook = Playback {
			tweens: vec![Tween {
				target: "Crater.Plane.Veins".into(),
				property: "Texture".into(),
				keys: (0..3).map(|frame| TweenKey { at: frame as f32 * 0.1, value: serde_json::json!(format!("rbxassetid://{}", 30 + frame)), ..Default::default() }).collect(),
				..Default::default()
			}],
			..Default::default()
		};
		for (time, frame) in [(0.0, 30), (0.15, 31), (0.2, 32), (0.5, 32)] {
			let undo = state_at(&flipbook, &mut document, time);
			assert_eq!(drawn(&document, "Crater.Plane.Veins", "Texture"), Some(format!("rbxassetid://{frame}")), "at {time} s");
			undo.restore(&mut document);
		}
		// a model built in code under the old name alone (Lune's way): the Content name sets it, both
		// names hold the new value, and an empty one draws none
		let mut model = ModelBuilder::new("Built");
		let part = model.part(&PartSpec { name: "Plane".into(), ..PartSpec::default() });
		model.child(part, InstanceBuilder::new("Decal").with_name("Veins").with_property("Texture", ContentId::from("rbxassetid://1")));
		let mut built = model.into_document();
		let playback = Playback {
			actions: vec![
				Action { at: 0.1, target: "Built.Plane.Veins".into(), set: [("TextureContent".to_string(), serde_json::json!("rbxassetid://2"))].into(), ..Default::default() },
				Action { at: 0.3, target: "Built.Plane.Veins".into(), set: [("Texture".to_string(), serde_json::json!(""))].into(), ..Default::default() },
			],
			..Default::default()
		};
		assert_eq!(drawn(&built, "Built.Plane.Veins", "Texture").as_deref(), Some("rbxassetid://1"));
		let undo = state_at(&playback, &mut built, 0.2);
		assert_eq!(drawn(&built, "Built.Plane.Veins", "Texture").as_deref(), Some("rbxassetid://2"));
		let decal = built.dom.get_by_ref(resolve(&built, "Built.Plane.Veins").unwrap()[0]).unwrap();
		assert_eq!(decal.properties.get(&rbx_dom_weak::ustr("Texture")), Some(&Variant::ContentId(ContentId::from("rbxassetid://2"))));
		undo.restore(&mut built);
		let undo = state_at(&playback, &mut built, 0.4);
		assert_eq!(drawn(&built, "Built.Plane.Veins", "Texture"), None);
		undo.restore(&mut built);
		assert_eq!(drawn(&built, "Built.Plane.Veins", "Texture").as_deref(), Some("rbxassetid://1"));
	}

	/// A relative local path, the file's or one a playback sets, is read from the folder the document
	/// names ([`Document::read_relative_to`]: a job's caller's working directory), never the
	/// process's (a service's state dir): `file://` and the absolute path, `%` escaped for its readers,
	/// a `#` fragment kept. Assets and absolute paths stay as they are.
	#[test]
	fn relative_local_paths_are_read_from_the_documents_folder() {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		use rbx_dom_weak::types::{Content, ContentId};
		let folder = std::env::temp_dir().join(format!("ror-relative-{}-100%", std::process::id()));
		std::fs::create_dir_all(&folder).unwrap();
		std::fs::write(folder.join("Loose.png"), b"").unwrap();
		let absolute = folder.join("Absolute.png").display().to_string();
		let mut model = ModelBuilder::new("Built");
		let plane = model.part(&PartSpec { name: "Plane".into(), ..PartSpec::default() });
		let decals = [("Path", "output/a.png"), ("Uri", "file://output/b.png"), ("Lod", "./c.mesh#lod=2"), ("Loose", "Loose.png"), ("Asset", "rbxassetid://5"), ("Absolute", absolute.as_str()), ("Unknown", "Nothing.png")];
		for (name, uri) in decals {
			model.child(plane, InstanceBuilder::new("Decal").with_name(name).with_property("TextureContent", Content::from_uri(uri)));
		}
		model.child(plane, InstanceBuilder::new("Decal").with_name("Legacy").with_property("Texture", ContentId::from("output/d.png")));
		let mut document = model.into_document();
		assert_eq!(document.read_relative_to(&folder), 5);
		let file = |relative: &str| format!("file://{}", std::path::absolute(folder.join(relative)).unwrap().display().to_string().replace('%', "%25"));
		for (name, expected) in [
			("Path", file("output/a.png")),
			("Uri", file("output/b.png")),
			("Lod", format!("{}#lod=2", file("c.mesh"))),
			("Loose", file("Loose.png")),
			("Asset", "rbxassetid://5".into()),
			("Absolute", absolute.clone()),
			("Unknown", "Nothing.png".into()),
			("Legacy", file("output/d.png")),
		] {
			assert_eq!(drawn(&document, &format!("Built.Plane.{name}"), "Texture"), Some(expected), "{name}");
		}
		// as the renderers read it back: the folder's own path, `%` and all
		let read = drawn(&document, "Built.Plane.Path", "Texture").unwrap();
		assert_eq!(ror_assets::ContentId::parse(&read).unwrap(), ror_assets::ContentId::File(std::path::absolute(folder.join("output/a.png")).unwrap()));
		// a playback's set is read from there too
		let playback = Playback { actions: vec![Action { at: 0.1, target: "Built.Plane.Asset".into(), set: [("TextureContent".to_string(), serde_json::json!("output/later.png"))].into(), ..Default::default() }], ..Default::default() };
		let undo = state_at(&playback, &mut document, 0.2);
		assert_eq!(drawn(&document, "Built.Plane.Asset", "Texture"), Some(file("output/later.png")));
		undo.restore(&mut document);
		assert_eq!(drawn(&document, "Built.Plane.Asset", "Texture").as_deref(), Some("rbxassetid://5"));
		let _ = std::fs::remove_dir_all(&folder);
	}

	/// A mirror effect's floor rings as its recording brings them: a SurfaceGui on the top of a
	/// clear plate it is adorned to, its rings hidden and 0 in size in the file, a pool of light shown
	/// from the start; saved and read back, as a recording tool hands ror a recording's scene.
	fn floor_rings() -> Document {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		use rbx_dom_weak::types::{Content, Enum, UDim, UDim2, Vector2};
		let square = |scale: f32| UDim2::new(UDim::new(scale, 0), UDim::new(scale, 0));
		let image = |name: &str, visible: bool, size: f32| {
			InstanceBuilder::new("ImageLabel")
				.with_name(name)
				.with_property("Visible", visible)
				.with_property("Size", square(size))
				.with_property("Position", square(0.5))
				.with_property("AnchorPoint", Vector2::new(0.5, 0.5))
				.with_property("BackgroundTransparency", 1.0f32)
				.with_property("ImageContent", Content::from_uri("rbxassetid://123456789"))
		};
		let mut model = ModelBuilder::new("Live");
		let plate = model.part(&PartSpec { name: "HeartFloor".into(), size: Vec3::new(26.0, 0.1, 26.0), transparency: 1.0, ..PartSpec::default() });
		let root = model.model;
		let gui = InstanceBuilder::new("SurfaceGui").with_name("FloorRings").with_property("Adornee", Variant::Ref(plate)).with_property("Face", Enum::from_u32(1)).with_property("SizingMode", Enum::from_u32(1)).with_property("PixelsPerStud", 24.0f32);
		let gui = model.child(root, gui);
		for (name, visible, size) in [("Ring1", false, 0.0), ("Ring2", false, 0.0), ("Pool", true, 0.75)] {
			model.child(gui, image(name, visible, size));
		}
		crate::scene::load::read_bytes(&model.to_rbxm().unwrap()).unwrap()
	}

	#[test]
	fn a_surface_guis_content_follows_the_playback() {
		// the recording's ring: shown and turned at 0.88 s, its Size, ImageTransparency and ImageColor3
		// keyed. A render of it at 1.2 s that came out as at 0 s (a caller's render took --time,
		// not --at) was taken for a GUI the playback cannot animate; its content is read again at every
		// moment, so rur draws a new image whenever the playback changes it
		let mut document = floor_rings();
		let recording: Playback = serde_json::from_value(serde_json::json!({
			"actions": [
				{"at": 0.88333, "target": "Live.FloorRings.Ring1", "set": {"Rotation": 345.0}},
				{"at": 0.88333, "target": "Live.FloorRings.Ring1", "set": {"Visible": true}}
			],
			"tweens": [
				{"target": "Live.FloorRings.Ring1", "property": "Size", "keys": [
					{"at": 0.86667, "value": [0.0, 0, 0.0, 0]}, {"at": 1.16667, "value": [0.17, 0, 0.17, 0]}, {"at": 2.0, "value": [0.33767, 0, 0.33767, 0]}]},
				{"target": "Live.FloorRings.Ring1", "property": "ImageTransparency", "keys": [
					{"at": 0.86667, "value": 0.0}, {"at": 0.88333, "value": 0.15}, {"at": 2.0, "value": 0.56402}]},
				{"target": "Live.FloorRings.Ring1", "property": "ImageColor3", "keys": [
					{"at": 0.86667, "value": [1.0, 1.0, 1.0]}, {"at": 2.0, "value": [1.0, 0.67827, 0.31087]}]}
			]
		}))
		.unwrap();
		// the GUI rur draws at a moment (read back as a model of its own), its content hash and the
		// scene's warnings
		let drawn = |playback: &Playback, document: &mut Document, time: f32| -> (Document, u64, Vec<String>) {
			let undo = state_at(playback, document, time);
			let scene = crate::scene::extract::extract(document);
			undo.restore(document);
			assert_eq!(scene.guis.len(), 1, "at {time} s: {:?}", scene.warnings);
			(crate::scene::load::read_bytes(&scene.guis[0].content).unwrap(), scene.guis[0].content_hash, scene.warnings)
		};
		let label = |gui: &Document, name: &str| {
			let instance = gui.dom.get_by_ref(resolve(gui, &format!("FloorRings.{name}")).unwrap()[0]).unwrap();
			(props::bool(instance, "Visible", true), props::udim2(instance, "Size").unwrap(), props::f32(instance, "ImageTransparency", f32::NAN), props::f32(instance, "Rotation", f32::NAN))
		};
		let as_authored = crate::scene::extract::extract(&document).guis[0].content_hash;
		let (before, hash_before, _) = drawn(&recording, &mut document, 0.5);
		assert_eq!(label(&before, "Ring1"), (false, [0.0; 4], 0.0, 0.0), "hidden before 0.88 s");
		assert!(label(&before, "Pool").0);
		assert_eq!(drawn(&recording, &mut document, 0.2).1, hash_before, "one image while nothing changes");
		let (after, hash_after, warnings) = drawn(&recording, &mut document, 1.2);
		let (visible, size, transparency, rotation) = label(&after, "Ring1");
		assert!(visible && rotation == 345.0, "shown and turned at 1.2 s");
		// 0.04 of the way from 0.17 to 0.33767 (4.6 studs of the 26-stud plate), from 0.15 to 0.56402
		assert!((size[0] - 0.176707).abs() < 1e-4 && size[0] == size[2] && size[1] == 0.0 && size[3] == 0.0, "{size:?}");
		assert!((transparency - 0.267406).abs() < 1e-4, "{transparency}");
		assert_ne!(hash_after, hash_before, "a new image for rur");
		assert!(warnings.is_empty(), "{warnings:?}");
		assert_eq!(crate::scene::extract::extract(&document).guis[0].content_hash, as_authored, "put back");

		// the command line's JSON arrays, as a playback file writes them: --set's [0.3,0,0.3,0] was text,
		// which the UDim2 Size took, and the SurfaceGui could not be written for rur (not drawn)
		let mut shown = recording.clone();
		shown.actions.extend([spec::set("Live.FloorRings.Ring2.Visible=true@0.5").unwrap(), spec::set("Live.FloorRings.Ring2.Size=[0.3,0,0.3,0]@0.5").unwrap()]);
		let (gui, _, warnings) = drawn(&shown, &mut document, 1.2);
		let (visible, size, ..) = label(&gui, "Ring2");
		assert!(visible && size == [0.3, 0.0, 0.3, 0.0], "{size:?}");
		assert!(warnings.is_empty(), "{warnings:?}");
		// a value no UDim2 takes is an error naming the property and the forms it takes
		let mut wrong = recording.clone();
		wrong.actions.push(spec::set("Live.FloorRings.Ring2.Size=big@0.5").unwrap());
		let error = Playback { time: 1.2, ..wrong }.resolve(&document).unwrap().apply_state(&mut document).err().map(|error| format!("{error:#}"));
		assert_eq!(error.as_deref(), Some("playback: Live.FloorRings.Ring2.Size: \"big\" is not a UDim2 ([xScale, xOffset, yScale, yOffset])"));
	}

	/// Emitters as effects carry them: EmitCount / EmitDelay / EmitDuration attributes, one
	/// streaming from the file.
	fn attributed_effect() -> Document {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		use rbx_dom_weak::types::Attributes;
		let mut model = ModelBuilder::new("Hit");
		let core = model.part(&PartSpec { name: "Core".into(), ..PartSpec::default() });
		let emitter = |name: &str, enabled: bool, attributes: &[(&str, f64)]| {
			let attributes = attributes.iter().fold(Attributes::new(), |all, (name, value)| all.with(*name, Variant::Float64(*value)));
			InstanceBuilder::new("ParticleEmitter").with_name(name).with_property("Enabled", enabled).with_property("Rate", 30.0f32).with_property("Attributes", attributes)
		};
		model.child(core, emitter("Both", false, &[("EmitCount", 3.0), ("EmitDelay", 0.2), ("EmitDuration", 0.3)]));
		model.child(core, emitter("Plain", false, &[]));
		model.child(core, emitter("Late", false, &[("EmitDelay", 0.2)]));
		model.child(core, emitter("Held", false, &[("EmitDuration", 0.4)]));
		model.child(core, emitter("Stream", true, &[]));
		model.into_document()
	}

	/// Per emitter, its bursts (time, count) and whether it is enabled at each moment asked.
	type Timing = BTreeMap<String, (Vec<(f32, u32)>, Vec<bool>)>;

	/// Each emitter's [`Timing`] at `moments` for an emit of `mode` at `at`.
	fn emit_timing(document: &Document, mode: &str, at: f32, moments: &[f32]) -> Timing {
		let playback = Playback { time: 1.0, actions: vec![Action { at, target: "Hit".into(), emit: Some(Emit::Mode(mode.into())), ..Default::default() }], ..Default::default() };
		let resolved = playback.resolve(document).unwrap();
		let mut timing = BTreeMap::new();
		for name in ["Both", "Plain", "Late", "Held", "Stream"] {
			let timeline = &resolved.timelines[&resolve(document, &format!("Hit.Core.{name}")).unwrap()[0]];
			let bursts = timeline.emits.iter().map(|burst| ((burst.at * 60.0).round() / 60.0, burst.count)).collect();
			timing.insert(name.to_string(), (bursts, moments.iter().map(|&moment| timeline.enabled_at(moment)).collect()));
		}
		timing
	}

	#[test]
	fn emits_by_attributes_follow_the_games_helpers() {
		let document = attributed_effect();
		let moments = [0.1, 0.25, 0.35, 0.45, 0.6];
		type Row<'a> = (&'a str, Vec<(f32, u32)>, [bool; 5]);
		let expect = |rows: [Row; 5]| -> Timing { rows.into_iter().map(|(name, bursts, enabled)| (name.to_string(), (bursts, enabled.to_vec()))).collect() };
		let (off, on) = (false, true);
		// "attributes": EmitCount after EmitDelay; Enabled from the call
		// until EmitDuration after it, 0 without one, the file's stream too
		assert_eq!(
			emit_timing(&document, "attributes", 0.0, &moments),
			expect([
				("Both", vec![(0.2, 3)], [on, on, off, off, off]),
				("Plain", vec![(0.0, 5)], [off; 5]),
				("Late", vec![(0.2, 5)], [off; 5]),
				("Held", vec![(0.0, 5)], [on, on, on, off, off]),
				("Stream", vec![(0.0, 5)], [off; 5]),
			])
		);
		// "immediate": the same, its EmitDelay not waited for
		assert_eq!(
			emit_timing(&document, "immediate", 0.0, &moments),
			expect([
				("Both", vec![(0.0, 3)], [on, on, off, off, off]),
				("Plain", vec![(0.0, 5)], [off; 5]),
				("Late", vec![(0.0, 5)], [off; 5]),
				("Held", vec![(0.0, 5)], [on, on, on, off, off]),
				("Stream", vec![(0.0, 5)], [off; 5]),
			])
		);
		// the cutscene library: after EmitDelay, enabled for EmitDuration or else emitted (16 without
		// an EmitCount); what streams from the file goes on
		assert_eq!(
			emit_timing(&document, "cutscene", 0.0, &moments),
			expect([
				("Both", vec![], [off, on, on, on, off]),
				("Plain", vec![(0.0, 16)], [off; 5]),
				("Late", vec![(0.2, 16)], [off; 5]),
				("Held", vec![], [on, on, on, off, off]),
				("Stream", vec![(0.0, 16)], [on; 5]),
			])
		);
		// the calibration runs' (ror's rule before): EmitCount after EmitDelay, enabled from then
		assert_eq!(
			emit_timing(&document, "delayed", 0.0, &moments),
			expect([
				("Both", vec![(0.2, 3)], [off, on, on, on, off]),
				("Plain", vec![(0.0, 5)], [off; 5]),
				("Late", vec![(0.2, 5)], [off; 5]),
				("Held", vec![(0.0, 5)], [on, on, on, off, off]),
				("Stream", vec![(0.0, 5)], [on; 5]),
			])
		);
		// an emit later in the timeline counts from its call
		let later = emit_timing(&document, "attributes", 0.5, &[0.45, 0.55, 0.75, 0.85]);
		assert_eq!(later["Both"], (vec![(0.7, 3)], vec![false, true, true, false]));
		assert!(Playback { actions: vec![Action { target: "Hit".into(), emit: Some(Emit::Mode("sparkle".into())), ..Default::default() }], ..Default::default() }.resolve(&document).is_err());
	}

	#[test]
	fn emitters_stream_through_their_window() {
		// the particles drawn at 0.6 s: the window's Rate stream and the bursts (Lifetime 5-10 s)
		let mut document = attributed_effect();
		let births = |document: &mut Document, mode: &str| -> BTreeMap<String, usize> {
			let playback = Playback { time: 0.6, actions: vec![Action { target: "Hit".into(), emit: Some(Emit::Mode(mode.into())), ..Default::default() }], ..Default::default() };
			let resolved = playback.resolve(document).unwrap();
			let undo = resolved.apply_state(document).unwrap();
			let mut scene = crate::scene::extract::extract(document);
			resolved.apply_history(document, &mut scene);
			undo.restore(document);
			scene.particles.iter().map(|emitter| (emitter.path.rsplit('.').next().unwrap().to_string(), emitter.births.as_ref().map_or(0, Vec::len))).collect()
		};
		let effect_utils = births(&mut document, "attributes");
		// Both: 0.3 s of Rate 30 from the call (9, a frame either way) and its burst of 3; Held 0.4 s
		// and 5; the file's stream switched off at once: its burst alone
		assert!((11..=13).contains(&effect_utils["Both"]), "{effect_utils:?}");
		assert!((16..=18).contains(&effect_utils["Held"]), "{effect_utils:?}");
		assert_eq!((effect_utils["Plain"], effect_utils["Late"], effect_utils["Stream"]), (5, 5, 5));
		let cutscene = births(&mut document, "cutscene");
		// Both: 0.3 s of Rate 30 from its delay, no burst; the stream runs on for 0.6 s
		assert!((8..=10).contains(&cutscene["Both"]), "{cutscene:?}");
		assert!(cutscene["Stream"] >= 16 + 16, "{cutscene:?}");
	}

	#[test]
	fn emitters_out_of_the_viewers_view_stand_still() {
		// particle-offscreen: a Rate 10 stream and a burst at 0.2 s while the camera looks away, the
		// camera on them from 1 s: at 1.5 s the stream has emitted for 0.5 s and the burst aged 0.5 s
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		use rbx_dom_weak::types::NumberRange;
		let mut model = ModelBuilder::new("Fx");
		let anchor = model.part(&PartSpec { name: "Anchor".into(), size: Vec3::splat(0.2), ..PartSpec::default() });
		let attachment = model.child(anchor, InstanceBuilder::new("Attachment").with_name("At"));
		model.child(attachment, InstanceBuilder::new("ParticleEmitter").with_name("Stream").with_property("Rate", 10.0f32).with_property("Lifetime", NumberRange::new(5.0, 5.0)));
		model.child(attachment, InstanceBuilder::new("ParticleEmitter").with_name("Burst").with_property("Rate", 0.0f32).with_property("Lifetime", NumberRange::new(5.0, 5.0)));
		let document = model.into_document();
		let away = ViewerKey { at: 0.0, cframe: [0.0, 0.0, -30.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], fov: 30.0, aspect: 1.5 };
		let watching = ViewerKey { at: 1.0, cframe: [0.0, 0.0, -30.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0], fov: 30.0, aspect: 1.5 };
		let drawn = |viewer: Vec<ViewerKey>| -> (usize, f32) {
			let playback = Playback { time: 1.5, actions: vec![Action { at: 0.2, target: "Fx.Anchor.At.Burst".into(), emit: Some(Emit::Count(3)), ..Default::default() }], viewer, ..Default::default() };
			let resolved = playback.resolve(&document).unwrap();
			let mut scene = crate::scene::extract::extract(&document);
			resolved.apply_history(&document, &mut scene);
			let births = |name: &str| scene.particles.iter().find(|emitter| emitter.path.ends_with(name)).and_then(|emitter| emitter.births.clone()).unwrap_or_default();
			(births("Stream").len(), births("Burst").first().map_or(-1.0, |birth| birth.age))
		};
		let (stream, burst) = drawn(Vec::new());
		assert!((14..=16).contains(&stream) && (burst - 1.3).abs() < 0.02, "unwatched: {stream} particles, the burst {burst} s old");
		let (stream, burst) = drawn(vec![away.clone(), watching.clone()]);
		assert!((4..=6).contains(&stream) && (burst - 0.5).abs() < 0.02, "watched from 1 s: {stream} particles, the burst {burst} s old");
		// watched all along, nothing changes
		let (stream, burst) = drawn(vec![ViewerKey { at: 0.0, ..watching }]);
		assert!((14..=16).contains(&stream) && (burst - 1.3).abs() < 0.02, "watched: {stream} particles, the burst {burst} s old");
		// a ball reaches into the view by its radius
		assert!(!away.sees(Vec3::ZERO, 1.0) && away.sees(Vec3::ZERO, 31.0));
	}

	#[test]
	fn a_long_running_stream_keeps_emitting() {
		// Rate 400 (the most) and Lifetime 1 s drawn at 20 s: 400 particles alive, though 8,000 were born
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		use rbx_dom_weak::types::NumberRange;
		let mut model = ModelBuilder::new("Map");
		let ground = model.part(&PartSpec { name: "Ground".into(), ..PartSpec::default() });
		model.child(ground, InstanceBuilder::new("ParticleEmitter").with_name("Mist").with_property("Rate", 400.0f32).with_property("Lifetime", NumberRange::new(1.0, 1.0)));
		let document = model.into_document();
		let playback = Playback { time: 20.0, ..Default::default() };
		let resolved = playback.resolve(&document).unwrap();
		let mut scene = crate::scene::extract::extract(&document);
		resolved.apply_history(&document, &mut scene);
		let alive = crate::render::particles::simulate(&scene.particles[0]).len();
		assert!((390..=410).contains(&alive), "{alive}");
	}

	/// A recording's bursts leave from where their emitter was at the call (`cframe`, and a part's
	/// `partSize`): an attachment moved to four corners in one frame, emitting after each move (an effect
	/// that locks its rectangle so), draws four bursts at four corners where its motion
	/// holds the last corner only; without them every burst is where the motion has the emitter, as
	/// before.
	#[test]
	fn each_burst_leaves_from_where_its_emit_says() {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use rbx_dom_weak::InstanceBuilder;
		use rbx_dom_weak::types::NumberRange;
		let mut model = ModelBuilder::new("Fx");
		let plate = model.part(&PartSpec { name: "Plate".into(), size: Vec3::splat(0.2), ..PartSpec::default() });
		let corner = model.child(plate, InstanceBuilder::new("Attachment").with_name("Corner"));
		model.child(corner, InstanceBuilder::new("ParticleEmitter").with_name("Burst").with_property("Rate", 0.0f32).with_property("Lifetime", NumberRange::new(5.0, 5.0)));
		let field = model.part(&PartSpec { name: "Field".into(), size: Vec3::splat(1.0), ..PartSpec::default() });
		model.child(field, InstanceBuilder::new("ParticleEmitter").with_name("Lift").with_property("Rate", 0.0f32).with_property("Lifetime", NumberRange::new(5.0, 5.0)));
		let mut document = model.into_document();
		let corners = [Vec3::new(-6.0, 0.0, -4.0), Vec3::new(6.0, 0.0, -4.0), Vec3::new(6.0, 0.0, 4.0), Vec3::new(-6.0, 0.0, 4.0)];
		let at = |position: Vec3| -> Vec<f32> { cframe_json(Affine3A::from_translation(position)).as_array().unwrap().iter().map(|n| n.as_f64().unwrap() as f32).collect() };
		// the corner's track: still, then at the frame's last corner
		let motion = vec![Motion { target: "Fx.Plate.Corner".into(), keys: vec![MotionKey { at: 0.0, cframe: vec![0.0, 0.0, 0.0], ..Default::default() }, MotionKey { at: 0.1, cframe: corners[3].to_array().to_vec(), ..Default::default() }], ..Default::default() }];
		let burst = |placed: bool, corner: Vec3| Action { at: 0.1, target: "Fx.Plate.Corner.Burst".into(), emit: Some(Emit::Count(8)), cframe: placed.then(|| at(corner)), ..Default::default() };
		let lift = |count: u32, position: Vec3, size: [f32; 3]| Action { at: 0.1, target: "Fx.Field.Lift".into(), emit: Some(Emit::Count(count)), cframe: Some(at(position)), part_size: Some(size), ..Default::default() };
		let births = |document: &mut Document, placed: bool| -> (Vec<Birth>, Vec<Birth>) {
			let mut actions: Vec<Action> = corners.iter().map(|&corner| burst(placed, corner)).collect();
			actions.extend([lift(3, Vec3::new(0.0, 0.2, 0.0), [12.0, 0.1, 8.0]), lift(5, Vec3::new(10.0, 0.2, 0.0), [4.0, 0.1, 4.0])]);
			let playback = Playback { time: 0.5, actions, motion: motion.clone(), ..Default::default() };
			let resolved = playback.resolve(document).unwrap();
			let undo = resolved.apply_state(document).unwrap();
			let mut scene = crate::scene::extract::extract(document);
			resolved.apply_history(document, &mut scene);
			undo.restore(document);
			let of = |name: &str| scene.particles.iter().find(|emitter| emitter.path.ends_with(name)).and_then(|emitter| emitter.births.clone()).unwrap_or_default();
			(of("Burst"), of("Lift"))
		};
		let (bursts, lifts) = births(&mut document, true);
		assert_eq!(bursts.len(), 32);
		for (index, corner) in corners.iter().enumerate() {
			let here = bursts.iter().filter(|birth| (Vec3::from(birth.cframe.translation) - *corner).length() < 1e-4).count();
			assert_eq!(here, 8, "corner {index} ({corner})");
		}
		// a part's emitter over the part as it was at each call
		assert_eq!(lifts.len(), 8);
		assert!(lifts.iter().filter(|birth| birth.size == Vec3::new(12.0, 0.1, 8.0) && birth.cframe.translation.x == 0.0).count() == 3, "{lifts:?}");
		assert!(lifts.iter().filter(|birth| birth.size == Vec3::new(4.0, 0.1, 4.0) && birth.cframe.translation.x == 10.0).count() == 5, "{lifts:?}");
		// a recording without them: every burst where the motion has the corner at that frame
		let (bursts, _) = births(&mut document, false);
		assert_eq!(bursts.len(), 32);
		assert!(bursts.iter().all(|birth| (Vec3::from(birth.cframe.translation) - corners[3]).length() < 1e-4));
		// a cframe goes with a count, and has twelve numbers
		let mode = Action { emit: Some(Emit::Mode("attributes".into())), ..burst(true, corners[0]) };
		assert!(Playback { actions: vec![mode], ..Default::default() }.resolve(&document).is_err());
		let short = Action { cframe: Some(vec![1.0, 2.0, 3.0]), ..burst(true, corners[0]) };
		assert!(Playback { actions: vec![short], ..Default::default() }.resolve(&document).is_err());
	}

	/// An instance reference set by a playback names the instance by its path, null for none: a beam
	/// parented without attachments and given them later (a recording of a script that does so: ree),
	/// drawn from then on, until one of them is taken away.
	#[test]
	fn references_set_by_path_play_back() {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use crate::scene::extract::extract;
		use rbx_dom_weak::InstanceBuilder;
		use rbx_dom_weak::types::{CFrame, Matrix3, Vector3};
		let mut model = ModelBuilder::new("Fx");
		let holder = model.part(&PartSpec { name: "Holder".into(), ..PartSpec::default() });
		let at = |x: f32| CFrame::new(Vector3::new(x, 0.0, 0.0), Matrix3::identity());
		model.child(holder, InstanceBuilder::new("Attachment").with_name("A").with_property("CFrame", at(-4.0)));
		model.child(holder, InstanceBuilder::new("Attachment").with_name("B").with_property("CFrame", at(4.0)));
		model.child(holder, InstanceBuilder::new("Beam").with_name("Beam"));
		let mut document = model.into_document();
		let set = |at: f32, property: &str, value: serde_json::Value| Action { at, target: "Fx.Holder.Beam".into(), set: [(property.to_string(), value)].into_iter().collect(), ..Default::default() };
		let playback = Playback {
			actions: vec![
				set(0.1, "Attachment0", serde_json::json!("Fx.Holder.A")),
				set(0.1, "Attachment1", serde_json::json!("Fx.Holder.B")),
				set(0.3, "Attachment1", serde_json::Value::Null),
			],
			..Default::default()
		};
		let beams_at = |document: &mut Document, time: f32| -> Vec<(Vec3, Vec3)> {
			let undo = state_at(&playback, document, time);
			let beams = extract(document).beams.iter().map(|beam| (Vec3::from(beam.attachment0.translation), Vec3::from(beam.attachment1.translation))).collect();
			undo.restore(document);
			beams
		};
		assert!(beams_at(&mut document, 0.05).is_empty(), "no attachments yet");
		assert_eq!(beams_at(&mut document, 0.2), [(Vec3::new(-4.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0))]);
		assert!(beams_at(&mut document, 0.4).is_empty(), "Attachment1 taken away");
		// put back as the file has it
		let beam = resolve(&document, "Fx.Holder.Beam").unwrap()[0];
		assert!(props::reference(document.dom.get_by_ref(beam).unwrap(), "Attachment0").is_none());
		// a reference takes a path or null, and the path names an instance
		assert!(Playback { actions: vec![set(0.1, "Attachment0", serde_json::json!(4))], ..Default::default() }.resolve(&document).is_err());
		assert!(Playback { actions: vec![set(0.1, "Attachment0", serde_json::json!("Fx.Holder.C"))], ..Default::default() }.resolve(&document).is_err());
	}

	#[test]
	fn events_wait_for_the_next_frame() {
		let frames = [0.0, 0.016, 0.033, 0.05];
		assert_eq!(quantize(&frames, 0.02), 0.033);
		assert_eq!(quantize(&frames, 0.016), 0.016);
		// a TimeScale set in a frame governs that frame's step, from the frame before
		assert_eq!(quantize_step(&frames, 0.02), 0.016);
		assert_eq!(frame_times(None, 0.05, 0.0).len(), 4);
		assert_eq!(frame_times(None, 0.05, 0.05)[0], -3.0 / 60.0);
	}
}
