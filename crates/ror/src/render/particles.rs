//! ParticleEmitters (notes/render.md "Particles"): a deterministic, seeded snapshot of each emitter's
//! steady state (Studio's own particles are random), turned into quads for every view on the CPU
//! (their orientation depends on the camera) and drawn with the transparent geometry into the
//! √-space scene target.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};

use super::camera::Camera;
use super::pipelines::{DEPTH_FORMAT, HDR_FORMAT, SAMPLE_COUNTS};
use super::resources::TextureUse;
use crate::scene::particles::{ParticleEmitter, color_at, number_at};
use crate::scene::Scene;

const COMMON: &str = include_str!("../../shaders/common.wgsl");
const PARTICLE: &str = include_str!("../../shaders/particle.wgsl");
const LEGACY: &str = include_str!("../../shaders/legacy.wgsl");

/// Unlit particles emit Color² · Brightness at this radiance, like beams and sky texels.
pub const PARTICLE_RADIANCE: f32 = 1.935;

/// No emitter keeps more particles than this in a snapshot.
const MAX_PARTICLES: usize = 4096;
/// Studio's near plane, studs from the camera: what ZOffset brings this close is not drawn (cut
/// between 0.1 and 0.12, a little over 0.1 against rounding).
const STUDIO_NEAR: f32 = 0.11;

/// Random flipbooks step at floor(age · FlipbookFramerate + this) (particle-flipbook-clock half: a
/// Loop grid beside the Random one pins the age at 0.4902-0.4906 s, where every Random cell fits a
/// phase of 0.42-0.45 and none fits floor + 1 or rounding: framerates 46-52 would step at 0.5. With
/// ror's own ages, the 256 Random cells of particle-flipbook-clock and particle-random-frames show
/// Studio's frame at 230 / 239 / 250 / 250 / 249 / 246 / 236 of 256 for phases 0.28 / 0.32 / 0.36 /
/// 0.38 / 0.40 / 0.43 / 0.5, 200 for floor + 1).
const RANDOM_PHASE: f32 = 0.38;

/// Soft particles: a particle's opacity fades in over this many times its Size of view depth in front
/// of the surface behind it (particle-soft, particle-soft-size: linear, exactly one Size).
const SOFT_DEPTH: f32 = 1.0;

/// A flat quad (VelocityPerpendicular, VelocityParallel) does not fade that way, or only within this many studs: an attack's
/// ground ring or scorch mark a fraction of a stud over the floor draws plainly in the engine, where fading over its Size
/// left almost nothing (notes/render.md).
const FLAT_SOFT_DEPTH: f32 = 1e-3;

/// Huge particles fade by their Size alone (vfx-particle-cover, -cover2: a white square covering the
/// view, 6, 60 or 600 studs off, shows 0.99 / 0.94 / 0.83 / 0.65 / 0.50 / 0.30 / 0.11 / 0.01 of itself
/// at Size 1,000 / 3,000 / 10,000 / 20,000 / 30,000 / 45,000 / 60,000 / 70,000 and none beyond;
/// 1 - (Size / 67,441)^0.883 within 0.010 rms. One effect's Flares1 grows to 124,113 studs:
/// ror drew it over the whole view, Studio nothing).
fn huge_fade(size: f32) -> f32 {
	(1.0 - (size.max(0.0) / 67_441.0).powf(0.883)).clamp(0.0, 1.0)
}

/// The most a view's particles may cover, in viewports' worth of blended pixels: beyond it every
/// emitter is thinned evenly, so that no render pass runs long (a GPU pass of seconds can reset the
/// driver for every application). Studio thins nothing (particle-overdraw: 50 to 600 faint particles
/// over the whole view add up in full), so the cap only guards the driver: 4,096 views is 0.35 s of
/// blending at 4K on a 16 GB GPU. A character's aura covers about 6, an orb 1.2, a cutscene's
/// explosion 200-280, its steady emitters 730.
pub const MAX_OVERDRAW: f32 = 4096.0;
/// The seeds of each particle's fixed draws against the thinnings' shares (`thinned`), mixed with its
/// emitter's: the overdraw cap's, the far Rate's.
const OVERDRAW_SEED: u64 = 0x6F76_6572_6472_6177;
const RATE_SEED: u64 = 0x7261_7465_5F6C_6F64;
/// The share of a thinning's kept particles that fade (`thinned`).
const THIN_MARGIN: f32 = 0.2;

/// How many evenly spaced values Size, Squash, Transparency and Color are sampled into.
const SEQUENCE_SAMPLES: usize = 32;

/// A VelocityPerpendicular particle moving within this angle (radians) of straight up or down takes
/// its image's turn from its emitter's frame (`quads`): an effect's portal, 4.4e-8 off, draws nothing in
/// Studio, a copy turned 1e-6 off draws as one tilted by that (vfx-tilt).
const VERTICAL: f32 = 2.5e-7;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ParticleVertex {
	pub position: [f32; 3],
	/// This corner's texture coordinates in the current flipbook frame (xy) and the next (zw).
	pub uv: [f32; 4],
	/// sRGB Color and opacity (1 - Transparency).
	pub color: [f32; 4],
	/// Towards the camera, for LightInfluence (xyz), and the depth over which the particle fades
	/// in front of what is behind it (w, studs: soft particles).
	pub normal: [f32; 4],
	/// x emission (Brightness · PARTICLE_RADIANCE), y LightInfluence, z LightEmission, w the blend
	/// towards the next flipbook frame.
	pub params: [f32; 4],
	/// x the most light emitted per channel (the emitter's cap), y 1 for a texture of premultiplied
	/// colour (legacy effects), z and w the width and threshold of the texel alpha that shows (z 0:
	/// the alpha as is).
	pub extra: [f32; 4],
}

impl ParticleVertex {
	const ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4];

	fn layout() -> wgpu::VertexBufferLayout<'static> {
		wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<ParticleVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &Self::ATTRIBUTES }
	}
}

/// One particle of a snapshot.
#[derive(Clone, Copy, Debug)]
pub struct Particle {
	pub position: Vec3,
	/// How far it moved since it spawned.
	pub moved: Vec3,
	pub velocity: Vec3,
	pub age: f32,
	/// Side of the square, studs, and Squash.
	pub size: f32,
	pub squash: f32,
	/// Degrees, clockwise on screen.
	pub rotation: f32,
	/// sRGB.
	pub color: Vec3,
	pub opacity: f32,
	/// Flipbook frame, the next one and the blend towards it.
	pub frames: (u32, u32, f32),
	/// Emitted by Rate, not by Emit (`quads` thins those far off).
	pub rate: bool,
	/// Its emission's stream (Rate's count up from 1, bursts from 2^40): which particle it is from frame
	/// to frame, whatever dies around it (`quads` thins by it).
	pub stream: u64,
}

/// SplitMix64: a small, fast, seedable generator (the snapshot only needs to be repeatable).
struct Rng(u64);

impl Rng {
	fn new(seed: u64, stream: u64) -> Self {
		let mut rng = Rng(seed ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15));
		rng.next();
		rng
	}

	fn next(&mut self) -> u64 {
		self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
		let mut z = self.0;
		z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
		z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
		z ^ (z >> 31)
	}

	/// Uniform in [0, 1).
	fn unit(&mut self) -> f32 {
		(self.next() >> 40) as f32 / (1u64 << 24) as f32
	}

	fn range(&mut self, (min, max): (f32, f32)) -> f32 {
		min + (max - min) * self.unit()
	}
}

/// The face normals of `Enum.NormalId`.
const NORMALS: [Vec3; 6] = [Vec3::X, Vec3::Y, Vec3::Z, Vec3::NEG_X, Vec3::NEG_Y, Vec3::NEG_Z];

/// A Cylinder's and a Disc's frame in the part for each EmissionDirection (`Enum.NormalId` order):
/// its X axis, the Cylinder's (from its full end to the one ShapePartial scales), and its Y axis, the
/// direction itself, across the Disc (particle-shape-partial: cones from each face; Top keeps the
/// part's axes, Bottom turns them about X, the four sides put Z up).
const SHAPE_FRAMES: [[Vec3; 3]; 6] = [
	[Vec3::Z, Vec3::X, Vec3::Y],
	[Vec3::X, Vec3::Y, Vec3::Z],
	[Vec3::NEG_X, Vec3::Z, Vec3::Y],
	[Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y],
	[Vec3::X, Vec3::NEG_Y, Vec3::NEG_Z],
	[Vec3::X, Vec3::NEG_Z, Vec3::Y],
];

/// Where along a Cylinder's axis a particle starts (0 at its full end, 1 at the end ShapePartial
/// scales) and the radius's scale there, for a uniform `u`: the scale runs linearly from 1 to
/// ShapePartial, through 0 and out again for a negative one (particle-shape-taper: -0.5 closes two
/// thirds along and opens to 0.3 of the radius by the end).
///
/// From ShapePartial 0 to 1 Studio does not spread them evenly: it takes a scale σ evenly between
/// ShapePartial and 1 and puts the particle as far along the axis as the share of the volume (the
/// side's area for the surface style) between the narrow end and the scale σ, so
/// t = (σ³ - p³) / (1 - p³), (σ² - p²) / (1 - p²) on the side: a cone (0) crowds its full end as u³
/// (a fifth of its particles in the first 1 % of its length, the rest thinning towards the tip but
/// never gone; particle-cylinder-axis: 2000 faint particles per Cylinder, their light along the axis
/// read as counts, Studio's quantiles of the cone 0.002 / 0.020 / 0.100 / 0.316 / 0.728 at 0.1 / 0.3 /
/// 0.5 / 0.7 / 0.9 where u³ gives 0.001 / 0.027 / 0.125 / 0.343 / 0.729 and an even spread 0.035 /
/// 0.112 / 0.206 / 0.331 / 0.536; the side's u²: 0.013 / 0.085 / 0.239 / 0.470 / 0.811 against
/// 0.01 / 0.09 / 0.25 / 0.49 / 0.81), and near 1 it is all but even (0.75 and 0.9 within 0.02 of an
/// even spread's quantiles). Past 1 and below 0 the even spread stays: Studio's there crowd a slab of
/// two to three fifths of the particles up to a tenth of the length beyond the wide end, not
/// modelled (14 of one game's Cylinders are above 1, none below 0).
fn taper(partial: f32, surface: bool, u: f32) -> (f32, f32) {
	if (partial - 1.0).abs() < 1e-3 {
		return (u, 1.0);
	}
	if (0.0..1.0).contains(&partial) {
		let power = if surface { 2 } else { 3 };
		let sigma = partial + (1.0 - partial) * u;
		let along = (sigma.powi(power) - partial.powi(power)) / (1.0 - partial.powi(power));
		return (along, 1.0 + (partial - 1.0) * along);
	}
	let scale = if surface {
		// s |s| / 2, a primitive of |s|, from its value at 1 to its value at ShapePartial
		let primitive = 0.5 + u * (partial * partial.abs() * 0.5 - 0.5);
		primitive.signum() * (2.0 * primitive.abs()).sqrt()
	} else {
		(1.0 + u * (partial.powi(3) - 1.0)).cbrt()
	};
	((scale - 1.0) / (partial - 1.0), scale)
}

/// Where a particle starts (emitter space) and the direction it leaves in, before the spread.
fn spawn(emitter: &ParticleEmitter, size: Vec3, rng: &mut Rng) -> (Vec3, Vec3) {
	let normal = NORMALS[emitter.emission_direction.min(5) as usize];
	let half = size * 0.5;
	let sign = match emitter.shape_in_out {
		1 => -1.0,
		2 if rng.unit() < 0.5 => -1.0,
		_ => 1.0,
	};
	if emitter.attachment {
		return (Vec3::ZERO, normal * sign);
	}
	let surface = emitter.shape_style == 1;
	match emitter.shape {
		// Sphere: radial from the centre; ShapePartial keeps a cap about the emission direction
		// (particle-shapes: 0.5 is the upper dome). In a part that is not a cube the ball stretches with
		// it and its particles leave along the stretched radius (particle-shape-ellipse leaving-sphere:
		// a 2 x 1 x 8 part's ball sends them mostly along its 8-stud axis)
		1 => {
			let cap = (std::f32::consts::PI * emitter.shape_partial.clamp(0.0, 1.0)).cos();
			let cos_polar = 1.0 - (1.0 - cap) * rng.unit();
			let azimuth = rng.unit() * std::f32::consts::TAU;
			let sin_polar = (1.0 - cos_polar * cos_polar).max(0.0).sqrt();
			let (a, b) = (normal.any_orthonormal_vector(), normal.cross(normal.any_orthonormal_vector()));
			let direction = normal * cos_polar + (a * azimuth.cos() + b * azimuth.sin()) * sin_polar;
			let radius = if surface { 1.0 } else { rng.unit().cbrt() };
			(direction * radius * half, (direction * half).normalize_or(direction) * sign)
		}
		// Cylinder: along its frame's X axis through the part, round across it, emitting away from the
		// axis; ShapePartial scales the radius from 1 at the axis's start to ShapePartial at its end
		// (particle-shape-partial: 0 a cone, 2 twice as wide at the end), even over the volume or the
		// surface style's slanted side. Across a part that is not square the round stretches to the
		// part's extents and "away from the axis" stretches with it: along the stretched radius, so
		// a flat part sends its particles mostly along its wide side (particle-shape-ellipse leaving:
		// a 2 x 1 x 8 part's Cylinder at 10 studs a second for 0.5 s spreads 6.49 / 1.98 studs along
		// its 8 / 1-stud sides in Studio, 6.55 / 1.87 so, 5.47 / 3.77 by the round's own angle)
		2 => {
			let [axis, across, other] = SHAPE_FRAMES[emitter.emission_direction.min(5) as usize];
			let (along, scale) = taper(emitter.shape_partial, surface, rng.unit());
			let along = along * 2.0 - 1.0;
			let radius = scale.abs() * if surface { 1.0 } else { rng.unit().sqrt() };
			let angle = rng.unit() * std::f32::consts::TAU;
			let stretched = across * (angle.cos() * across.abs().dot(half)) + other * (angle.sin() * other.abs().dot(half));
			let round = stretched.normalize_or(across * angle.cos() + other * angle.sin());
			let point = axis * (along * axis.abs().dot(half)) + stretched * radius;
			(point, round * sign)
		}
		// Disc: round across EmissionDirection, through the part along it, or for the surface style on
		// that face, emitting along it; between |1 - ShapePartial| and 1 of the radius, even over the
		// area (particle-shape-partial: 1 a full disc, 0 and 2 the rim, 0.5 a ring from half the
		// radius, -0.5 one from the radius out to 1.5 times it)
		3 => {
			let [first, across, second] = SHAPE_FRAMES[emitter.emission_direction.min(5) as usize];
			let inner = (1.0 - emitter.shape_partial).powi(2);
			let radius = (inner + (1.0 - inner) * rng.unit()).sqrt();
			let angle = rng.unit() * std::f32::consts::TAU;
			let height = if surface { 1.0 } else { rng.unit() * 2.0 - 1.0 };
			let point = across * (height * across.abs().dot(half)) + (first * (angle.cos() * first.abs().dot(half)) + second * (angle.sin() * second.abs().dot(half))) * radius;
			(point, normal * sign)
		}
		// Box: its volume, or for Surface the face EmissionDirection names (particle-shapes:
		// box-surface leaves as one sheet)
		_ => {
			let mut point = Vec3::new(rng.unit(), rng.unit(), rng.unit()) * 2.0 - 1.0;
			if surface {
				let axis = emitter.emission_direction.min(5) as usize % 3;
				point[axis] = normal[axis];
			}
			(point * half, normal * sign)
		}
	}
}

/// The direction turned by random angles within ±SpreadAngle (degrees): X about the emitter's X axis
/// (its Z for a direction along X), Y about the axis across both (particle-shapes: (30, 0) fans across
/// Z from Top and across Y from Front, (30, 30) fills a square). With both angles set the direction
/// is uniform over the solid angle they bound, the first angle a latitude (its sine uniform, up to
/// 90 degrees) and the second a longitude, where independent angles crowd it towards the poles
/// across both axes (particle-spread-sphere: 300 particles per spread seen from above and the front;
/// Studio's (90, 90) to (360, 360) clouds are even, anisotropy -0.16 to 0 and 13.8-16.8 % of their
/// light within half their radius from the front, ror's +0.29 to +0.44 and 29-46 %, now -0.08 to
/// +0.06 and 13-16 %; a single angle still fans evenly in its plane).
fn spread(direction: Vec3, angles: glam::Vec2, rng: &mut Rng) -> Vec3 {
	let first = if direction.x.abs() > 0.9 { Vec3::Z } else { Vec3::X };
	let second = direction.cross(first).normalize_or(direction.any_orthonormal_vector());
	let (a, b) = if angles.x != 0.0 && angles.y != 0.0 {
		let limit = angles.x.abs().min(90.0).to_radians().sin();
		(((rng.unit() * 2.0 - 1.0) * limit).asin(), (rng.unit() * 2.0 - 1.0) * angles.y.to_radians())
	} else {
		((rng.unit() * 2.0 - 1.0) * angles.x.to_radians(), (rng.unit() * 2.0 - 1.0) * angles.y.to_radians())
	};
	Quat::from_axis_angle(second, b) * (Quat::from_axis_angle(first, a) * direction)
}

/// Studio's particle step under Drag (seconds): a frame at 60 a second.
const DRAG_STEP: f32 = 1.0 / 60.0;

/// Position and velocity after `age` seconds from `start` with velocity `velocity`, under world
/// acceleration and Drag (the speed halves every 1 / Drag seconds). Studio moves a particle a frame at
/// its speed before the drag takes that frame's share, so it goes k·dt / (1 - e^(-k·dt)) as far as
/// the smooth decay would (particle-drag: 300 particles at 20 studs a second in every direction,
/// frozen 0.4 s after the emit, lie on a shell 0.98, 1.045 and 1.10 times ror's at Drag 3, 10 and 20,
/// where these factors are 1.017, 1.059 and 1.120, and 0.96 at Drag 0: the rest is a frame less of
/// motion in that experiment's protocol; 1.39 and 1.80 times at Drag 60 and 115.2, now 1.005 and
/// 0.992). The pull towards the terminal velocity keeps the smooth law (particle-drag-accel: particles
/// at rest pulled up at 40 studs a second² rise 2.467 / 0.953 / 0.361 studs in 0.4 s at Drag 3 / 20 /
/// 60, the smooth law's 2.470 / 0.947 / 0.362; stepped as their speed, 2.421 / 1.033 / 0.489). A
/// legacy effect's systems (`stepped` false) keep the smooth decay their Drag was fitted with.
fn motion(start: Vec3, velocity: Vec3, acceleration: Vec3, drag: f32, age: f32, stepped: bool) -> (Vec3, Vec3) {
	let k = drag * std::f32::consts::LN_2;
	if k.abs() < 1e-5 {
		return (start + velocity * age + acceleration * (0.5 * age * age), velocity + acceleration * age);
	}
	let terminal = acceleration / k;
	let decay = (-k * age).exp();
	let steps = if stepped { k * DRAG_STEP / (1.0 - (-k * DRAG_STEP).exp()) } else { 1.0 };
	let position = start + terminal * (age - (1.0 - decay) / k) + velocity * ((1.0 - decay) / k * steps);
	(position, terminal + (velocity - terminal) * decay)
}

/// The flipbook frame at `age` (of `lifetime`), the next one and the blend towards it.
fn flipbook(emitter: &ParticleEmitter, age: f32, lifetime: f32, rng: &mut Rng) -> (u32, u32, f32) {
	let (columns, rows) = emitter.flipbook_grid();
	let count = (columns * rows).max(1);
	let framerate = rng.range(emitter.flipbook_framerate).max(0.0);
	let start = if emitter.flipbook_start_random { (rng.unit() * count as f32).floor() } else { 0.0 };
	if count == 1 {
		return (0, 0, 0.0);
	}
	let (frame, next, blend) = match emitter.flipbook_mode {
		// OneShot: the frames spread evenly over the lifetime; with FlipbookStartRandom from a random
		// one, the rest of the sheet spread over the life, ending on the last frame (particle-startrandom:
		// the swirl sheet's frames at 0.25, 0.5, 0.75 and 0.95 of the life, χ² 17 / 6 / 3 / 0 against 41 /
		// 66 / 188 / 448 for running on round the sheet; particle-flipbook-random: every first frame)
		1 => {
			let life = (age / lifetime.max(1e-6)).clamp(0.0, 1.0);
			let position = start + (count as f32 - start) * life;
			let frame = (position.floor() as u32).min(count - 1);
			(frame, (frame + 1).min(count - 1), position.fract())
		}
		// PingPong: first to last and back
		2 => {
			let period = 2 * (count - 1);
			let position = start + age * framerate;
			let step = position.floor() as u32 % period;
			let frame = if step < count { step } else { period - step };
			let next = if step + 1 < count { step + 1 } else { period - (step + 1) % period };
			(frame, next.min(count - 1), position.fract())
		}
		// Random: not random. Step n = floor(age · framerate + RANDOM_PHASE) shows frame (n mod count)
		// xor (n / count mod count), the same for every particle of that age, from a random step with
		// FlipbookStartRandom (particle-random-frames: 4 x 4 swirl particles at framerates 1 to 64
		// frozen near 1 s show frames 1-15, then 16-31 with the last bit flipped, 32-47 xor 2, 48-63
		// xor 3, all 64 of two captures; particle-flipbook-random random: 64 emitters alike)
		3 => {
			let position = age * framerate + RANDOM_PHASE;
			let step = position.floor() as u64 + start as u64;
			let count = count as u64;
			let pick = |n: u64| ((n % count) ^ (n / count % count)) as u32 % count as u32;
			(pick(step), pick(step + 1), position.fract())
		}
		// Loop
		_ => {
			let position = start + age * framerate;
			let frame = position.floor() as u32 % count;
			(frame, (frame + 1) % count, position.fract())
		}
	};
	(frame, next, if emitter.flipbook_blend_frames { blend } else { 0.0 })
}

/// A particle emitted `age` seconds ago by the emitter at `frame` (its part's or attachment's world
/// CFrame then, the part of `size` then) moving at `inherited` (its share of the emitter's velocity),
/// or `None` once its lifetime is over (by `lived`, the spawning clock's age, where that is longer).
fn particle(emitter: &ParticleEmitter, frame: &glam::Affine3A, size: Vec3, inherited: Vec3, age: f32, lived: f32, rng: &mut Rng) -> Option<Particle> {
	let lifetime = rng.range(emitter.lifetime);
	if lifetime <= 0.0 || age.max(lived) > lifetime {
		return None;
	}
	let speed = rng.range(emitter.speed);
	let (local, direction) = spawn(emitter, size, rng);
	let direction = spread(direction, emitter.spread_angle, rng);
	let rotation = frame.matrix3;
	let start = Vec3::from(frame.translation) + Vec3::from(rotation * glam::Vec3A::from(local));
	let velocity = Vec3::from(rotation * glam::Vec3A::from(direction)).normalize_or_zero() * speed + inherited;
	// a LockedToPart particle is pulled in its emitter's frame, as it moves with it (particle-locked-
	// accel: pulled at 20 studs a second² along Y from an attachment whose up is the world's -X, Studio's
	// locked particles went 2.33 studs along -X, its free ones 2.34 up; the same in a turned part and
	// with a dragon fire's turn, where Acceleration (0, 1000, 0) lays its flames along the flight)
	let acceleration = if emitter.locked_to_part { Vec3::from(rotation * glam::Vec3A::from(emitter.acceleration)) } else { emitter.acceleration };
	// a ParticleEmitter has no mount; a legacy effect's system does
	let (position, velocity) = motion(start, velocity, acceleration, emitter.drag, age, emitter.mount.is_none());
	let spin = rng.range(emitter.rotation) + rng.range(emitter.rot_speed) * age;
	// envelopes: one random share per particle and sequence places every keypoint at that share of
	// its range, value ± envelope cut to the property's own range (particle-envelope: at half life the
	// sizes spread as widely as when young, where a share per keypoint would narrow them; Size 1 ± 1.5
	// spreads evenly over 0..2.5, none vanish)
	let mut offsets = |keys: &[(f32, f32, f32)], low: f32, high: f32| -> Vec<f32> {
		let share = rng.unit();
		keys.iter()
			.map(|&(_, value, envelope)| {
				let envelope = envelope.abs();
				let (from, to) = ((value - envelope).max(low), (value + envelope).min(high));
				from + (to - from) * share - value
			})
			.collect()
	};
	let size_offsets = offsets(&emitter.size_keys, 0.0, f32::INFINITY);
	let transparency_offsets = offsets(&emitter.transparency, 0.0, 1.0);
	let squash_offsets = offsets(&emitter.squash, f32::NEG_INFINITY, f32::INFINITY);
	let life = (age / lifetime).clamp(0.0, 1.0);
	let frames = flipbook(emitter, age, lifetime, rng);
	// sequences sampled at 32 points (k / 31) and interpolated, as for beams (particle-age: a Size
	// peak at half life shows 2.95 of 3 studs, 2.94 predicted)
	let sampled = |at: &dyn Fn(f32) -> f32| {
		let position = life * (SEQUENCE_SAMPLES - 1) as f32;
		let index = (position.floor() as usize).min(SEQUENCE_SAMPLES - 2);
		let (a, b) = (at(index as f32 / (SEQUENCE_SAMPLES - 1) as f32), at((index + 1) as f32 / (SEQUENCE_SAMPLES - 1) as f32));
		a + (b - a) * (position - index as f32)
	};
	let color = {
		let position = life * (SEQUENCE_SAMPLES - 1) as f32;
		let index = (position.floor() as usize).min(SEQUENCE_SAMPLES - 2);
		let (a, b) = (color_at(&emitter.color, index as f32 / (SEQUENCE_SAMPLES - 1) as f32), color_at(&emitter.color, (index + 1) as f32 / (SEQUENCE_SAMPLES - 1) as f32));
		a.lerp(b, position - index as f32)
	};
	Some(Particle {
		position,
		moved: position - start,
		velocity,
		age,
		size: sampled(&|t| number_at(&emitter.size_keys, t, &size_offsets)).max(0.0),
		squash: sampled(&|t| number_at(&emitter.squash, t, &squash_offsets)),
		rotation: spin,
		color,
		opacity: whole_255ths(emitter, 1.0 - sampled(&|t| number_at(&emitter.transparency, t, &transparency_offsets)).clamp(0.0, 1.0)),
		frames,
		rate: false,
		stream: 0,
	})
}

/// A ParticleEmitter's particle keeps its opacity (1 - Transparency) in whole 255ths, cut down
/// (particle-faint-stack: white additive squares of opacity 0.99 / 1.01 / 1.5 / 1.99 / 2.01 / 3.5
/// 255ths, 40 at one place, show 0 / 24 / 24 / 24 / 73 / 125 over black, nothing below one 255th
/// however many are stacked, where summing them ror showed 24 / 25 / 47 / 72 / 74 / 149; over a lit gray
/// wall alike). A stack of faint particles drew a glow where Studio draws nothing: a
/// cutscene's flare, 40 quads of Transparency 0.97 to 1, its lobes 1.5 times Studio's light.
/// Legacy effects keep their fitted opacities.
fn whole_255ths(emitter: &ParticleEmitter, opacity: f32) -> f32 {
	if emitter.premultiplied {
		return opacity;
	}
	// (a hair over, so that an exact 255th computed in floats keeps itself)
	(opacity * 255.0 + 1e-3).floor() / 255.0
}

/// Every emitter's particles (`simulate`): each on its own core (each seeded by itself: the same
/// snapshot in any order; one after another, a recorded fight's 368 emitters took 1.6 ms of each
/// frame's draws). A live scene keeps to its thread, as its parts do (render::each_part).
pub fn simulate_all(scene: &Scene) -> Vec<Vec<Particle>> {
	if scene.live || scene.particles.len() < 8 || !crate::par::spread(scene.particles.len(), crate::par::CLIP_EMITTERS) {
		scene.particles.iter().map(simulate).collect()
	} else {
		use rayon::prelude::*;
		scene.particles.par_iter().map(simulate).collect()
	}
}

/// An emitter's particles in the snapshot, oldest first: a playback's (each where and when it was
/// emitted), or its steady state (one particle every 1 / Rate seconds of particle time, at a phase
/// from its seed) and the calibration bursts.
pub fn simulate(emitter: &ParticleEmitter) -> Vec<Particle> {
	let mut particles = Vec::new();
	if let Some(births) = &emitter.births {
		for birth in births {
			let mut rng = Rng::new(emitter.seed, birth.stream);
			// a part moved by its CFrame lends its particles no velocity, whatever VelocityInheritance
			// (particle-moving inherit: the same still line as without it)
			let Some(mut particle) = particle(emitter, &birth.cframe, birth.size, Vec3::ZERO, birth.age, birth.lived, &mut rng) else {
				continue;
			};
			// a burst's streams start at 2^40 (scene::playback), Rate's count up from 1
			particle.rate = birth.stream < 1 << 40;
			particle.stream = birth.stream;
			// LockedToPart: the particles move with the emitter since their emission (not as physics moves it)
			if emitter.locked_to_part {
				let moved = birth.follow.unwrap_or(emitter.cframe) * birth.cframe.inverse();
				particle.position = moved.transform_point3(particle.position);
				particle.velocity = moved.transform_vector3(particle.velocity);
			}
			particles.push(particle);
			if particles.len() >= MAX_PARTICLES {
				break;
			}
		}
		particles.sort_by(|a, b| b.age.total_cmp(&a.age));
		return particles;
	}
	for (burst, &(age, count)) in emitter.bursts.iter().enumerate() {
		for index in 0..count {
			let stream = (1 << 40) + ((burst as u64) << 20) + index as u64;
			let mut rng = Rng::new(emitter.seed, stream);
			if let Some(particle) = particle(emitter, &emitter.cframe, emitter.size, Vec3::ZERO, age, age, &mut rng) {
				particles.push(Particle { stream, ..particle });
			}
		}
	}
	if emitter.enabled && emitter.rate > 0.0 {
		let longest = emitter.lifetime.0.max(emitter.lifetime.1);
		let phase = Rng::new(emitter.seed, 0).unit();
		let mut index = 0u64;
		loop {
			let age = (index as f32 + phase) / emitter.rate;
			if age > longest || particles.len() >= MAX_PARTICLES {
				break;
			}
			let mut rng = Rng::new(emitter.seed, index + 1);
			if let Some(particle) = particle(emitter, &emitter.cframe, emitter.size, Vec3::ZERO, age, age, &mut rng) {
				particles.push(Particle { rate: true, stream: index + 1, ..particle });
			}
			index += 1;
		}
	}
	particles.sort_by(|a, b| b.age.total_cmp(&a.age));
	particles
}

/// The particle pipelines (per MSAA sample count, and whether the draw's flipbook cross-fades), their
/// texture bind group layout and sampler; the legacy effects' linear-light pipelines and the composite
/// of their layer over the scene.
pub struct ParticlePipelines {
	layout: wgpu::BindGroupLayout,
	sampler: wgpu::Sampler,
	pipelines: Vec<(u32, bool, wgpu::RenderPipeline)>,
	linear: Vec<(u32, bool, wgpu::RenderPipeline)>,
	pub composite_layout: wgpu::BindGroupLayout,
	pub composite: wgpu::RenderPipeline,
}

impl ParticlePipelines {
	pub fn new(device: &wgpu::Device, frame_layout: &wgpu::BindGroupLayout, cache: Option<&wgpu::PipelineCache>) -> Self {
		let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("particle texture"),
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
					count: None,
				},
				wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
			],
		});
		let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
			label: Some("particle"),
			address_mode_u: wgpu::AddressMode::ClampToEdge,
			address_mode_v: wgpu::AddressMode::ClampToEdge,
			mag_filter: wgpu::FilterMode::Linear,
			min_filter: wgpu::FilterMode::Linear,
			mipmap_filter: wgpu::FilterMode::Linear,
			..Default::default()
		});
		let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("particle"), source: wgpu::ShaderSource::Wgsl(format!("{COMMON}\n{PARTICLE}\n{}", super::pipelines::VOXEL_READ).into()) });
		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("particle"), bind_group_layouts: &[frame_layout, &layout], push_constant_ranges: &[] });
		// premultiplied in √x; the alpha written is opacity · (1 - LightEmission)
		let blend = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add };
		// legacy effects' layer: premultiplied linear light over the layer, its alpha the transmittance
		// of what is behind (cleared to 1)
		let transmittance = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add };
		let pipeline = |samples: u32, cross_fade: bool, entry: &str, blend: wgpu::BlendState| {
			let constants = [("CROSS_FADE", if cross_fade { 1.0 } else { 0.0 })];
			device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
				label: Some(entry),
				layout: Some(&pipeline_layout),
				vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_particle"), buffers: &[ParticleVertex::layout()], compilation_options: Default::default() },
				primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
				depth_stencil: Some(wgpu::DepthStencilState {
					format: DEPTH_FORMAT,
					depth_write_enabled: false,
					depth_compare: wgpu::CompareFunction::GreaterEqual,
					stencil: Default::default(),
					bias: Default::default(),
				}),
				multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
				fragment: Some(wgpu::FragmentState {
					module: &module,
					entry_point: Some(entry),
					targets: &[Some(wgpu::ColorTargetState { format: HDR_FORMAT, blend: Some(blend), write_mask: wgpu::ColorWrites::ALL })],
					compilation_options: wgpu::PipelineCompilationOptions { constants: &constants, ..Default::default() },
				}),
				multiview: None,
				cache,
			})
		};
		let variants = || SAMPLE_COUNTS.iter().flat_map(|&samples| [(samples, false), (samples, true)]);
		let pipelines = variants().map(|(samples, cross_fade)| (samples, cross_fade, pipeline(samples, cross_fade, "fs_particle", wgpu::BlendState { color: blend, alpha: blend }))).collect();
		let linear = variants().map(|(samples, cross_fade)| (samples, cross_fade, pipeline(samples, cross_fade, "fs_particle_linear", wgpu::BlendState { color: blend, alpha: transmittance }))).collect();
		let texture = |binding: u32| wgpu::BindGroupLayoutEntry {
			binding,
			visibility: wgpu::ShaderStages::FRAGMENT,
			ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
			count: None,
		};
		let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("legacy composite"), entries: &[texture(0), texture(1)] });
		let composite_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("legacy composite"), source: wgpu::ShaderSource::Wgsl(format!("{}\n{LEGACY}", super::pipelines::FULLSCREEN).into()) });
		let composite_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("legacy composite"), bind_group_layouts: &[&composite_layout], push_constant_ranges: &[] });
		let composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("legacy composite"),
			layout: Some(&composite_pipeline_layout),
			vertex: wgpu::VertexState { module: &composite_module, entry_point: Some("vs_fullscreen"), buffers: &[], compilation_options: Default::default() },
			primitive: Default::default(),
			depth_stencil: None,
			multisample: Default::default(),
			fragment: Some(wgpu::FragmentState {
				module: &composite_module,
				entry_point: Some("fs_legacy_composite"),
				targets: &[Some(wgpu::ColorTargetState { format: HDR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
				compilation_options: Default::default(),
			}),
			multiview: None,
			cache,
		});
		Self { layout, sampler, pipelines, linear, composite_layout, composite }
	}

	pub fn pipeline(&self, samples: u32, cross_fade: bool) -> &wgpu::RenderPipeline {
		Self::find(&self.pipelines, samples, cross_fade)
	}

	/// The legacy effects' pipeline into their linear-light layer.
	pub fn linear(&self, samples: u32, cross_fade: bool) -> &wgpu::RenderPipeline {
		Self::find(&self.linear, samples, cross_fade)
	}

	fn find(pipelines: &[(u32, bool, wgpu::RenderPipeline)], samples: u32, cross_fade: bool) -> &wgpu::RenderPipeline {
		let found = pipelines.iter().find(|(count, fades, _)| *count == samples && *fades == cross_fade);
		&found.or_else(|| pipelines.iter().find(|(_, fades, _)| *fades == cross_fade)).unwrap_or(&pipelines[0]).2
	}
}

/// Every emitter's particles (camera-independent) and texture bind group.
pub struct PreparedParticles {
	particles: Vec<Vec<Particle>>,
	groups: Vec<Arc<wgpu::BindGroup>>,
}

impl PreparedParticles {
	/// How many particles draw (any opacity).
	pub(super) fn drawn(&self) -> usize {
		self.particles.iter().flatten().filter(|particle| particle.opacity > 0.0).count()
	}

	/// The box the live particles fill at this moment (each a square of its size round its place).
	pub(super) fn bounds(&self) -> Option<(Vec3, Vec3)> {
		self.particles.iter().flatten().filter(|particle| particle.opacity > 0.0).fold(None, |bounds, particle| {
			let half = Vec3::splat(particle.size.max(0.0) * 0.5);
			let (low, high) = (particle.position - half, particle.position + half);
			Some(bounds.map_or((low, high), |(min, max): (Vec3, Vec3)| (min.min(low), max.max(high))))
		})
	}
}

/// A view's particle quads: one vertex buffer, each emitter's vertex range and the point it sorts by.
pub struct ViewParticles {
	buffer: Option<wgpu::Buffer>,
	pub draws: Vec<ParticleDraw>,
	/// The box where its particles with LightInfluence read the lights' voxels (render::voxel_field).
	pub lit: Option<(Vec3, Vec3)>,
}

pub struct ParticleDraw {
	pub emitter: usize,
	pub vertices: Range<u32>,
	/// Whether any of its particles shows two flipbook frames cross-faded (the others read one)
	pub cross_fades: bool,
	pub center: Vec3,
	/// How far its particles moved along the view since they spawned, on average (away from the
	/// camera positive): what orders emitters at one place before their sizes.
	pub moved: f32,
}

impl super::Renderer {
	/// Simulates every emitter and loads its texture; one without a texture draws no particles, as
	/// Studio's (particle-orient-texture: TextureContent none or an empty URI, or the old Texture empty,
	/// in every Orientation, where ror drew solid quads), nor one whose texture cannot be loaded
	/// (an attack run: an emitter whose flipbook the account may not download).
	/// Each emitter's texture bind group, and whether its texture loaded (one that did not draws no
	/// particle).
	pub(super) fn particle_groups(&self, scene: &Scene, missing: &mut Vec<String>) -> Vec<(Arc<wgpu::BindGroup>, bool)> {
		let mut cache: HashMap<Option<String>, (Arc<wgpu::BindGroup>, bool)> = HashMap::new();
		let groups: Vec<(Arc<wgpu::BindGroup>, bool)> = scene
			.particles
			.iter()
			.map(|emitter| {
				cache
					.entry(emitter.texture.clone())
					.or_insert_with(|| {
						let texture = emitter.texture.as_ref().map(|uri| match self.load_whole(&super::transcode::image_key(uri), TextureUse::Color) {
							Ok(texture) => Some(texture),
							Err(error) => {
								missing.push(format!("{}: {error:#} (its particles are not drawn)", emitter.path));
								None
							}
						});
						let loaded = matches!(texture, Some(Some(_)));
						let view = texture.as_ref().and_then(|texture| texture.as_ref()).map_or_else(|| self.defaults.white.view.clone(), |texture| texture.view());
						let group = Arc::new(self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
							label: Some("particle texture"),
							layout: &self.particles.layout,
							entries: &[
								wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
								wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.particles.sampler) },
							],
						}));
						(group, loaded)
					})
					.clone()
			})
			.collect();
		groups
	}

	/// The particles of every emitter (made ahead, a clip's frames on its producer's threads: `ahead`;
	/// else here) with their bind groups (`particle_groups`): none for an emitter whose texture did not
	/// load.
	pub(super) fn prepare_particles(&self, scene: &Scene, groups: &[(Arc<wgpu::BindGroup>, bool)], ahead: Option<Vec<Vec<Particle>>>) -> PreparedParticles {
		let particles = match ahead {
			Some(ahead) if ahead.len() == groups.len() => ahead.into_iter().zip(groups).map(|(particles, (_, loaded))| if *loaded { particles } else { Vec::new() }).collect(),
			_ => {
				let simulated = |(emitter, (_, loaded)): (&ParticleEmitter, &(Arc<wgpu::BindGroup>, bool))| if *loaded { simulate(emitter) } else { Vec::new() };
				if scene.live || scene.particles.len() < 8 || !crate::par::spread(scene.particles.len(), crate::par::CLIP_EMITTERS) {
					scene.particles.iter().zip(groups).map(simulated).collect()
				} else {
					use rayon::prelude::*;
					scene.particles.par_iter().zip(groups.par_iter()).map(simulated).collect()
				}
			}
		};
		PreparedParticles { particles, groups: groups.iter().map(|(group, _)| group.clone()).collect() }
	}

	/// The particles' quads seen from `camera` (a view of `aspect` width / height). Particles that
	/// would cover more than [`MAX_OVERDRAW`] viewports are thinned evenly, noted in `missing`.
	pub(super) fn particle_geometry(&self, scene: &Scene, prepared: &PreparedParticles, camera: &Camera, aspect: f32, missing: &mut Vec<String>, lease: &super::reuse::Lease) -> ViewParticles {
		// emitters on parts following the camera stand under this one (`followed`)
		let moved: Vec<Option<(ParticleEmitter, Vec<Particle>)>> = scene.particles.iter().zip(&prepared.particles).map(|(emitter, particles)| followed(emitter, particles, camera)).collect();
		let emitters = || scene.particles.iter().zip(&prepared.particles).zip(&moved).map(|((emitter, particles), moved)| match moved {
			Some((emitter, particles)) => (emitter, particles.as_slice()),
			None => (emitter, particles.as_slice()),
		});
		let covered: f32 = emitters().map(|(emitter, particles)| coverage(emitter, particles, camera, aspect)).sum();
		let keep = if covered > MAX_OVERDRAW { MAX_OVERDRAW / covered } else { 1.0 };
		if keep < 1.0 {
			missing.push(format!("particles: they would cover the view {covered:.0} times over; drawing {:.1} % of each emitter's", keep * 100.0));
		}
		let mut vertices = Vec::new();
		let mut draws = Vec::new();
		let forward = -Vec3::from(camera.world.matrix3.z_axis);
		// the frame's own projection (render_view's frame uniform): quads off the view are left out
		let view_proj = camera.view_projection(aspect);
		for (index, (emitter, particles)) in emitters().enumerate() {
			let start = vertices.len() as u32;
			quads(emitter, particles, camera, keep, Some(&view_proj), &mut vertices);
			if vertices.len() as u32 > start {
				let moved = particles.iter().map(|particle| particle.moved.dot(forward)).sum::<f32>() / particles.len().max(1) as f32;
				let cross_fades = vertices[start as usize..].iter().any(|vertex| vertex.params[3] != 0.0);
				draws.push(ParticleDraw { emitter: index, vertices: start..vertices.len() as u32, cross_fades, center: Vec3::from(emitter.cframe.translation), moved });
			}
		}
		let buffer = (!vertices.is_empty()).then(|| lease.buffer("particles", wgpu::BufferUsages::VERTEX, bytemuck::cast_slice(&vertices)));
		// each corner read up to PARTICLE_READ studs towards the camera
		let lit = super::voxel_field::bounds(vertices.iter().filter(|vertex| vertex.params[1] > 0.0).map(|vertex| Vec3::from(vertex.position)), super::voxel_field::PARTICLE_READ);
		ViewParticles { buffer, draws, lit }
	}

	/// Draws one emitter of `view` into the main pass, or a legacy effect's into its linear-light layer
	/// (the frame bind group is set).
	pub(super) fn draw_particles(&self, pass: &mut wgpu::RenderPass<'_>, view: &ViewParticles, draw: &ParticleDraw, prepared: &PreparedParticles, samples: u32, linear: bool) {
		let Some(buffer) = &view.buffer else {
			return;
		};
		pass.set_pipeline(if linear { self.particles.linear(samples, draw.cross_fades) } else { self.particles.pipeline(samples, draw.cross_fades) });
		pass.set_vertex_buffer(0, buffer.slice(..));
		pass.set_bind_group(1, prepared.groups[draw.emitter].as_ref(), &[]);
		pass.draw(draw.vertices.clone(), 0..1);
	}
}

/// An emitter on a part a game keeps under the camera (scene::follow), as seen from `camera`: its
/// steady state moved to where the part stands for it. None for any other emitter, and for one a
/// playback emitted (its births were made where the part moved, frame by frame).
pub(crate) fn followed(emitter: &ParticleEmitter, particles: &[Particle], camera: &Camera) -> Option<(ParticleEmitter, Vec<Particle>)> {
	let follow = emitter.follow.filter(|_| emitter.births.is_none())?;
	let offset = follow.offset(camera.eye, -Vec3::from(camera.world.matrix3.z_axis));
	let mut moved = emitter.clone();
	moved.cframe.translation += glam::Vec3A::from(offset);
	let particles = particles.iter().map(|particle| Particle { position: particle.position + offset, ..*particle }).collect();
	Some((moved, particles))
}

/// The corners of a frame's cell in the flipbook grid (left-to-right, top-to-bottom).
fn cell(frame: u32, (columns, rows): (u32, u32)) -> [f32; 4] {
	let (column, row) = (frame % columns, (frame / columns) % rows);
	[column as f32 / columns as f32, row as f32 / rows as f32, (column + 1) as f32 / columns as f32, (row + 1) as f32 / rows as f32]
}

/// How many viewports' worth of pixels an emitter's `particles` cover seen from `camera` (each quad at
/// most one): the size of the blended work they make. A quad counts the part of it on screen (a map's
/// mist, beside and behind a first-person camera as much as before it, counted 160-345 views where
/// the view showed a fraction of that), less what the far Rate thins.
pub(crate) fn coverage(emitter: &ParticleEmitter, particles: &[Particle], camera: &Camera, aspect: f32) -> f32 {
	let m = camera.world.matrix3;
	let (right, up, forward) = (Vec3::from(m.x_axis), Vec3::from(m.y_axis), -Vec3::from(m.z_axis));
	let tan = (camera.fov.to_radians() * 0.5).tan();
	let aspect = aspect.max(1e-3);
	let rate_share = super::distance_lod(camera.eye.distance(Vec3::from(emitter.cframe.translation)));
	particles
		.iter()
		.map(|particle| {
			let drawn = if particle.rate { thinned(rate_share, draw(emitter, RATE_SEED, particle)) } else { 1.0 };
			// ZOffset and Squash keep the area on screen
			let offset = particle.position - camera.eye;
			let depth = offset.dot(forward);
			let size = particle.size;
			if drawn <= 0.0 || depth <= -2.0 * size {
				return 0.0;
			}
			// a quad facing the camera lies at one depth: behind the near plane none of it shows
			if emitter.orientation == 0 && depth < camera.near {
				return 0.0;
			}
			let near = depth.max(camera.near);
			// turned otherwise, a quad reaching past the eye may cover the view from beside it
			if emitter.orientation != 0 && depth < size {
				let view_height = 2.0 * near * tan;
				return drawn * (4.0 * size * size / (view_height * view_height * aspect)).min(1.0);
			}
			// its square on screen (half extents in the view's -1..1), clipped to the view
			let (scale_x, scale_y) = (1.0 / (near * tan * aspect), 1.0 / (near * tan));
			let (x, y, half_x, half_y) = (offset.dot(right) * scale_x, offset.dot(up) * scale_y, size * scale_x, size * scale_y);
			let across = ((x + half_x).min(1.0) - (x - half_x).max(-1.0)).max(0.0);
			let tall = ((y + half_y).min(1.0) - (y - half_y).max(-1.0)).max(0.0);
			drawn * across * tall * 0.25
		})
		.sum()
}

/// How many viewports' worth of pixels `triangles` shade from `view`, overlaps counted each time: each
/// one clipped to the view and measured as it lands on screen. [`coverage`] takes a particle for a
/// square facing the camera, which is cheap and enough to bound a render; a ring lying flat or a quad
/// seen edge-on shades a fraction of that (one effect's peak: 101 viewports by the square, 24 by
/// the engine's pixel-shader count), so what the GPU really does is priced here.
pub(crate) fn shaded_area(view: &Mat4, triangles: impl Iterator<Item = [Vec3; 3]>) -> f32 {
	let inside: [fn(glam::Vec4) -> f32; 5] = [|c| c.w + c.x, |c| c.w - c.x, |c| c.w + c.y, |c| c.w - c.y, |c| c.w - 1e-4];
	let mut area = 0.0;
	let (mut polygon, mut next): (Vec<glam::Vec4>, Vec<glam::Vec4>) = (Vec::with_capacity(9), Vec::with_capacity(9));
	for triangle in triangles {
		polygon.clear();
		polygon.extend(triangle.iter().map(|corner| *view * corner.extend(1.0)));
		for side in inside {
			next.clear();
			for (index, &a) in polygon.iter().enumerate() {
				let b = polygon[(index + 1) % polygon.len()];
				let (from, to) = (side(a), side(b));
				if from >= 0.0 {
					next.push(a);
				}
				if (from >= 0.0) != (to >= 0.0) {
					next.push(a + (b - a) * (from / (from - to)));
				}
			}
			std::mem::swap(&mut polygon, &mut next);
			if polygon.is_empty() {
				break;
			}
		}
		if polygon.len() < 3 {
			continue;
		}
		let (mut twice, first) = (0.0, polygon[0]);
		let first = (first.x / first.w, first.y / first.w);
		for pair in polygon[1..].windows(2) {
			let (a, b) = ((pair[0].x / pair[0].w - first.0, pair[0].y / pair[0].w - first.1), (pair[1].x / pair[1].w - first.0, pair[1].y / pair[1].w - first.1));
			twice += a.0 * b.1 - a.1 * b.0;
		}
		// the view is 2 by 2 in these units
		area += twice.abs() * 0.125;
	}
	area
}

/// The pixels an emitter's particles really shade, in viewports: their quads as drawn.
pub(crate) fn shaded(emitter: &ParticleEmitter, particles: &[Particle], camera: &Camera, aspect: f32) -> f32 {
	let view = camera.view_projection(aspect);
	let mut vertices = Vec::new();
	quads(emitter, particles, camera, 1.0, Some(&view), &mut vertices);
	shaded_area(&view, vertices.chunks_exact(3).map(|t| [Vec3::from(t[0].position), Vec3::from(t[1].position), Vec3::from(t[2].position)]))
}

/// A particle's fixed draw in [0, 1) against a thinning's share, by its emission (`stream`): which
/// particle it is whatever dies around it, from frame to frame.
fn draw(emitter: &ParticleEmitter, seed: u64, particle: &Particle) -> f32 {
	Rng::new(emitter.seed ^ seed, particle.stream).unit()
}

/// How much of a particle a thinning to `share` draws (its opacity scaled), by its `draw`: all of
/// those well below the share, none above, and a margin of THIN_MARGIN of the share between fading
/// out, placed so that the share's worth is drawn (and at most 1.1 times the share's quads). As the
/// share moves with the view, a particle at the margin fades over frames instead of popping in and
/// out (a map's mist, cut to 37-79 % of its particles by the overdraw cap as a camera turned, its
/// clouds popping in a frame and out the next).
fn thinned(share: f32, draw: f32) -> f32 {
	if share >= 1.0 {
		return 1.0;
	}
	if share <= 0.0 {
		return 0.0;
	}
	let margin = THIN_MARGIN * share;
	// the ramp's top: the share drawn is the area under clamp((top - draw) / margin, 0, 1) over [0, 1)
	let top = if 1.0 - share <= margin * 0.5 { 1.0 + margin - (2.0 * margin * (1.0 - share)).sqrt() } else { share + margin * 0.5 };
	((top - draw) / margin).clamp(0.0, 1.0)
}

/// Appends an emitter's particles as quads (triangle lists) seen from `camera`, in draw order; `keep`
/// below 1 draws that share of them (`thinned`).
/// The particles' quads seen from `camera`, as two triangles each, in their order; with `view` (the
/// frame's view-projection) leaving out those no pixel of the view can show ([`off_view`]).
pub fn quads(emitter: &ParticleEmitter, particles: &[Particle], camera: &Camera, keep: f32, view: Option<&Mat4>, out: &mut Vec<ParticleVertex>) {
	let m = camera.world.matrix3;
	let (camera_right, camera_up, camera_back) = (Vec3::from(m.x_axis), Vec3::from(m.y_axis), Vec3::from(m.z_axis));
	let grid = emitter.flipbook_grid();
	let params3 = [emitter.brightness.max(0.0) * PARTICLE_RADIANCE, emitter.light_influence.clamp(0.0, 1.0), emitter.light_emission];
	// far off, an emitter shows that share of what its Rate emits, evenly over its particles' ages, by
	// the camera's distance to its centre (super::distance_lod), and all of a burst (particle-far-rate:
	// squares over boxes 50 to 2,000 studs off, the same size on screen, Rate 40 for 1 s against 40
	// emitted at once: Studio's Rate light 0.88 / 0.64 / 0.38 of the near boxes' at 300 / 500 / 700
	// studs, nothing from 1,000, its bursts alike at every distance; an effect's cloud anchor 1,100 studs
	// off; particle-far-box: 600-stud boxes reaching from 800 and 1,000 studs to 1,400 and 1,600 show
	// nothing, the one centred at 700 its share all along, so neither the nearest point nor each
	// particle's own distance). Legacy effects alike (legacy-far: Size 25 Fires 100 to 1,300 studs
	// off, Studio's light 0.98 / 0.82 / 0.59 / 0.22 of ror's unthinned at 300 / 500 / 700 / 900 studs
	// and none from 1,100; legacy-far-fire: Size 8 ones dim more, by their size, from 200 studs, open)
	// Both thinnings pick particles by a fixed draw of their emission's (`thinned`), not by their place
	// among the living: as the oldest die each frame that place shifts, and picking by it swapped which
	// particles showed from frame to frame (a turning video's mist, vapor and far emitters flickering);
	// nor every 1 / share-th emitted, which swaps neighbours as the share moves with the camera
	let rate_share = super::distance_lod(camera.eye.distance(Vec3::from(emitter.cframe.translation)));
	for particle in particles {
		let rate = if particle.rate { thinned(rate_share, draw(emitter, RATE_SEED, particle)) } else { 1.0 };
		let fade = huge_fade(particle.size) * rate * thinned(keep, draw(emitter, OVERDRAW_SEED, particle));
		if particle.size <= 0.0 || particle.opacity <= 0.0 || fade <= 0.0 {
			continue;
		}
		let mut center = particle.position;
		let mut size = particle.size;
		// ZOffset moves the particle along its view ray by view depth, keeping its size on screen: its
		// depth along the camera's axis goes from z to z - ZOffset; one in front of the camera that this
		// brings to Studio's near plane, 0.1 studs, or past it is not drawn, at any ZOffset
		// (particle-zoffset-near: a square 10 studs off shows at ZOffset 9.88, 0.12 off, not at 9.9;
		// particle-zoffset-close: one 1 stud above the camera and 0.05-0.3 before it goes behind a wall
		// 6 studs off at ZOffset -10, so to depth 10, and at 0.05 without ZOffset shows nothing)
		let depth = (center - camera.eye).dot(-camera_back);
		let moved = depth - emitter.z_offset;
		if depth <= 0.0 || moved <= STUDIO_NEAR {
			continue;
		}
		if emitter.z_offset != 0.0 {
			center = camera.eye + (center - camera.eye) * (moved / depth);
			size *= moved / depth;
		}
		let velocity = particle.velocity.normalize_or_zero();
		// a velocity-oriented particle at rest has no orientation and draws nothing (particle-slow-
		// orient: VelocityParallel and VelocityPerpendicular at Speed 0 show nothing; at 0.0635 studs a
		// second they turn as faster ones)
		if matches!(emitter.orientation, 2 | 3) && velocity == Vec3::ZERO {
			continue;
		}
		// the image's right (u) and up (-v) axes before the rotation
		let (right, up) = match emitter.orientation {
			// about the world's up axis only
			1 => {
				let right = Vec3::Y.cross(camera.eye - center).normalize_or(camera_right);
				(right, Vec3::Y)
			}
			// the image's u along the motion itself (particle-basic: parallel shows the arrow's right
			// edge pointing up the motion), the quad turned about it to face the camera, as a FaceCamera
			// beam (particle-locked-orient side: particles moving half towards the camera show
			// foreshortened, turned 7-16 degrees from their motion's projection on the screen)
			2 => {
				let ray = (camera.eye - center).normalize_or(camera_back);
				let along = if velocity.cross(ray).length_squared() > 1e-8 { velocity } else { camera_up };
				let up = along.cross(ray).normalize_or(camera_up);
				(along, up)
			}
			// facing along the motion, up as near the world's up as it can be (particle-basic:
			// perpendicular, moving at the camera, shows the image mirrored). Moving straight up or down
			// (within VERTICAL), as if tilted towards its emitter's Z axis: up against that when emitted
			// upwards and along it when emitted downwards, by the face it leaves from whatever the sign of
			// its Speed (vfx-perpendicular-reversed: the thirteen frames at Speed -0.5, every image as its
			// face puts it; an effect's aura discs, Top at a negative Speed, keep -Z: the black hole's run at
			// 0.55 s 7.6 of 255 from Studio over the pixels taking the velocity's sign moved, 20.1 with it);
			// along the motion that axis gives way to -X for an emitter moving along its +Z, and one moving
			// along its look vector (-Z) draws nothing (vfx-perpendicular-frames:
			// arrows moving up and down off attachments in thirteen exact frames, every image turned as
			// this puts it, the two emitting Front missing; vfx-tilt: an effect's portal, on a part turned so
			// its Front is the world's up, 4.4e-8 off, draws nothing, 1e-6 off it does). Unturned that is
			// up along -Z either way (vfx-perpendicular-top: seen from above Studio's images point as when
			// tilted towards +Z going up and towards -Z going down), and the residue left along the motion
			// where its length rounds off 1 is taken out (an effect's Burn at 0.112 studs a second drew a
			// quad of no width; 5,984 of one game's emitters move straight up or down)
			3 => {
				let residue = Vec3::Y - velocity * velocity.y;
				let residue = residue - velocity * velocity.dot(residue);
				let up = if residue.length_squared() > VERTICAL * VERTICAL {
					residue.normalize()
				} else {
					// by the face it is emitted from, whichever way its Speed sends it
					let m = emitter.cframe.matrix3;
					let (x, z) = (Vec3::from(m.x_axis).normalize_or(Vec3::X), Vec3::from(m.z_axis).normalize_or(Vec3::Z));
					let emitted = (glam::Mat3::from(m) * NORMALS[emitter.emission_direction.min(5) as usize]).normalize_or(velocity);
					let across = |axis: Vec3| axis - velocity * velocity.dot(axis);
					let tilt = if across(z).length_squared() > 1e-6 {
						across(z)
					} else if emitted.dot(z) > 0.0 {
						across(-x)
					} else {
						continue;
					};
					(tilt * -emitted.y.signum()).normalize()
				};
				(velocity.cross(up), up)
			}
			_ => (camera_right, camera_up),
		};
		// Rotation turns the image counterclockwise on screen (particle-basic: arrow-rot45)
		let (sin, cos) = particle.rotation.to_radians().sin_cos();
		// a legacy effect's spin turns the texture within the upright quad instead
		let (right, up) = if emitter.spin_texture { (right, up) } else { (right * cos + up * sin, up * cos - right * sin) };
		// the quad spans twice Size (particle-basic: square, Size 4 is 8 studs across); Squash s
		// stretches it by 1 + s and narrows it as much, or for s < 0 squashes it to 1 / (1 - s)
		// (particle-basic: arrow-squash 0.5 is 1.52 tall and 0.65 wide; particle-orient: -0.5 the
		// reverse, 1 and 2 stretch 2 and 3 times)
		let stretch = if particle.squash >= 0.0 { 1.0 + particle.squash } else { 1.0 / (1.0 - particle.squash) };
		let (width, height) = (2.0 * size / stretch, 2.0 * size * stretch);
		let (half_right, half_up) = (right * (width * 0.5), up * (height * 0.5));
		let normal = (camera.eye - center).normalize_or(camera_back);
		let (frame, next, blend) = particle.frames;
		let (a, b) = (cell(frame, grid), cell(next, grid));
		let corner = |position: Vec3, u: f32, v: f32| ParticleVertex {
			position: position.into(),
			uv: {
				let (u, v) = if emitter.spin_texture { (0.5 + (u - 0.5) * cos - (v - 0.5) * sin, 0.5 + (u - 0.5) * sin + (v - 0.5) * cos) } else { (u, v) };
				[a[0] + (a[2] - a[0]) * u, a[1] + (a[3] - a[1]) * v, b[0] + (b[2] - b[0]) * u, b[1] + (b[3] - b[1]) * v]
			},
			color: [particle.color.x, particle.color.y, particle.color.z, particle.opacity * fade],
			// its Size before ZOffset's scaling (particle-soft zoffset-gap0.5 fades like gap0.5)
			normal: [normal.x, normal.y, normal.z, if matches!(emitter.orientation, 2 | 3) { FLAT_SOFT_DEPTH } else { particle.size * SOFT_DEPTH }],
			params: [params3[0], params3[1], params3[2], blend],
			extra: [emitter.cap, if emitter.premultiplied { 1.0 } else { 0.0 }, emitter.alpha_ramp.1, emitter.alpha_ramp.0],
		};
		let corners = [center - half_right + half_up, center + half_right + half_up, center - half_right - half_up, center + half_right - half_up];
		if view.is_some_and(|view| off_view(view, &corners)) {
			continue;
		}
		let top_left = corner(corners[0], 0.0, 0.0);
		let top_right = corner(corners[1], 1.0, 0.0);
		let bottom_left = corner(corners[2], 0.0, 1.0);
		let bottom_right = corner(corners[3], 1.0, 1.0);
		out.extend_from_slice(&[top_left, bottom_left, bottom_right, top_left, bottom_right, top_right]);
	}
}

/// Whether a quad's corners all lie beyond one side of the view's clip volume (left, right, bottom or
/// top; in clip space, so corners behind the eye count too), by a margin far past the GPU's own
/// rounding of the same projection: its triangles then cover no pixel, and drawing it changes
/// nothing. SV's views drew all of its 4,800 particles each, 2.6 MB of vertices a view.
fn off_view(view: &Mat4, corners: &[Vec3; 4]) -> bool {
	let clips = corners.map(|corner| *view * corner.extend(1.0));
	let beyond = |side: &dyn Fn(glam::Vec4) -> f32| clips.iter().all(|clip| side(*clip) > 1e-4 * (clip.x.abs() + clip.y.abs() + clip.w.abs()) + 1e-6);
	beyond(&|clip| clip.x - clip.w) || beyond(&|clip| -clip.x - clip.w) || beyond(&|clip| clip.y - clip.w) || beyond(&|clip| -clip.y - clip.w)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A still white emitter on an attachment at the origin, emitting upwards.
	fn emitter() -> ParticleEmitter {
		ParticleEmitter {
			path: "Test.Emitter".into(),
			cframe: glam::Affine3A::IDENTITY,
			size: Vec3::ZERO,
			attachment: true,
			enabled: true,
			rate: 0.0,
			lifetime: (10.0, 10.0),
			speed: (0.0, 0.0),
			spread_angle: glam::Vec2::ZERO,
			acceleration: Vec3::ZERO,
			drag: 0.0,
			velocity_inheritance: 0.0,
			locked_to_part: false,
			emission_direction: 1,
			shape: 0,
			shape_in_out: 0,
			shape_style: 0,
			shape_partial: 1.0,
			rotation: (0.0, 0.0),
			rot_speed: (0.0, 0.0),
			size_keys: vec![(0.0, 1.0, 0.0), (1.0, 1.0, 0.0)],
			squash: vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
			transparency: vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
			color: vec![(0.0, Vec3::ONE)],
			light_emission: 0.0,
			light_influence: 0.0,
			brightness: 1.0,
			texture: None,
			orientation: 0,
			z_offset: 0.0,
			time_scale: 1.0,
			flipbook_layout: 0,
			flipbook_mode: 0,
			flipbook_framerate: (1.0, 1.0),
			flipbook_start_random: false,
			flipbook_blend_frames: true,
			flipbook_size: (1, 1),
			bursts: Vec::new(),
			referent: None,
			births: None,
			follow: None,
			seed: 7,
			emit_count: None,
			mount: None,
			cap: 7.84,
			linear: false,
			premultiplied: false,
			alpha_ramp: (0.0, 0.0),
			spin_texture: false,
		}
	}

	#[test]
	fn cylinders_and_discs_take_their_frame_and_shape_partial() {
		let part = |shape: u32, partial: f32, direction: u32| ParticleEmitter { attachment: false, shape, shape_partial: partial, emission_direction: direction, ..emitter() };
		let size = Vec3::splat(4.0);
		let mut rng = Rng::new(3, 1);
		// a cone (ShapePartial 0) emitting Right runs along the part's Z axis, its tip at +Z
		let cone = part(2, 0.0, 0);
		for _ in 0..200 {
			let (point, _) = spawn(&cone, size, &mut rng);
			let radius = glam::Vec2::new(point.x, point.y).length();
			assert!(radius <= (2.0 - point.z) * 0.5 + 1e-4, "{point}");
		}
		// a Disc at ShapePartial -0.5 emitting Top lies across Y, from the radius out to 1.5 times it
		let ring = part(3, -0.5, 1);
		for _ in 0..200 {
			let (point, direction) = spawn(&ring, size, &mut rng);
			let radius = glam::Vec2::new(point.x, point.z).length();
			assert!((2.0 - 1e-3..=3.0 + 1e-3).contains(&radius), "{point}");
			assert_eq!(direction, Vec3::Y);
		}
		// Studio's cone crowds its full end as u³: half of its particles within an eighth of its length
		let near = (0..2000).filter(|_| taper(0.0, false, rng.unit()).0 < 0.125).count();
		assert!((900..1100).contains(&near), "{near}");
		// near ShapePartial 1 the spread is all but even, and past 1 it is even over the volume
		let (along, scale) = taper(0.999_9, false, 0.5);
		assert!((along - 0.5).abs() < 1e-3 && (scale - 1.0).abs() < 1e-3);
		let wide = (0..2000).filter(|_| taper(2.0, false, rng.unit()).0 < (1.0 + 0.5 * 7.0f32).cbrt() - 1.0).count();
		assert!((900..1100).contains(&wide), "{wide}");
		// across a flat part a Cylinder's particles leave along the stretched radius, mostly along its
		// wide side (a 2 x 1 x 8 part from Top: 92 % of them nearer Z than Y)
		let flat = part(2, 1.0, 1);
		let along_wide = (0..400)
			.filter(|_| {
				let (_, direction) = spawn(&flat, Vec3::new(2.0, 1.0, 8.0), &mut rng);
				direction.z.abs() > direction.y.abs()
			})
			.count();
		assert!(along_wide > 340, "{along_wide}");
	}

	#[test]
	fn steady_state_holds_rate_times_lifetime() {
		let emitter = ParticleEmitter { rate: 40.0, lifetime: (1.5, 1.5), ..emitter() };
		let count = simulate(&emitter).len();
		assert!((59..=61).contains(&count), "{count}");
		assert!(simulate(&ParticleEmitter { enabled: false, ..emitter }).is_empty());
	}

	#[test]
	fn opacity_keeps_whole_255ths() {
		// particle-faint-stack: opacities of 0.99, 1.01, 1.99 and 2.01 255ths draw as 0, 1, 1 and 2
		let opacity = |level: f32| {
			let emitter = ParticleEmitter { rate: 1.0, lifetime: (100.0, 100.0), transparency: vec![(0.0, 1.0 - level / 255.0, 0.0), (1.0, 1.0 - level / 255.0, 0.0)], ..emitter() };
			simulate(&emitter).first().map(|particle| particle.opacity * 255.0)
		};
		for (level, kept) in [(0.99f32, 0.0f32), (1.01, 1.0), (1.99, 1.0), (2.01, 2.0), (255.0, 255.0)] {
			let drawn = opacity(level).expect("a particle");
			assert!((drawn - kept).abs() < 1e-3, "{level} 255ths drew {drawn}");
		}
		// Transparency 0.5 and 0.8, exact in 255ths, keep themselves
		assert!((opacity(127.5).unwrap() - 127.0).abs() < 1e-3);
		assert!((opacity(51.0).unwrap() - 51.0).abs() < 1e-3);
		// a legacy effect's fitted opacities stay as they are
		let legacy = ParticleEmitter { premultiplied: true, ..emitter() };
		assert!((whole_255ths(&legacy, 0.5 / 255.0) - 0.5 / 255.0).abs() < 1e-9);
	}

	#[test]
	fn drag_halves_the_speed_every_one_over_drag_seconds() {
		let (_, velocity) = motion(Vec3::ZERO, Vec3::new(8.0, 0.0, 0.0), Vec3::ZERO, 2.0, 0.5, true);
		assert!((velocity.x - 4.0).abs() < 1e-4, "{velocity}");
		// with acceleration the velocity tends to Acceleration / (Drag · ln 2)
		let (_, velocity) = motion(Vec3::ZERO, Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), 1.0, 100.0, true);
		assert!((velocity.y + 1.0 / std::f32::consts::LN_2).abs() < 1e-3, "{velocity}");
		// the distance as Studio's frames add it up: 24 steps of a sixtieth of a second at Drag 20
		let (position, _) = motion(Vec3::ZERO, Vec3::new(20.0, 0.0, 0.0), Vec3::ZERO, 20.0, 0.4, true);
		let step = (-20.0 * std::f32::consts::LN_2 / 60.0).exp();
		let frames: f32 = (0..24).map(|frame| 20.0 / 60.0 * step.powi(frame)).sum();
		assert!((position.x - frames).abs() < 1e-3, "{position} against {frames}");
	}

	#[test]
	fn flipbook_frames() {
		// OneShot: the 16 frames of a 4 x 4 sheet over the lifetime
		let emitter = ParticleEmitter { flipbook_layout: 2, flipbook_mode: 1, flipbook_blend_frames: false, ..emitter() };
		let mut rng = Rng::new(1, 1);
		assert_eq!(flipbook(&emitter, 0.0, 16.0, &mut rng).0, 0);
		assert_eq!(flipbook(&emitter, 5.5, 16.0, &mut rng).0, 5);
		assert_eq!(flipbook(&emitter, 16.0, 16.0, &mut rng).0, 15);
		// Loop at 2 frames a second wraps after 8 seconds
		let looping = ParticleEmitter { flipbook_mode: 0, flipbook_framerate: (2.0, 2.0), ..emitter };
		assert_eq!(flipbook(&looping, 8.75, 100.0, &mut rng).0, 1);
	}

	#[test]
	fn spread_stays_within_its_angles() {
		let mut rng = Rng::new(3, 3);
		for _ in 0..1000 {
			// (30, 0) from Top turns about X only: no X component, at most 30 degrees off Y
			let direction = spread(Vec3::Y, glam::Vec2::new(30.0, 0.0), &mut rng);
			assert!(direction.x.abs() < 1e-5 && direction.y >= 30f32.to_radians().cos() - 1e-5, "{direction}");
		}
	}

	#[test]
	fn quads_span_twice_the_size() {
		let emitter = ParticleEmitter { size_keys: vec![(0.0, 4.0, 0.0), (1.0, 4.0, 0.0)], bursts: vec![(0.0, 1)], ..emitter() };
		let camera = super::super::camera::CameraSpec::LookAt { eye: [0.0, 0.0, -40.0], target: [0.0, 0.0, 0.0], fov: 20.0 }.place(None, 100, 100);
		let mut vertices = Vec::new();
		quads(&emitter, &simulate(&emitter), &camera, 1.0, None, &mut vertices);
		let xs: Vec<f32> = vertices.iter().map(|vertex| vertex.position[0]).collect();
		let width = xs.iter().cloned().fold(f32::MIN, f32::max) - xs.iter().cloned().fold(f32::MAX, f32::min);
		assert!((width - 8.0).abs() < 1e-4, "{width}");
	}

	#[test]
	fn flat_quads_do_not_fade_by_depth() {
		let camera = super::super::camera::CameraSpec::LookAt { eye: [0.0, 20.0, -40.0], target: [0.0, 0.0, 0.0], fov: 20.0 }.place(None, 100, 100);
		let fade_depth = |orientation| {
			let emitter = ParticleEmitter { orientation, speed: (1.0, 1.0), size_keys: vec![(0.0, 4.0, 0.0), (1.0, 4.0, 0.0)], bursts: vec![(0.0, 1)], ..emitter() };
			let mut vertices = Vec::new();
			quads(&emitter, &simulate(&emitter), &camera, 1.0, None, &mut vertices);
			assert!(!vertices.is_empty(), "orientation {orientation} drew nothing");
			vertices[0].normal[3]
		};
		// camera-facing quads fade in over their Size in front of what is behind them
		assert_eq!(fade_depth(0), 4.0 * SOFT_DEPTH);
		// flat ones, velocity-perpendicular and -parallel, do not: at Size 4 and a fraction of a stud off the floor
		// that fade would leave almost nothing of a ground ring, and the engine draws it plainly
		for orientation in [2, 3] {
			assert!(fade_depth(orientation) < 0.01, "orientation {orientation}");
		}
	}

	#[test]
	fn shaded_area_is_what_lands_on_screen() {
		let camera = super::super::camera::CameraSpec::LookAt { eye: [0.0, 0.0, -40.0], target: [0.0, 0.0, 0.0], fov: 20.0 }.place(None, 100, 100);
		let view = camera.view_projection(1.0);
		let half = 40.0 * 10f32.to_radians().tan();
		let square = |depth: f32, shift: f32, size: f32| {
			let corner = |x: f32, y: f32| Vec3::new(x + shift, y, depth);
			[[corner(-size, -size), corner(size, -size), corner(size, size)], [corner(-size, -size), corner(size, size), corner(-size, size)]]
		};
		// a square a quarter of the view's height across, at the camera's target: 1/16 of the view
		let whole = shaded_area(&view, square(0.0, 0.0, half * 0.25).into_iter());
		assert!((whole - 1.0 / 16.0).abs() < 1e-3, "{whole}");
		// half off the view's right edge: half of it
		let cut = shaded_area(&view, square(0.0, half, half * 0.25).into_iter());
		assert!((cut - 1.0 / 32.0).abs() < 1e-3, "{cut}");
		// wider than the view: all of it, once
		let over = shaded_area(&view, square(0.0, 0.0, half * 5.0).into_iter());
		assert!((over - 1.0).abs() < 1e-3, "{over}");
		// lying flat, seen from 60 degrees off its face: half the area
		let flat = |x: f32, z: f32| {
			let angle = 60f32.to_radians();
			Vec3::new(x * 0.25 * half, z * angle.cos() * 0.25 * half, z * angle.sin() * 0.25 * half)
		};
		let tilted = shaded_area(&view, [[flat(-1.0, -1.0), flat(1.0, -1.0), flat(1.0, 1.0)], [flat(-1.0, -1.0), flat(1.0, 1.0), flat(-1.0, 1.0)]].into_iter());
		assert!((tilted - 1.0 / 32.0).abs() < 2e-3, "{tilted}");
		// edge on, and behind the eye: nothing
		let edge = |x: f32, z: f32| Vec3::new(x, 0.0, z);
		assert!(shaded_area(&view, [[edge(-1.0, -1.0), edge(1.0, -1.0), edge(1.0, 1.0)]].into_iter()) < 1e-6);
		assert!(shaded_area(&view, square(-60.0, 0.0, 1.0).into_iter()) < 1e-6);
	}

	#[test]
	fn overdraw_is_capped() {
		// 20000 particles each filling the view: thinned to about MAX_OVERDRAW of them
		let emitter = ParticleEmitter { size_keys: vec![(0.0, 50.0, 0.0), (1.0, 50.0, 0.0)], bursts: vec![(0.0, 20000)], ..emitter() };
		let camera = super::super::camera::CameraSpec::LookAt { eye: [0.0, 0.0, -40.0], target: [0.0, 0.0, 0.0], fov: 20.0 }.place(None, 100, 100);
		let particles = simulate(&emitter);
		let covered = coverage(&emitter, &particles, &camera, 1.0);
		assert!((covered - 20000.0).abs() < 1e-2, "{covered}");
		let mut vertices = Vec::new();
		quads(&emitter, &particles, &camera, MAX_OVERDRAW / covered, None, &mut vertices);
		// their opacity summed: the cap's worth, in about a tenth more quads (each particle's draw is
		// random: within a few percent of the cap)
		let drawn = vertices.iter().map(|vertex| vertex.color[3]).sum::<f32>() / 6.0;
		assert!((drawn - MAX_OVERDRAW).abs() < MAX_OVERDRAW * 0.05, "{drawn}");
		assert!(vertices.len() / 6 <= (MAX_OVERDRAW * 1.12) as usize, "{}", vertices.len() / 6);
	}

	#[test]
	fn coverage_counts_what_is_on_screen() {
		let camera = super::super::camera::CameraSpec::LookAt { eye: [0.0, 0.0, -40.0], target: [0.0, 0.0, 0.0], fov: 20.0 }.place(None, 100, 100);
		let at = |x: f32, z: f32, size: f32| {
			let emitter = ParticleEmitter { cframe: glam::Affine3A::from_translation(Vec3::new(x, 0.0, z)), size_keys: vec![(0.0, size, 0.0), (1.0, size, 0.0)], bursts: vec![(0.0, 10)], ..emitter() };
			coverage(&emitter, &simulate(&emitter), &camera, 1.0)
		};
		// in the middle of the view, 40 studs off: 10 quads a quarter of its height across
		let tan = 10f32.to_radians().tan();
		let quarter = 40.0 * tan * 0.25;
		assert!((at(0.0, 0.0, quarter) - 10.0 / 16.0).abs() < 1e-4, "{}", at(0.0, 0.0, quarter));
		// as big beside the view, half on it, and wholly off it
		let edge = 40.0 * tan;
		assert!((at(edge, 0.0, quarter) - 10.0 / 32.0).abs() < 1e-4, "{}", at(edge, 0.0, quarter));
		assert_eq!(at(edge + 2.0 * quarter, 0.0, quarter), 0.0);
		// a huge one beside the eye, and one behind it, show nothing
		assert_eq!(at(60.0, -38.0, 20.0), 0.0);
		assert_eq!(at(0.0, -50.0, 20.0), 0.0);
		// one round the eye fills the view
		assert!((at(0.0, -39.0, 20.0) - 10.0).abs() < 1e-4, "{}", at(0.0, -39.0, 20.0));
	}

	#[test]
	fn thinning_fades_a_margin_and_keeps_the_share() {
		let draws: Vec<f32> = (0..100_000).map(|index| (index as f32 + 0.5) / 100_000.0).collect();
		for share in [0.0, 0.01, 0.1, 0.3, 0.5, 0.8, 0.9, 0.93, 0.97, 0.999, 1.0] {
			let drawn = draws.iter().map(|&draw| thinned(share, draw)).sum::<f32>() / draws.len() as f32;
			let quads = draws.iter().filter(|&&draw| thinned(share, draw) > 0.0).count() as f32 / draws.len() as f32;
			assert!((drawn - share).abs() < 1e-3, "{share}: {drawn}");
			assert!(quads <= share * (1.0 + THIN_MARGIN * 0.5) + 1e-3, "{share}: {quads} of the quads");
		}
		// a particle draws more as the share grows, never less
		for &draw in draws.iter().step_by(997) {
			let mut last = 0.0;
			for step in 0..=200 {
				let now = thinned(step as f32 / 200.0, draw);
				assert!(now >= last - 1e-6, "{draw}");
				last = now;
			}
			assert_eq!(last, 1.0);
		}
	}

	#[test]
	fn thinning_keeps_the_same_particles_as_others_die() {
		// the oldest gone (as from one frame to the next) and the cap's share a little different: the
		// particles still drawn are the ones drawn before, but for the few at the share's margin
		let emitter = ParticleEmitter { size_keys: vec![(0.0, 50.0, 0.0), (1.0, 50.0, 0.0)], bursts: vec![(0.0, 2000)], ..emitter() };
		let camera = super::super::camera::CameraSpec::LookAt { eye: [0.0, 0.0, -40.0], target: [0.0, 0.0, 0.0], fov: 20.0 }.place(None, 100, 100);
		let particles = simulate(&emitter);
		let drawn = |list: &[Particle], keep: f32| -> std::collections::HashSet<u64> {
			list.iter()
				.filter(|particle| {
					let mut vertices = Vec::new();
					quads(&emitter, std::slice::from_ref(*particle), &camera, keep, None, &mut vertices);
					!vertices.is_empty()
				})
				.map(|particle| particle.stream)
				.collect()
		};
		let before = drawn(&particles, 0.30);
		let after = drawn(&particles[200..], 0.31);
		let survivors: std::collections::HashSet<u64> = before.iter().filter(|stream| particles[200..].iter().any(|particle| particle.stream == **stream)).copied().collect();
		let lost = survivors.difference(&after).count();
		let gained = after.difference(&survivors).count();
		assert_eq!(lost, 0, "particles that stopped drawing");
		assert!(gained < 40, "{gained} particles started drawing for a 0.01 change of the share");
	}
}
