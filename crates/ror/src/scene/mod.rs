//! A Roblox model → the scene description the renderer draws.
//!
//! [`load`] reads `.rbxm` / `.rbxmx` / `.rbxl` / `.rbxlx` into a DOM, [`extract`] walks it into a
//! [`Scene`]: every visible part with its geometry, material and surface decorations, the lights,
//! and the environment (Lighting, Sky, Atmosphere, post effects). Asset URIs stay unresolved here;
//! the renderer fetches and decodes them through its caches.

pub mod bones;
pub mod beams;
pub mod build;
pub mod character;
pub mod collision;
pub mod environment;
pub mod extract;
pub mod follow;
pub mod guis;
pub mod legacy;
pub mod load;
pub mod particles;
pub mod playback;
pub mod props;
pub mod trails;

use std::sync::Arc;

use glam::{Affine3A, Vec3};

pub use bones::Skeleton;
pub use beams::Beam;
pub use environment::Environment;
pub use particles::ParticleEmitter;
pub use playback::Playback;
pub use trails::Trail;

/// Parts up to this Transparency draw as opaque ones, ambient occlusion included (ao-transparency:
/// 0.005 and 0.01 as 0; 0.015 blends), and only those and Glass take Highlights
/// (vfx-highlight-transparent: 0.01 highlighted, 0.015 not).
pub const OPAQUE_TRANSPARENCY: f32 = 0.0125;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
	pub parts: Vec<Arc<Part>>,
	pub lights: Vec<Light>,
	/// Beams (ribbons between two attachments), drawn with the transparent geometry.
	pub beams: Vec<Beam>,
	/// Highlights (fill and outline over their adornee's parts), drawn over the finished image.
	pub highlights: Vec<Highlight>,
	/// Glass parts at Transparency 1 or more that a Highlight keeps drawing as Glass, a disabled one
	/// too (indices into `parts`; notes/materials.md "Glass").
	pub kept_glass: Vec<usize>,
	/// Particle emitters (the renderer simulates their particles).
	pub particles: Vec<ParticleEmitter>,
	/// Trails (ribbons left by moving attachments: only a playback moves them).
	pub trails: Vec<Trail>,
	/// SurfaceGuis and BillboardGuis (their content drawn by rur).
	pub guis: Vec<guis::Gui>,
	pub environment: Environment,
	/// MaterialService: the material generation and the MaterialVariants parts can use.
	pub materials: Materials,
	/// Things the extraction skipped or could not represent (unsupported classes, odd values).
	pub warnings: Vec<String>,
	/// Each emitter's, beam's and trail's path to the paths of what holds it, the file's topmost
	/// first and its own last (`Document::lineage`): the lints group by what holds an instance
	/// without reading its path back (crate::lint).
	pub lineage: std::collections::HashMap<String, Vec<String>>,
	/// Drawn again and again with its poses changing (a live preview scrubbing an animation): each
	/// skinned part keeps one vertex buffer, written over for a new pose, instead of a cached copy per
	/// pose (allocated and uploaded anew), and its parts' meshes and materials are looked up on the
	/// rendering thread (its first render or a prefetch loads them; render::each_part). Its renders
	/// and prefetches must not overlap one another.
	pub live: bool,
	/// The path segment of the Folder holding another file's world drawn around the model
	/// (`--map`, [`load::Document::add_setting`]): drawn, never framed ([`Scene::framing_subject`]).
	pub setting: Option<String>,
	/// Drawn as a running game draws it (a recording of play, the playback's `play`): a
	/// SurfaceAppearance map that cannot be loaded is left out, where Studio's edit mode hides the part
	/// (measured in a playtest 2026-10-16: refused, unauthorized and missing Color, Normal and Metalness
	/// maps all draw; notes/render.md "SurfaceAppearance").
	pub play: bool,
}

impl Scene {
	/// The scene as a ViewportFrame draws it (notes/render.md "ViewportFrames"): lit by the frame's own
	/// light alone ([`Environment::viewport_frame`]), its parts without local lights, Highlights,
	/// GUIs, particles, beams or trails.
	pub fn in_viewport_frame(&mut self, lighting: environment::ViewportLighting) {
		self.environment = Environment::viewport_frame(lighting);
		self.keep_parts_only();
	}

	/// Whether the scene holds what a ViewportFrame leaves out (local lights, Highlights and the Glass
	/// they keep, GUIs, particles, beams, trails).
	pub(crate) fn has_more_than_parts(&self) -> bool {
		!(self.lights.is_empty() && self.highlights.is_empty() && self.kept_glass.is_empty() && self.guis.is_empty() && self.particles.is_empty() && self.beams.is_empty() && self.trails.is_empty())
	}

	/// Leaves out what a ViewportFrame does not draw ([`Scene::has_more_than_parts`]).
	pub(crate) fn keep_parts_only(&mut self) {
		self.lights.clear();
		self.highlights.clear();
		self.kept_glass.clear();
		self.guis.clear();
		self.particles.clear();
		self.beams.clear();
		self.trails.clear();
	}

	/// What an orbit frames when the scene has a setting ([`Scene::setting`]): its parts, particles,
	/// beams and trails outside it, nothing else. `None` without a setting.
	pub fn framing_subject(&self) -> Option<Scene> {
		let setting = self.setting.as_deref()?;
		let under = format!("{setting}.");
		let outside = |path: &str| path != setting && !path.starts_with(&under);
		Some(Scene {
			parts: self.parts.iter().filter(|part| outside(&part.path)).cloned().collect(),
			particles: self.particles.iter().filter(|emitter| outside(&emitter.path)).cloned().collect(),
			beams: self.beams.iter().filter(|beam| outside(&beam.path)).cloned().collect(),
			trails: self.trails.iter().filter(|trail| outside(&trail.path)).cloned().collect(),
			environment: self.environment.clone(),
			..Default::default()
		})
	}

	/// World-space bounds of every part (each part's oriented box), `None` for an empty scene.
	pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
		let mut min = Vec3::splat(f32::INFINITY);
		let mut max = Vec3::splat(f32::NEG_INFINITY);
		for part in &self.parts {
			let half = part.size * 0.5;
			let extent = part.cframe.matrix3.abs() * glam::Vec3A::from(half);
			let center = Vec3::from(part.cframe.translation);
			min = min.min(center - Vec3::from(extent));
			max = max.max(center + Vec3::from(extent));
		}
		(min.x <= max.x).then_some((min, max))
	}

	/// The box of what is at or under an instance (`--focus`): `path` dotted from the file's root, or
	/// its last names (`Head`, `Armor.Torso`). Its parts' boxes, else (an effect whose parts draw
	/// nothing) its emitters' places and its beams' and trails' ends, at least a stud across; None
	/// when nothing is there.
	pub fn bounds_at(&self, path: &str) -> Option<(Vec3, Vec3)> {
		let path = path.trim_matches('.');
		let (under, last, inside) = (format!("{path}."), format!(".{path}"), format!(".{path}."));
		let at = |other: &str| other == path || other.starts_with(&under) || other.ends_with(&last) || other.contains(&inside);
		let mut boxes: Vec<(Vec3, Vec3)> = self
			.parts
			.iter()
			.filter(|part| at(&part.path))
			.map(|part| {
				let extent = Vec3::from(part.cframe.matrix3.abs() * glam::Vec3A::from(part.size * 0.5));
				let center = Vec3::from(part.cframe.translation);
				(center - extent, center + extent)
			})
			.collect();
		if boxes.is_empty() {
			let point = |cframe: &Affine3A, half: Vec3| {
				let center = Vec3::from(cframe.translation);
				(center - half, center + half)
			};
			boxes.extend(self.particles.iter().filter(|emitter| at(&emitter.path)).map(|emitter| point(&emitter.cframe, emitter.size * 0.5)));
			boxes.extend(self.beams.iter().filter(|beam| at(&beam.path)).flat_map(|beam| [point(&beam.attachment0, Vec3::ZERO), point(&beam.attachment1, Vec3::ZERO)]));
			boxes.extend(self.trails.iter().filter(|trail| at(&trail.path)).flat_map(|trail| [point(&trail.cframe0, Vec3::ZERO), point(&trail.cframe1, Vec3::ZERO)]));
		}
		let (low, high) = boxes.into_iter().reduce(|(low, high), (min, max)| (low.min(min), high.max(max)))?;
		let (middle, half) = ((low + high) * 0.5, ((high - low) * 0.5).max(Vec3::splat(0.5)));
		Some((middle - half, middle + half))
	}

	/// Bounds for framing the camera and fitting the shadow map: [`Scene::bounds`] without extreme
	/// outliers, parts whose centre lies more than ten times the 90th-percentile distance (and at
	/// least 50 studs) from the median centre — a place's kill brick a million studs below the map
	/// would otherwise shrink everything else to a dot.
	pub fn framing_bounds(&self) -> Option<(Vec3, Vec3)> {
		self.framing_bounds_drawn(&[])
	}

	/// [`Scene::framing_bounds`] by what each part draws: `drawn[i]` is part i's drawn box in its own
	/// space, its box where `None`. A mesh can reach far outside its part's box (a SpecialMesh's file
	/// mesh at its Scale: an accessory's Handle) or fill little of it (a sword's thin blade).
	pub fn framing_bounds_drawn(&self, drawn: &[Option<(Vec3, Vec3)>]) -> Option<(Vec3, Vec3)> {
		let world_box = |index: usize, part: &Part| {
			let half = part.size * 0.5;
			let (low, high) = match drawn.get(index) {
				Some(Some(drawn)) => *drawn,
				_ => (-half, half),
			};
			let center = Vec3::from(part.cframe.transform_point3a(glam::Vec3A::from((low + high) * 0.5)));
			let extent = Vec3::from(part.cframe.matrix3.abs() * glam::Vec3A::from((high - low) * 0.5));
			(center - extent, center + extent)
		};
		let union = |parts: &mut dyn Iterator<Item = (usize, &Arc<Part>)>| {
			let mut min = Vec3::splat(f32::INFINITY);
			let mut max = Vec3::splat(f32::NEG_INFINITY);
			for (index, part) in parts {
				let (low, high) = world_box(index, part);
				min = min.min(low);
				max = max.max(high);
			}
			(min.x <= max.x).then_some((min, max))
		};
		if self.parts.len() <= 8 {
			return union(&mut self.parts.iter().enumerate());
		}
		let centers: Vec<Vec3> = self.parts.iter().map(|part| Vec3::from(part.cframe.translation)).collect();
		let median = |axis: usize| {
			let mut values: Vec<f32> = centers.iter().map(|center| center[axis]).collect();
			// the value sorting puts at the middle, found without sorting them all (a 10,636-part map's clip frames)
			let middle = values.len() / 2;
			*values.select_nth_unstable_by(middle, f32::total_cmp).1
		};
		let middle = Vec3::new(median(0), median(1), median(2));
		let mut distances: Vec<f32> = centers.iter().map(|center| center.distance(middle)).collect();
		let tenth = distances.len() * 9 / 10;
		let limit = (*distances.select_nth_unstable_by(tenth, f32::total_cmp).1 * 10.0).max(50.0);
		let mut kept = self.parts.iter().enumerate().filter(|(index, _)| centers[*index].distance(middle) <= limit);
		union(&mut kept).or_else(|| union(&mut self.parts.iter().enumerate()))
	}
}

#[derive(Clone, Debug, PartialEq)]
pub struct Part {
	/// Dotted path from the model root, for diagnostics and `inspect`.
	pub path: String,
	pub class: String,
	/// The part's CFrame (rotation + position), without its size.
	pub cframe: Affine3A,
	pub size: Vec3,
	pub geometry: Geometry,
	/// Part colour, sRGB 0..1.
	pub color: Vec3,
	/// `Enum.Material` value.
	pub material: u32,
	pub material_variant: String,
	pub transparency: f32,
	pub reflectance: f32,
	pub cast_shadow: bool,
	pub double_sided: bool,
	/// `MeshPart.TextureID` / `SpecialMesh.TextureId`.
	pub texture: Option<String>,
	/// `SpecialMesh.VertexColor` (multiplies the mesh texture), 1,1,1 otherwise.
	pub vertex_color: Vec3,
	pub surface_appearance: Option<SurfaceAppearance>,
	pub decals: Vec<Decal>,
	/// `Enum.SurfaceType` per face, in `Enum.NormalId` order (Right, Top, Back, Left, Bottom, Front).
	pub surfaces: [u32; 6],
	/// MeshParts saved with `HasSkinnedMesh`: the Bones that pose their skinned mesh (`bones`).
	pub skeleton: Option<Arc<Skeleton>>,
	/// A Humanoid's body part (a limb of its rig type, R6 `Torso`, R15 `UpperTorso`, ..., a child of
	/// the Humanoid's model): the rig type, `Enum.HumanoidRigType` (0 R6, 1 R15).
	pub body_part: Option<u32>,
	/// A dressed character's body part: the composited texture (body colours and clothing) it draws.
	pub composite: Option<character::Composite>,
	/// Stands in for what Studio draws at random per part (some materials' texture offset, the
	/// Organic noise offset; notes/materials.md "Random offsets"): [`part_seed`] of the part's position
	/// and size as loaded. Studio draws them once per part and load (material-offsets), not again as
	/// the part moves, so a playback moving or resizing it leaves its seed as it was
	/// ([`load::Document::loaded_seed`]).
	pub seed: u64,
	/// What it collides with, for the collisions-only view ([`collision`]); None for parts made in code
	pub collision: Option<Arc<collision::PartCollision>>,
	/// The instance it was read from: what holds it (a Humanoid's model, a Highlight) for the cost
	/// model (crate::cost); None for parts made in code
	pub referent: Option<rbx_dom_weak::types::Ref>,
}

/// A part's seed ([`Part::seed`]) from its position and size: FNV-1a over their bits.
pub fn part_seed(position: Vec3, size: Vec3) -> u64 {
	let mut hash = 0xcbf29ce484222325u64;
	for value in [position.x, position.y, position.z, size.x, size.y, size.z] {
		for byte in value.to_bits().to_le_bytes() {
			hash ^= byte as u64;
			hash = hash.wrapping_mul(0x100000001b3);
		}
	}
	hash
}

#[derive(Clone, Debug, PartialEq)]
pub enum Geometry {
	Block,
	/// Sphere of diameter `min(size)`.
	Ball,
	/// Cylinder along the part's X axis.
	Cylinder,
	Wedge,
	CornerWedge,
	Truss { style: u32 },
	/// A MeshPart: the mesh asset scaled by `size / initial_size`.
	MeshPart { mesh: String, initial_size: Vec3, fidelity: u32 },
	/// A part modified by a SpecialMesh / BlockMesh / CylinderMesh (`mesh_type` is `Enum.MeshType`).
	Special { mesh_type: u32, mesh: Option<String>, scale: Vec3, offset: Vec3 },
	/// A union / negate / intersect operation: its mesh data, inline or behind `AssetId`.
	Union { asset: Option<String>, mesh_data: Option<Arc<Vec<u8>>>, initial_size: Vec3, use_part_color: bool },
	/// The collisions-only view's MeshPart or union: its decoded collision triangles (each hull its own
	/// vertex colour), centred on the part and scaled by `size / initial_size`; `key` names the data
	Collision { mesh: Arc<collision::Triangles>, key: u64, initial_size: Vec3 },
}

/// A `Highlight`: the parts of its adornee it draws on (indices into [`Scene::parts`]) and how.
#[derive(Clone, Debug, PartialEq)]
pub struct Highlight {
	pub parts: Vec<usize>,
	/// sRGB 0..1
	pub fill_color: Vec3,
	pub fill_transparency: f32,
	/// sRGB 0..1
	pub outline_color: Vec3,
	pub outline_transparency: f32,
	/// `Enum.HighlightDepthMode`: 0 AlwaysOnTop, 1 Occluded.
	pub depth_mode: u32,
}

/// `MaterialService`: which built-in material textures apply and the place's MaterialVariants.
#[derive(Clone, Debug, PartialEq)]
pub struct Materials {
	/// `MaterialService.Use2022Materials` (true for places without a MaterialService: every place
	/// made since 2022 uses them).
	pub use_2022: bool,
	pub variants: Vec<MaterialVariant>,
	/// `MaterialService.<Material>Name`: the variant a base material uses unless a part names one.
	pub overrides: Vec<(u32, String)>,
}

impl Default for Materials {
	fn default() -> Self {
		Self { use_2022: true, variants: Vec::new(), overrides: Vec::new() }
	}
}

impl Materials {
	/// The variant a part with `material` and `MaterialVariant = name` uses: the named one, else the
	/// base material's override; `None` for the built-in material.
	pub fn variant(&self, material: u32, name: &str) -> Option<&MaterialVariant> {
		let find = |name: &str| self.variants.iter().find(|variant| variant.base_material == material && variant.name == name);
		if !name.is_empty()
			&& let Some(variant) = find(name)
		{
			return Some(variant);
		}
		let (_, name) = self.overrides.iter().find(|(base, _)| *base == material)?;
		find(name)
	}
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialVariant {
	pub name: String,
	/// `Enum.Material` it applies to.
	pub base_material: u32,
	pub color_map: Option<String>,
	pub normal_map: Option<String>,
	pub metalness_map: Option<String>,
	pub roughness_map: Option<String>,
	pub emissive_mask: Option<String>,
	/// The pack published with the variant (its maps when the per-map properties are empty).
	pub texture_pack: Option<String>,
	pub studs_per_tile: f32,
	/// `Enum.MaterialPattern`: 0 Regular, 1 Organic.
	pub pattern: u32,
	pub emissive_strength: f32,
	/// sRGB 0..1.
	pub emissive_tint: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceAppearance {
	pub color_map: Option<String>,
	pub normal_map: Option<String>,
	pub metalness_map: Option<String>,
	pub roughness_map: Option<String>,
	/// `SurfaceAppearance.TexturePack`: the published pack (its maps when the per-map properties are
	/// empty).
	pub texture_pack: Option<String>,
	/// `Enum.AlphaMode`
	pub alpha_mode: u32,
	/// Tint, sRGB 0..1.
	pub color: Vec3,
	/// `SurfaceAppearance.EmissiveMaskContent`: where it glows (render::part_material).
	pub emissive_mask: Option<String>,
	pub emissive_strength: f32,
	/// `EmissiveTint`, as stored (0..1).
	pub emissive_tint: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Decal {
	/// `Enum.NormalId` of the face it is drawn on.
	pub face: u32,
	pub texture: String,
	/// `Decal.Color3`, sRGB 0..1.
	pub color: Vec3,
	pub transparency: f32,
	pub z_index: i32,
	/// Set for `Texture` (tiled) instances.
	pub tiling: Option<Tiling>,
	/// `UVOffset` and `UVScale`: a Decal shows its image at uv * scale + offset of the face's own uv
	/// (notes/render.md "Decals").
	pub uv_offset: [f32; 2],
	pub uv_scale: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tiling {
	pub studs_per_tile_u: f32,
	pub studs_per_tile_v: f32,
	pub offset_studs_u: f32,
	pub offset_studs_v: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
	Point,
	Spot,
	Surface,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Light {
	pub path: String,
	pub kind: LightKind,
	/// CFrame of what the light is attached to (a part's CFrame, or an Attachment's world CFrame).
	pub cframe: Affine3A,
	/// Size of the parent part (SurfaceLight emits from one of its faces); zero for attachments.
	pub part_size: Vec3,
	/// sRGB 0..1.
	pub color: Vec3,
	pub brightness: f32,
	pub range: f32,
	/// Spot / surface cone angle in degrees.
	pub angle: f32,
	/// `Enum.NormalId` the spot / surface light faces.
	pub face: u32,
	pub shadows: bool,
	/// Placed with an effect at the start of a playback: Studio's voxels of far lights do not take it
	/// up while the camera stays (notes/render.md "Local lights": light-latency).
	pub placed: bool,
}
