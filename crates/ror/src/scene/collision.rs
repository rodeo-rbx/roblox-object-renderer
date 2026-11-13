//! The collisions-only view (`render --collision`): every part drawn as what Roblox's physics collides
//! with instead of what it shows. A MeshPart or union is its saved collision data (PhysicalConfigData /
//! PhysicsData) decoded into its convex hulls, each in its own colour, or its box at Box fidelity; a Part
//! is its shape (a SpecialMesh changes what a part draws, not what it collides with; a truss collides as
//! its box). Invisible parts draw too: they often are the collision (its scene is extracted keeping
//! them, `load::Document::keep_invisible`). Effects, lights in the model and textures are left out.
//!
//! Decoding the saved data needs a decoder that is not part of this repository: a ror built without it
//! refuses the view ([`require`]).

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec3;
use rbx_dom_weak::Instance;
use rbx_dom_weak::types::{SharedString, Variant};
use serde::{Deserialize, Serialize};

use super::{Geometry, Part, Scene, props};

/// Saved collision data, as the file holds it (a shared string stays shared between parts)
#[derive(Clone, Debug, PartialEq)]
pub enum Data {
	Shared(SharedString),
	Owned(Arc<Vec<u8>>),
}

impl Data {
	pub fn bytes(&self) -> &[u8] {
		match self {
			Data::Shared(shared) => shared.data(),
			Data::Owned(bytes) => bytes,
		}
	}
}

#[derive(Clone, Debug, PartialEq)]
pub struct PartCollision {
	/// PhysicalConfigData (MeshParts) or PhysicsData (unions); None for other parts
	pub data: Option<Data>,
	/// A part's own shape when a SpecialMesh / BlockMesh / CylinderMesh changes what it draws
	pub shape: Option<Geometry>,
	pub can_collide: bool,
	pub can_query: bool,
	pub can_touch: bool,
}

/// A part's collision properties from its instance; `shape` is its geometry before mesh modifiers
pub fn of(instance: &Instance, shape: Option<Geometry>) -> PartCollision {
	let data = ["PhysicalConfigData", "PhysicsData"].iter().find_map(|name| match instance.properties.get(&rbx_dom_weak::ustr(name))? {
		Variant::SharedString(shared) if !shared.data().is_empty() => Some(Data::Shared(shared.clone())),
		Variant::BinaryString(bytes) if !AsRef::<[u8]>::as_ref(bytes).is_empty() => Some(Data::Owned(Arc::new(AsRef::<[u8]>::as_ref(bytes).to_vec()))),
		_ => None,
	});
	PartCollision {
		data,
		shape,
		can_collide: props::bool(instance, "CanCollide", true),
		can_query: props::bool(instance, "CanQuery", true),
		can_touch: props::bool(instance, "CanTouch", true),
	}
}

/// Decoded collision triangles with a colour per hull, in the mesh's unscaled space centred on its box
#[derive(Clone, Debug, PartialEq)]
pub struct Triangles {
	pub positions: Vec<[f32; 3]>,
	pub normals: Vec<[f32; 3]>,
	pub colors: Vec<[u8; 4]>,
	pub indices: Vec<u32>,
	pub hulls: u32,
}

/// Which parts the view draws: those that collide (the default), are queried, are touched, or all
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Filter {
	#[default]
	Collide,
	Query,
	Touch,
	All,
}

impl Filter {
	pub fn parse(text: &str) -> Option<Filter> {
		match text {
			"collide" => Some(Filter::Collide),
			"query" => Some(Filter::Query),
			"touch" => Some(Filter::Touch),
			"all" => Some(Filter::All),
			_ => None,
		}
	}

	fn keeps(self, collision: &PartCollision) -> bool {
		match self {
			Filter::Collide => collision.can_collide,
			Filter::Query => collision.can_query,
			Filter::Touch => collision.can_touch,
			Filter::All => true,
		}
	}
}

/// SmoothPlastic: an even surface the hulls' colours show on
const SMOOTH_PLASTIC: u32 = 272;
/// Parts that collide as a box or their shape: a neutral grey
const SHAPE_COLOR: Vec3 = Vec3::new(0.72, 0.72, 0.74);
/// MeshParts and unions at Box fidelity: a warmer grey, to tell them from Parts
const BOX_COLOR: Vec3 = Vec3::new(0.80, 0.74, 0.62);


fn fnv1a(bytes: &[u8]) -> u64 {
	let mut hash = 0xcbf29ce484222325u64;
	for &byte in bytes {
		hash ^= byte as u64;
		hash = hash.wrapping_mul(0x100000001b3);
	}
	hash
}

/// Whether this ror can draw the collisions-only view (it decodes saved collision data).
pub const AVAILABLE: bool = false;

/// Fails unless this ror can draw the collisions-only view ([`AVAILABLE`]).
pub fn require() -> anyhow::Result<()> {
	if !AVAILABLE {
		anyhow::bail!("--collision: this ror is built without collision support (decoding saved collision data needs a decoder that is not part of this repository)");
	}
	Ok(())
}

/// Saved collision data decoded: without a decoder, never ([`require`] refuses the view first)
fn decode(_bytes: &[u8], _initial_size: Vec3) -> Result<(String, Option<Triangles>), String> {
	Err("this ror is built without collision support".into())
}

/// Counts of what a view drew
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Summary {
	pub parts: usize,
	pub hulls: u64,
	/// MeshParts and unions per fidelity (as their data says)
	pub fidelities: std::collections::BTreeMap<String, usize>,
	/// Parts the filter left out
	pub filtered: usize,
}

/// The collisions-only scene of `scene`, and what it drew
pub fn view(scene: &Scene, filter: Filter) -> (Scene, Summary) {
	let mut summary = Summary::default();
	let mut warnings = Vec::new();
	// per blob: its fidelity's name and, unless it is Box, its triangles
	let mut decoded: HashMap<u64, (String, Option<Arc<Triangles>>)> = HashMap::new();
	let mut parts = Vec::with_capacity(scene.parts.len());
	for part in &scene.parts {
		let Some(collision) = &part.collision else { continue };
		if !filter.keeps(collision) {
			summary.filtered += 1;
			continue;
		}
		let mut color = SHAPE_COLOR;
		let geometry = match &part.geometry {
			Geometry::MeshPart { initial_size, .. } | Geometry::Union { initial_size, .. } => match &collision.data {
				Some(data) => {
					let bytes = data.bytes();
					let key = fnv1a(bytes);
					let (fidelity, mesh) = decoded.entry(key).or_insert_with(|| match decode(bytes, *initial_size) {
						Ok((fidelity, None)) => (fidelity, None),
						Ok((fidelity, Some(triangles))) => (fidelity, Some(Arc::new(triangles))),
						Err(error) => {
							warnings.push(format!("{}: collision data not decoded ({error}); drawn as its box", part.path));
							("undecoded".into(), None)
						}
					});
					*summary.fidelities.entry(fidelity.clone()).or_default() += 1;
					match mesh {
						Some(mesh) => {
							summary.hulls += mesh.hulls as u64;
							Geometry::Collision { mesh: mesh.clone(), key, initial_size: *initial_size }
						}
						None => {
							color = BOX_COLOR;
							Geometry::Block
						}
					}
				}
				None => {
					warnings.push(format!("{}: no saved collision data; drawn as its box", part.path));
					color = BOX_COLOR;
					Geometry::Block
				}
			},
			Geometry::Truss { .. } => Geometry::Block,
			Geometry::Special { .. } => collision.shape.clone().unwrap_or(Geometry::Block),
			other => other.clone(),
		};
		summary.parts += 1;
		parts.push(Arc::new(Part {
			geometry,
			color,
			material: SMOOTH_PLASTIC,
			material_variant: String::new(),
			transparency: 0.0,
			reflectance: 0.0,
			cast_shadow: true,
			double_sided: true,
			texture: None,
			vertex_color: Vec3::ONE,
			surface_appearance: None,
			decals: Vec::new(),
			surfaces: [0; 6],
			skeleton: None,
			body_part: None,
			composite: None,
			..Part::clone(part)
		}));
	}
	warnings.push(format!(
		"collision view ({filter:?}): {} parts, {} hulls, {} left out by the filter",
		summary.parts, summary.hulls, summary.filtered
	));
	let view = Scene { parts, environment: scene.environment.clone(), materials: scene.materials.clone(), warnings, ..Default::default() };
	(view, summary)
}

#[cfg(test)]
mod tests {
	use super::*;


	fn part(geometry: Geometry, collision: PartCollision) -> Part {
		let mut model = crate::scene::build::ModelBuilder::new("Model");
		model.part(&crate::scene::build::PartSpec { class: "Part", name: "P".into(), size: Vec3::splat(2.0), ..Default::default() });
		let mut part = Part::clone(&crate::scene::extract::extract(&model.into_document()).parts[0]);
		part.geometry = geometry;
		part.collision = Some(Arc::new(collision));
		part
	}


	#[test]
	fn special_meshes_collide_as_their_part() {
		let special = part(
			Geometry::Special { mesh_type: 5, mesh: Some("rbxassetid://1".into()), scale: Vec3::ONE, offset: Vec3::ZERO },
			PartCollision { data: None, shape: Some(Geometry::Cylinder), can_collide: true, can_query: true, can_touch: true },
		);
		let (view, _) = view(&Scene { parts: vec![Arc::new(special)], ..Default::default() }, Filter::Collide);
		assert!(matches!(view.parts[0].geometry, Geometry::Cylinder));
	}
}
