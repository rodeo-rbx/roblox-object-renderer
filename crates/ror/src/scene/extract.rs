//! Walks a loaded document into a [`Scene`].

use std::sync::Arc;

use glam::{Affine3A, Vec3};
use rayon::prelude::*;
use rbx_dom_weak::Instance;
use rbx_dom_weak::types::Ref;

use super::bones::Skeletons;
use super::character::{Character, Characters};
use super::load::Document;
use super::{Decal, Environment, Geometry, Highlight, Light, LightKind, MaterialVariant, Materials, Part, Scene, SurfaceAppearance, Tiling, props};

/// `Enum.MeshType` values
pub mod mesh_type {
	pub const HEAD: u32 = 0;
	pub const TORSO: u32 = 1;
	pub const WEDGE: u32 = 2;
	pub const SPHERE: u32 = 3;
	pub const CYLINDER: u32 = 4;
	pub const FILE_MESH: u32 = 5;
	pub const BRICK: u32 = 6;
}

/// Studio draws at most this many Highlights at once, the first in the file's order
/// (vfx-highlight-many: 300 highlighted cubes, the first 255 drawn; vfx-highlight-cap: all 40 of 40,
/// where Roblox's documentation says 31).
const MAX_HIGHLIGHTS: usize = 255;

/// `Enum.Material` values
const SMOOTH_PLASTIC: u32 = 272;
pub(crate) const GLASS: u32 = 1568;
const FORCEFIELD: u32 = 1584;

pub fn extract(document: &Document) -> Scene {
	extract_roots(document, &document.roots)
}

/// What a walk through one of the document's roots adds to a scene ([`extract_reusing`]): its
/// entries in the scene's lists in walk order, its drawn parts' indices (from the root's first part)
/// and the Highlights and invisible Glass met on the way.
#[derive(Clone)]
struct Piece {
	parts: Vec<Arc<Part>>,
	lights: Vec<Light>,
	beams: Vec<super::beams::Beam>,
	particles: Vec<super::particles::ParticleEmitter>,
	trails: Vec<super::trails::Trail>,
	guis: Vec<super::guis::Gui>,
	variants: Vec<MaterialVariant>,
	warnings: Vec<String>,
	part_indices: Vec<(Ref, usize)>,
	invisible_glass: Vec<(Ref, Part)>,
	highlights: Vec<Ref>,
	any_highlights: Vec<Ref>,
}

/// A stretch of a walk ([`RootReuse`]): one instance alone (a touched root's or container's own step,
/// its children following as stretches of their own), or a whole subtree, its entries kept while
/// nothing in it changes when `kept`.
#[derive(Clone, Copy, Debug)]
enum Unit {
	One(Ref),
	Tree { root: Ref, kept: bool },
}

/// What a moment's state changed inside a stretch: each property (with the value it then holds) and
/// each parent, in the order the state changed them ([`playback::Undo::changes`]). Two moments with
/// the same changes inside a stretch read the same instances there.
type Signature = Vec<(Ref, rbx_dom_weak::Ustr, Option<rbx_dom_weak::types::Variant>)>;

/// A clip's document's walk cut into stretches, the entries of those that read only their own
/// instances as last walked, and what the moment's state had changed in them then
/// ([`extract_reusing`]).
pub struct RootReuse {
	units: Vec<Unit>,
	pieces: Vec<Option<(Signature, Piece)>>,
	/// The kept stretch each instance is in
	unit_of: std::collections::HashMap<Ref, usize>,
}

impl RootReuse {
	/// The stretches of `document`'s walk and those whose entries are kept: its roots no instance a
	/// playback touches (`touched`: `Resolved::touched`) is above and none of whose instances names (a
	/// Ref property) an instance outside it; and inside a root something touched is in, cut at the
	/// touched instances' ancestors, each Model and each part (not a character's limb: its look reads
	/// its model's Humanoid, clothing and body colours) that names nothing outside it and that nothing
	/// outside names (a joint elsewhere poses a skinned part by its bones, a Highlight elsewhere
	/// colours it). A kept stretch is walked again only at a moment whose state changes something in
	/// it otherwise than at its last walk ([`Signature`]): a cutscene's map, its grass and its parts at
	/// rest are walked once, its animated pieces every frame. A walk through one reads only its own
	/// instances (and the document's names and services, which a clip's playback changes alike for
	/// every root, or reads per frame).
	pub fn new(document: &Document, touched: &[Ref]) -> Self {
		use std::collections::{HashMap, HashSet};
		let touched_set: HashSet<Ref> = touched.iter().copied().collect();
		// every touched instance and its ancestors: what a moment changes something at or under
		let mut dirty: HashSet<Ref> = HashSet::new();
		for &referent in touched {
			let mut current = referent;
			while document.dom.get_by_ref(current).is_some() && dirty.insert(current) {
				current = document.dom.get_by_ref(current).map(|instance| instance.parent()).unwrap_or_else(Ref::none);
			}
		}
		// what names each instance (only needed inside a touched root)
		let mut named_by: HashMap<Ref, Vec<Ref>> = HashMap::new();
		if document.roots.iter().any(|root| dirty.contains(root)) {
			for instance in document.dom.descendants() {
				for value in instance.properties.values() {
					if let rbx_dom_weak::types::Variant::Ref(target) = value
						&& target.is_some()
					{
						named_by.entry(*target).or_default().push(instance.referent());
					}
				}
			}
		}
		let mut units = Vec::new();
		for &root in &document.roots {
			// a touched instance above the root (a Workspace's property): walked every time
			let mut above = false;
			let mut current = document.dom.get_by_ref(root).map(|instance| instance.parent()).unwrap_or_else(Ref::none);
			while let Some(instance) = document.dom.get_by_ref(current) {
				above |= touched_set.contains(&current);
				current = instance.parent();
			}
			if dirty.contains(&root) {
				expand(document, root, &dirty, &named_by, above, &mut units);
			} else {
				units.push(Unit::Tree { root, kept: !above && self_contained(document, root, None) });
			}
		}
		let mut unit_of = HashMap::new();
		for (index, unit) in units.iter().enumerate() {
			if let Unit::Tree { root, kept: true } = *unit {
				unit_of.extend(walk_all(document, root).into_iter().map(|referent| (referent, index)));
			}
		}
		Self { pieces: vec![None; units.len()], units, unit_of }
	}

	/// What `undo`'s moment changed in each kept stretch ([`Signature`]; a stretch it changed nothing
	/// in is not listed). A changed parent counts in the stretches of the instance, of its old parent
	/// and of its new one.
	fn signatures(&self, document: &Document, undo: &super::playback::Undo) -> std::collections::HashMap<usize, Signature> {
		let mut signatures: std::collections::HashMap<usize, Signature> = std::collections::HashMap::new();
		for change in undo.changes() {
			match change {
				super::playback::Change::Property(referent, name) => {
					if let Some(&unit) = self.unit_of.get(&referent) {
						let value = document.dom.get_by_ref(referent).and_then(|instance| instance.properties.get(&name).cloned());
						signatures.entry(unit).or_default().push((referent, name, value));
					}
				}
				super::playback::Change::Parent(referent, old) => {
					let new = document.dom.get_by_ref(referent).map(|instance| instance.parent()).unwrap_or_else(Ref::none);
					let entry = (referent, rbx_dom_weak::ustr("\0Parent"), Some(rbx_dom_weak::types::Variant::Ref(new)));
					let mut units: Vec<usize> = [referent, old, new].iter().filter_map(|at| self.unit_of.get(at).copied()).collect();
					units.dedup();
					for unit in units {
						signatures.entry(unit).or_default().push(entry.clone());
					}
				}
			}
		}
		signatures
	}

	/// How many stretches of the walk are kept.
	pub fn count(&self) -> usize {
		self.units.iter().filter(|unit| matches!(unit, Unit::Tree { kept: true, .. })).count()
	}
}

/// The stretches of the walk at and under `referent`, inside a touched root ([`RootReuse::new`]).
fn expand(document: &Document, referent: Ref, dirty: &std::collections::HashSet<Ref>, named_by: &std::collections::HashMap<Ref, Vec<Ref>>, above: bool, units: &mut Vec<Unit>) {
	let Some(instance) = document.dom.get_by_ref(referent) else {
		return;
	};
	let class = instance.class.as_str();
	// a part that is no character's limb (its look reads its model's Humanoid, clothing and body
	// colours: scene::character): kept, touched or not (a part's children draw with it)
	let part = props::class_is_a(class, "BasePart");
	let limb = || document.dom.get_by_ref(instance.parent()).is_some_and(|parent| parent.children().iter().any(|&child| document.dom.get_by_ref(child).is_some_and(|child| child.class == "Humanoid")));
	if part {
		units.push(Unit::Tree { root: referent, kept: !above && !limb() && self_contained(document, referent, Some(named_by)) });
		return;
	}
	if !dirty.contains(&referent) {
		if class == "Model" {
			units.push(Unit::Tree { root: referent, kept: !above && self_contained(document, referent, Some(named_by)) });
			return;
		}
		if class != "Folder" {
			units.push(Unit::Tree { root: referent, kept: false });
			return;
		}
	} else if props::class_is_a(class, "LayerCollector") {
		// (a GUI's walk stops at it)
		units.push(Unit::Tree { root: referent, kept: false });
		return;
	}
	units.push(Unit::One(referent));
	for &child in instance.children() {
		expand(document, child, dirty, named_by, above, units);
	}
}

/// Whether nothing at or under `root` names an instance outside it, nor (with `named_by`) does
/// anything outside it name one inside.
fn self_contained(document: &Document, root: Ref, named_by: Option<&std::collections::HashMap<Ref, Vec<Ref>>>) -> bool {
	let inside: std::collections::HashSet<Ref> = walk_all(document, root).into_iter().collect();
	let outward = inside.iter().filter_map(|&referent| document.dom.get_by_ref(referent)).any(|instance| {
		instance.properties.values().any(|value| matches!(value, rbx_dom_weak::types::Variant::Ref(target) if target.is_some() && !inside.contains(target)))
	});
	let inward = named_by.is_some_and(|named_by| inside.iter().any(|referent| named_by.get(referent).is_some_and(|sources| sources.iter().any(|source| !inside.contains(source)))));
	!outward && !inward
}

/// Every instance at or under `root` (GUI trees' insides too, unlike [`walk_order`]).
fn walk_all(document: &Document, root: Ref) -> Vec<Ref> {
	let mut found = Vec::new();
	let mut stack = vec![root];
	while let Some(current) = stack.pop() {
		let Some(instance) = document.dom.get_by_ref(current) else {
			continue;
		};
		found.push(current);
		stack.extend(instance.children().iter().copied());
	}
	found
}

/// [`extract`], each root `reuse` names walked at its first call only and its entries taken as they
/// were after: a walk through the roots one after another is the walk through all of them
/// ([`walk_order`]), and what the walk keeps across instances (skeletons and characters, each found
/// by its own model, the parts' indices, the Highlights) is carried as the whole walk carries it.
pub fn extract_reusing(document: &Document, reuse: &mut RootReuse, undo: &super::playback::Undo) -> Scene {
	let signatures = reuse.signatures(document, undo);
	extract_with(document, &document.roots, Some((reuse, signatures)))
}

/// The scene of what is at or under `roots` (a ViewportFrame's world: scene::guis), lit by the
/// document's Lighting.
pub fn extract_roots(document: &Document, roots: &[Ref]) -> Scene {
	extract_with(document, roots, None)
}

fn extract_with(document: &Document, roots: &[Ref], reuse: Option<(&mut RootReuse, std::collections::HashMap<usize, Signature>)>) -> Scene {
	let (mut reuse, signatures) = match reuse {
		Some((reuse, signatures)) => (Some(reuse), signatures),
		None => (None, std::collections::HashMap::new()),
	};
	let unchanged: Signature = Vec::new();
	let signature_of = |index: usize| signatures.get(&index).unwrap_or(&unchanged);
	let mut scene = Scene { environment: environment(document).unwrap_or_default(), setting: document.setting.as_deref().map(super::load::path_segment), play: document.play, ..Default::default() };
	if let Some(service) = document.material_service.and_then(|service| document.dom.get_by_ref(service)) {
		scene.materials = read_material_service(document, service);
	}
	let mut skeletons = Skeletons::default();
	let mut characters = Characters::default();
	// each drawn part's index by its referent, and the Highlights met on the way (the enabled ones,
	// and all of them)
	let mut part_indices = std::collections::HashMap::new();
	let mut highlights = Vec::new();
	let mut any_highlights = Vec::new();
	// invisible Glass parts: a Highlight still covers them (vfx-highlights: an effect's black disc is a
	// Highlight at FillTransparency -5 on a Transparency 1 Glass sphere; on SmoothPlastic, and on
	// invisible balls of other materials, Studio draws no Highlight; 53 of one game's 64 effect Highlights
	// adorn only invisible Glass parts), and any Highlight, a disabled one too, draws them as Glass
	// (glass-quirks: balls at Transparency 1, 1.5 and 3 with disabled Highlights; none without one in
	// glass-transparent; an effect's rings are eight such spheres)
	let mut invisible_glass: std::collections::HashMap<Ref, Part> = std::collections::HashMap::new();
	// the roots walked now (all of them, or those `reuse` has no entries of), each with the place its
	// reused entries go: walked roots one after another are the whole walk (walk_order)
	let units: Vec<Unit> = match reuse.as_ref() {
		Some(reuse) => reuse.units.clone(),
		None => roots.iter().map(|&root| Unit::Tree { root, kept: false }).collect(),
	};
	// (each unit, and whether its kept piece stands: read in place below, not copied first)
	let mut plan: Vec<(usize, bool)> = Vec::new();
	let mut order = Vec::new();
	let mut spans: Vec<(usize, usize, std::ops::Range<usize>)> = Vec::new();
	for (index, unit) in units.iter().enumerate() {
		// a kept stretch's entries as its last walk left them, while the moment changes in it what that
		// walk's did
		let kept = reuse
			.as_ref()
			.and_then(|reuse| reuse.pieces.get(index))
			.and_then(Option::as_ref)
			.is_some_and(|(signature, _)| signature == signature_of(index));
		if !kept {
			let start = order.len();
			match *unit {
				Unit::One(referent) => order.push(referent),
				Unit::Tree { root, .. } => order.extend(walk_order(document, std::slice::from_ref(&root))),
			}
			spans.push((index, plan.len(), start..order.len()));
		}
		plan.push((index, kept));
	}
	// the emitters' reads, most of a walk through an effect, on every core ahead of it
	let read = |&referent: &Ref| {
		let instance = document.dom.get_by_ref(referent).filter(|instance| instance.class == "ParticleEmitter")?;
		let mut warnings = Vec::new();
		Some((referent, (super::particles::read(document, referent, instance, &mut warnings), warnings)))
	};
	let mut emitters: std::collections::HashMap<Ref, (Option<super::particles::ParticleEmitter>, Vec<String>)> =
		if crate::par::spread(order.len(), 4096) { order.par_iter().filter_map(read).collect() } else { order.iter().filter_map(read).collect() };
	scene.particles.reserve_exact(emitters.len());
	let mut spans = spans.into_iter().peekable();
	// the kept pieces' parts' indices, wanted only by Highlights (finish_walk): a map's thousands of
	// them put in a map every frame took a millisecond
	let mut kept_indices: Vec<(Ref, usize)> = Vec::new();
	for (step, (index, kept)) in plan.into_iter().enumerate() {
		// a root's entries as its first walk left them (each piece's lists copied straight in: the
		// parts are shared, `Arc`s)
		if kept && let Some((_, piece)) = reuse.as_deref().and_then(|reuse| reuse.pieces[index].as_ref()) {
			let base = scene.parts.len();
			kept_indices.extend(piece.part_indices.iter().map(|&(referent, at)| (referent, base + at)));
			invisible_glass.extend(piece.invisible_glass.iter().cloned());
			highlights.extend_from_slice(&piece.highlights);
			any_highlights.extend_from_slice(&piece.any_highlights);
			scene.parts.extend_from_slice(&piece.parts);
			scene.lights.extend_from_slice(&piece.lights);
			scene.beams.extend_from_slice(&piece.beams);
			scene.particles.extend_from_slice(&piece.particles);
			scene.trails.extend_from_slice(&piece.trails);
			scene.guis.extend_from_slice(&piece.guis);
			scene.materials.variants.extend_from_slice(&piece.variants);
			scene.warnings.extend_from_slice(&piece.warnings);
			continue;
		}
		let Some((_, _, range)) = spans.next_if(|span| span.1 == step) else {
			continue;
		};
		// what a kept stretch's walk adds, with what the moment changed in it
		let keep = matches!(units.get(index), Some(Unit::Tree { kept: true, .. }));
		let before = (scene.parts.len(), scene.lights.len(), scene.beams.len(), scene.particles.len(), scene.trails.len(), scene.guis.len(), scene.materials.variants.len(), scene.warnings.len(), highlights.len(), any_highlights.len());
		let mut met: Vec<Ref> = Vec::new();
		for &referent in &order[range] {
			if keep {
				met.push(referent);
			}
			walk_one(document, referent, &mut scene, &mut skeletons, &mut characters, &mut part_indices, &mut highlights, &mut any_highlights, &mut invisible_glass, &mut emitters);
		}
		if keep && let Some(reuse) = reuse.as_deref_mut() {
			let base = before.0;
			reuse.pieces[index] = Some((signature_of(index).clone(), Piece {
				parts: scene.parts[before.0..].to_vec(),
				lights: scene.lights[before.1..].to_vec(),
				beams: scene.beams[before.2..].to_vec(),
				particles: scene.particles[before.3..].to_vec(),
				trails: scene.trails[before.4..].to_vec(),
				guis: scene.guis[before.5..].to_vec(),
				variants: scene.materials.variants[before.6..].to_vec(),
				warnings: scene.warnings[before.7..].to_vec(),
				part_indices: met.iter().filter_map(|referent| part_indices.get(referent).map(|&at| (*referent, at - base))).collect(),
				invisible_glass: met.iter().filter_map(|referent| invisible_glass.get(referent).map(|part| (*referent, part.clone()))).collect(),
				highlights: highlights[before.8..].to_vec(),
				any_highlights: any_highlights[before.9..].to_vec(),
			}));
		}
	}
	if !highlights.is_empty() {
		part_indices.extend(kept_indices);
	}
	finish_walk(document, &mut scene, part_indices, highlights, any_highlights, invisible_glass);
	scene
}

/// One instance of a walk ([`extract_with`]): what it draws added to `scene` and the walk's state.
#[allow(clippy::too_many_arguments)]
fn walk_one(
	document: &Document,
	referent: Ref,
	scene: &mut Scene,
	skeletons: &mut Skeletons,
	characters: &mut Characters,
	part_indices: &mut std::collections::HashMap<Ref, usize>,
	highlights: &mut Vec<Ref>,
	any_highlights: &mut Vec<Ref>,
	invisible_glass: &mut std::collections::HashMap<Ref, Part>,
	emitters: &mut std::collections::HashMap<Ref, (Option<super::particles::ParticleEmitter>, Vec<String>)>,
) {
	{
		let Some(instance) = document.dom.get_by_ref(referent) else {
			return;
		};
		let class = instance.class.as_str();
		// GUI trees (ScreenGui / SurfaceGui / BillboardGui, ViewportFrame worlds) are not world geometry;
		// SurfaceGuis and BillboardGuis draw their content in the world, a PlayerGui's ScreenGuis on the
		// screen (render::guis)
		if props::class_is_a(class, "LayerCollector") {
			if matches!(class, "SurfaceGui" | "BillboardGui" | "ScreenGui")
				&& let Some(gui) = super::guis::read(document, referent, instance, &mut scene.warnings)
			{
				scene.guis.push(gui);
			}
			return;
		}
		if props::class_is_a(class, "BasePart") && class != "Terrain" {
			let count = scene.parts.len();
			if let Some(part) = extract_part(document, referent, instance, skeletons, characters, scene) {
				invisible_glass.insert(referent, part);
			}
			if scene.parts.len() > count {
				part_indices.insert(referent, count);
			}
		}
		// an attachment outside any part (a Model's, a Folder's) lights from the frame above it times its
		// CFrame, where its particles spawn (light-bones: lights on an attachment under a Model and in a
		// Folder in one shone from its pivot in Studio, drawn by none in ror)
		if matches!(class, "Attachment" | "Bone")
			&& document.dom.get_by_ref(instance.parent()).is_some_and(|parent| !props::is_a(parent, "BasePart") && !props::is_a(parent, "Attachment"))
			&& let Some(world) = super::beams::world_cframe(document, referent)
		{
			attachment_lights(document, referent, world, &mut scene.lights);
		}
		if class == "Highlight" {
			any_highlights.push(referent);
			if props::bool(instance, "Enabled", true) {
				highlights.push(referent);
			}
		}
		if class == "Beam"
			&& let Some(beam) = super::beams::read(document, referent, instance, &mut scene.warnings)
		{
			scene.beams.push(beam);
		}
		if class == "ParticleEmitter"
			&& let Some((emitter, warnings)) = emitters.remove(&referent)
		{
			scene.warnings.extend(warnings);
			scene.particles.extend(emitter);
		}
		if class == "Trail"
			&& let Some(trail) = super::trails::read(document, referent, instance, &mut scene.warnings)
		{
			scene.trails.push(trail);
		}
		// Fire, Smoke, Sparkles, Explosion and the ForceField instance: Studio's built-in particle
		// effects, drawn as the emitters they are made of
		if matches!(class, "Fire" | "Smoke" | "Sparkles" | "Explosion" | "ForceField") {
			let emitters = super::legacy::read(document, referent, instance, &mut scene.warnings);
			scene.particles.extend(emitters);
		}
		// variants exported along with a model (outside a MaterialService) apply after the place's own
		if class == "MaterialVariant" {
			scene.materials.variants.push(read_material_variant(instance));
		}
	}
}

/// After the walk: invisible Glass a Highlight covers, the Highlights, the effects' lineage.
fn finish_walk(document: &Document, scene: &mut Scene, mut part_indices: std::collections::HashMap<Ref, usize>, highlights: Vec<Ref>, any_highlights: Vec<Ref>, mut invisible_glass: std::collections::HashMap<Ref, Part>) {
	if !invisible_glass.is_empty() {
		for &referent in &any_highlights {
			for adorned in adorned_parts(document, referent) {
				if let Some(part) = invisible_glass.remove(&adorned) {
					part_indices.insert(adorned, scene.parts.len());
					scene.kept_glass.push(scene.parts.len());
					scene.parts.push(Arc::new(part));
				}
			}
		}
	}
	scene.highlights = highlights.into_iter().filter_map(|referent| read_highlight(document, referent, &part_indices, &scene.parts)).collect();
	if scene.highlights.len() > MAX_HIGHLIGHTS {
		scene.warnings.push(format!("{} Highlights: Studio draws at most {MAX_HIGHLIGHTS} at once (the first {MAX_HIGHLIGHTS} in the file drawn)", scene.highlights.len()));
		scene.highlights.truncate(MAX_HIGHLIGHTS);
	}
	let effects = scene.particles.iter().filter_map(|emitter| emitter.referent.map(|referent| (&emitter.path, referent)));
	let mut named = std::collections::HashMap::new();
	for (path, referent) in effects.chain(scene.beams.iter().map(|beam| (&beam.path, beam.referent))).chain(scene.trails.iter().map(|trail| (&trail.path, trail.referent))) {
		named.entry(path.clone()).or_insert(referent);
	}
	let lineage = |(path, referent): (String, Ref)| (path, document.lineage(referent));
	scene.lineage = if crate::par::spread(named.len(), 1024) { named.into_par_iter().map(lineage).collect() } else { named.into_iter().map(lineage).collect() };
}

/// The instances at or under `roots` in the order a walk through them meets them: depth first, each
/// instance before its children in their order, a GUI tree's root without what is under it.
fn walk_order(document: &Document, roots: &[Ref]) -> Vec<Ref> {
	let mut order = Vec::new();
	let mut stack: Vec<Ref> = roots.iter().rev().copied().collect();
	while let Some(referent) = stack.pop() {
		let Some(instance) = document.dom.get_by_ref(referent) else {
			continue;
		};
		order.push(referent);
		if props::class_is_a(instance.class.as_str(), "LayerCollector") {
			continue;
		}
		stack.extend(instance.children().iter().rev().copied());
	}
	order
}

/// The instance a Highlight adorns (its Adornee, else its parent) and everything under it.
fn adorned_parts(document: &Document, referent: Ref) -> Vec<Ref> {
	let Some(instance) = document.dom.get_by_ref(referent) else {
		return Vec::new();
	};
	let adornee = match props::get(instance, "Adornee") {
		Some(rbx_dom_weak::types::Variant::Ref(adornee)) if adornee.is_some() => *adornee,
		_ => instance.parent(),
	};
	let mut found = Vec::new();
	let mut stack = vec![adornee];
	while let Some(current) = stack.pop() {
		let Some(node) = document.dom.get_by_ref(current) else {
			continue;
		};
		found.push(current);
		stack.extend(node.children().iter().copied());
	}
	found
}

/// A Highlight over its Adornee, or its parent without one: that instance's parts and every part
/// under it, those that draw opaque or are Glass: a part of Transparency over
/// [`super::OPAQUE_TRANSPARENCY`] takes none (vfx-highlight-transparent: gray cubes at 0.015 to 0.99,
/// AlwaysOnTop and Occluded, show neither fill nor outline, Glass ones at any Transparency both; a
/// Model's cube at 0.5 is left out of its outline; vfx-parity's half-transparent parts showed none).
fn read_highlight(document: &Document, referent: Ref, part_indices: &std::collections::HashMap<Ref, usize>, scene_parts: &[Arc<Part>]) -> Option<Highlight> {
	let instance = document.dom.get_by_ref(referent)?;
	let mut parts: Vec<usize> = adorned_parts(document, referent)
		.into_iter()
		.filter_map(|adorned| part_indices.get(&adorned).copied())
		.filter(|&index| scene_parts[index].transparency <= super::OPAQUE_TRANSPARENCY || scene_parts[index].material == GLASS)
		.collect();
	if parts.is_empty() {
		return None;
	}
	parts.sort_unstable();
	// transparencies below 0 extrapolate the blend (vfx-highlights visible-neg2, an effect's -5)
	Some(Highlight {
		parts,
		fill_color: props::color(instance, "FillColor", Vec3::new(1.0, 0.0, 0.0)),
		fill_transparency: props::f32(instance, "FillTransparency", 0.5).min(1.0),
		outline_color: props::color(instance, "OutlineColor", Vec3::ONE),
		outline_transparency: props::f32(instance, "OutlineTransparency", 0.0).min(1.0),
		depth_mode: props::enum_value(instance, "DepthMode", 0),
	})
}

/// Studio's rounded meshes for an R6 rig's block limbs, and their size.
fn r6_limb_mesh(name: &str) -> Option<(&'static str, Vec3)> {
	match name {
		"Torso" => Some(("rbxasset://avatar/meshes/torso.mesh", Vec3::new(2.0, 2.0, 1.0))),
		"Left Arm" => Some(("rbxasset://avatar/meshes/leftarm.mesh", Vec3::new(1.0, 2.0, 1.0))),
		"Right Arm" => Some(("rbxasset://avatar/meshes/rightarm.mesh", Vec3::new(1.0, 2.0, 1.0))),
		"Left Leg" => Some(("rbxasset://avatar/meshes/leftleg.mesh", Vec3::new(1.0, 2.0, 1.0))),
		"Right Leg" => Some(("rbxasset://avatar/meshes/rightleg.mesh", Vec3::new(1.0, 2.0, 1.0))),
		_ => None,
	}
}

/// A Humanoid's body part as its character draws it: in its BodyColors colour, an R6 rig's block
/// limbs as Studio's rounded limb meshes (or the character's CharacterMesh for the limb), and the
/// composited texture of its body colours and clothing on R6 limb meshes and R15 MeshPart limbs.
fn dress(part: &mut Part, name: &str, character: &Character) {
	if character.body_colors.is_some()
		&& let Some(color) = character.color(name)
	{
		part.color = color;
	}
	if character.rig_type == 0 {
		if part.class != "Part" || !matches!(part.geometry, Geometry::Block) {
			return;
		}
		// the rounded limb meshes, scaled to the part (the R6 compare: Studio shows their bevels and
		// the seams between them); a CharacterMesh's mesh instead, authored at the same size
		let Some((mesh, initial_size)) = r6_limb_mesh(name) else {
			return;
		};
		let mesh = character.limb_mesh(name).map(|limb| limb.mesh.clone()).unwrap_or_else(|| mesh.into());
		part.geometry = Geometry::MeshPart { mesh, initial_size, fidelity: 0 };
	} else if part.class != "MeshPart" || part.texture.is_some() {
		// an R15 limb with a TextureID of its own keeps it (clothing over it is not modelled)
		return;
	}
	if part.surface_appearance.is_none() {
		part.composite = character.composite(name);
	}
}

/// A part's Size (Part's default without one).
fn part_size(instance: &Instance) -> Vec3 {
	props::vec3(instance, "Size", Vec3::new(4.0, 1.0, 2.0))
}

/// The seed ([`Part::seed`]) of a part's CFrame and Size as they are now; `None` without a CFrame.
pub(super) fn current_seed(instance: &Instance) -> Option<u64> {
	let cframe = props::cframe(instance, "CFrame")?;
	Some(super::part_seed(Vec3::from(cframe.translation), part_size(instance)))
}

/// Adds a part to the scene; returns it instead when it is an invisible Glass part (only a Highlight
/// draws it) and the document does not keep invisible parts ([`Document::keep_invisible`]).
fn extract_part(document: &Document, referent: Ref, instance: &Instance, skeletons: &mut Skeletons, characters: &mut Characters, scene: &mut Scene) -> Option<Part> {
	let path = document.path_of(referent);
	let Some(cframe) = props::cframe(instance, "CFrame") else {
		scene.warnings.push(format!("{path}: no CFrame"));
		return None;
	};
	let size = part_size(instance);
	let class = instance.class.to_string();
	let mut texture = None;
	let mut vertex_color = Vec3::ONE;
	let geometry = match class.as_str() {
		"MeshPart" => match props::content(instance, &["MeshContent", "MeshId"]) {
			Some(mesh) => Geometry::MeshPart {
				mesh,
				initial_size: props::vec3(instance, "InitialSize", size),
				fidelity: props::enum_value(instance, "RenderFidelity", 0),
			},
			// A MeshPart without a mesh draws as a block (as Studio does).
			None => Geometry::Block,
		},
		"UnionOperation" | "IntersectOperation" | "PartOperation" => Geometry::Union {
			// current Studio saves the union's asset as Content (a large place: 153 of 156 unions), older
			// files as AssetId
			asset: props::content(instance, &["Content", "AssetId"]),
			// newer files keep the mesh in MeshData2 (a shared string), unpublished unions in
			// SolidMeshHolder (union-sources "holed", "joined")
			mesh_data: ["MeshData", "MeshData2", "SolidMeshHolder"].iter().find_map(|name| props::binary(instance, name)).map(Arc::new),
			initial_size: props::vec3(instance, "InitialSize", size),
			use_part_color: props::bool(instance, "UsePartColor", false),
		},
		"NegateOperation" => return None,
		"WedgePart" => Geometry::Wedge,
		"CornerWedgePart" => Geometry::CornerWedge,
		"TrussPart" => Geometry::Truss { style: props::enum_value(instance, "Style", 0) },
		_ => match props::enum_value(instance, "Shape", 1) {
			0 => Geometry::Ball,
			2 => Geometry::Cylinder,
			3 => Geometry::Wedge,
			4 => Geometry::CornerWedge,
			_ => Geometry::Block,
		},
	};
	// a union without a mesh (only PhysicalConfigData, or ChildData2 sources: Studio does not re-solve
	// them) draws nothing in Studio (union-sources "chair-no-content", "suntemple-no-meshdata",
	// "holed-no-solid")
	if let Geometry::Union { asset: None, mesh_data: None, .. } = geometry {
		scene.warnings.push(format!("{path}: union without a mesh (not drawn, as in Studio)"));
		return None;
	}
	if class == "MeshPart" {
		texture = props::content(instance, &["TextureContent", "TextureID"]);
	}
	let character = characters.body_part(document, instance);
	// what it collides with: its saved data, or its shape for the view's sake should a mesh modifier
	// replace what it draws (scene::collision)
	let shape = (!matches!(geometry, Geometry::MeshPart { .. } | Geometry::Union { .. })).then(|| geometry.clone());
	let collision = Some(Arc::new(super::collision::of(instance, shape)));

	let mut part = Part {
		path: path.clone(),
		class,
		cframe,
		size,
		geometry,
		color: props::color(instance, "Color", Vec3::splat(163.0 / 255.0)),
		material: props::enum_value(instance, "Material", 256),
		// files store the name as MaterialVariantSerialized; MaterialVariant's reflection default is ""
		material_variant: ["MaterialVariantSerialized", "MaterialVariant"].iter().find_map(|name| props::string(instance, name).filter(|value| !value.is_empty())).unwrap_or_default(),
		transparency: props::f32(instance, "Transparency", 0.0),
		reflectance: props::f32(instance, "Reflectance", 0.0),
		cast_shadow: props::bool(instance, "CastShadow", true),
		double_sided: props::bool(instance, "DoubleSided", false),
		texture: None,
		vertex_color,
		surface_appearance: None,
		decals: Vec::new(),
		surfaces: [
			props::enum_value(instance, "RightSurface", 0),
			props::enum_value(instance, "TopSurface", 0),
			props::enum_value(instance, "BackSurface", 0),
			props::enum_value(instance, "LeftSurface", 0),
			props::enum_value(instance, "BottomSurface", 0),
			props::enum_value(instance, "FrontSurface", 0),
		],
		skeleton: None,
		body_part: character.as_ref().map(|character| character.rig_type),
		composite: None,
		seed: document.loaded_seed(referent).unwrap_or_else(|| super::part_seed(Vec3::from(cframe.translation), size)),
		collision,
		referent: Some(referent),
	};
	// Studio deforms only MeshParts saved with HasSkinnedMesh (skin-pose "flag-off")
	if matches!(part.geometry, Geometry::MeshPart { .. }) && props::bool(instance, "HasSkinnedMesh", false) {
		part.skeleton = skeletons.for_part(document, referent);
	}

	let is_mesh_capable = !matches!(part.geometry, Geometry::MeshPart { .. } | Geometry::Union { .. });
	let mut has_mesh_modifier = false;
	for &child_ref in instance.children() {
		let Some(child) = document.dom.get_by_ref(child_ref) else {
			continue;
		};
		match child.class.as_str() {
			// FileMesh, SpecialMesh's base class, draws as a SpecialMesh of MeshType FileMesh
			"SpecialMesh" | "FileMesh" | "BlockMesh" | "CylinderMesh" if is_mesh_capable && !has_mesh_modifier => {
				has_mesh_modifier = true;
				let mesh_type = match child.class.as_str() {
					"BlockMesh" => mesh_type::BRICK,
					"CylinderMesh" => mesh_type::CYLINDER,
					"FileMesh" => mesh_type::FILE_MESH,
					_ => props::enum_value(child, "MeshType", mesh_type::HEAD),
				};
				let mesh = if mesh_type == mesh_type::FILE_MESH { props::content(child, &["MeshContent", "MeshId"]) } else { None };
				texture = props::content(child, &["TextureContent", "TextureId"]);
				// VertexColor stops at 2 per channel (forcefield-specialmesh: an effect's (0.1, 0.9, 12)
				// shows the blue of a VertexColor of 2, on ForceField and transparent Plastic alike)
				vertex_color = props::vec3(child, "VertexColor", Vec3::ONE).clamp(Vec3::ZERO, Vec3::splat(2.0));
				part.geometry = Geometry::Special {
					mesh_type,
					mesh,
					scale: props::vec3(child, "Scale", Vec3::ONE),
					offset: props::vec3(child, "Offset", Vec3::ZERO),
				};
			}
			"SurfaceAppearance" => {
				part.surface_appearance = Some(SurfaceAppearance {
					color_map: props::content(child, &["ColorMapContent", "ColorMap"]),
					normal_map: props::content(child, &["NormalMapContent", "NormalMap"]),
					metalness_map: props::content(child, &["MetalnessMapContent", "MetalnessMap"]),
					roughness_map: props::content(child, &["RoughnessMapContent", "RoughnessMap"]),
					texture_pack: props::content(child, &["TexturePackContent", "TexturePack"]),
					alpha_mode: props::enum_value(child, "AlphaMode", 0),
					color: props::color(child, "Color", Vec3::ONE),
					emissive_mask: props::content(child, &["EmissiveMaskContent", "EmissiveMask"]),
					emissive_strength: props::f32(child, "EmissiveStrength", 1.0),
					emissive_tint: props::color(child, "EmissiveTint", Vec3::ONE),
				});
			}
			"Decal" | "Texture" => {
				let Some(uri) = props::content(child, &["TextureContent", "Texture"]) else {
					continue;
				};
				let tiling = (child.class.as_str() == "Texture").then(|| Tiling {
					studs_per_tile_u: props::f32(child, "StudsPerTileU", 2.0),
					studs_per_tile_v: props::f32(child, "StudsPerTileV", 2.0),
					offset_studs_u: props::f32(child, "OffsetStudsU", 0.0),
					offset_studs_v: props::f32(child, "OffsetStudsV", 0.0),
				});
				part.decals.push(Decal {
					face: props::enum_value(child, "Face", 5),
					texture: uri,
					color: props::color(child, "Color3", Vec3::ONE),
					transparency: props::f32(child, "Transparency", 0.0),
					z_index: props::i32(child, "ZIndex", 1),
					tiling,
					uv_offset: props::vec2(child, "UVOffset", [0.0, 0.0]),
					uv_scale: props::vec2(child, "UVScale", [1.0, 1.0]),
				});
			}
			"PointLight" | "SpotLight" | "SurfaceLight" => {
				if let Some(light) = read_light(document, child_ref, child, cframe, size) {
					scene.lights.push(light);
				}
			}
			// a Bone is an Attachment: lights on bones shine from them too
			"Attachment" | "Bone" => attachment_lights(document, child_ref, cframe * props::cframe(child, "CFrame").unwrap_or(Affine3A::IDENTITY), &mut scene.lights),
			_ => {}
		}
	}
	// a ForceField part drawn by a FileMesh without a texture is not a ForceField: Studio draws it lit,
	// as SmoothPlastic (forcefield-specialmesh plain, vc: black in the dark, glossy under the sun; a
	// MeshType Sphere without a texture stays a ForceField)
	if part.material == FORCEFIELD
		&& texture.is_none()
		&& matches!(part.geometry, Geometry::Special { mesh_type: mesh_type::FILE_MESH, .. })
	{
		part.material = SMOOTH_PLASTIC;
	}
	part.texture = texture;
	part.vertex_color = vertex_color;
	if let Some(character) = &character {
		dress(&mut part, instance.name.as_str(), character);
	}

	let visible = part.transparency < 1.0 || !part.decals.is_empty();
	if visible || document.keep_invisible {
		scene.parts.push(Arc::new(part));
		return None;
	}
	(part.material == GLASS).then_some(part)
}

/// A document's MaterialService (or, without one, the MaterialVariants anywhere in it).
pub fn materials(document: &Document) -> Materials {
	if let Some(service) = document.material_service.and_then(|service| document.dom.get_by_ref(service)) {
		return read_material_service(document, service);
	}
	let mut materials = Materials::default();
	let mut stack = vec![document.dom.root_ref()];
	while let Some(referent) = stack.pop() {
		let Some(instance) = document.dom.get_by_ref(referent) else {
			continue;
		};
		if instance.class == "MaterialVariant" {
			materials.variants.push(read_material_variant(instance));
		}
		stack.extend(instance.children().iter().rev().copied());
	}
	materials
}

/// A document's Lighting as an environment, if it has one.
pub fn environment(document: &Document) -> Option<Environment> {
	let lighting = document.lighting.and_then(|lighting| document.dom.get_by_ref(lighting))?;
	let mut environment = Environment::from_lighting(&document.dom, lighting);
	// the Terrain's Clouds belong to the place's look too
	environment.clouds = document.clouds().and_then(|clouds| document.dom.get_by_ref(clouds)).and_then(crate::scene::environment::Clouds::read);
	Some(environment)
}

/// `MaterialService`: the generation, its MaterialVariants and the base material overrides.
fn read_material_service(document: &Document, service: &Instance) -> Materials {
	let mut materials = Materials { use_2022: props::bool(service, "Use2022MaterialsXml", true), ..Materials::default() };
	// depth first in file order (the first variant of a name wins)
	let mut stack: Vec<Ref> = service.children().iter().rev().copied().collect();
	while let Some(referent) = stack.pop() {
		let Some(instance) = document.dom.get_by_ref(referent) else {
			continue;
		};
		if instance.class == "MaterialVariant" {
			materials.variants.push(read_material_variant(instance));
		}
		stack.extend(instance.children().iter().rev().copied());
	}
	if let Some(descriptor) = props::database().enums.get("Material") {
		for (name, value) in &descriptor.items {
			if let Some(variant) = props::string(service, &format!("{name}Name"))
				&& !variant.is_empty()
				&& variant != *name
			{
				materials.overrides.push((*value, variant));
			}
		}
	}
	materials
}

fn read_material_variant(instance: &Instance) -> MaterialVariant {
	MaterialVariant {
		name: instance.name.clone(),
		base_material: props::enum_value(instance, "BaseMaterial", 256),
		color_map: props::content(instance, &["ColorMapContent", "ColorMap"]),
		normal_map: props::content(instance, &["NormalMapContent", "NormalMap"]),
		metalness_map: props::content(instance, &["MetalnessMapContent", "MetalnessMap"]),
		roughness_map: props::content(instance, &["RoughnessMapContent", "RoughnessMap"]),
		emissive_mask: props::content(instance, &["EmissiveMaskContent", "EmissiveMask"]),
		texture_pack: props::content(instance, &["TexturePackContent", "TexturePack"]),
		studs_per_tile: props::f32(instance, "StudsPerTile", 10.0),
		pattern: props::enum_value(instance, "MaterialPattern", 0),
		emissive_strength: props::f32(instance, "EmissiveStrength", 1.0),
		emissive_tint: props::color(instance, "EmissiveTint", Vec3::ONE),
	}
}

/// The lights on an attachment (or bone) at world CFrame `world` and on the attachments inside it, each
/// relative to the one it is in (13 of one game's 316 attack PointLights sit two attachments deep).
fn attachment_lights(document: &Document, referent: Ref, world: Affine3A, lights: &mut Vec<Light>) {
	let mut attachments = vec![(referent, world)];
	while let Some((attachment_ref, attachment)) = attachments.pop() {
		let Some(holder) = document.dom.get_by_ref(attachment_ref) else {
			continue;
		};
		for &child_ref in holder.children() {
			let Some(child) = document.dom.get_by_ref(child_ref) else {
				continue;
			};
			if matches!(child.class.as_str(), "Attachment" | "Bone") {
				attachments.push((child_ref, attachment * props::cframe(child, "CFrame").unwrap_or(Affine3A::IDENTITY)));
			} else if matches!(child.class.as_str(), "PointLight" | "SpotLight" | "SurfaceLight")
				&& let Some(light) = read_light(document, child_ref, child, attachment, Vec3::ZERO)
			{
				lights.push(light);
			}
		}
	}
}

fn read_light(document: &Document, referent: Ref, light: &Instance, cframe: Affine3A, part_size: Vec3) -> Option<Light> {
	if !props::bool(light, "Enabled", true) {
		return None;
	}
	let kind = match light.class.as_str() {
		"PointLight" => LightKind::Point,
		"SpotLight" => LightKind::Spot,
		_ => LightKind::Surface,
	};
	Some(Light {
		path: document.path_of(referent),
		kind,
		cframe,
		part_size,
		color: props::color(light, "Color", Vec3::ONE),
		brightness: props::f32(light, "Brightness", 1.0),
		range: props::f32(light, "Range", 8.0),
		angle: props::f32(light, "Angle", 90.0),
		face: props::enum_value(light, "Face", 5),
		shadows: props::bool(light, "Shadows", false),
		placed: false,
	})
}

#[cfg(test)]
mod tests {
	use rbx_dom_weak::InstanceBuilder;
	use rbx_dom_weak::types::{Content, NetAssetRef, SharedString, Variant};

	use crate::scene::Geometry;
	use crate::scene::build::ModelBuilder;

	/// A union with the given geometry properties; returns its extracted geometry, if drawn.
	fn union(properties: Vec<(&str, Variant)>) -> Option<Geometry> {
		let mut model = ModelBuilder::new("Test");
		let root = model.model;
		let mut builder = InstanceBuilder::new("UnionOperation").with_name("Union").with_property("PhysicalConfigData", Variant::SharedString(SharedString::new(b"CSGPHS".to_vec())));
		for (name, value) in properties {
			builder = builder.with_property(name, value);
		}
		model.child(root, builder);
		let scene = super::extract(&model.into_document());
		scene.parts.into_iter().next().map(|part| part.geometry.clone())
	}

	#[test]
	fn body_parts_follow_the_rig_type() {
		use crate::scene::build::PartSpec;
		use rbx_dom_weak::types::Enum;

		let scene = |rig_type: Option<u32>, humanoid: &str| {
			let mut model = ModelBuilder::new("Rig");
			let root = model.model;
			for name in ["Torso", "UpperTorso", "Head", "Handle"] {
				model.part(&PartSpec { name: name.into(), ..Default::default() });
			}
			let mut builder = InstanceBuilder::new(humanoid).with_name("Humanoid");
			if let Some(rig_type) = rig_type {
				builder = builder.with_property("RigType", Enum::from_u32(rig_type));
			}
			model.child(root, builder);
			let scene = super::extract(&model.into_document());
			scene.parts.into_iter().map(|part| (part.path.rsplit('.').next().unwrap_or_default().to_string(), part.body_part, part.geometry.clone())).collect::<Vec<_>>()
		};
		// R6 by default: Torso and Head are limbs, the Torso drawn as Studio's rounded torso mesh
		let r6 = scene(None, "Humanoid");
		assert_eq!(r6.iter().filter(|(_, body, _)| *body == Some(0)).map(|(name, ..)| name.as_str()).collect::<Vec<_>>(), ["Torso", "Head"]);
		assert!(matches!(&r6[0].2, Geometry::MeshPart { mesh, .. } if mesh.ends_with("torso.mesh")));
		let r15 = scene(Some(1), "Humanoid");
		assert_eq!(r15.iter().filter(|(_, body, _)| *body == Some(1)).map(|(name, ..)| name.as_str()).collect::<Vec<_>>(), ["UpperTorso", "Head"]);
		assert!(matches!(r15[0].2, Geometry::Block));
		// an AnimationController named Humanoid is no Humanoid
		assert!(scene(None, "AnimationController").iter().all(|(_, body, _)| body.is_none()));
	}

	#[test]
	fn highlights_cover_their_adornee() {
		use crate::scene::build::PartSpec;
		use rbx_dom_weak::types::Color3;

		let mut model = ModelBuilder::new("Lit");
		let root = model.model;
		let group = model.child(root, InstanceBuilder::new("Model").with_name("Group"));
		let a = model.part(&PartSpec { name: "A".into(), ..Default::default() });
		let b = model.part(&PartSpec { name: "B".into(), ..Default::default() });
		let c = model.part(&PartSpec { name: "C".into(), ..Default::default() });
		model.dom.transfer_within(a, group);
		model.dom.transfer_within(b, group);
		model.child(group, InstanceBuilder::new("Highlight").with_property("OutlineColor", Color3::new(0.0, 0.0, 0.0)));
		let adorning = model.child(root, InstanceBuilder::new("Highlight").with_name("Elsewhere"));
		model.set(adorning, "Adornee", c);
		model.child(root, InstanceBuilder::new("Highlight").with_name("Off").with_property("Enabled", false));
		let scene = super::extract(&model.into_document());
		assert_eq!(scene.highlights.len(), 2);
		let names = |highlight: &crate::scene::Highlight| highlight.parts.iter().map(|&index| scene.parts[index].path.rsplit('.').next().unwrap_or_default().to_string()).collect::<Vec<_>>();
		assert_eq!(names(&scene.highlights[0]), ["A", "B"]);
		assert_eq!(scene.highlights[0].outline_color, glam::Vec3::ZERO);
		assert_eq!(names(&scene.highlights[1]), ["C"]);
		assert_eq!(scene.highlights[1].fill_transparency, 0.5);
	}

	/// Only parts that draw opaque, and Glass at any Transparency, take a Highlight
	/// (vfx-highlight-transparent)
	#[test]
	fn highlights_skip_see_through_parts() {
		use crate::scene::build::PartSpec;

		let mut model = ModelBuilder::new("Lit");
		let root = model.model;
		for (name, transparency, material) in [("Opaque", 0.0, 256), ("Faint", 0.01, 256), ("Half", 0.5, 256), ("Below", -0.5, 256), ("Glass", 0.5, 1568)] {
			model.part(&PartSpec { name: name.into(), transparency, material, ..Default::default() });
		}
		model.child(root, InstanceBuilder::new("Highlight"));
		let lone = model.part(&PartSpec { name: "Lone".into(), transparency: 0.5, ..Default::default() });
		model.child(lone, InstanceBuilder::new("Highlight"));
		let scene = super::extract(&model.into_document());
		assert_eq!(scene.highlights.len(), 1);
		let names: Vec<&str> = scene.highlights[0].parts.iter().map(|&index| scene.parts[index].path.rsplit('.').next().unwrap_or_default()).collect();
		assert_eq!(names, ["Opaque", "Faint", "Below", "Glass"]);
	}

	/// A light two attachments deep in a part sits at part · outer · inner, as Studio places
	/// attachments in attachments
	#[test]
	fn lights_in_attachments_in_attachments() {
		use crate::scene::build::{PartSpec, cframe};

		let mut model = ModelBuilder::new("Lit");
		let part = model.part(&PartSpec { name: "Carrier".into(), cframe: glam::Affine3A::from_translation(glam::Vec3::new(10.0, 0.0, 0.0)), transparency: 1.0, ..Default::default() });
		let outer = model.child(part, InstanceBuilder::new("Attachment").with_name("Outer").with_property("CFrame", cframe(glam::Affine3A::from_translation(glam::Vec3::Y))));
		let inner = model.child(outer, InstanceBuilder::new("Attachment").with_name("Inner").with_property("CFrame", cframe(glam::Affine3A::from_translation(glam::Vec3::Z * 2.0))));
		model.child(inner, InstanceBuilder::new("PointLight").with_name("Light"));
		let scene = super::extract(&model.into_document());
		assert_eq!(scene.lights.len(), 1);
		assert_eq!(glam::Vec3::from(scene.lights[0].cframe.translation), glam::Vec3::new(10.0, 1.0, 2.0));
	}

	/// A light on a Bone (an Attachment) shines from it, and one on a bone inside another from both
	#[test]
	fn lights_on_bones() {
		use crate::scene::build::{PartSpec, cframe};

		let mut model = ModelBuilder::new("Rig");
		let part = model.part(&PartSpec { name: "Body".into(), cframe: glam::Affine3A::from_translation(glam::Vec3::new(0.0, 5.0, 0.0)), ..Default::default() });
		let arm = model.child(part, InstanceBuilder::new("Bone").with_name("Arm").with_property("CFrame", cframe(glam::Affine3A::from_translation(glam::Vec3::X))));
		let hand = model.child(arm, InstanceBuilder::new("Bone").with_name("Hand").with_property("CFrame", cframe(glam::Affine3A::from_translation(glam::Vec3::X * 2.0))));
		model.child(arm, InstanceBuilder::new("PointLight").with_name("Glow"));
		model.child(hand, InstanceBuilder::new("SpotLight").with_name("Torch"));
		let scene = super::extract(&model.into_document());
		let mut placed: Vec<glam::Vec3> = scene.lights.iter().map(|light| glam::Vec3::from(light.cframe.translation)).collect();
		placed.sort_by(|a, b| a.x.total_cmp(&b.x));
		assert_eq!(placed, [glam::Vec3::new(1.0, 5.0, 0.0), glam::Vec3::new(3.0, 5.0, 0.0)]);
	}

	/// An attachment under a Model (not in a part) lights from the model's PrimaryPart, else the centre
	/// of the box its parts fill, whatever its WorldPivotData (vfx-model-frames, light-bones)
	#[test]
	fn lights_on_model_attachments() {
		use crate::scene::build::cframe;
		use rbx_dom_weak::types::{Variant, Vector3};

		let place = |primary: bool| {
			let mut model = ModelBuilder::new("Effect");
			let group = model.child(model.model, InstanceBuilder::new("Model").with_name("Group").with_property("WorldPivotData", Variant::OptionalCFrame(Some(cframe(glam::Affine3A::from_translation(glam::Vec3::new(-8.0, 3.0, 0.0)))))));
			let mut parts = Vec::new();
			for (x, y) in [(6.0f32, 1.0f32), (12.0, 3.0)] {
				parts.push(model.child(group, InstanceBuilder::new("Part").with_property("CFrame", cframe(glam::Affine3A::from_translation(glam::Vec3::new(x, y, 0.0)))).with_property("Size", Vector3::new(1.0, 1.0, 1.0))));
			}
			if primary {
				model.set(group, "PrimaryPart", Variant::Ref(parts[1]));
			}
			let folder = model.child(group, InstanceBuilder::new("Folder"));
			let attachment = model.child(folder, InstanceBuilder::new("Attachment").with_property("CFrame", cframe(glam::Affine3A::from_translation(glam::Vec3::Z * 2.0))));
			model.child(attachment, InstanceBuilder::new("PointLight"));
			let scene = super::extract(&model.into_document());
			assert_eq!(scene.lights.len(), 1);
			glam::Vec3::from(scene.lights[0].cframe.translation)
		};
		assert_eq!(place(false), glam::Vec3::new(9.0, 2.0, 2.0));
		assert_eq!(place(true), glam::Vec3::new(12.0, 3.0, 2.0));
	}

	/// FileMesh, SpecialMesh's base class, draws its mesh as a SpecialMesh of MeshType FileMesh
	#[test]
	fn file_mesh_instances_draw_their_mesh() {
		use crate::scene::build::PartSpec;
		use rbx_dom_weak::types::ContentId;

		let mut model = ModelBuilder::new("Old");
		let part = model.part(&PartSpec { name: "Rock".into(), ..Default::default() });
		model.child(part, InstanceBuilder::new("FileMesh").with_property("MeshId", ContentId::from("rbxassetid://123")).with_property("Scale", rbx_dom_weak::types::Vector3::new(2.0, 2.0, 2.0)));
		let scene = super::extract(&model.into_document());
		match &scene.parts[0].geometry {
			Geometry::Special { mesh_type, mesh, scale, .. } => {
				assert_eq!(*mesh_type, super::mesh_type::FILE_MESH);
				assert_eq!(mesh.as_deref(), Some("rbxassetid://123"));
				assert_eq!(*scale, glam::Vec3::splat(2.0));
			}
			other => panic!("drawn as {other:?}"),
		}
	}

	/// Studio draws at most 255 Highlights at once: the first 255 in the file are kept, with a warning
	/// (vfx-highlight-many)
	#[test]
	fn at_most_255_highlights() {
		use crate::scene::build::PartSpec;

		let mut model = ModelBuilder::new("Many");
		for index in 0..300 {
			let part = model.part(&PartSpec { name: format!("P{index}"), ..Default::default() });
			model.child(part, InstanceBuilder::new("Highlight"));
		}
		let scene = super::extract(&model.into_document());
		assert_eq!(scene.highlights.len(), 255);
		let first: Vec<String> = scene.highlights.iter().map(|highlight| scene.parts[highlight.parts[0]].path.rsplit('.').next().unwrap_or_default().to_string()).collect();
		assert_eq!(first[0], "P0");
		assert_eq!(first[254], "P254");
		assert!(scene.warnings.iter().any(|warning| warning.contains("300 Highlights")), "{:?}", scene.warnings);
	}

	#[test]
	fn union_sources() {
		// only PhysicalConfigData: Studio draws nothing (union-sources "chair-no-content")
		assert!(union(Vec::new()).is_none());
		// a published union's asset, saved as Content
		match union(vec![("Content", Variant::Content(Content::from_uri("https://www.roblox.com//asset/?id=123456789")))]) {
			Some(Geometry::Union { asset: Some(asset), mesh_data: None, .. }) => assert!(asset.ends_with("id=123456789")),
			other => panic!("{other:?}"),
		}
		// an unpublished union's mesh inline in SolidMeshHolder
		match union(vec![("SolidMeshHolder", Variant::NetAssetRef(NetAssetRef::new(b"\x03\x01SolidMesh".to_vec())))]) {
			Some(Geometry::Union { mesh_data: Some(data), .. }) => assert!(data.starts_with(b"\x03\x01")),
			other => panic!("{other:?}"),
		}
	}

	/// A part's composite as a test sees it: its layout, the templates it draws in the shirt's and the
	/// pants' place, and whether it draws the graphic.
	type Worn = Option<(crate::scene::character::Layout, Option<String>, Option<String>, bool)>;

	/// A rig of `rig_type` with the named limbs (blocks for R6, MeshParts for R15) wearing `clothes`
	/// (Shirt rbxassetid://1, Pants 2, ShirtGraphic 3); returns each extracted part's name and what it
	/// wears.
	fn dressed(rig_type: u32, limbs: &[&str], clothes: &[&str]) -> Vec<(String, Worn)> {
		use rbx_dom_weak::types::Enum;

		use crate::scene::build::PartSpec;
		let mut model = ModelBuilder::new("Rig");
		let root = model.model;
		model.child(root, InstanceBuilder::new("Humanoid").with_property("RigType", Enum::from_u32(rig_type)));
		for &class in clothes {
			let (property, uri) = match class {
				"Shirt" => ("ShirtTemplateContent", "rbxassetid://1"),
				"Pants" => ("PantsTemplateContent", "rbxassetid://2"),
				_ => ("TextureContent", "rbxassetid://3"),
			};
			model.child(root, InstanceBuilder::new(class).with_property(property, Variant::Content(Content::from_uri(uri))));
		}
		for name in limbs {
			model.part(&PartSpec { class: if rig_type == 1 { "MeshPart" } else { "Part" }, name: (*name).into(), size: glam::Vec3::new(1.0, 2.0, 1.0), ..Default::default() });
		}
		let scene = super::extract(&model.into_document());
		scene
			.parts
			.into_iter()
			.map(|part| {
				let worn = part.composite.clone().map(|composite| (composite.layout, composite.shirt.map(|layer| layer.image), composite.pants.map(|layer| layer.image), composite.graphic.is_some()));
				(part.path.rsplit('.').next().unwrap_or_default().to_string(), worn)
			})
			.collect()
	}

	#[test]
	fn clothing_by_limb() {
		use crate::scene::character::Layout;
		let (shirt, pants) = (Some("rbxassetid://1".to_string()), Some("rbxassetid://2".to_string()));
		// R6: every limb samples the one atlas holding both templates and the graphic; the head is not
		// composited
		for (name, composite) in dressed(0, &["Torso", "Left Arm", "Right Leg", "Head", "Block"], &["Shirt", "Pants", "ShirtGraphic"]) {
			match name.as_str() {
				"Torso" | "Left Arm" | "Right Leg" => assert_eq!(composite, Some((Layout::R6, shirt.clone(), pants.clone(), true)), "{name}"),
				_ => assert_eq!(composite, None, "{name}"),
			}
		}
		// R15: a shirt dresses the torso and arms, pants the torso and legs, the graphic the torso
		for (name, composite) in dressed(1, &["UpperTorso", "LowerTorso", "LeftLowerArm", "RightFoot", "Head"], &["Shirt", "Pants", "ShirtGraphic"]) {
			let expected = match name.as_str() {
				"UpperTorso" | "LowerTorso" => Some((Layout::R15Torso, shirt.clone(), pants.clone(), true)),
				"LeftLowerArm" => Some((Layout::R15LeftArm, shirt.clone(), None, false)),
				"RightFoot" => Some((Layout::R15RightLeg, None, pants.clone(), false)),
				_ => None,
			};
			assert_eq!(composite, expected, "{name}");
		}
		// R15 Pants without a Shirt dress the torso and, in a shirt's place, the arms; not the legs
		// (characters: r15-pants-grid)
		for (name, composite) in dressed(1, &["UpperTorso", "LeftLowerArm", "RightFoot"], &["Pants"]) {
			let expected = match name.as_str() {
				"UpperTorso" => Some((Layout::R15Torso, None, pants.clone(), false)),
				"LeftLowerArm" => Some((Layout::R15LeftArm, pants.clone(), None, false)),
				_ => None,
			};
			assert_eq!(composite, expected, "{name}");
		}
	}

	#[test]
	fn body_colors_and_character_meshes() {
		use rbx_dom_weak::types::{Color3, Enum};

		use crate::scene::build::PartSpec;
		let mut model = ModelBuilder::new("Rig");
		let root = model.model;
		model.child(root, InstanceBuilder::new("Humanoid"));
		model.child(root, InstanceBuilder::new("BodyColors").with_property("HeadColor3", Color3::new(1.0, 0.0, 0.0)).with_property("TorsoColor3", Color3::new(0.0, 1.0, 0.0)));
		model.child(root, InstanceBuilder::new("CharacterMesh").with_property("BodyPart", Enum::from_u32(1)).with_property("MeshContent", Variant::Content(Content::from_uri("rbxassetid://48112070"))));
		for (name, size) in [("Head", glam::Vec3::new(2.0, 1.0, 1.0)), ("Torso", glam::Vec3::new(2.0, 2.0, 1.0)), ("Left Arm", glam::Vec3::new(1.0, 2.0, 1.0))] {
			model.part(&PartSpec { name: name.into(), size, color: [10, 10, 10], ..Default::default() });
		}
		let scene = super::extract(&model.into_document());
		let part = |name: &str| scene.parts.iter().find(|part| part.path.ends_with(name)).expect(name);
		// BodyColors over the parts' own colours, the head's too (the left arm takes its default)
		assert_eq!(part("Head").color, glam::Vec3::new(1.0, 0.0, 0.0));
		assert_eq!(part("Torso").color, glam::Vec3::new(0.0, 1.0, 0.0));
		assert!((part("Left Arm").color - glam::Vec3::new(253.0, 234.0, 141.0) / 255.0).length() < 1e-3);
		// the torso draws its CharacterMesh, the arm its rounded limb mesh; neither is composited
		assert!(matches!(&part("Torso").geometry, Geometry::MeshPart { mesh, .. } if mesh == "rbxassetid://48112070"));
		assert!(matches!(&part("Left Arm").geometry, Geometry::MeshPart { mesh, .. } if mesh.ends_with("leftarm.mesh")));
		assert!(scene.parts.iter().all(|part| part.composite.is_none()));
	}
}

#[cfg(test)]
mod reuse_tests {
	use glam::{Affine3A, Vec3};
	use rbx_dom_weak::InstanceBuilder;
	use rbx_dom_weak::types::Variant;

	use super::{RootReuse, extract, extract_reusing};
	use crate::scene::build::{ModelBuilder, PartSpec};

	#[test]
	fn untouched_roots_extract_once_and_the_scene_is_the_whole_one() {
		// three roots: a part a playback changes, a still one with an emitter, and a still one naming
		// the first (an ObjectValue), which therefore is walked every time
		let mut model = ModelBuilder::new("Roots");
		let spec = |name: &str, x: f32| PartSpec { name: name.into(), cframe: Affine3A::from_translation(Vec3::new(x, 0.0, 0.0)), ..Default::default() };
		let moving = model.part(&spec("Moving", 0.0));
		let still = model.part(&spec("Still", 6.0));
		let attachment = model.child(still, InstanceBuilder::new("Attachment"));
		model.child(attachment, InstanceBuilder::new("ParticleEmitter").with_property("Rate", 5.0f32));
		let naming = model.part(&spec("Naming", 12.0));
		model.child(naming, InstanceBuilder::new("ObjectValue").with_property("Value", Variant::Ref(moving)));
		let mut document = model.into_document();
		document.roots = vec![moving, still, naming];
		let mut reuse = RootReuse::new(&document, &[moving]);
		// (the moving part is named from outside: never kept)
		assert_eq!(reuse.count(), 1);
		for transparency in [0.0f32, 0.5, 0.25] {
			document.dom.get_by_ref_mut(moving).unwrap().properties.insert(rbx_dom_weak::ustr("Transparency"), Variant::Float32(transparency));
			let reused = extract_reusing(&document, &mut reuse, &crate::scene::playback::Undo::default());
			assert!(reused == extract(&document), "at transparency {transparency}");
			assert_eq!(reused.parts.iter().map(|part| part.path.as_str()).collect::<Vec<_>>(), ["Roots.Moving", "Roots.Still", "Roots.Naming"]);
			assert_eq!(reused.parts[0].transparency, transparency);
		}
		assert!(reuse.pieces[1].is_some() && reuse.pieces[0].is_none() && reuse.pieces[2].is_none());
		// a touched instance above the roots leaves none to reuse
		let parent = document.dom.get_by_ref(moving).unwrap().parent();
		assert_eq!(RootReuse::new(&document, &[parent]).count(), 0);
	}

	/// A kept part a playback changes at some moments is walked again at the moments its state differs
	/// from its last walk's, and its entries are kept at the others: the scene is the whole extraction's
	/// at every moment, in any order of moments.
	#[test]
	fn a_touched_part_is_walked_again_only_when_its_state_changes() {
		use crate::scene::playback::{Action, Playback};
		let mut model = ModelBuilder::new("Roots");
		let spec = |name: &str, x: f32| PartSpec { name: name.into(), cframe: Affine3A::from_translation(Vec3::new(x, 0.0, 0.0)), ..Default::default() };
		let fading = model.part(&spec("Fading", 0.0));
		let still = model.part(&spec("Still", 6.0));
		let mut document = model.into_document();
		document.roots = vec![fading, still];
		let set = |at: f32, value: f32| Action { at, target: "Roots.Fading".into(), set: [("Transparency".to_string(), serde_json::json!(value))].into(), ..Default::default() };
		let playback = Playback { actions: vec![set(1.0, 0.5), set(2.0, 0.0)], ..Default::default() };
		let mut reuse: Option<RootReuse> = None;
		for time in [0.0f32, 0.5, 1.5, 1.25, 3.0, 1.75, 0.25] {
			let resolved = playback.resolve_at(&document, time).unwrap();
			let undo = resolved.apply_state(&mut document).unwrap();
			let reuse = reuse.get_or_insert_with(|| RootReuse::new(&document, &resolved.touched()));
			assert_eq!(reuse.count(), 2, "both parts kept");
			let reused = extract_reusing(&document, reuse, &undo);
			assert!(reused == extract(&document), "at {time} s");
			let expected = if (1.0..2.0).contains(&time) { 0.5 } else { 0.0 };
			assert_eq!(reused.parts[0].transparency, expected, "at {time} s");
			undo.restore(&mut document);
		}
	}
}
