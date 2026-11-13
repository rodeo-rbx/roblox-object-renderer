//! Reading Roblox model and place files (binary and XML) into a DOM.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use rbx_dom_weak::types::{ContentType, Ref, Variant, VariantType};
use rbx_dom_weak::{InstanceBuilder, Ustr, WeakDom};
use rbx_reflection::{ClassDescriptor, DataType, PropertyKind, PropertyMigration, PropertySerialization, ReflectionDatabase};

/// A loaded file: the DOM and the instances to draw.
pub struct Document {
	pub dom: WeakDom,
	/// [`super::playback::Playback::salt`] of the playback this document is drawn for.
	pub salt: String,
	/// True for place files (`.rbxl` / `.rbxlx`): the root's children are services.
	pub is_place: bool,
	/// The instances whose subtrees are drawn: a model's top-level instances, a place's Workspace
	/// children (without Terrain and the Camera), or the `select`ed instance.
	pub roots: Vec<Ref>,
	/// The place's Lighting service, if any.
	pub lighting: Option<Ref>,
	/// The place's MaterialService, if any.
	pub material_service: Option<Ref>,
	/// The name of the Folder holding another file's world drawn around this one's (`--map`,
	/// [`Document::add_setting`]): framing leaves out what is under it.
	pub setting: Option<String>,
	/// The seeds ([`super::Part::seed`]) of the parts a playback has moved or resized, as loaded; they
	/// stay when it puts them back.
	seeds: HashMap<Ref, u64>,
	/// Drawn as a running game draws it, a recording of play (the playback's `play`): see
	/// [`super::Scene::play`].
	pub play: bool,
	/// The folder its content's relative local paths are read from ([`Document::read_relative_to`]);
	/// `None`: the process's working directory.
	pub relative_to: Option<PathBuf>,
	/// Drawn as the collisions-only view ([`super::collision`]): its scene keeps the parts that draw
	/// nothing (Transparency 1 or more, no decals) among its parts, since they often are the collision.
	pub keep_invisible: bool,
}

/// A place's Lighting and what is in it, and its Terrain's Clouds, copied out of the place to give to
/// the models a playback changes them for ([`Document::give_lighting`]).
pub struct LightingCopy {
	dom: WeakDom,
	lighting: Ref,
	clouds: Option<Ref>,
}

impl LightingCopy {
	pub fn of(document: &Document) -> Option<Self> {
		let lighting = document.lighting?;
		let mut dom = WeakDom::new(InstanceBuilder::new("DataModel"));
		let copy = document.dom.clone_into_external(lighting, &mut dom);
		dom.transfer_within(copy, dom.root_ref());
		let clouds = document.clouds().map(|clouds| document.dom.clone_into_external(clouds, &mut dom));
		if let Some(clouds) = clouds {
			dom.transfer_within(clouds, dom.root_ref());
		}
		Some(Self { dom, lighting: copy, clouds })
	}
}

/// Where a playback finds the Terrain's Clouds: their path in a place (a recording's target).
pub const CLOUDS_PATH: [&str; 3] = ["Workspace", "Terrain", "Clouds"];

/// The first child of `parent` of class `class`.
fn child_of_class(dom: &WeakDom, parent: Ref, class: &str) -> Option<Ref> {
	dom.get_by_ref(parent)?.children().iter().copied().find(|&child| dom.get_by_ref(child).is_some_and(|instance| instance.class == class))
}

pub fn read_file(path: &Path) -> Result<Document> {
	let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
	read_bytes(&bytes).with_context(|| format!("parsing {}", path.display()))
}

pub fn read_bytes(bytes: &[u8]) -> Result<Document> {
	// within the machine's free memory (crate::budget); a caller holding a reservation for the whole load adds none
	let _budget = crate::budget::reserve(bytes.len() as u64);
	let reading = reading();
	let mut dom = if bytes.starts_with(b"<roblox!") {
		rbx_binary::Deserializer::new().reflection_database(&reading.database).deserialize(bytes).map_err(|error| anyhow::anyhow!("binary model: {error}"))?
	} else if is_xml(bytes) {
		rbx_xml::from_reader(bytes, rbx_xml::DecodeOptions::new().reflection_database(&reading.database)).map_err(|error| anyhow::anyhow!("XML model: {error}"))?
	} else {
		bail!("not a Roblox model or place (expected <roblox! or <roblox ...> header)");
	};
	fold_content_ids(&mut dom, reading);
	Ok(Document::new(dom))
}

/// How files are read: the reflection database, with the content IDs Roblox replaced by Content
/// properties (ParticleEmitter.Texture by TextureContent, MeshPart.MeshId by MeshContent, ...) read
/// under their own names, and per class those properties with the migration that moves each.
struct Reading {
	database: ReflectionDatabase<'static>,
	content_ids: HashMap<&'static str, Vec<(Ustr, PropertyMigration<'static>)>>,
}

fn reading() -> &'static Reading {
	static READING: OnceLock<Reading> = OnceLock::new();
	READING.get_or_init(|| {
		let shipped = rbx_reflection_database::get().unwrap_or_else(|_| rbx_reflection_database::get_bundled());
		let is_content = |class: &ClassDescriptor<'static>, name: &str| {
			shipped.superclasses_iter(class).find_map(|class| class.properties.get(name)).is_some_and(|property| matches!(property.data_type, DataType::Value(VariantType::Content)))
		};
		let mut own: Vec<(&'static str, &'static str, PropertyMigration<'static>)> = Vec::new();
		for (&class_name, class) in &shipped.classes {
			for (&name, property) in &class.properties {
				if let PropertyKind::Canonical { serialization: PropertySerialization::Migrate(migration) } = &property.kind
					&& migration.new_property_names().iter().all(|new| is_content(class, new))
				{
					own.push((class_name, name, migration.clone()));
				}
			}
		}
		let mut database = shipped.clone();
		for &(class_name, name, _) in &own {
			if let Some(property) = database.classes.get_mut(class_name).and_then(|class| class.properties.get_mut(name)) {
				property.kind = PropertyKind::Canonical { serialization: PropertySerialization::Serializes };
			}
		}
		// each class with its superclasses' (a Texture's Decal.Texture, a Tool's BackpackItem.TextureId)
		let mut content_ids: HashMap<&'static str, Vec<(Ustr, PropertyMigration<'static>)>> = HashMap::new();
		for (&class_name, class) in &shipped.classes {
			let ancestry: Vec<&str> = shipped.superclasses_iter(class).map(|class| class.name).collect();
			let migrations: Vec<_> = own.iter().filter(|(owner, ..)| ancestry.contains(owner)).map(|(_, name, migration)| (rbx_dom_weak::ustr(name), migration.clone())).collect();
			if !migrations.is_empty() {
				content_ids.insert(class_name, migrations);
			}
		}
		Reading { database, content_ids }
	})
}

/// Moves each content ID read under its old name into its Content property, as rbx_binary and
/// rbx_xml migrate one, except that an empty Content does not hide it. A binary file holds a property
/// for every instance of a class or for none, so a writer that knows only the old names (Lune
/// 0.10.2), saving script-made instances beside Studio-saved ones, gives each both: its own and the
/// other generation's empty default. The readers keep the new property whenever it is there, so a
/// script-made emitter's Texture was lost under that empty TextureContent and drew nothing. Where
/// both hold content the new one stays, as there.
fn fold_content_ids(dom: &mut WeakDom, reading: &Reading) {
	let holders: Vec<Ref> = dom.descendants().filter(|instance| reading.content_ids.contains_key(instance.class.as_str())).map(|instance| instance.referent()).collect();
	for referent in holders {
		let Some(instance) = dom.get_by_ref_mut(referent) else {
			continue;
		};
		let Some(migrations) = reading.content_ids.get(instance.class.as_str()) else {
			continue;
		};
		for (old, migration) in migrations {
			let Some(value) = instance.properties.remove(old) else {
				continue;
			};
			let Ok(migrated) = migration.perform(&value) else {
				continue;
			};
			for &new in migration.new_property_names() {
				let new = rbx_dom_weak::ustr(new);
				let replace = match instance.properties.get(&new) {
					None => true,
					Some(current) => !holds_content(current) && holds_content(&migrated),
				};
				if replace {
					instance.properties.insert(new, migrated.clone());
				}
			}
		}
	}
}

/// The two names of one content value: the Content property and the content ID it replaced
/// (ImageLabel: ImageContent and Image; ShirtGraphic: TextureContent and Graphic).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContentNames {
	pub content: &'static str,
	pub legacy: &'static str,
}

/// The names of the content value `name` (either of them) is on `class` or a superclass (a Texture's
/// Decal.Texture), for every content ID the reflection database migrates to a Content property; None
/// for any other property. In Roblox they are one value: setting `Image` sets `ImageContent`, and
/// files read here hold the Content name alone ([`fold_content_ids`]), which ror draws from
/// (`props::content`).
pub fn content_names(class: &str, name: &str) -> Option<ContentNames> {
	reading().content_ids.get(class)?.iter().find_map(|(legacy, migration)| {
		let content = *migration.new_property_names().first()?;
		(legacy.as_str() == name || content == name).then_some(ContentNames { content, legacy: legacy.as_str() })
	})
}

/// A Content or content ID naming a relative local path, as `file://` and that path in `folder`;
/// None for anything else (an asset, an absolute path, another type).
fn local_variant(value: &Variant, folder: &Path) -> Option<Variant> {
	match value {
		Variant::Content(content) => match content.value() {
			ContentType::Uri(uri) => relative_file(uri, folder).map(|uri| rbx_dom_weak::types::Content::from_uri(uri).into()),
			_ => None,
		},
		Variant::ContentId(id) => relative_file(id.as_str(), folder).map(|uri| rbx_dom_weak::types::ContentId::from(uri).into()),
		_ => None,
	}
}

/// `uri` as `file://` and its absolute path when it names a relative local file (ror_assets'
/// [`ror_assets::ContentId::File`], or a bare name `folder` holds), its `#` fragment kept (a mesh's
/// `#lod=N`); `%` is escaped as the file URIs' readers decode it.
fn relative_file(uri: &str, folder: &Path) -> Option<String> {
	let text = uri.trim();
	// most content is an asset: no parse, no look on disk
	let starts = |prefix: &str| text.as_bytes().get(..prefix.len()).is_some_and(|head| head.eq_ignore_ascii_case(prefix.as_bytes()));
	if text.is_empty() || starts("rbxasset") || starts("rbxthumb://") || starts("http") {
		return None;
	}
	let (path, fragment) = match text.split_once('#') {
		Some((path, fragment)) => (path, Some(fragment)),
		None => (text, None),
	};
	let relative = match ror_assets::ContentId::parse(path) {
		Ok(ror_assets::ContentId::File(file)) => file,
		Err(_) if !path.contains("://") && folder.join(path).is_file() => PathBuf::from(path),
		_ => return None,
	};
	if relative.is_absolute() {
		return None;
	}
	let joined = folder.join(relative);
	let joined = std::path::absolute(&joined).unwrap_or(joined);
	let mut resolved = format!("file://{}", joined.display().to_string().replace('%', "%25"));
	if let Some(fragment) = fragment {
		resolved.push('#');
		resolved.push_str(fragment);
	}
	Some(resolved)
}

/// A Content that names something: an asset URI that is not blank, or an object.
fn holds_content(value: &Variant) -> bool {
	match value {
		Variant::Content(content) => match content.value() {
			ContentType::None => false,
			ContentType::Uri(uri) => !uri.trim().is_empty(),
			_ => true,
		},
		_ => true,
	}
}

fn is_xml(bytes: &[u8]) -> bool {
	let text = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
	let start = text.iter().position(|byte| !byte.is_ascii_whitespace()).unwrap_or(text.len());
	text[start..].starts_with(b"<roblox")
}

impl Document {
	pub fn new(dom: WeakDom) -> Self {
		let root = dom.root();
		let mut workspace = None;
		let mut lighting = None;
		let mut material_service = None;
		for &child in root.children() {
			let Some(instance) = dom.get_by_ref(child) else {
				continue;
			};
			match instance.class.as_str() {
				"Workspace" => workspace = Some(child),
				"Lighting" => lighting = Some(child),
				"MaterialService" => material_service = Some(child),
				_ => {}
			}
		}
		let is_place = workspace.is_some();
		let roots = match workspace {
			Some(workspace) => dom
				.get_by_ref(workspace)
				.map(|workspace| {
					workspace
						.children()
						.iter()
						.copied()
						.filter(|&child| dom.get_by_ref(child).is_some_and(|instance| !matches!(instance.class.as_str(), "Terrain" | "Camera")))
						.collect()
				})
				.unwrap_or_default(),
			None => root.children().to_vec(),
		};
		Self { dom, salt: String::new(), is_place, roots, lighting, material_service, setting: None, seeds: HashMap::new(), play: false, relative_to: None, keep_invisible: false }
	}

	/// Reads its content's relative local paths from `folder` (a job's caller's working directory: a
	/// service runs in its state dir): every Content and content ID naming one (`output/x.mesh`,
	/// `file://output/x.mesh`, as a game's asset table may hold a file not uploaded yet) becomes `file://` and the
	/// absolute path, and so does what a playback sets later ([`Document::local_content`]). Returns
	/// how many it changed. A map added after keeps its own.
	pub fn read_relative_to(&mut self, folder: &Path) -> usize {
		let folder = std::path::absolute(folder).unwrap_or_else(|_| folder.to_path_buf());
		let mut changed = 0;
		let referents: Vec<Ref> = self.dom.descendants().map(|instance| instance.referent()).collect();
		for referent in referents {
			let Some(instance) = self.dom.get_by_ref_mut(referent) else {
				continue;
			};
			for value in instance.properties.values_mut() {
				if let Some(resolved) = local_variant(value, &folder) {
					*value = resolved;
					changed += 1;
				}
			}
		}
		self.relative_to = Some(folder);
		changed
	}

	/// `value` with a relative local path read from [`Document::relative_to`] (a playback's set of a
	/// Content or content ID); other values as they are.
	pub(crate) fn local_content(&self, value: Variant) -> Variant {
		match &self.relative_to {
			Some(folder) => local_variant(&value, folder).unwrap_or(value),
			None => value,
		}
	}

	/// Draws another file's world around this one's, as the setting it sits in (`--map`: a game map):
	/// a copy of what `map` draws (a place's Workspace children, a model's top-level instances) in a
	/// Folder at the root named `Map` (`Map (setting)` when the root has a `Map`: a recording's
	/// mounted map), drawn after this document's own. `origin` (x, y, z, yaw in degrees) is the point
	/// of the map that lands on this document's origin, the map turned by the yaw about it. Returns
	/// the Folder's name, which [`Document::setting`] keeps.
	pub fn add_setting(&mut self, map: &Document, origin: Option<[f32; 4]>) -> String {
		let root = self.dom.root_ref();
		let taken = |name: &str| self.dom.root().children().iter().any(|&child| self.dom.get_by_ref(child).is_some_and(|instance| instance.name == name));
		let name = if taken("Map") { "Map (setting)" } else { "Map" }.to_string();
		let holder = self.dom.insert(root, InstanceBuilder::new("Folder").with_name(name.clone()));
		for &child in &map.roots {
			let copy = map.dom.clone_into_external(child, &mut self.dom);
			self.dom.transfer_within(copy, holder);
		}
		if let Some([x, y, z, yaw]) = origin {
			let shift = glam::Affine3A::from_rotation_y(-yaw.to_radians()) * glam::Affine3A::from_translation(glam::Vec3::new(-x, -y, -z));
			let parts: Vec<Ref> = self.dom.descendants_of(holder).filter(|instance| super::props::is_a(instance, "BasePart")).map(|instance| instance.referent()).collect();
			for referent in parts {
				let Some(instance) = self.dom.get_by_ref_mut(referent) else {
					continue;
				};
				if let Some(Variant::CFrame(cframe)) = instance.properties.get(&rbx_dom_weak::ustr("CFrame")) {
					let moved = super::build::cframe(shift * super::props::to_affine(cframe));
					instance.properties.insert(rbx_dom_weak::ustr("CFrame"), Variant::CFrame(moved));
				}
			}
		}
		self.roots.push(holder);
		self.setting = Some(name.clone());
		name
	}

	/// The seed ([`super::Part::seed`]) of a part a playback has moved or resized, from its position and
	/// size as loaded; `None` for one it has not (its seed is that of where it is).
	pub fn loaded_seed(&self, referent: Ref) -> Option<u64> {
		self.seeds.get(&referent).copied()
	}

	/// Keeps a part's seed from its CFrame and Size as they are, before a playback first changes
	/// either: a lerp's rounding, let alone a motion, then leaves its random draws as they were.
	pub(crate) fn keep_seed(&mut self, referent: Ref) {
		if self.seeds.contains_key(&referent) {
			return;
		}
		let seed = self.dom.get_by_ref(referent).filter(|instance| super::props::is_a(instance, "BasePart")).and_then(super::extract::current_seed);
		if let Some(seed) = seed {
			self.seeds.insert(referent, seed);
		}
	}

	/// A copy of the document with the same referents, so a scene extracted from this one names the
	/// copy's instances too (a clip's threads each change a copy of the document its scene came
	/// from; a file read again gets new referents).
	pub fn copy(&self) -> Document {
		let builder = |instance: &rbx_dom_weak::Instance| {
			InstanceBuilder::new(instance.class)
				.with_referent(instance.referent())
				.with_name(instance.name.clone())
				.with_properties(instance.properties.iter().map(|(name, value)| (*name, value.clone())))
		};
		let root = self.dom.root();
		let mut dom = WeakDom::new(builder(root));
		// parents before their children, children in their order
		let mut queue: std::collections::VecDeque<Ref> = root.children().iter().copied().collect();
		while let Some(referent) = queue.pop_front() {
			if let Some(instance) = self.dom.get_by_ref(referent) {
				dom.insert(instance.parent(), builder(instance));
				queue.extend(instance.children().iter().copied());
			}
		}
		Document { dom, salt: self.salt.clone(), is_place: self.is_place, roots: self.roots.clone(), lighting: self.lighting, material_service: self.material_service, setting: self.setting.clone(), seeds: self.seeds.clone(), play: self.play, relative_to: self.relative_to.clone(), keep_invisible: self.keep_invisible }
	}

	/// Gives the document a Lighting for a playback to change, in place of its own: a copy of
	/// `place`'s (a model drawn in that place's light, `--lighting-from`), else, when it has none, one
	/// holding ror's default environment, the light a model is drawn in. It is not drawn. The Terrain's
	/// Clouds go with it, at [`CLOUDS_PATH`]: a copy of `place`'s, else the document's own, and a
	/// disabled Clouds (no clouds, as ror's default environment and a place without any draw) when there
	/// are none.
	pub fn give_lighting(&mut self, place: Option<&LightingCopy>) {
		let lighting = match place {
			Some(place) => {
				if let Some(own) = self.lighting {
					self.dom.transfer_within(own, Ref::none());
					self.roots.retain(|&root| root != own);
				}
				let copy = place.dom.clone_into_external(place.lighting, &mut self.dom);
				self.dom.transfer_within(copy, self.dom.root_ref());
				Some(copy)
			}
			None if self.lighting.is_none() => {
				let properties = super::Environment::default().lighting_properties();
				Some(self.dom.insert(self.dom.root_ref(), InstanceBuilder::new("Lighting").with_properties(properties)))
			}
			None => None,
		};
		if lighting.is_some() {
			self.lighting = lighting;
		}
		let own = self.clouds();
		if place.is_none() && own.is_some() {
			return;
		}
		if let Some(own) = own {
			self.dom.transfer_within(own, Ref::none());
		}
		let terrain = self.terrain();
		let clouds = match place.and_then(|place| place.clouds.map(|clouds| (place, clouds))) {
			Some((place, clouds)) => {
				let copy = place.dom.clone_into_external(clouds, &mut self.dom);
				self.dom.transfer_within(copy, terrain);
				copy
			}
			None => {
				// (the values a new Clouds holds in Roblox, as Clouds::read reads one without them)
				let off = InstanceBuilder::new("Clouds")
					.with_property("Enabled", false)
					.with_property("Cover", 0.5f32)
					.with_property("Density", 0.7f32)
					.with_property("Color", rbx_dom_weak::types::Color3::new(1.0, 1.0, 1.0));
				self.dom.insert(terrain, off)
			}
		};
		if let Some(instance) = self.dom.get_by_ref_mut(clouds) {
			instance.name = CLOUDS_PATH[2].into();
		}
	}

	/// The Terrain's Clouds it draws: a Clouds in its Workspace's Terrain.
	pub fn clouds(&self) -> Option<Ref> {
		let dom = &self.dom;
		child_of_class(dom, dom.root_ref(), "Workspace").and_then(|workspace| child_of_class(dom, workspace, "Terrain")).and_then(|terrain| child_of_class(dom, terrain, "Clouds"))
	}

	/// Its Workspace's Terrain, made (and a Workspace for a model, which is not drawn) when there is none.
	fn terrain(&mut self) -> Ref {
		let root = self.dom.root_ref();
		let workspace = match child_of_class(&self.dom, root, "Workspace") {
			Some(workspace) => workspace,
			None => self.dom.insert(root, InstanceBuilder::new("Workspace").with_name(CLOUDS_PATH[0])),
		};
		match child_of_class(&self.dom, workspace, "Terrain") {
			Some(terrain) => terrain,
			None => self.dom.insert(workspace, InstanceBuilder::new("Terrain").with_name(CLOUDS_PATH[1])),
		}
	}

	/// Restricts drawing to one instance, found by a dotted path of names from the file's root
	/// (`Workspace.Map.House`, or `House.Door` in a model). A segment holding dots can be quoted:
	/// `Folder["Item.Name"]`.
	pub fn select(&mut self, path: &str) -> Result<()> {
		let segments = split_path(path)?;
		let mut current = self.dom.root_ref();
		for segment in &segments {
			let instance = self.dom.get_by_ref(current).context("broken DOM")?;
			let next = instance.children().iter().copied().find(|&child| self.dom.get_by_ref(child).is_some_and(|child| &child.name == segment));
			match next {
				Some(next) => current = next,
				None => {
					let names: Vec<_> = instance.children().iter().filter_map(|&child| self.dom.get_by_ref(child)).map(|child| child.name.as_str()).take(20).collect();
					bail!("no instance {segment:?} under {:?} (children: {})", instance.name, names.join(", "));
				}
			}
		}
		self.roots = vec![current];
		Ok(())
	}

	/// Sets TextureSpeed to 0 on every drawn Beam that scrolls (TextureSpeed defaults to 1) and returns
	/// how many: Studio scrolls a beam's texture from when it enters the world, so its frames never
	/// settle, while a still render draws the texture at rest (notes/render.md "Beams").
	pub fn hold_beams(&mut self) -> usize {
		let mut scrolling = Vec::new();
		let mut stack = self.roots.clone();
		while let Some(referent) = stack.pop() {
			let Some(instance) = self.dom.get_by_ref(referent) else {
				continue;
			};
			stack.extend(instance.children().iter().copied());
			let speed = match instance.properties.get(&rbx_dom_weak::ustr("TextureSpeed")) {
				Some(rbx_dom_weak::types::Variant::Float32(speed)) => *speed,
				_ => 1.0,
			};
			if instance.class == "Beam" && speed != 0.0 {
				scrolling.push(referent);
			}
		}
		for &referent in &scrolling {
			if let Some(instance) = self.dom.get_by_ref_mut(referent) {
				instance.properties.insert(rbx_dom_weak::ustr("TextureSpeed"), rbx_dom_weak::types::Variant::Float32(0.0));
			}
		}
		scrolling.len()
	}

	/// The dotted path of an instance from the file's root (services included for places).
	pub fn path_of(&self, referent: Ref) -> String {
		let mut path = String::new();
		for name in self.names_to(referent) {
			push_segment(&mut path, name);
		}
		path
	}

	/// The paths of an instance and its ancestors, the file's topmost first and its own last, each as
	/// [`Document::path_of`] writes it: what holds a path, known from the instances rather than by
	/// reading the path back.
	pub fn lineage(&self, referent: Ref) -> Vec<String> {
		let mut path = String::new();
		self.names_to(referent).into_iter().map(|name| push_segment(&mut path, name).clone()).collect()
	}

	/// The names from the file's topmost instance down to `referent`.
	fn names_to(&self, referent: Ref) -> Vec<&str> {
		let mut names = Vec::new();
		let mut current = referent;
		while let Some(instance) = self.dom.get_by_ref(current) {
			if current == self.dom.root_ref() {
				break;
			}
			names.push(instance.name.as_str());
			current = instance.parent();
		}
		names.reverse();
		names
	}
}

/// A name as one step of a dotted path, as [`split_path`] reads it back: quoted (`["a.b"]`, or with
/// single quotes when it holds a double one) when a dot or bracket would split it, edge spaces would
/// be trimmed, or it is empty, as ree writes a playback's paths (ree's record::segment). One game's assets
/// hold such names: `Exp4 [5]`.
pub fn path_segment(name: &str) -> String {
	if name.is_empty() || name.contains(['.', '[', ']']) || name.starts_with(char::is_whitespace) || name.ends_with(char::is_whitespace) {
		let quote = if name.contains('"') { '\'' } else { '"' };
		format!("[{quote}{name}{quote}]")
	} else {
		name.to_string()
	}
}

/// `path` with `name` as its next step.
pub fn push_segment<'a>(path: &'a mut String, name: &str) -> &'a String {
	let segment = path_segment(name);
	if !path.is_empty() && !segment.starts_with('[') {
		path.push('.');
	}
	path.push_str(&segment);
	path
}

pub(crate) fn split_path(path: &str) -> Result<Vec<String>> {
	let mut segments = Vec::new();
	let mut rest = path.trim();
	while !rest.is_empty() {
		if let Some(after) = rest.strip_prefix('[') {
			let quote = after.chars().next().filter(|c| *c == '"' || *c == '\'').context("expected a quoted name after [")?;
			let end = after[1..].find(quote).context("unterminated quoted name")? + 1;
			segments.push(after[1..end].to_string());
			rest = after[end + 1..].strip_prefix(']').context("expected ] after a quoted name")?;
		} else {
			let end = rest.find(['.', '[']).unwrap_or(rest.len());
			if end > 0 {
				segments.push(rest[..end].to_string());
			}
			rest = &rest[end..];
		}
		rest = rest.strip_prefix('.').unwrap_or(rest);
	}
	if segments.first().is_some_and(|first| first == "game") {
		segments.remove(0);
	}
	Ok(segments)
}

#[cfg(test)]
mod tests {
	use super::split_path;

	/// A file whose ParticleEmitters mix the two generations of their texture: a writer that knows only
	/// the old Texture (Lune 0.10.2's) saving a script-made emitter beside ones Studio saved with
	/// TextureContent. The binary format holds a property for every instance of a class, so each emitter
	/// gets the other generation's empty default; every emitter keeps its own texture, and one saved
	/// without any stays without.
	#[test]
	fn mixed_content_generations_keep_each_instances_texture() {
		use rbx_dom_weak::types::{Content, ContentId};
		use rbx_dom_weak::{InstanceBuilder, WeakDom};
		use rbx_reflection::{PropertyKind, PropertySerialization};

		let mut writer = rbx_reflection_database::get_bundled().clone();
		let emitter = writer.classes.get_mut("ParticleEmitter").unwrap();
		emitter.properties.get_mut("Texture").unwrap().kind = PropertyKind::Canonical { serialization: PropertySerialization::Serializes };
		emitter.default_properties.remove("Texture");
		emitter.default_properties.remove("TextureContent");
		let mut dom = WeakDom::new(InstanceBuilder::new("DataModel"));
		let root = dom.root_ref();
		let model = dom.insert(root, InstanceBuilder::new("Model").with_name("Effect"));
		let part = dom.insert(model, InstanceBuilder::new("Part").with_name("Rock"));
		dom.insert(part, InstanceBuilder::new("ParticleEmitter").with_name("Scripted").with_property("Texture", ContentId::from("rbxassetid://1")));
		dom.insert(part, InstanceBuilder::new("ParticleEmitter").with_name("Saved").with_property("TextureContent", Content::from_uri("rbxassetid://2")));
		dom.insert(part, InstanceBuilder::new("ParticleEmitter").with_name("Bare").with_property("TextureContent", Content::none()));
		let mut bytes = Vec::new();
		rbx_binary::Serializer::new().reflection_database(&writer).serialize(&mut bytes, &dom, &[model]).unwrap();

		let document = super::read_bytes(&bytes).unwrap();
		let scene = crate::scene::extract::extract(&document);
		let texture = |name: &str| scene.particles.iter().find(|emitter| emitter.path.ends_with(name)).unwrap().texture.clone();
		assert_eq!(texture("Scripted").as_deref(), Some("rbxassetid://1"));
		assert_eq!(texture("Saved").as_deref(), Some("rbxassetid://2"));
		assert_eq!(texture("Bare"), None);
		// the old property folds into the new one, as the readers' own migration leaves a file
		assert!(document.dom.descendants().all(|instance| !instance.properties.contains_key(&rbx_dom_weak::ustr("Texture"))));
	}

	/// A content ID's old name and its Content property name one value, either way round, on the class
	/// that has it and the classes inheriting it; other properties are none.
	#[test]
	fn content_ids_and_their_content_properties() {
		use super::{ContentNames, content_names};
		let names = |content, legacy| Some(ContentNames { content, legacy });
		for class in ["ImageLabel", "ImageButton", "ImageHandleAdornment"] {
			assert_eq!(content_names(class, "Image"), names("ImageContent", "Image"), "{class}");
			assert_eq!(content_names(class, "ImageContent"), names("ImageContent", "Image"), "{class}");
		}
		assert_eq!(content_names("ImageButton", "HoverImage"), names("HoverImageContent", "HoverImage"));
		assert_eq!(content_names("ImageButton", "PressedImage"), names("PressedImageContent", "PressedImage"));
		for class in ["Decal", "Texture", "ParticleEmitter", "Beam", "Trail"] {
			assert_eq!(content_names(class, "Texture"), names("TextureContent", "Texture"), "{class}");
			assert_eq!(content_names(class, "TextureContent"), names("TextureContent", "Texture"), "{class}");
		}
		assert_eq!(content_names("MeshPart", "TextureID"), names("TextureContent", "TextureID"));
		assert_eq!(content_names("MeshPart", "MeshId"), names("MeshContent", "MeshId"));
		assert_eq!(content_names("MeshPart", "MeshContent"), names("MeshContent", "MeshId"));
		assert_eq!(content_names("SpecialMesh", "TextureId"), names("TextureContent", "TextureId"));
		assert_eq!(content_names("SpecialMesh", "MeshId"), names("MeshContent", "MeshId"));
		assert_eq!(content_names("ShirtGraphic", "Graphic"), names("TextureContent", "Graphic"));
		assert_eq!(content_names("Sky", "SkyboxBk"), names("SkyboxBackContent", "SkyboxBk"));
		assert_eq!(content_names("SurfaceAppearance", "ColorMap"), names("ColorMapContent", "ColorMap"));
		assert_eq!(content_names("UnionOperation", "AssetId"), names("Content", "AssetId"));
		// a class's own content IDs only, and only content IDs
		assert_eq!(content_names("Part", "Texture"), None);
		assert_eq!(content_names("MeshPart", "Texture"), None);
		assert_eq!(content_names("ParticleEmitter", "Rate"), None);
		assert_eq!(content_names("NotAClass", "Image"), None);
	}

	/// `--map`: another file's world drawn around the model, moved so its origin point lands on the
	/// model's origin, in a Folder named apart from a root `Map` the model has (a recording's mounted
	/// map); what an orbit frames, and the lint's emitters, are the model's alone.
	#[test]
	fn a_map_is_drawn_around_the_model_and_never_framed() {
		use crate::scene::build::{ModelBuilder, PartSpec};
		use glam::{Affine3A, Vec3};

		let mut map = ModelBuilder::new("Bridge");
		map.part(&PartSpec { name: "Deck".into(), cframe: Affine3A::from_translation(Vec3::new(100.0, 20.0, 50.0)), size: Vec3::new(60.0, 2.0, 20.0), ..Default::default() });
		let tower = map.part(&PartSpec { name: "Tower".into(), cframe: Affine3A::from_translation(Vec3::new(130.0, 40.0, 50.0)), size: Vec3::new(4.0, 40.0, 4.0), ..Default::default() });
		map.child(tower, rbx_dom_weak::InstanceBuilder::new("ParticleEmitter").with_name("Smoke"));
		let map = map.into_document();

		let mut model = ModelBuilder::new("Map");
		model.part(&PartSpec { name: "Unit".into(), size: Vec3::new(2.0, 5.0, 1.0), ..Default::default() });
		let mut document = model.into_document();
		let name = document.add_setting(&map, Some([100.0, 21.0, 50.0, 90.0]));
		assert_eq!(name, "Map (setting)", "the model's own root is named Map");

		let scene = crate::scene::extract::extract(&document);
		assert_eq!(scene.parts.len(), 3);
		let deck = scene.parts.iter().find(|part| part.path.ends_with("Deck")).unwrap();
		assert!(Vec3::from(deck.cframe.translation).abs_diff_eq(Vec3::new(0.0, -1.0, 0.0), 1e-3), "the deck's middle lands under the origin: {:?}", deck.cframe.translation);
		// turned 90 degrees about the origin point: the tower, 30 studs along +X, is 30 studs along +Z
		let tower = scene.parts.iter().find(|part| part.path.ends_with("Tower")).unwrap();
		assert!(Vec3::from(tower.cframe.translation).abs_diff_eq(Vec3::new(0.0, 19.0, 30.0), 1e-3), "{:?}", tower.cframe.translation);

		let subject = scene.framing_subject().expect("a scene with a map has a subject");
		assert_eq!(subject.parts.iter().map(|part| part.path.as_str()).collect::<Vec<_>>(), ["Map.Unit"]);
		assert!(subject.particles.is_empty(), "the map's emitters are not the model's");
		let (min, max) = subject.bounds().unwrap();
		assert!(min.abs_diff_eq(Vec3::new(-1.0, -2.5, -0.5), 1e-3) && max.abs_diff_eq(Vec3::new(1.0, 2.5, 0.5), 1e-3));
		assert!(crate::scene::extract::extract(&map).framing_subject().is_none(), "no map, no subject");
	}

	#[test]
	fn paths() {
		assert_eq!(split_path("Workspace.Map.House").unwrap(), ["Workspace", "Map", "House"]);
		assert_eq!(split_path("game.Workspace[\"A.B\"].C").unwrap(), ["Workspace", "A.B", "C"]);
		assert_eq!(split_path("['x.y']").unwrap(), ["x.y"]);
	}

	#[test]
	fn a_model_is_given_the_light_it_is_drawn_in() {
		use crate::scene::Environment;
		use crate::scene::build::{ModelBuilder, PartSpec};
		let mut model = ModelBuilder::new("M");
		model.part(&PartSpec::default());
		let mut document = model.into_document();
		document.give_lighting(None);
		let lit = crate::scene::extract::environment(&document).unwrap();
		let default = Environment::default();
		assert_eq!((lit.ambient, lit.outdoor_ambient, lit.brightness, lit.clock_time, lit.exposure_compensation), (default.ambient, default.outdoor_ambient, default.brightness, default.clock_time, default.exposure_compensation));
		assert_eq!((lit.technology, lit.global_shadows, lit.fog_end, lit.environment_specular_scale), (default.technology, default.global_shadows, default.fog_end, default.environment_specular_scale));
		assert!(lit.sky.is_none() && lit.atmosphere.is_none() && lit.effects.is_empty());
		assert!(!document.roots.contains(&document.lighting.unwrap()), "not drawn");
		// a copy keeps every referent, name, parent and property
		let copy = document.copy();
		assert_eq!((copy.lighting, &copy.roots), (document.lighting, &document.roots));
		for instance in document.dom.descendants() {
			let other = copy.dom.get_by_ref(instance.referent()).unwrap();
			assert_eq!((&other.name, other.parent(), other.children(), other.properties.len()), (&instance.name, instance.parent(), instance.children(), instance.properties.len()));
		}
	}
}
