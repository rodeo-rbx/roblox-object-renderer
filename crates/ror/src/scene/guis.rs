//! SurfaceGuis and BillboardGuis (notes/render.md "GUIs"): where each one draws and how it is lit;
//! and the ScreenGuis of a PlayerGui, drawn over the finished image ("ScreenGuis"). Their 2D content
//! is the GUI's own subtree, which rur lays out and rasterises (render::guis).

use std::sync::Arc;

use glam::{Affine3A, Vec3};
use rbx_dom_weak::types::{Ref, Variant};
use rbx_dom_weak::{Instance, InstanceBuilder, WeakDom};

use super::environment::ViewportLighting;
use super::load::Document;
use super::{Scene, beams, props};

/// Per `Enum.NormalId`: the face's normal and the canvas's right and down directions in the part's
/// space. The side faces run like decals; Top and Bottom are turned a quarter, their canvas's right
/// along -Z (gui-surface-basics: block-face0..5, turned-face1).
pub const FACE_NORMAL: [Vec3; 6] = [Vec3::X, Vec3::Y, Vec3::Z, Vec3::NEG_X, Vec3::NEG_Y, Vec3::NEG_Z];
pub const FACE_RIGHT: [Vec3; 6] = [Vec3::NEG_Z, Vec3::NEG_Z, Vec3::X, Vec3::Z, Vec3::NEG_Z, Vec3::NEG_X];
pub const FACE_DOWN: [Vec3; 6] = [Vec3::NEG_Y, Vec3::X, Vec3::NEG_Y, Vec3::NEG_Y, Vec3::NEG_X, Vec3::NEG_Y];

#[derive(Clone, Debug, PartialEq)]
pub struct Gui {
	pub path: String,
	pub kind: GuiKind,
	/// The GUI instance and its descendants as a binary model (references outside it cleared): what
	/// rur draws.
	pub content: Arc<Vec<u8>>,
	/// FNV-1a of `content`.
	pub content_hash: u64,
	pub light_influence: f32,
	pub brightness: f32,
	pub always_on_top: bool,
	/// Not drawn farther than this from the camera (studs); 0 or less: no limit.
	pub max_distance: f32,
	/// `ClipsDescendants`.
	pub clips: bool,
	/// `Enum.ZIndexBehavior`: 0 Global, 1 Sibling.
	pub z_index_behavior: u32,
	/// Its ViewportFrames that draw a world (render::guis draws them with rur's image of the rest):
	/// windows first, then the framed ones in the order the GUI draws them.
	pub viewports: Vec<Viewport>,
}

/// A SurfaceGui's or BillboardGui's ViewportFrame that draws a world.
#[derive(Clone, Debug, PartialEq)]
pub struct Viewport {
	/// The frame's place in its GUI: each GuiObject's Size, Position and AnchorPoint from the GUI's child
	/// down to the frame ([`Viewport::rect`] lays them out in a canvas).
	pub placement: Vec<Placement>,
	/// `ImageColor3` (sRGB 0..1) and the image's opacity, 1 - `ImageTransparency`.
	pub tint: [f32; 4],
	/// Its world as the frame draws it ([`Scene::in_viewport_frame`]: the frame's own light).
	pub scene: Arc<Scene>,
	pub kind: ViewportKind,
	/// Its UICorner's CornerRadius (scale, offset), which rounds its world's corners as it rounds its
	/// background; `None` without one.
	pub corner_radius: Option<[f32; 2]>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ViewportKind {
	/// A SurfaceGui's window into another space: a world posed round the part the GUI lies on, whose
	/// camera follows the player's every frame so that the face shows that world as if seen through it
	/// (a rift or a portal; EgoMoose's ViewportWindow is a public example). Drawn so: its world as the view's own camera sees it,
	/// inside the frame's rect, over the GUI's image.
	Window,
	/// What Roblox's ViewportFrame draws: its world through its own camera, fitted to the frame and
	/// clipped to it, at the frame's place among what the GUI draws (an emote's portrait of the player in a
	/// BillboardGui behind them, eyes and smoke over it).
	Framed(Framed),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Framed {
	/// Its camera: the CurrentCamera's CFrame (files keep it as CameraCFrame) as components, x y z
	/// R00 R01 R02 ... R22, and its vertical field of view (CameraFieldOfView), degrees.
	pub camera: [f32; 12],
	pub fov: f32,
	/// What the GUI draws over the frame: its content with everything drawn up to and including the
	/// frame drawing none of its own (where each is and what it holds kept), and that content's FNV-1a;
	/// `None` when nothing draws after the frame.
	pub over: Option<(Arc<Vec<u8>>, u64)>,
	/// Its UIGradient, which shades its picture as it shades an image.
	pub gradient: Option<Gradient>,
}

/// A UIGradient: its colour and transparency across the object (rur's gradient_plane: `t` 0 to 1 along
/// Rotation's direction over the object's span that way, its middle moved by Offset).
#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
	pub rotation: f32,
	pub offset: [f32; 2],
	/// Color's keypoints (time, sRGB) and Transparency's (time, value), at most [`GRADIENT_KEYS`] each.
	pub colors: Vec<(f32, [f32; 3])>,
	pub transparency: Vec<(f32, f32)>,
}

/// The keypoints a sequence holds at most (Roblox's limit).
pub const GRADIENT_KEYS: usize = 20;

impl Gradient {
	/// `t` across an object of `size` (any unit) as `a · x + b · y + c`, x and y its fractions right and
	/// down (rur's gradient_plane over its rect).
	pub fn plane(&self, size: [f32; 2]) -> [f32; 3] {
		let angle = self.rotation.to_radians();
		let (dx, dy) = (angle.cos(), angle.sin());
		let [width, height] = size.map(|side| side.max(1e-6));
		let span = (if dx.abs() > 1e-9 { width / dx.abs() } else { f32::INFINITY }).min(if dy.abs() > 1e-9 { height / dy.abs() } else { f32::INFINITY });
		let (cx, cy) = ((0.5 + self.offset[0]) * width, (0.5 + self.offset[1]) * height);
		[dx * width / span, dy * height / span, 0.5 - (cx * dx + cy * dy) / span]
	}
}

/// A GuiObject's Size, Position and AnchorPoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
	pub size: [f32; 4],
	pub position: [f32; 4],
	pub anchor: [f32; 2],
}

impl Placement {
	fn of(instance: &Instance) -> Self {
		// (a GuiObject's default Size is 100 x 100 pixels)
		Self { size: props::udim2(instance, "Size").unwrap_or([0.0, 100.0, 0.0, 100.0]), position: props::udim2(instance, "Position").unwrap_or([0.0; 4]), anchor: props::vec2(instance, "AnchorPoint", [0.0, 0.0]) }
	}

	/// Its rect (x, y, width, height) inside its parent's.
	fn layout(&self, parent: [f32; 4]) -> [f32; 4] {
		let [size_x, offset_x, size_y, offset_y] = self.size;
		let [position_x, position_offset_x, position_y, position_offset_y] = self.position;
		let (width, height) = (size_x * parent[2] + offset_x, size_y * parent[3] + offset_y);
		let x = parent[0] + position_x * parent[2] + position_offset_x - self.anchor[0] * width;
		let y = parent[1] + position_y * parent[3] + position_offset_y - self.anchor[1] * height;
		[x, y, width, height]
	}
}

impl Viewport {
	/// The frame's rect (x, y, width, height) in a canvas of `canvas` GUI pixels.
	pub fn pixels(&self, canvas: [f32; 2]) -> [f32; 4] {
		self.placement.iter().fold([0.0, 0.0, canvas[0], canvas[1]], |rect, placement| placement.layout(rect))
	}

	/// The frame's rect in canvas fractions: left, top, right, bottom.
	pub fn rect(&self, canvas: [f32; 2]) -> [f32; 4] {
		let [x, y, width, height] = self.pixels(canvas);
		[x / canvas[0], y / canvas[1], (x + width) / canvas[0], (y + height) / canvas[1]]
	}

	/// Its corners' radius in canvas fractions (across, down), as rur rounds the background: the scale of
	/// the frame's shorter side plus the offset, at most half that side; zero without a UICorner.
	pub fn corner(&self, canvas: [f32; 2]) -> [f32; 2] {
		let Some([scale, offset]) = self.corner_radius else {
			return [0.0, 0.0];
		};
		let [_, _, width, height] = self.pixels(canvas);
		let shortest = width.min(height).max(0.0);
		let radius = (scale * shortest + offset).clamp(0.0, shortest / 2.0);
		[radius / canvas[0].max(1e-6), radius / canvas[1].max(1e-6)]
	}
}

#[derive(Clone, Debug, PartialEq)]
pub enum GuiKind {
	Surface(SurfaceGui),
	Billboard(BillboardGui),
	Screen(ScreenGui),
}

/// A ScreenGui of a PlayerGui: drawn over the finished image, post effects included, at the view's
/// size (notes/render.md "ScreenGuis").
#[derive(Clone, Debug, PartialEq)]
pub struct ScreenGui {
	pub display_order: i32,
	/// Below the top bar (ScreenInsets CoreUISafeInsets, IgnoreGuiInset off), else the whole view.
	pub inset: bool,
}

/// The top bar's height in GUI pixels a ScreenGui below it leaves free: GuiService:GetGuiInset() in a
/// running game (Roblox's GetInsetArea samples, -58 at the top; not measured here), 0 in Studio's
/// edit viewport (screen-gui-inset: GetGuiInset 0, 0 and a frame at (0, 0) drawn at the top edge).
pub fn top_bar_inset(play: bool) -> f32 {
	if play { 58.0 } else { 0.0 }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceGui {
	/// The part the canvas lies on: its CFrame and size (the canvas covers a face of its box, whatever
	/// the part's shape; gui-surface-basics: shapes).
	pub cframe: Affine3A,
	pub size: Vec3,
	/// `Enum.NormalId`.
	pub face: u32,
	/// Canvas size in GUI pixels: CanvasSize, or the face's size times PixelsPerStud.
	pub canvas: [f32; 2],
	pub z_offset: f32,
}

impl SurfaceGui {
	/// The face's width and height in studs (along the canvas's right and down).
	pub fn face_size(&self) -> [f32; 2] {
		face_size(self.size, self.face)
	}

	/// World position of a canvas point (fractions of the canvas, 0..1 right and down).
	pub fn point(&self, u: f32, v: f32) -> Vec3 {
		let face = self.face.min(5) as usize;
		let [width, height] = self.face_size();
		let center = FACE_NORMAL[face] * (self.size.abs().dot(FACE_NORMAL[face].abs()) * 0.5);
		let local = center + FACE_RIGHT[face] * (u - 0.5) * width + FACE_DOWN[face] * (v - 0.5) * height;
		self.cframe.transform_point3(local)
	}

	pub fn normal(&self) -> Vec3 {
		self.cframe.transform_vector3(FACE_NORMAL[self.face.min(5) as usize]).normalize_or_zero()
	}
}

fn face_size(size: Vec3, face: u32) -> [f32; 2] {
	let face = face.min(5) as usize;
	[size.abs().dot(FACE_RIGHT[face].abs()), size.abs().dot(FACE_DOWN[face].abs())]
}

#[derive(Clone, Debug, PartialEq)]
pub struct BillboardGui {
	/// World position of the adornee: a part's centre, an attachment.
	pub anchor: Vec3,
	/// Half the adornee's world-space bounding box (the ExtentsOffset unit); zero for attachments.
	pub extents: Vec3,
	/// `Size`: x studs, x pixels, y studs, y pixels.
	pub size: [f32; 4],
	/// `SizeOffset`: the billboard moves by these fractions of its size, right and up.
	pub size_offset: [f32; 2],
	/// In the camera's axes (x right, y up, z towards the camera), studs: StudsOffset plus
	/// ExtentsOffset times the extents.
	pub camera_offset: Vec3,
	/// In world axes, studs: StudsOffsetWorldSpace plus ExtentsOffsetWorldSpace times the extents.
	pub world_offset: Vec3,
}

fn fnv1a(bytes: &[u8]) -> u64 {
	let mut hash = 0xcbf29ce484222325u64;
	for &byte in bytes {
		hash ^= byte as u64;
		hash = hash.wrapping_mul(0x100000001b3);
	}
	hash
}

/// Whether an instance is part of a GUI's 2D content (a GUI object, a UI component, a LayerCollector),
/// not of a ViewportFrame's world.
fn is_gui(instance: &Instance) -> bool {
	props::class_is_a(instance.class.as_str(), "GuiBase") || props::class_is_a(instance.class.as_str(), "UIComponent")
}

/// The GUI's subtree as a binary model of its own (references to instances outside it cleared). A
/// ViewportFrame's world is left out: rur draws a ViewportFrame's background alone (render::guis draws
/// the world), and a world that moves would otherwise make a new image (a new rur call) every frame.
/// Fails as rbx_binary does on a property holding a value of another type than its class gives it.
fn content(document: &Document, referent: Ref) -> Result<Vec<u8>, rbx_binary::EncodeError> {
	let (dom, clone) = subtree(document, referent);
	encode(dom, clone)
}

/// A copy of the GUI's subtree in a model of its own, and the copy's referent.
fn subtree(document: &Document, referent: Ref) -> (WeakDom, Ref) {
	let mut dom = WeakDom::new(InstanceBuilder::new("DataModel"));
	let clone = document.dom.clone_into_external(referent, &mut dom);
	let root = dom.root_ref();
	dom.transfer_within(clone, root);
	(dom, clone)
}

/// What only a ViewportFrame has: its world's light and camera, and its image's tint.
const VIEWPORT_ONLY: [&str; 8] = ["Ambient", "LightColor", "LightDirection", "ImageColor3", "ImageTransparency", "CurrentCamera", "CameraCFrame", "CameraFieldOfView"];

/// [`subtree`]'s model as rbx_binary writes it, its ViewportFrames as plain Frames without their
/// worlds: what rur draws of them is their background (and what they hold of GUI), the same pixels as
/// a ViewportFrame's, and render::guis draws their worlds. Given a ViewportFrame, rur can ask a ROR
/// worker for a picture of each new state of it (its `rur-viewport://` images), which nothing here uses.
fn encode(mut dom: WeakDom, clone: Ref) -> Result<Vec<u8>, rbx_binary::EncodeError> {
	let mut worlds = Vec::new();
	let mut frames = Vec::new();
	let mut stack = vec![clone];
	while let Some(current) = stack.pop() {
		let Some(instance) = dom.get_by_ref(current) else {
			continue;
		};
		let viewport = instance.class == "ViewportFrame";
		if viewport {
			frames.push(current);
		}
		for &child in instance.children() {
			if viewport && dom.get_by_ref(child).is_some_and(|child| !is_gui(child)) {
				worlds.push(child);
			} else {
				stack.push(child);
			}
		}
	}
	for world in worlds {
		dom.destroy(world);
	}
	for frame in frames {
		if let Some(instance) = dom.get_by_ref_mut(frame) {
			instance.class = rbx_dom_weak::ustr("Frame");
			for name in VIEWPORT_ONLY {
				instance.properties.remove(&rbx_dom_weak::ustr(name));
			}
		}
	}
	let mut bytes = Vec::new();
	rbx_binary::to_writer(&mut bytes, &dom, &[clone])?;
	Ok(bytes)
}

/// The GuiObjects under a GUI in the order it draws them, as rur draws them (its export's walk): with
/// Sibling ZIndexBehavior each object and then what it holds, siblings by ZIndex then child order;
/// with Global all of them by ZIndex, then that order. A Folder holds its objects where it stands; a
/// hidden object (Visible off) draws nothing of itself or what it holds and is left out.
fn draw_order(dom: &WeakDom, gui: Ref, global: bool) -> Vec<Ref> {
	fn children(dom: &WeakDom, referent: Ref, out: &mut Vec<Ref>) {
		for &child in dom.get_by_ref(referent).map(Instance::children).unwrap_or_default() {
			match dom.get_by_ref(child) {
				Some(instance) if props::class_is_a(instance.class.as_str(), "GuiObject") => out.push(child),
				Some(instance) if instance.class == "Folder" => children(dom, child, out),
				_ => {}
			}
		}
	}
	let z_index = |referent: &Ref| dom.get_by_ref(*referent).map_or(1, |instance| props::i32(instance, "ZIndex", 1));
	fn walk(dom: &WeakDom, referent: Ref, global: bool, z_index: &dyn Fn(&Ref) -> i32, out: &mut Vec<Ref>) {
		let mut list = Vec::new();
		children(dom, referent, &mut list);
		list.retain(|child| dom.get_by_ref(*child).is_some_and(|instance| props::bool(instance, "Visible", true)));
		if !global {
			list.sort_by_key(|child| z_index(child));
		}
		for child in list {
			out.push(child);
			walk(dom, child, global, z_index, out);
		}
	}
	let mut order = Vec::new();
	walk(dom, gui, global, &z_index, &mut order);
	if global {
		order.sort_by_key(|referent| z_index(referent));
	}
	order
}

/// Has a GuiObject draw none of its own, its place and what it holds kept (no layout moves): its
/// background and border, image, text, scroll bars and strokes. A ViewportFrame draws only its
/// background in rur, so what draws over two frames with nothing shown between them (an emote
/// portrait's rim and bust) is one content, one image.
fn clear_own(dom: &mut WeakDom, referent: Ref) {
	let strokes: Vec<Ref> = dom.get_by_ref(referent).map(Instance::children).unwrap_or_default().iter().copied().filter(|&child| dom.get_by_ref(child).is_some_and(|child| child.class == "UIStroke")).collect();
	for stroke in strokes {
		if let Some(stroke) = dom.get_by_ref_mut(stroke) {
			stroke.properties.insert(rbx_dom_weak::ustr("Enabled"), Variant::Bool(false));
		}
	}
	let Some(instance) = dom.get_by_ref_mut(referent) else {
		return;
	};
	let clear: &[&str] = match instance.class.as_str() {
		"ImageLabel" | "ImageButton" => &["BackgroundTransparency", "ImageTransparency"],
		"TextLabel" | "TextButton" | "TextBox" => &["BackgroundTransparency", "TextTransparency", "TextStrokeTransparency"],
		"ScrollingFrame" => &["BackgroundTransparency", "ScrollBarImageTransparency"],
		_ => &["BackgroundTransparency"],
	};
	for name in clear {
		instance.properties.insert(rbx_dom_weak::ustr(name), Variant::Float32(1.0));
	}
}

/// What a GUI draws over one of its ViewportFrames (`frame`), as content of its own ([`Framed::over`]):
/// what the GUI draws up to and including the frame draws none of its own. `None` when nothing draws
/// after the frame.
fn over_content(document: &Document, gui: Ref, frame: Ref, global: bool) -> Option<Result<Vec<u8>, rbx_binary::EncodeError>> {
	// the frame's copy, found by its child indices from the GUI
	let mut path = Vec::new();
	let mut current = frame;
	while current != gui {
		let parent = document.dom.get_by_ref(current)?.parent();
		path.push(document.dom.get_by_ref(parent)?.children().iter().position(|&child| child == current)?);
		current = parent;
	}
	let (mut dom, clone) = subtree(document, gui);
	let copy = path.iter().rev().try_fold(clone, |at, &index| dom.get_by_ref(at)?.children().get(index).copied())?;
	let order = draw_order(&dom, clone, global);
	let at = order.iter().position(|&referent| referent == copy)?;
	if at + 1 == order.len() {
		return None;
	}
	for &referent in &order[..=at] {
		clear_own(&mut dom, referent);
	}
	Some(encode(dom, clone))
}

/// A ViewportFrame's UIGradient (an enabled one), as it shades the frame's picture.
fn gradient(document: &Document, frame: &Instance) -> Option<Gradient> {
	let gradient = frame.children().iter().filter_map(|&child| document.dom.get_by_ref(child)).find(|child| child.class == "UIGradient" && props::bool(child, "Enabled", true))?;
	let colors = super::particles::color_keys(gradient, "Color").into_iter().take(GRADIENT_KEYS).map(|(time, color)| (time, color.to_array())).collect();
	let transparency = match props::get(gradient, "Transparency") {
		Some(Variant::NumberSequence(sequence)) => sequence.keypoints.iter().take(GRADIENT_KEYS).map(|key| (key.time, key.value)).collect(),
		_ => vec![(0.0, 0.0)],
	};
	Some(Gradient { rotation: props::f32(gradient, "Rotation", 0.0), offset: props::vec2(gradient, "Offset", [0.0, 0.0]), colors, transparency })
}

/// A ViewportFrame's own camera: its CurrentCamera's CFrame and FieldOfView when the file keeps the
/// reference, else CameraCFrame and CameraFieldOfView, as Roblox files keep the CurrentCamera's (a
/// recording's follow it as the camera moves). `None` without either: Roblox draws a ViewportFrame's
/// world only through a camera.
fn frame_camera(document: &Document, frame: &Instance) -> Option<([f32; 12], f32)> {
	let camera = props::reference(frame, "CurrentCamera").and_then(|camera| document.dom.get_by_ref(camera)).filter(|camera| camera.class == "Camera");
	let (cframe, fov) = match camera {
		Some(camera) => (props::cframe(camera, "CFrame")?, props::f32(camera, "FieldOfView", 70.0)),
		None => (props::cframe(frame, "CameraCFrame")?, props::f32(frame, "CameraFieldOfView", 70.0)),
	};
	let m = cframe.matrix3;
	let p = cframe.translation;
	Some(([p.x, p.y, p.z, m.x_axis.x, m.y_axis.x, m.z_axis.x, m.x_axis.y, m.y_axis.y, m.z_axis.y, m.x_axis.z, m.y_axis.z, m.z_axis.z], fov))
}


/// Whether a ViewportFrame's world is a window's: a portal window built the common way (EgoMoose's
/// ViewportWindow) gives its camera a skewed CFrame each frame (its CameraCFrame), whose axes are no
/// longer unit and square, so that the frame shows its world as the player's camera would see it
/// through the face.
fn is_window(document: &Document, frame: Ref) -> bool {
	if skewed_camera(document, frame) {
		return true;
	}
	false
}

/// Whether a ViewportFrame's camera has a skewed CFrame (axes off unit length or square): its
/// CameraCFrame, as files keep the CurrentCamera's (the reference itself is not saved).
fn skewed_camera(document: &Document, frame: Ref) -> bool {
	document.dom.get_by_ref(frame).and_then(|frame| props::cframe(frame, "CameraCFrame")).is_some_and(|cframe| super::playback::skewed(&cframe))
}

/// A GUI's visible ViewportFrames that draw a world ([`Viewport`]), each with its world extracted and
/// lit as the frame lights it (Roblox's defaults: Ambient 200, LightColor 140, LightDirection (-1, -1,
/// -1)): windows (a SurfaceGui's alone, `windows`), then the framed ones in the order the GUI draws
/// them, each with what the GUI draws over it. A framed one drawing nothing (no camera, no parts, its
/// image transparent) is left to rur, which draws its background.
fn viewports(document: &Document, gui: Ref, windows: bool, global: bool, warnings: &mut Vec<String>) -> Vec<Viewport> {
	let mut found = Vec::new();
	let mut framed = Vec::new();
	let mut stack = vec![(gui, Vec::new())];
	while let Some((referent, placement)) = stack.pop() {
		let Some(instance) = document.dom.get_by_ref(referent) else {
			continue;
		};
		for &child in instance.children() {
			let Some(item) = document.dom.get_by_ref(child) else {
				continue;
			};
			// GUI objects lay out inside their parent's rect; a Folder holds them where it is
			if item.class == "Folder" {
				stack.push((child, placement.clone()));
				continue;
			}
			if !props::class_is_a(item.class.as_str(), "GuiObject") || !props::bool(item, "Visible", true) {
				continue;
			}
			let mut own = placement.clone();
			own.push(Placement::of(item));
			if item.class != "ViewportFrame" {
				stack.push((child, own));
				continue;
			}
			let color = props::color(item, "ImageColor3", Vec3::ONE);
			let opacity = 1.0 - props::f32(item, "ImageTransparency", 0.0).clamp(0.0, 1.0);
			// a window, else the frame's own camera (what draws over it is found once its world draws)
			let window = is_window(document, child);
			let camera = if window {
				if !windows {
					continue;
				}
				None
			} else {
				let Some(camera) = frame_camera(document, item).filter(|_| opacity > 0.0) else {
					continue;
				};
				Some(camera)
			};
			let lighting = ViewportLighting {
				ambient: props::color(item, "Ambient", Vec3::splat(200.0 / 255.0)),
				light_color: props::color(item, "LightColor", Vec3::splat(140.0 / 255.0)),
				light_direction: props::vec3(item, "LightDirection", Vec3::NEG_ONE),
				sky: None,
			};
			let roots: Vec<Ref> = item.children().iter().copied().filter(|&root| document.dom.get_by_ref(root).is_some_and(|root| !is_gui(root))).collect();
			let mut scene = super::extract::extract_roots(document, &roots);
			scene.in_viewport_frame(lighting);
			warnings.append(&mut scene.warnings);
			// Glass draws as in the world, opaque at Transparency 0 (Studio captures: Glass, Plastic,
			// Neon and five more materials alike in a window). What hides the walls of an
			// effect's rift and crater is their meshes' vertex alpha 0: a hole through the frame's
			// background that still hides what is behind it, their decals drawn over it (render::guis)
			scene.parts.retain(|part| part.transparency < 1.0 || !part.decals.is_empty());
			let (placement, tint, scene) = (own, [color.x, color.y, color.z, opacity], Arc::new(scene));
			let corner_radius = corner_radius(document, item);
			let Some((camera, fov)) = camera else {
				found.push(Viewport { placement, tint, scene, kind: ViewportKind::Window, corner_radius });
				continue;
			};
			if scene.parts.is_empty() {
				continue;
			}
			let over = match over_content(document, gui, child, global) {
				Some(Ok(bytes)) => Some((fnv1a(&bytes), Arc::new(bytes))),
				Some(Err(error)) => {
					warnings.push(format!("{}: could not serialise what draws over its ViewportFrame (drawn under its world): {error}", document.path_of(child)));
					None
				}
				None => None,
			};
			let kind = ViewportKind::Framed(Framed { camera, fov, over: over.map(|(hash, bytes)| (bytes, hash)), gradient: gradient(document, item) });
			framed.push((child, Viewport { placement, tint, scene, kind, corner_radius }));
		}
	}
	if !framed.is_empty() {
		let order = draw_order(&document.dom, gui, global);
		framed.sort_by_key(|(frame, _)| order.iter().position(|referent| referent == frame));
		found.extend(framed.into_iter().map(|(_, viewport)| viewport));
	}
	found
}

/// A ViewportFrame's first UICorner's radius (scale, offset): its TopLeftRadius, which files from a
/// current engine keep the radius in (rur reads the four corners), else its CornerRadius, else Roblox's
/// default (0, 8).
fn corner_radius(document: &Document, frame: &Instance) -> Option<[f32; 2]> {
	let corner = frame.children().iter().filter_map(|&child| document.dom.get_by_ref(child)).find(|child| child.class == "UICorner")?;
	// what the file holds (props::udim falls back to the reflection database's default)
	let held = |name: &str| match corner.properties.get(&rbx_dom_weak::ustr(name)) {
		Some(rbx_dom_weak::types::Variant::UDim(value)) => Some([value.scale, value.offset as f32]),
		_ => None,
	};
	Some(held("TopLeftRadius").or_else(|| held("CornerRadius")).unwrap_or([0.0, 8.0]))
}

/// A world-space bounding box's half size and centre of a part.
fn part_box(instance: &Instance) -> Option<(Vec3, Vec3)> {
	let cframe = props::cframe(instance, "CFrame")?;
	let size = props::vec3(instance, "Size", Vec3::new(4.0, 1.0, 2.0));
	let extent = Vec3::from(cframe.matrix3.abs() * glam::Vec3A::from(size * 0.5));
	Some((Vec3::from(cframe.translation), extent))
}

/// A Model's parts' combined world bounding box (centre, half size).
fn model_box(document: &Document, referent: Ref) -> Option<(Vec3, Vec3)> {
	let mut min = Vec3::splat(f32::INFINITY);
	let mut max = Vec3::splat(f32::NEG_INFINITY);
	let mut stack = vec![referent];
	while let Some(current) = stack.pop() {
		let Some(instance) = document.dom.get_by_ref(current) else {
			continue;
		};
		if props::is_a(instance, "BasePart")
			&& let Some((center, extent)) = part_box(instance)
		{
			min = min.min(center - extent);
			max = max.max(center + extent);
		}
		stack.extend(instance.children().iter().copied());
	}
	(min.x <= max.x).then(|| ((min + max) * 0.5, (max - min) * 0.5))
}

/// Where a BillboardGui sits: its Adornee (a part, an attachment or a model), else its parent (a part
/// or an attachment). Returns the anchor and the extents.
fn billboard_anchor(document: &Document, instance: &Instance) -> Option<(Vec3, Vec3)> {
	let target = props::reference(instance, "Adornee").unwrap_or_else(|| instance.parent());
	let adornee = document.dom.get_by_ref(target)?;
	if props::is_a(adornee, "BasePart") {
		return part_box(adornee);
	}
	if props::is_a(adornee, "Attachment") {
		return beams::world_cframe(document, target).map(|cframe| (Vec3::from(cframe.translation), Vec3::ZERO));
	}
	if props::is_a(adornee, "Model") && props::reference(instance, "Adornee").is_some() {
		return model_box(document, target);
	}
	None
}

/// A SurfaceGui or BillboardGui as drawn, or `None` when Studio draws nothing for it (disabled, or
/// nothing to draw on).
pub fn read(document: &Document, referent: Ref, instance: &Instance, warnings: &mut Vec<String>) -> Option<Gui> {
	if !props::bool(instance, "Enabled", true) {
		return None;
	}
	let path = document.path_of(referent);
	let kind = match instance.class.as_str() {
		// only a PlayerGui's: a ScreenGui anywhere else (the Workspace, a model) draws nothing
		"ScreenGui" => {
			if !document.dom.get_by_ref(instance.parent()).is_some_and(|parent| parent.class == "PlayerGui") {
				return None;
			}
			// IgnoreGuiInset is a view of ScreenInsets: off is CoreUISafeInsets (2), on DeviceSafeInsets (1)
			let fallback = if props::bool(instance, "IgnoreGuiInset", false) { 1 } else { 2 };
			GuiKind::Screen(ScreenGui { display_order: props::i32(instance, "DisplayOrder", 0), inset: props::enum_value(instance, "ScreenInsets", fallback) == 2 })
		}
		"SurfaceGui" => {
			let target = props::reference(instance, "Adornee").unwrap_or_else(|| instance.parent());
			let part = document.dom.get_by_ref(target).filter(|part| props::is_a(part, "BasePart"));
			let Some(part) = part else {
				warnings.push(format!("{path}: SurfaceGui without a part to draw on (not drawn)"));
				return None;
			};
			if !beams::drawn(document, target) {
				return None;
			}
			let cframe = props::cframe(part, "CFrame")?;
			let size = props::vec3(part, "Size", Vec3::new(4.0, 1.0, 2.0));
			let face = props::enum_value(instance, "Face", 5).min(5);
			// FixedSize: CanvasSize; PixelsPerStud: the face's size in studs times PixelsPerStud
			// (a 5.3 x 3.7 face at 50 reads AbsoluteSize 265, 185)
			let canvas = if props::enum_value(instance, "SizingMode", 0) == 1 {
				let pixels_per_stud = props::f32(instance, "PixelsPerStud", 50.0);
				face_size(size, face).map(|studs| studs * pixels_per_stud)
			} else {
				props::vec2(instance, "CanvasSize", [800.0, 600.0])
			};
			if canvas[0] <= 0.0 || canvas[1] <= 0.0 {
				return None;
			}
			GuiKind::Surface(SurfaceGui { cframe, size, face, canvas, z_offset: props::f32(instance, "ZOffset", 0.0) })
		}
		_ => {
			let Some((anchor, extents)) = billboard_anchor(document, instance) else {
				warnings.push(format!("{path}: BillboardGui without a part or attachment to follow (not drawn)"));
				return None;
			};
			let size = props::udim2(instance, "Size").unwrap_or([0.0; 4]);
			let extents_offset = props::vec3(instance, "ExtentsOffset", Vec3::ZERO);
			let extents_offset_world = props::vec3(instance, "ExtentsOffsetWorldSpace", Vec3::ZERO);
			GuiKind::Billboard(BillboardGui {
				anchor,
				extents,
				size,
				size_offset: props::vec2(instance, "SizeOffset", [0.0; 2]),
				camera_offset: props::vec3(instance, "StudsOffset", Vec3::ZERO) + extents_offset * extents,
				world_offset: props::vec3(instance, "StudsOffsetWorldSpace", Vec3::ZERO) + extents_offset_world * extents,
			})
		}
	};
	let content = match content(document, referent) {
		Ok(content) => content,
		Err(error) => {
			warnings.push(format!("{path}: could not serialise the GUI (not drawn): {error}"));
			return None;
		}
	};
	let z_index_behavior = props::enum_value(instance, "ZIndexBehavior", 0);
	let viewports = match &kind {
		GuiKind::Surface(_) => viewports(document, referent, true, z_index_behavior == 0, warnings),
		GuiKind::Billboard(_) => viewports(document, referent, false, z_index_behavior == 0, warnings),
		GuiKind::Screen(_) => Vec::new(),
	};
	Some(Gui {
		viewports,
		path,
		kind,
		content_hash: fnv1a(&content),
		content: Arc::new(content),
		light_influence: props::f32(instance, "LightInfluence", 0.0).clamp(0.0, 1.0),
		brightness: props::f32(instance, "Brightness", 1.0).max(0.0),
		always_on_top: props::bool(instance, "AlwaysOnTop", false),
		max_distance: props::f32(instance, "MaxDistance", 0.0),
		clips: props::bool(instance, "ClipsDescendants", false),
		z_index_behavior,
	})
}

#[cfg(test)]
mod tests {
	use glam::{Affine3A, Vec3};
	use rbx_dom_weak::InstanceBuilder;
	use rbx_dom_weak::types::{Enum, UDim, UDim2, Vector2, Vector3};

	use super::GuiKind;
	use crate::scene::build::{ModelBuilder, PartSpec, cframe};

	fn extract(model: ModelBuilder) -> crate::scene::Scene {
		crate::scene::extract::extract(&model.into_document())
	}

	#[test]
	fn surface_canvas_and_basis() {
		let mut model = ModelBuilder::new("Gui");
		let part = model.part(&PartSpec { size: Vec3::new(8.0, 6.0, 4.0), transparency: 1.0, ..Default::default() });
		model.child(part, InstanceBuilder::new("SurfaceGui").with_property("Face", Enum::from_u32(1)).with_property("SizingMode", Enum::from_u32(1)).with_property("PixelsPerStud", 50.0f32));
		model.child(part, InstanceBuilder::new("SurfaceGui").with_property("Face", Enum::from_u32(5)).with_property("CanvasSize", Vector2::new(320.0, 240.0)));
		let scene = extract(model);
		// an invisible part still holds its GUIs
		assert!(scene.parts.is_empty());
		assert_eq!(scene.guis.len(), 2);
		let GuiKind::Surface(top) = &scene.guis[0].kind else {
			panic!("not a SurfaceGui");
		};
		// Top: the canvas runs along -Z (4 studs) and down along +X (8 studs)
		assert_eq!(top.canvas, [200.0, 400.0]);
		assert!(top.point(0.0, 0.0).distance(Vec3::new(-4.0, 3.0, 2.0)) < 1e-5);
		assert!(top.point(1.0, 1.0).distance(Vec3::new(4.0, 3.0, -2.0)) < 1e-5);
		assert!(top.normal().distance(Vec3::Y) < 1e-5);
		let GuiKind::Surface(front) = &scene.guis[1].kind else {
			panic!("not a SurfaceGui");
		};
		// Front: right is -X, down is -Y, as for decals
		assert_eq!(front.canvas, [320.0, 240.0]);
		assert!(front.point(0.0, 0.0).distance(Vec3::new(4.0, 3.0, -2.0)) < 1e-5);
	}

	#[test]
	fn billboard_anchors_and_offsets() {
		let mut model = ModelBuilder::new("Gui");
		let part = model.part(&PartSpec { cframe: Affine3A::from_translation(Vec3::new(10.0, 0.0, 0.0)), size: Vec3::new(2.0, 4.0, 6.0), ..Default::default() });
		let attachment = model.child(part, InstanceBuilder::new("Attachment").with_property("CFrame", cframe(Affine3A::from_translation(Vec3::new(0.0, 3.0, 0.0)))));
		let size = UDim2::new(UDim::new(4.0, 10), UDim::new(1.0, 0));
		model.child(attachment, InstanceBuilder::new("BillboardGui").with_property("Size", size));
		model.child(
			part,
			InstanceBuilder::new("BillboardGui")
				.with_property("Size", size)
				.with_property("StudsOffset", Vector3::new(0.0, 2.0, 0.0))
				.with_property("ExtentsOffset", Vector3::new(0.0, 1.0, 0.0))
				.with_property("ExtentsOffsetWorldSpace", Vector3::new(1.0, 0.0, 0.0)),
		);
		// a disabled one draws nothing
		model.child(part, InstanceBuilder::new("BillboardGui").with_property("Enabled", false));
		let mut scene = extract(model);
		assert_eq!(scene.guis.len(), 2);
		let GuiKind::Billboard(on_attachment) = scene.guis.remove(0).kind else {
			panic!("not a BillboardGui");
		};
		assert_eq!(on_attachment.anchor, Vec3::new(10.0, 3.0, 0.0));
		assert_eq!(on_attachment.extents, Vec3::ZERO);
		assert_eq!(on_attachment.size, [4.0, 10.0, 1.0, 0.0]);
		let GuiKind::Billboard(on_part) = scene.guis.remove(0).kind else {
			panic!("not a BillboardGui");
		};
		// extents are half the part's box: camera-space y 2 + 2, world x 1
		assert_eq!(on_part.camera_offset, Vec3::new(0.0, 4.0, 0.0));
		assert_eq!(on_part.world_offset, Vec3::new(1.0, 0.0, 0.0));
	}

	#[test]
	fn screen_guis_draw_from_a_player_gui_alone() {
		// a recording's PlayerGui (ree keeps the ScreenGuis the player's PlayerGui shows in one): a
		// cutscene's fade over the whole view, a HUD below the top bar; a ScreenGui elsewhere draws nothing
		let mut model = ModelBuilder::new("Scene");
		let player_gui = model.child(model.model, InstanceBuilder::new("PlayerGui").with_name("PlayerGui"));
		let fade = model.child(player_gui, InstanceBuilder::new("ScreenGui").with_name("Transition").with_property("DisplayOrder", 3i32).with_property("ScreenInsets", Enum::from_u32(1)));
		model.child(fade, InstanceBuilder::new("Frame").with_name("LOADING").with_property("Size", UDim2::new(UDim::new(1.0, 0), UDim::new(1.0, 0))));
		model.child(player_gui, InstanceBuilder::new("ScreenGui").with_name("Hud").with_property("IgnoreGuiInset", false));
		model.child(player_gui, InstanceBuilder::new("ScreenGui").with_name("Off").with_property("Enabled", false));
		model.child(model.model, InstanceBuilder::new("ScreenGui").with_name("Stray"));
		let scene = extract(model);
		let screens: Vec<(&str, &super::ScreenGui)> = scene
			.guis
			.iter()
			.filter_map(|gui| match &gui.kind {
				GuiKind::Screen(screen) => Some((gui.path.as_str(), screen)),
				_ => None,
			})
			.collect();
		assert_eq!(screens.len(), 2, "{:?}", scene.guis.iter().map(|gui| &gui.path).collect::<Vec<_>>());
		assert!(screens[0].0.ends_with("PlayerGui.Transition") && screens[0].1 == &super::ScreenGui { display_order: 3, inset: false });
		assert!(screens[1].0.ends_with("PlayerGui.Hud") && screens[1].1 == &super::ScreenGui { display_order: 0, inset: true });
		assert!(scene.warnings.is_empty(), "{:?}", scene.warnings);
		assert_eq!((super::top_bar_inset(true), super::top_bar_inset(false)), (58.0, 0.0));
	}

	/// A BillboardGui as an emote builds its portrait: a backdrop, a ViewportFrame holding a
	/// WorldModel and a camera, its UIGradient, a label over it.
	fn portrait(frame_transparency: f32, label: bool) -> crate::scene::Scene {
		use rbx_dom_weak::types::{CFrame, Color3, Matrix3, NumberSequence, NumberSequenceKeypoint, Vector2};
		let mut model = ModelBuilder::new("Portrait");
		let holder = model.part(&PartSpec { transparency: 1.0, ..Default::default() });
		let gui = model.child(holder, InstanceBuilder::new("BillboardGui").with_name("Billboard").with_property("Size", UDim2::new(UDim::new(8.0, 0), UDim::new(6.0, 0))).with_property("ZIndexBehavior", Enum::from_u32(1)));
		model.child(gui, InstanceBuilder::new("Frame").with_name("Backdrop").with_property("Size", UDim2::new(UDim::new(1.0, 0), UDim::new(1.0, 0))).with_property("BackgroundTransparency", 0.0f32).with_property("ZIndex", 1i32));
		let camera = CFrame::new(Vector3::new(0.0, 1.0, 8.0), Matrix3::identity());
		let frame = model.child(
			gui,
			InstanceBuilder::new("ViewportFrame")
				.with_name("Portrait")
				.with_property("AnchorPoint", Vector2::new(0.5, 0.5))
				.with_property("Position", UDim2::new(UDim::new(0.5, 0), UDim::new(0.5, 0)))
				.with_property("Size", UDim2::new(UDim::new(0.9, 0), UDim::new(0.8, 0)))
				.with_property("BackgroundTransparency", 1.0f32)
				.with_property("ImageColor3", Color3::new(1.0, 0.5, 0.25))
				.with_property("ImageTransparency", frame_transparency)
				.with_property("Ambient", Color3::new(0.5, 0.5, 0.5))
				.with_property("CameraCFrame", camera)
				.with_property("CameraFieldOfView", 40.0f32)
				.with_property("ZIndex", 2i32),
		);
		model.child(
			frame,
			InstanceBuilder::new("UIGradient").with_property("Rotation", 90.0f32).with_property("Transparency", NumberSequence { keypoints: vec![NumberSequenceKeypoint::new(0.0, 0.0, 0.0), NumberSequenceKeypoint::new(1.0, 1.0, 0.0)] }),
		);
		let world = model.child(frame, InstanceBuilder::new("WorldModel"));
		model.child(world, InstanceBuilder::new("Part").with_name("Head").with_property("Size", Vector3::new(2.0, 2.0, 2.0)).with_property("CFrame", CFrame::new(Vector3::new(0.0, 0.0, 0.0), Matrix3::identity())));
		if label {
			model.child(gui, InstanceBuilder::new("ImageLabel").with_name("Eye").with_property("ZIndex", 6i32));
		}
		extract(model)
	}

	#[test]
	fn a_billboards_viewport_frame_draws_its_world_through_its_own_camera() {
		let scene = portrait(0.25, true);
		assert!(scene.parts.is_empty(), "the world is the frame's, not the Workspace's");
		let [viewport] = scene.guis[0].viewports.as_slice() else {
			panic!("one ViewportFrame: {:?}", scene.guis[0].viewports.len());
		};
		let super::ViewportKind::Framed(framed) = &viewport.kind else {
			panic!("not framed");
		};
		assert_eq!(framed.camera, [0.0, 1.0, 8.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
		assert_eq!(framed.fov, 40.0);
		assert_eq!(viewport.tint, [1.0, 0.5, 0.25, 0.75]);
		assert!(viewport.rect([800.0, 600.0]).iter().zip([0.05, 0.1, 0.95, 0.9]).all(|(a, b)| (a - b).abs() < 1e-6));
		// lit by the frame's own light, its one part
		assert_eq!(viewport.scene.parts.len(), 1);
		assert!(viewport.scene.environment.viewport.is_some());
		let gradient = framed.gradient.as_ref().expect("its UIGradient");
		assert_eq!((gradient.rotation, gradient.transparency.as_slice()), (90.0, &[(0.0, 0.0), (1.0, 1.0)][..]));
		// rur draws the frame as a plain Frame without its world (its background alone), and what draws
		// over it is the GUI with the backdrop and the frame drawing none of their own, the label as it is
		let names = |bytes: &[u8]| -> Vec<(String, String, Option<f32>)> {
			let dom = rbx_binary::from_reader(bytes).unwrap();
			dom.descendants()
				.filter(|instance| instance.class != "DataModel")
				.map(|instance| {
					let background = match instance.properties.get(&rbx_dom_weak::ustr("BackgroundTransparency")) {
						Some(rbx_dom_weak::types::Variant::Float32(value)) => Some(*value),
						_ => None,
					};
					(instance.class.to_string(), instance.name.clone(), background)
				})
				.collect()
		};
		let content = names(&scene.guis[0].content);
		assert!(content.iter().any(|(class, name, _)| class == "Frame" && name == "Portrait"), "{content:?}");
		assert!(content.iter().all(|(class, _, _)| !matches!(class.as_str(), "ViewportFrame" | "WorldModel" | "Part" | "Camera")), "{content:?}");
		assert!(content.iter().any(|(_, name, background)| name == "Backdrop" && *background == Some(0.0)));
		let (over, hash) = framed.over.as_ref().expect("the label draws over it");
		assert_eq!(*hash, super::fnv1a(over));
		let over = names(over);
		assert!(over.iter().any(|(_, name, background)| name == "Backdrop" && *background == Some(1.0)), "{over:?}");
		assert!(over.iter().any(|(class, name, _)| class == "ImageLabel" && name == "Eye"));
	}

	#[test]
	fn a_viewport_frames_ui_corner_rounds_its_world() {
		let corner = |radius: Option<UDim>, top_left: Option<UDim>| {
			let mut model = ModelBuilder::new("Rounded");
			let holder = model.part(&PartSpec { transparency: 1.0, ..Default::default() });
			let gui = model.child(holder, InstanceBuilder::new("BillboardGui").with_property("Size", UDim2::new(UDim::new(8.0, 0), UDim::new(6.0, 0))));
			let frame = model.child(
				gui,
				InstanceBuilder::new("ViewportFrame")
					.with_property("Size", UDim2::new(UDim::new(0.5, 0), UDim::new(0.5, 0)))
					.with_property("CameraCFrame", rbx_dom_weak::types::CFrame::new(Vector3::new(0.0, 0.0, 8.0), rbx_dom_weak::types::Matrix3::identity())),
			);
			let mut builder = InstanceBuilder::new("UICorner");
			if let Some(radius) = radius {
				builder = builder.with_property("CornerRadius", radius);
			}
			if let Some(radius) = top_left {
				builder = builder.with_property("TopLeftRadius", radius);
			}
			model.child(frame, builder);
			model.child(frame, InstanceBuilder::new("Part").with_property("Size", Vector3::new(2.0, 2.0, 2.0)));
			extract(model).guis[0].viewports[0].clone()
		};
		// rur's radius: the scale of the shorter side plus the offset, at most half that side; 400 x 300
		// in an 800 x 600 canvas
		let circle = corner(Some(UDim::new(0.5, 0)), None);
		assert_eq!(circle.corner_radius, Some([0.5, 0.0]));
		assert_eq!(circle.corner([800.0, 600.0]), [150.0 / 800.0, 150.0 / 600.0]);
		assert_eq!(corner(Some(UDim::new(0.1, 20)), None).corner([800.0, 600.0]), [50.0 / 800.0, 50.0 / 600.0]);
		assert_eq!(corner(Some(UDim::new(1.0, 0)), None).corner([800.0, 600.0]), [150.0 / 800.0, 150.0 / 600.0]);
		// a UICorner the file leaves at its default: 8 pixels
		assert_eq!(corner(None, None).corner([800.0, 600.0]), [8.0 / 800.0, 8.0 / 600.0]);
		// a current engine's file: the radius in the corners, CornerRadius left at its default
		assert_eq!(corner(Some(UDim::new(0.0, 8)), Some(UDim::new(0.5, 0))).corner_radius, Some([0.5, 0.0]));
	}

	#[test]
	fn a_viewport_frame_drawing_nothing_is_left_to_rur() {
		// nothing drawn after it: no image of what draws over it
		let scene = portrait(0.0, false);
		let super::ViewportKind::Framed(framed) = &scene.guis[0].viewports[0].kind else {
			panic!("not framed");
		};
		assert!(framed.over.is_none());
		// its image transparent: no world to draw
		assert!(portrait(1.0, true).guis[0].viewports.is_empty());
	}

	#[test]
	fn guis_draw_their_objects_in_rurs_order() {
		// A (ZIndex 2) holding A1 (1), B (1), a Folder holding C (3), D hidden holding D1
		let mut model = ModelBuilder::new("Order");
		let gui = model.child(model.model, InstanceBuilder::new("ScreenGui").with_name("Gui"));
		let frame = |name: &str, z: i32| InstanceBuilder::new("Frame").with_name(name).with_property("ZIndex", z);
		let a = model.child(gui, frame("A", 2));
		model.child(a, frame("A1", 1));
		model.child(gui, frame("B", 1));
		let folder = model.child(gui, InstanceBuilder::new("Folder"));
		model.child(folder, frame("C", 3));
		let d = model.child(gui, frame("D", 1).with_property("Visible", false));
		model.child(d, frame("D1", 1));
		let document = model.into_document();
		let gui = document.dom.descendants().find(|instance| instance.name == "Gui").unwrap().referent();
		let names = |global: bool| -> Vec<String> { super::draw_order(&document.dom, gui, global).iter().map(|&referent| document.dom.get_by_ref(referent).unwrap().name.clone()).collect() };
		// Sibling: siblings by ZIndex, each followed by what it holds
		assert_eq!(names(false), ["B", "A", "A1", "C"]);
		// Global: everything by ZIndex, ties in tree order
		assert_eq!(names(true), ["A1", "B", "A", "C"]);
	}

	#[test]
	fn a_gradient_runs_across_its_object_as_rurs() {
		// rur's gradient_plane: t at a point of a rect (x, y, width, height) in pixels
		let rur = |rotation: f32, offset: [f32; 2], rect: [f32; 4], point: [f32; 2]| {
			let angle = rotation.to_radians();
			let (dx, dy) = (angle.cos(), angle.sin());
			let span = (rect[2] / dx.abs().max(1e-12)).min(rect[3] / dy.abs().max(1e-12));
			let (cx, cy) = (rect[0] + rect[2] / 2.0 + offset[0] * rect[2], rect[1] + rect[3] / 2.0 + offset[1] * rect[3]);
			(dx * point[0] + dy * point[1]) / span - (cx * dx + cy * dy) / span + 0.5
		};
		for (rotation, offset) in [(0.0, [0.0, 0.0]), (90.0, [0.0, 0.0]), (45.0, [0.1, -0.2]), (200.0, [0.0, 0.3])] {
			let gradient = super::Gradient { rotation, offset, colors: Vec::new(), transparency: Vec::new() };
			let rect = [30.0, 40.0, 300.0, 120.0];
			let [a, b, c] = gradient.plane([rect[2], rect[3]]);
			for (x, y) in [(0.0, 0.0), (1.0, 0.0), (0.25, 0.75), (1.0, 1.0)] {
				let ours = a * x + b * y + c;
				let theirs = rur(rotation, offset, rect, [rect[0] + x * rect[2], rect[1] + y * rect[3]]);
				assert!((ours - theirs).abs() < 1e-4, "{rotation} {offset:?} ({x}, {y}): {ours} against {theirs}");
			}
		}
	}

	#[test]
	fn a_gui_that_cannot_be_written_says_why() {
		// a property holding a value of another type than its class gives it (text in a UDim2 Size, as
		// --set's "[0.3,0,0.3,0]" left one): rbx_binary refuses the GUI's model, rur cannot draw it,
		// and the warning names the property
		let mut model = ModelBuilder::new("Gui");
		let part = model.part(&PartSpec { transparency: 1.0, ..Default::default() });
		let gui = model.child(part, InstanceBuilder::new("SurfaceGui").with_name("Sign"));
		model.child(gui, InstanceBuilder::new("ImageLabel").with_name("Ring").with_property("Size", rbx_dom_weak::types::Variant::String("[0.3,0,0.3,0]".into())));
		let scene = extract(model);
		assert!(scene.guis.is_empty());
		assert_eq!(scene.warnings.len(), 1, "{:?}", scene.warnings);
		let warning = &scene.warnings[0];
		assert!(warning.contains("Sign: could not serialise the GUI (not drawn): ") && warning.contains("ImageLabel.Size to be of type UDim2"), "{warning}");
	}
}
