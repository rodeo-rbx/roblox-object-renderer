//! SurfaceGuis and BillboardGuis (notes/render.md "GUIs"). rur draws their content (render::rur)
//! into textures cached by content, canvas size and scale, in memory and on disk
//! (`<state dir>/guis/<rur build>-<code>`, render::gui_files; the content store's too, which a state
//! dir lacking one takes it from). A SurfaceGui is a quad on its part's face, a BillboardGui a
//! quad facing the camera at its anchor's depth, sized in whole pixels; both blend with the
//! transparent geometry (`shaders/gui.wgsl`), and AlwaysOnTop ones are a layer laid over the
//! finished image, Highlights included.
//!
//! Sizes in GUI pixels follow Studio's capture viewport: SurfaceGui content is drawn at 1.5 device
//! pixels per canvas pixel (the scale Studio snaps and rasterises it at here), and a BillboardGui's
//! pixel offsets, text and lines scale with the image height, `height / 744` device pixels per GUI
//! pixel (744 GUI pixels is the capture viewport's height at DPI 144, so a 1116 px tall render is
//! Studio's 1.5), unless a view sets `guiScale`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use parking_lot::Mutex;
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use super::camera::Camera;
use super::ids::{Readback, copy_out};
use super::pipelines::{DEPTH_FORMAT, HDR_FORMAT, OUTPUT_FORMAT, SAMPLE_COUNTS, SHADOW};
use super::gui_files::{GUI_CHUNK, GuiImage};
use super::resources::{GpuTexture, Weighed};
use super::{Prepared, Renderer, ViewRequest, rur};
use crate::scene::Scene;
use crate::scene::guis::{GRADIENT_KEYS, Gui, GuiKind, ViewportKind};

const COMMON: &str = include_str!("../../shaders/common.wgsl");
const GUI: &str = include_str!("../../shaders/gui.wgsl");

/// Device pixels per canvas pixel of SurfaceGui content: Studio snaps its rects to 1/1.5 canvas pixel
/// and rasterises its glyphs at 1.5 x TextSize (gui-surface-basics: magnified, one-pixel frames
/// come out 0.67 or 1.33 canvas pixels).
pub const SURFACE_SCALE: f32 = 1.5;
/// The capture viewport's height in GUI pixels (1116 device pixels at DPI 144).
pub const REFERENCE_HEIGHT: f32 = 744.0;
/// The largest GUI texture side (device pixels, margins included); bigger canvases are drawn at a
/// smaller scale.
const MAX_TEXTURE: f32 = 2048.0;
/// Device pixels of GUI content drawn by one rur call at most, and how many calls run at once: rur
/// draws a call's atlas on the GPU in one frame, kept about the size of a screen (a long GPU
/// submission can make the driver reset the device, every other app's included).
const BATCH_PIXELS: u64 = 2048 * 2048;
/// rur calls one resolution of GUIs makes at once (rur's service draws one atlas at a time; the
/// others run their Luau, are written and read meanwhile).
const RUR_WORKERS: usize = 3;
/// rur's revision while its native app has no build ([`rur::revision`]): nothing is cached on disk.
const UNBUILT: &str = "unbuilt";
/// How long a GUI whose content could not be drawn is not tried again.
const FAILURE_TTL: Duration = Duration::from_secs(60);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GuiVertex {
	position: [f32; 3],
	uv: [f32; 2],
	normal: [f32; 3],
	/// x LightInfluence, y Brightness, z 1 when the sun reaches it, w unused
	params: [f32; 4],
}

impl GuiVertex {
	const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3, 3 => Float32x4];

	fn layout() -> wgpu::VertexBufferLayout<'static> {
		wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<GuiVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &Self::ATTRIBUTES }
	}
}

/// What a GUI's texture holds: its content, canvas size (1/1024 GUI pixel), scale (1/1024) and
/// whether it clips (else the texture keeps margins for what overflows the canvas).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GuiKey {
	content: u64,
	width: u32,
	height: u32,
	scale: u32,
	clips: bool,
}

impl GuiKey {
	fn new(content: u64, canvas: [f32; 2], scale: f32, clips: bool) -> Self {
		// a canvas bigger than a texture can hold is drawn at a smaller scale (the margins: a quarter of
		// the canvas and 8 GUI pixels on each side, up to a device pixel more each when rounded up)
		let side = canvas[0].max(canvas[1]);
		let extent = if clips { side } else { side * 1.5 + 16.0 };
		let scale = scale.min((MAX_TEXTURE - 3.0) / extent.max(1.0));
		let quantize = |value: f32, steps: f32| (value * steps).round().clamp(1.0, u32::MAX as f32) as u32;
		// the scale rounded down, so a capped one stays within the limit
		let scale = (scale * 1024.0).floor().max(1.0) as u32;
		Self { content, width: quantize(canvas[0], 1024.0), height: quantize(canvas[1], 1024.0), scale, clips }
	}

	fn canvas(&self) -> [f32; 2] {
		[self.width as f32 / 1024.0, self.height as f32 / 1024.0]
	}

	fn scale(&self) -> f32 {
		self.scale as f32 / 1024.0
	}

	/// Device pixels kept on each side of a canvas that does not clip: Studio draws what overflows it
	/// (gui-surface-misc: overflow), here up to a quarter of the canvas and 8 GUI pixels.
	fn margin(&self) -> [u32; 2] {
		if self.clips {
			return [0, 0];
		}
		self.canvas().map(|size| ((size * 0.25 + 8.0) * self.scale()).ceil() as u32)
	}

	/// The image's extent in canvas fractions: (left, top, right, bottom), beyond 0..1 by the margins.
	fn extent(&self) -> [f32; 4] {
		let [width, height] = self.canvas().map(|size| size * self.scale());
		let [x, y] = self.margin().map(|margin| margin as f32);
		[-x / width, -y / height, 1.0 + x / width, 1.0 + y / height]
	}

	/// Its file in the disk cache ([`GuiImage`]'s `.gui`).
	fn file_name(&self) -> String {
		format!("{}.gui", self.stem())
	}

	fn stem(&self) -> String {
		format!("{:016x}-{}x{}-{}{}", self.content, self.width, self.height, self.scale, if self.clips { "" } else { "-overflow" })
	}
}

pub struct GuiTexture {
	texture: GpuTexture,
	group: wgpu::BindGroup,
	/// The share of the texture the canvas and its margins cover (u, v): the texture is whole device
	/// pixels.
	uv: [f32; 2],
	/// Where its texture goes once dropped: a frame's own image's, after that frame is submitted
	/// ([`GuiUploads`])
	recycle: Option<Arc<GuiUploads>>,
}

impl Drop for GuiTexture {
	fn drop(&mut self) {
		if let Some(uploads) = self.recycle.take()
			&& let Some(held) = self.texture.held()
		{
			uploads.give_back(held.texture, held.view, self.group.clone());
		}
	}
}

/// What a clip's GUI uploads go through again frame after frame ([`Renderer::upload_guis`]): two
/// staging buffers taking turns, each mapped again once its last copies are done, and the textures
/// of images only one frame drew, back from that frame once it is submitted (dropped after its
/// render: a later copy into one queues after the draws that read it), for the next image of their
/// size. An animated GUI is a new image every frame, its size the same: 13 telegraphs made a ~25 MB
/// staging buffer and 13 textures with their bind groups every frame, 1.1 ms of each frame's upload
/// for the buffer alone.
#[derive(Default)]
pub struct GuiUploads {
	staging: Mutex<StagingPair>,
	textures: Mutex<Vec<Recycled>>,
}

/// A GUI texture given back ([`GuiUploads`]): its size and levels, its view and bind group, when.
struct Recycled {
	size: (u32, u32, u32),
	texture: wgpu::Texture,
	view: wgpu::TextureView,
	group: wgpu::BindGroup,
	given: Instant,
}

#[derive(Default)]
struct StagingPair {
	buffers: [Option<(wgpu::Buffer, super::gpu::Submitted)>; 2],
	turn: usize,
	used: Option<Instant>,
}

/// Recycled GUI textures kept past this go (a clip's frames take theirs back within a frame or two)
const GUI_TEXTURES_KEPT: usize = 64;
/// What the uploads keep goes after this long unused ([`GuiUploads::trim`])
const GUI_UPLOADS_KEPT_FOR: std::time::Duration = std::time::Duration::from_secs(2);

impl GuiUploads {
	fn give_back(&self, texture: wgpu::Texture, view: wgpu::TextureView, group: wgpu::BindGroup) {
		let mut textures = self.textures.lock();
		if textures.len() >= GUI_TEXTURES_KEPT {
			textures.remove(0);
		}
		textures.push(Recycled { size: (texture.width(), texture.height(), texture.mip_level_count()), texture, view, group, given: Instant::now() });
	}

	fn take(&self, width: u32, height: u32, levels: u32) -> Option<(wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)> {
		let mut textures = self.textures.lock();
		let index = textures.iter().rposition(|entry| entry.size == (width, height, levels))?;
		let Recycled { texture, view, group, .. } = textures.remove(index);
		Some((texture, view, group))
	}

	/// Lets go of the staging buffers and textures unused for a while; returns their bytes.
	pub fn trim(&self, all: bool) -> u64 {
		let mut freed = 0;
		{
			let mut staging = self.staging.lock();
			if all || staging.used.is_some_and(|used| used.elapsed() > GUI_UPLOADS_KEPT_FOR) {
				for (buffer, _) in staging.buffers.iter_mut().filter_map(Option::take) {
					freed += buffer.size();
				}
				staging.used = None;
			}
		}
		self.textures.lock().retain(|entry| {
			let keep = !all && entry.given.elapsed() <= GUI_UPLOADS_KEPT_FOR;
			if !keep {
				let (width, height, levels) = entry.size;
				freed += (0..levels).map(|level| ((width >> level).max(1) as u64) * ((height >> level).max(1) as u64) * 4).sum::<u64>();
			}
			keep
		});
		freed
	}
}

impl Weighed for GuiTexture {
	fn bytes(&self) -> u64 {
		self.texture.bytes()
	}
}

/// What a ViewportFrame's world adds to a GUI draw (`shaders/gui.wgsl` GuiWindow): the frame's rect in
/// the quad's uv (left, top, right, bottom) and the image's tint (ImageColor3, and its opacity in w; 0:
/// no world); for a framed one (`mode` x 1) its UIGradient: `plane` gives t over the frame, `mode` y
/// and z count `colors` (t, r, g, b) and `transparency` (t, value) keypoints; `corner` xy its UICorner's
/// radius in the quad's uv (0: square corners).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct WindowUniform {
	rect: [f32; 4],
	tint: [f32; 4],
	mode: [f32; 4],
	plane: [f32; 4],
	colors: [[f32; 4]; GRADIENT_KEYS],
	transparency: [[f32; 4]; GRADIENT_KEYS],
	corner: [f32; 4],
}

/// The GUI pipelines: in the scene (per MSAA sample count) and over the finished image.
pub struct GuiPipelines {
	layout: wgpu::BindGroupLayout,
	sampler: wgpu::Sampler,
	/// A window's world as the view sees it (group 2), and the group for a GUI without one.
	window_layout: wgpu::BindGroupLayout,
	window_sampler: wgpu::Sampler,
	no_window: wgpu::BindGroup,
	/// A GUI texture of nothing (group 1): what draws over a framed ViewportFrame when nothing does
	no_texture: wgpu::BindGroup,
	/// 1 x 1, zero
	blank: wgpu::TextureView,
	scene: Vec<(u32, wgpu::RenderPipeline)>,
	overlay: wgpu::RenderPipeline,
}

impl GuiPipelines {
	pub fn new(device: &wgpu::Device, frame_layout: &wgpu::BindGroupLayout, cache: Option<&wgpu::PipelineCache>) -> Self {
		let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("gui texture"),
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
		let window_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("gui window"),
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
					count: None,
				},
				wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
				wgpu::BindGroupLayoutEntry {
					binding: 2,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
					count: None,
				},
				// the share of each pixel an opaque surface of the world covers (ViewImage::opaque_share)
				wgpu::BindGroupLayoutEntry {
					binding: 3,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
					count: None,
				},
			],
		});
		// the world is drawn at the view's size and read pixel for pixel: no mips, no anisotropy
		let window_sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("gui window"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
		// wgpu clears a new texture to zero: transparent, nothing opaque (a window drawn without its share)
		let blank = device
			.create_texture(&wgpu::TextureDescriptor {
				label: Some("gui no window"),
				size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
				mip_level_count: 1,
				sample_count: 1,
				dimension: wgpu::TextureDimension::D2,
				format: wgpu::TextureFormat::Rgba8Unorm,
				usage: wgpu::TextureUsages::TEXTURE_BINDING,
				view_formats: &[],
			})
			.create_view(&Default::default());
		let no_window = {
			// the uniform's zero opacity draws none
			let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("gui no window"), contents: bytemuck::bytes_of(&WindowUniform::default()), usage: wgpu::BufferUsages::UNIFORM });
			device.create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("gui no window"),
				layout: &window_layout,
				entries: &[
					wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&blank) },
					wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&window_sampler) },
					wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
					wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&blank) },
				],
			})
		};
		// trilinear and anisotropic: SurfaceGuis are mostly seen smaller than their texture, and at angles
		let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
			label: Some("gui"),
			mag_filter: wgpu::FilterMode::Linear,
			min_filter: wgpu::FilterMode::Linear,
			mipmap_filter: wgpu::FilterMode::Linear,
			anisotropy_clamp: 16,
			..Default::default()
		});
		let no_texture = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("gui no texture"),
			layout: &layout,
			entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&blank) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) }],
		});
		let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gui"), source: wgpu::ShaderSource::Wgsl(format!("{COMMON}\n{SHADOW}\n{GUI}").into()) });
		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("gui"), bind_group_layouts: &[frame_layout, &layout, &window_layout], push_constant_ranges: &[] });
		let premultiplied = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
		let pipeline = |label: &str, entry: &str, samples: u32, format: wgpu::TextureFormat, depth: Option<wgpu::DepthStencilState>| {
			device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
				label: Some(label),
				layout: Some(&pipeline_layout),
				vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_gui"), buffers: &[GuiVertex::layout()], compilation_options: Default::default() },
				primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
				depth_stencil: depth,
				multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
				fragment: Some(wgpu::FragmentState {
					module: &module,
					entry_point: Some(entry),
					targets: &[Some(wgpu::ColorTargetState { format, blend: premultiplied, write_mask: wgpu::ColorWrites::ALL })],
					compilation_options: Default::default(),
				}),
				multiview: None,
				cache,
			})
		};
		let scene = SAMPLE_COUNTS
			.iter()
			.map(|&samples| {
				let depth = wgpu::DepthStencilState {
					format: DEPTH_FORMAT,
					depth_write_enabled: false,
					depth_compare: wgpu::CompareFunction::GreaterEqual,
					stencil: Default::default(),
					bias: Default::default(),
				};
				(samples, pipeline("gui", "fs_gui", samples, HDR_FORMAT, Some(depth)))
			})
			.collect();
		let overlay = pipeline("gui overlay", "fs_gui_overlay", 1, OUTPUT_FORMAT, None);
		Self { layout, sampler, window_layout, window_sampler, no_window, no_texture, blank, scene, overlay }
	}

	fn scene(&self, samples: u32) -> &wgpu::RenderPipeline {
		&self.scene.iter().find(|(count, _)| *count == samples).unwrap_or(&self.scene[0]).1
	}
}

/// GUI textures that could not be drawn recently (not tried again for a while).
#[derive(Default)]
pub struct GuiFailures {
	keys: Mutex<HashMap<GuiKey, (Instant, String)>>,
}

/// A GUI as a view draws it.
pub(super) struct GuiDraw {
	/// The GUI's index in `Scene::guis`.
	gui: usize,
	pub(super) key: GuiKey,
	/// Top-left, top-right, bottom-left, bottom-right.
	corners: [Vec3; 4],
	normal: Vec3,
	params: [f32; 4],
	/// The point it sorts by among transparent geometry, and its order on a shared face (ZOffset).
	pub center: Vec3,
	pub z_offset: f32,
	/// AlwaysOnTop: in the layer over the finished image.
	pub overlay: bool,
	/// A ScreenGui's DisplayOrder: in that layer over the world's GUIs, by DisplayOrder.
	pub screen: Option<i32>,
	/// Its framed ViewportFrames as this view draws them, in the order the GUI draws them.
	framed: Vec<FramedDraw>,
}

/// A framed ViewportFrame of a GUI draw (scene::guis::Framed): its world drawn by its own camera at
/// about the size the view shows the frame, laid inside the frame's rect under what the GUI draws over
/// it (`over`'s image; none when nothing does).
struct FramedDraw {
	/// Its index in the GUI's viewports.
	viewport: usize,
	/// Its rect in canvas fractions (left, top, right, bottom).
	rect: [f32; 4],
	/// Its picture's size in pixels, at the frame's own proportions.
	size: (u32, u32),
	over: Option<GuiKey>,
	/// Its UIGradient's t over it (scene::guis::Gradient::plane).
	plane: [f32; 3],
}

impl GuiDraw {
	/// The GUI images it draws with: the GUI's own, and what draws over each framed ViewportFrame.
	pub(super) fn keys(&self) -> impl Iterator<Item = GuiKey> + '_ {
		std::iter::once(self.key).chain(self.framed.iter().filter_map(|framed| framed.over))
	}
}

/// A framed ViewportFrame's picture is drawn no smaller than this and no larger than a GUI image (pixels).
const FRAMED_MIN: f32 = 8.0;

/// A GUI's framed ViewportFrames as a draw of it shows them: on a canvas of `canvas` GUI pixels, its
/// image keyed as `key`, the frame's height on the screen given by `height` (its rect in canvas fractions
/// and in GUI pixels → device pixels).
fn plan_framed(gui: &Gui, canvas: [f32; 2], key: &GuiKey, height: &dyn Fn([f32; 4], [f32; 4]) -> f32) -> Vec<FramedDraw> {
	gui.viewports
		.iter()
		.enumerate()
		.filter_map(|(index, viewport)| {
			let ViewportKind::Framed(framed) = &viewport.kind else {
				return None;
			};
			let rect = viewport.rect(canvas);
			let pixels = viewport.pixels(canvas);
			let (width, tall) = (pixels[2], pixels[3]);
			if !(width > 0.0 && tall > 0.0) {
				return None;
			}
			// at the frame's own proportions (its camera's aspect), as tall as it shows
			let rows = height(rect, pixels).round().clamp(FRAMED_MIN, MAX_TEXTURE);
			let columns = (rows * width / tall).round().clamp(FRAMED_MIN, MAX_TEXTURE);
			let over = framed.over.as_ref().map(|(_, hash)| GuiKey { content: *hash, ..*key });
			let plane = framed.gradient.as_ref().map_or([0.0; 3], |gradient| gradient.plane([width, tall]));
			Some(FramedDraw { viewport: index, rect, size: (columns as u32, rows as u32), over, plane })
		})
		.collect()
}

/// Whether `scene` has GUIs to draw (ROR_NO_GUIS: the image without them, for before / after
/// comparisons).
fn drawn(scene: &Scene) -> bool {
	!scene.guis.is_empty() && std::env::var_os("ROR_NO_GUIS").is_none()
}

/// Where the scene's GUIs draw in a view and which textures they need: SurfaceGuis on their part's
/// face; BillboardGuis sized for this camera and image.
pub(super) fn plan(scene: &Scene, camera: &Camera, request: &ViewRequest) -> Vec<GuiDraw> {
	if !drawn(scene) {
		return Vec::new();
	}
	let supersample = request.supersample.clamp(1, 4);
	let (width, height) = (request.width.max(1), request.height.max(1));
	let right = Vec3::from(camera.world.x_axis).normalize_or(Vec3::X);
	let up = Vec3::from(camera.world.y_axis).normalize_or(Vec3::Y);
	let back = Vec3::from(camera.world.z_axis).normalize_or(Vec3::Z);
	let tan = (camera.fov.to_radians() * 0.5).tan().max(1e-4);
	let mut draws = Vec::new();
	for (index, gui) in scene.guis.iter().enumerate() {
		match &gui.kind {
			GuiKind::Surface(surface) => {
				let normal = surface.normal();
				let center = surface.point(0.5, 0.5);
				// one-sided: nothing shows from behind its face
				if (camera.eye - center).dot(normal) <= 0.0 {
					continue;
				}
				if gui.max_distance > 0.0 && camera.eye.distance(center) > gui.max_distance {
					continue;
				}
				// with GlobalShadows Studio lights a SurfaceGui from its light-grid voxel, which its own
				// part fills: no sun (gui-light: default, every orientation)
				let sun = if scene.environment.global_shadows { 0.0 } else { 1.0 };
				let key = GuiKey::new(gui.content_hash, surface.canvas, SURFACE_SCALE, gui.clips);
				let (across, upward) = (right, up);
				let [left, top, right, bottom] = key.extent();
				// a framed ViewportFrame's picture as tall as the view shows the frame (its sides projected),
				// else as tall as the GUI's image draws it
				let focal = (height * if gui.always_on_top { 1 } else { supersample }) as f32 * 0.5 / tan;
				let project = |point: Vec3| {
					let relative = point - camera.eye;
					let depth = -relative.dot(back);
					(depth > camera.near).then(|| glam::Vec2::new(relative.dot(across) / depth, relative.dot(upward) / depth) * focal)
				};
				let shown = |rect: [f32; 4], pixels: [f32; 4]| match [(rect[0], rect[1]), (rect[2], rect[1]), (rect[0], rect[3]), (rect[2], rect[3])].map(|(u, v)| project(surface.point(u, v))) {
					[Some(a), Some(b), Some(c), Some(d)] => a.distance(c).max(b.distance(d)),
					_ => pixels[3] * SURFACE_SCALE,
				};
				let framed = plan_framed(gui, surface.canvas, &key, &shown);
				draws.push(GuiDraw {
					gui: index,
					key,
					corners: [surface.point(left, top), surface.point(right, top), surface.point(left, bottom), surface.point(right, bottom)],
					normal,
					params: [gui.light_influence, gui.brightness, sun, 0.0],
					center,
					z_offset: surface.z_offset,
					overlay: gui.always_on_top,
					screen: None,
					framed,
				});
			}
			GuiKind::Billboard(billboard) => {
				// the pixels it is drawn in: the render target, or the image for AlwaysOnTop
				let overlay = gui.always_on_top;
				let grid = if overlay { 1 } else { supersample };
				let (grid_width, grid_height) = ((width * grid) as f32, (height * grid) as f32);
				// pixels per stud at one stud of depth
				let focal = grid_height * 0.5 / tan;
				// (in the key's 1/1024 steps, so the canvas at this scale is the whole pixels drawn)
				let gui_scale = (request.gui_scale.filter(|scale| *scale > 0.0).unwrap_or(height as f32 / REFERENCE_HEIGHT) * grid as f32 * 1024.0).floor().max(1.0) / 1024.0;
				// StudsOffset and ExtentsOffset in the camera's axes, z towards it (gui-billboard-basics:
				// studsoffset, extents; probes: StudsOffset z 5 sizes it as 5 studs nearer)
				let offset = billboard.camera_offset;
				let position = billboard.anchor + billboard.world_offset + right * offset.x + up * offset.y + back * offset.z;
				let relative = position - camera.eye;
				// sized by its view depth, not its distance (probes: 10 studs to the side reads as head-on)
				let depth = -relative.dot(back);
				if depth <= camera.near {
					continue;
				}
				if gui.max_distance > 0.0 && relative.length() > gui.max_distance {
					continue;
				}
				// whole device pixels: AbsoluteSize is round(1.5 · (studs · pixels per stud + offset)) / 1.5
				// (probes at depths 10-80 and fields of view 40 and 70)
				let [x_scale, x_offset, y_scale, y_offset] = billboard.size;
				let pixel_width = (x_scale * focal / depth + x_offset * gui_scale).round();
				let pixel_height = (y_scale * focal / depth + y_offset * gui_scale).round();
				if pixel_width < 1.0 || pixel_height < 1.0 {
					continue;
				}
				// centred on the projected anchor, SizeOffset moving it right and up by fractions of its
				// size, the corner on a whole pixel (gui-billboard-basics: sizeoffset, studs-12)
				let x = relative.dot(right) / depth * focal + grid_width * 0.5 + billboard.size_offset[0] * pixel_width;
				let y = -relative.dot(up) / depth * focal + grid_height * 0.5 - billboard.size_offset[1] * pixel_height;
				let key = GuiKey::new(gui.content_hash, [pixel_width / gui_scale, pixel_height / gui_scale], gui_scale, gui.clips);
				// the texture's margins in these pixels (wider than its texels when a big billboard's
				// texture is drawn at a smaller scale)
				let [margin_x, margin_y] = key.margin().map(|margin| margin as f32 * gui_scale / key.scale());
				// a half pixel rounds right and up (gui-billboard-basics: 566.5 -> 567 across, 532.5 -> 532
				// down the image, for stud, pixel and mixed sizes)
				let left = (x - pixel_width * 0.5 + 0.5).floor() - margin_x;
				let top = (y - pixel_height * 0.5 - 0.5).ceil() - margin_y;
				let (right_edge, bottom_edge) = (left + pixel_width + 2.0 * margin_x, top + pixel_height + 2.0 * margin_y);
				let corner = |px: f32, py: f32| camera.eye - back * depth + right * ((px - grid_width * 0.5) * depth / focal) + up * ((grid_height * 0.5 - py) * depth / focal);
				// a framed ViewportFrame's picture as tall as the frame is on the screen
				let framed = plan_framed(gui, key.canvas(), &key, &|rect: [f32; 4], _| (rect[3] - rect[1]) * pixel_height);
				draws.push(GuiDraw {
					gui: index,
					key,
					corners: [corner(left, top), corner(right_edge, top), corner(left, bottom_edge), corner(right_edge, bottom_edge)],
					// lit facing the camera (gui-light: bb-front, bb-behind)
					normal: back,
					params: [gui.light_influence, gui.brightness, 1.0, 0.0],
					center: position,
					z_offset: 0.0,
					overlay,
					screen: None,
					framed,
				});
			}
			GuiKind::Screen(screen) => {
				// the view in GUI pixels (whole 1/1024 steps of the scale), below the top bar when inset;
				// drawn 1:1 into the image's pixels over everything (screen-gui-post: no post effect touches it)
				let gui_scale = (request.gui_scale.filter(|scale| *scale > 0.0).unwrap_or(height as f32 / REFERENCE_HEIGHT) * 1024.0).floor().max(1.0) / 1024.0;
				let top = if screen.inset { (crate::scene::guis::top_bar_inset(scene.play) * gui_scale).min(height as f32) } else { 0.0 };
				if (height as f32 - top) < 1.0 {
					continue;
				}
				let key = GuiKey::new(gui.content_hash, [width as f32 / gui_scale, (height as f32 - top) / gui_scale], gui_scale, true);
				// the canvas as the texture holds it (a canvas past MAX_TEXTURE is drawn smaller and magnified)
				let [canvas_width, canvas_height] = key.canvas().map(|size| size * gui_scale);
				let focal = height as f32 * 0.5 / tan;
				// (any depth past the near plane: the layer has no depth test; ten studs keep the corners'
				// pixels exact for cameras thousands of studs out)
				let depth = (camera.near * 4.0).max(10.0);
				let corner = |px: f32, py: f32| camera.eye - back * depth + right * ((px - width as f32 * 0.5) * depth / focal) + up * ((height as f32 * 0.5 - py) * depth / focal);
				draws.push(GuiDraw {
					gui: index,
					key,
					corners: [corner(0.0, top), corner(canvas_width, top), corner(0.0, top + canvas_height), corner(canvas_width, top + canvas_height)],
					normal: back,
					params: [0.0, 1.0, 0.0, 0.0],
					center: camera.eye - back * depth,
					z_offset: 0.0,
					overlay: true,
					screen: Some(screen.display_order),
					framed: Vec::new(),
				});
			}
		}
	}
	draws
}

/// The GUI draws of a view, uploaded: each one's texture and vertex range.
pub(super) struct ViewGuis {
	buffer: Option<wgpu::Buffer>,
	pub draws: Vec<ViewGui>,
}

pub(super) struct ViewGui {
	texture: Arc<GuiTexture>,
	/// Its window's world as this view sees it (group 2), if it is one.
	window: Option<wgpu::BindGroup>,
	/// Its framed ViewportFrames, each drawn over the GUI's image in turn: what draws over the frame
	/// (group 1; none: nothing) and the frame's picture (group 2).
	layers: Vec<(Option<Arc<GuiTexture>>, wgpu::BindGroup)>,
	vertices: std::ops::Range<u32>,
	pub center: Vec3,
	pub z_offset: f32,
	pub overlay: bool,
	pub screen: Option<i32>,
}

/// A window's world drawn by a view's camera (scene::guis::Viewport): the texture and its opaque share
/// (given back once the view is drawn), and the frame's rect on the canvas and its tint.
pub(super) struct Window {
	pub texture: wgpu::Texture,
	view: wgpu::TextureView,
	pub share: Option<wgpu::Texture>,
	share_view: Option<wgpu::TextureView>,
	rect: [f32; 4],
	tint: [f32; 4],
	/// Its corners' radius in canvas fractions (scene::guis::Viewport::corner).
	corner: [f32; 2],
}

/// A framed ViewportFrame's picture, drawn by its own camera (given back once the view is drawn), and
/// which of its draw's framed ViewportFrames it is.
pub(super) struct Picture {
	pub texture: wgpu::Texture,
	view: wgpu::TextureView,
	framed: usize,
}

/// The worlds a GUI draw shows: its window's, and its framed ViewportFrames' pictures.
#[derive(Default)]
pub(super) struct Worlds {
	pub window: Option<Window>,
	pub pictures: Vec<Picture>,
}

/// GUI images a lookup or a rur call is about: each one's key, its GUI and the content drawn (the GUI's
/// own, or what draws over one of its framed ViewportFrames).
type Members<'a> = Vec<(GuiKey, &'a Gui, &'a Arc<Vec<u8>>)>;

/// The GUI images `draws` of `scene` need ([`GuiDraw::keys`]).
fn wanted<'a>(scene: &'a Scene, draws: &[&GuiDraw]) -> Members<'a> {
	let mut wanted = Vec::new();
	for draw in draws {
		let Some(gui) = scene.guis.get(draw.gui) else {
			continue;
		};
		wanted.push((draw.key, gui, &gui.content));
		for framed in &draw.framed {
			if let (Some(key), Some(ViewportKind::Framed(crate::scene::guis::Framed { over: Some((content, _)), .. }))) = (framed.over, gui.viewports.get(framed.viewport).map(|viewport| &viewport.kind)) {
				wanted.push((key, gui, content));
			}
		}
	}
	wanted
}

/// GUI images kept on disk; past that the least recently used go (a billboard has an image per size,
/// so per camera distance).
const DISK_IMAGES: usize = 8192;
/// How long the images of another rur build are kept after it last drew.
const OLD_BUILDS: Duration = Duration::from_secs(24 * 3600);
/// The disk cache is pruned at most this often: its listing (thousands of files) was taken after
/// every rur call, every frame of an animated GUI.
const PRUNE_EVERY: Duration = Duration::from_secs(60);
/// Other processes' rur work directories (render::rur keeps its own between calls) left this long go.
const OLD_WORK: Duration = Duration::from_secs(3600);

static LAST_PRUNE: Mutex<Option<Instant>> = Mutex::new(None);

/// The cache files this process marked used lately (their modification time, which the pruning goes
/// by), and when: marked again only after [`MARK_EVERY`]. Opening each file a second time to mark it
/// on every read (13 images a frame: two opens, a metadata write and a close each, scanned by the
/// antivirus) took a fifth of a GUI clip's CPU.
static MARKED: Mutex<Option<HashMap<PathBuf, Instant>>> = Mutex::new(None);
const MARK_EVERY: std::time::Duration = std::time::Duration::from_secs(600);

fn mark_used(file: &std::path::Path) {
	{
		let mut marked = MARKED.lock();
		let marked = marked.get_or_insert_with(HashMap::new);
		if marked.get(file).is_some_and(|when| when.elapsed() < MARK_EVERY) {
			return;
		}
		if marked.len() > 65_536 {
			marked.retain(|_, when| when.elapsed() < MARK_EVERY);
		}
		marked.insert(file.to_path_buf(), Instant::now());
	}
	let _ = std::fs::File::options().write(true).open(file).and_then(|handle| handle.set_modified(std::time::SystemTime::now()));
}

/// The folder of `revision`'s GUI images under `guis/`: rur's build and the code that makes the files
/// (render::gui_files, `ROR_GUI_CODE`), so no build reads images another build's code made otherwise.
fn images_folder(revision: &str) -> String {
	format!("{revision}-{}", &env!("ROR_GUI_CODE")[..8])
}

/// A GUI image kept as `name` in `folder` elsewhere: the content store's, else another state dir's that
/// used it, taken whole (a `.gui` file this build reads, all its bytes there) and made this state dir's
/// too at `own` (hard-linked), and the store's.
fn kept_elsewhere(folder: &str, name: &str, own: &std::path::Path) -> Option<GuiImage> {
	let store = crate::output::store_dir().map(|store| (store.join("guis").join(folder).join(name), true));
	let others = crate::output::other_state_dirs().iter().map(|dir| (dir.join("guis").join(folder).join(name), false));
	for (candidate, in_store) in store.into_iter().chain(others) {
		let Ok(bytes) = std::fs::read(&candidate) else { continue };
		let Some(image) = GuiImage::from_file(&bytes) else {
			if in_store {
				let _ = std::fs::remove_file(&candidate);
			}
			continue;
		};
		if ror_assets::store::link(&candidate, own).is_ok() {
			mark_used(own);
			if !in_store {
				publish(folder, own);
			}
		}
		return Some(image);
	}
	None
}

/// Gives the content store the GUI image kept at `file` (a name of the same file).
fn publish(folder: &str, file: &std::path::Path) {
	if let Some(root) = crate::output::store_dir()
		&& let Some(name) = file.file_name()
		&& ror_assets::store::link(file, &root.join("guis").join(folder).join(name)).is_ok()
	{
		ror_assets::store::maybe_prune(&root);
	}
}

fn prune_disk_cache(directory: &std::path::Path) {
	{
		let mut last = LAST_PRUNE.lock();
		if last.is_some_and(|when| when.elapsed() < PRUNE_EVERY) {
			return;
		}
		*last = Some(Instant::now());
	}
	let idle = |entry: &std::fs::DirEntry| entry.metadata().and_then(|metadata| metadata.modified()).ok().and_then(|modified| modified.elapsed().ok());
	// other builds' images (rur's project directory stays)
	if let Some(parent) = directory.parent()
		&& let Ok(entries) = std::fs::read_dir(parent)
	{
		for entry in entries.flatten() {
			let path = entry.path();
			if path == directory || entry.file_name() == "project" || !path.is_dir() {
				continue;
			}
			if idle(&entry).is_some_and(|idle| idle > OLD_BUILDS) {
				let _ = std::fs::remove_dir_all(&path);
			}
		}
	}
	// work directories of processes gone (a running one draws into its own every call)
	let own = format!("{}-", std::process::id());
	if let Ok(entries) = std::fs::read_dir(directory.join("work")) {
		for entry in entries.flatten() {
			if !entry.file_name().to_string_lossy().starts_with(&own) && idle(&entry).is_some_and(|idle| idle > OLD_WORK) {
				let _ = std::fs::remove_dir_all(entry.path());
			}
		}
	}
	let Ok(entries) = std::fs::read_dir(directory) else {
		return;
	};
	let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
		.flatten()
		.filter(|entry| entry.path().extension().is_some_and(|extension| extension == "gui" || extension == "png"))
		.filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
		.collect();
	if files.len() <= DISK_IMAGES {
		return;
	}
	files.sort();
	let excess = files.len() - DISK_IMAGES * 3 / 4;
	for (_, path) in files.into_iter().take(excess) {
		let _ = std::fs::remove_file(path);
	}
}


/// rur's batches for GUI images (frame, key, GUI, content) in frame order, made frame by frame as each
/// frame alone would make them: one scale each, at most BATCH_PIXELS of content, an image into the first
/// of its frame's batches it fits (scale, members, device pixels). A GUI's images (its own and what draws
/// over its framed ViewportFrames, nearly the same content) go into one batch while they fit twice that:
/// a rur call that finds no warm VM waiting runs in a new one, ~1 s, far more than drawing the second
/// image (an emote's portrait, two images of 2.1 M pixels a frame: a 3 s clip made 63 more calls, 14 ->
/// 65 s).
fn frame_batches<'a>(guis: Vec<(usize, GuiKey, &'a Gui, &'a Arc<Vec<u8>>)>) -> Vec<(u32, Members<'a>, u64)> {
	let mut batches: Vec<(u32, Members, u64)> = Vec::new();
	// the frame being batched and where its batches start
	let mut current = (usize::MAX, 0);
	for (frame, key, gui, content) in guis {
		if frame != current.0 {
			current = (frame, batches.len());
		}
		let (width, height) = rur::image_size(key.canvas(), key.scale(), key.margin());
		let pixels = width as u64 * height as u64;
		let fits = |scale: &u32, members: &Members, total: &u64| *scale == key.scale && (*total + pixels <= BATCH_PIXELS || (*total + pixels <= 2 * BATCH_PIXELS && members.iter().all(|(_, other, _)| std::ptr::eq(*other, gui))));
		match batches[current.1..].iter_mut().find(|(scale, members, total)| fits(scale, members, total)) {
			Some((_, members, total)) => {
				members.push((key, gui, content));
				*total += pixels;
			}
			None => batches.push((key.scale, vec![(key, gui, content)], pixels)),
		}
	}
	batches
}

/// The GUI textures a render draws with, by key: resolved ahead of it so that its GPU slot never
/// waits on rur ([`Renderer::resolve_frame_guis`], [`Renderer::upload_frame_guis`]).
pub type GuiTextures = HashMap<GuiKey, Arc<GuiTexture>>;

/// What a frame's GUIs need, found before it draws: the textures the GPU cache already holds (kept
/// here, so the cache may let them go meanwhile) and the images, from disk or rur, to upload when it
/// draws, each marked whether the GPU cache keeps it (an image other frames draw too); and what went
/// wrong finding them.
#[derive(Default)]
pub struct FrameGuis {
	textures: GuiTextures,
	images: Vec<(GuiKey, Arc<GuiImage>, bool)>,
	pub warnings: Vec<String>,
}

impl Renderer {
	/// A GUI's texture, its levels copied by `encoder` from `staged` (its decompressed bytes at `base`):
	/// one a frame gave back when `recycle` (a frame's own image's, given back in turn).
	fn staged_gui(&self, key: &GuiKey, image: &GuiImage, staged: &wgpu::Buffer, base: u64, encoder: &mut wgpu::CommandEncoder, recycle: Option<&Arc<GuiUploads>>) -> Result<GuiTexture> {
		let (width, height) = (image.width, image.height);
		let max_dimension = self.gpu.device.limits().max_texture_dimension_2d;
		if width > max_dimension || height > max_dimension {
			return Err(anyhow::anyhow!("gui: texture size {width}x{height} unsupported"));
		}
		let reused = recycle.and_then(|uploads| uploads.take(width, height, image.levels));
		let raw = match &reused {
			Some((texture, _, _)) => texture.clone(),
			None => self.gpu.device.create_texture(&wgpu::TextureDescriptor {
				label: Some("gui"),
				size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
				mip_level_count: image.levels,
				sample_count: 1,
				dimension: wgpu::TextureDimension::D2,
				format: wgpu::TextureFormat::Rgba8Unorm,
				usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
				view_formats: &[],
			}),
		};
		let (layout, size) = GuiImage::layout(width, height, image.levels);
		for (level, (w, h, row, offset)) in layout.into_iter().enumerate() {
			encoder.copy_buffer_to_texture(
				wgpu::TexelCopyBufferInfo { buffer: staged, layout: wgpu::TexelCopyBufferLayout { offset: base + offset, bytes_per_row: Some(row), rows_per_image: Some(h) } },
				wgpu::TexelCopyTextureInfo { texture: &raw, mip_level: level as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
				wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
			);
		}
		// what its levels hold, rows unpadded
		let bytes = (0..image.levels).map(|level| ((width >> level).max(1) as u64) * ((height >> level).max(1) as u64) * 4).sum::<u64>().min(size);
		let (texture, group) = match reused {
			Some((_, view, group)) => (GpuTexture::whole(raw, view, width, height, [true, false, false], bytes), group),
			None => {
				let view = raw.create_view(&wgpu::TextureViewDescriptor::default());
				let texture = GpuTexture::whole(raw, view, width, height, [true, false, false], bytes);
				let group = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
					label: Some("gui"),
					layout: &self.guis.layout,
					entries: &[
						wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&texture.view()) },
						wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.guis.sampler) },
					],
				});
				(texture, group)
			}
		};
		// the canvas and its margins cover the texture from its top-left corner (whole device pixels
		// rounded up)
		let [margin_x, margin_y] = key.margin().map(|margin| 2.0 * margin as f32);
		let [canvas_width, canvas_height] = key.canvas().map(|size| size * key.scale());
		let uv = [((canvas_width + margin_x) / width as f32).min(1.0), ((canvas_height + margin_y) / height as f32).min(1.0)];
		Ok(GuiTexture { texture, group, uv, recycle: recycle.cloned() })
	}

	/// What the GUIs of `frames` (each frame's GUIs in its draws' order) draw with: the textures the GPU
	/// cache holds, else their images from disk (read on every core), else drawn by rur. rur's batches
	/// are made frame by frame as each frame alone would have made them (one scale each, at most
	/// BATCH_PIXELS; a GUI with the first frame that needs it), so an image comes out of the same atlas
	/// as when every frame asked for its own: rur draws a GUI a little differently by what shares its
	/// atlas (render::rur). The batches of all the frames are drawn RUR_WORKERS at a time, each written
	/// to disk and mip-mapped as it arrives. Problems go into `warnings`.
	fn resolve_guis(&self, frames: &[Members], warnings: &mut Vec<String>) -> (GuiTextures, HashMap<GuiKey, Arc<GuiImage>>) {
		let mut textures = HashMap::new();
		let mut images = HashMap::new();
		let mut seen = std::collections::HashSet::new();
		// (frame, key, GUI, content) of what the GPU cache misses, in frame order
		let mut missing: Vec<(usize, GuiKey, &Gui, &Arc<Vec<u8>>)> = Vec::new();
		for (frame, members) in frames.iter().enumerate() {
			for &(key, gui, content) in members {
				if !seen.insert(key) {
					continue;
				}
				match self.gui_textures.get(&key) {
					Some(texture) => {
						textures.insert(key, texture);
					}
					None => missing.push((frame, key, gui, content)),
				}
			}
		}
		if missing.is_empty() {
			return (textures, images);
		}
		let Some(rur) = rur::locate() else {
			warnings.push(format!("{} GUIs not drawn: rur not found (set ROR_RUR to a roblox-ui-renderer checkout, or put one beside this one)", missing.len()));
			return (textures, images);
		};
		let profile = std::env::var_os("ROR_PROFILE").is_some();
		let started = Instant::now();
		// drawn before, by this or another process, with this rur: on disk (marked used, the cache
		// drops the least recently used); not while rur is between builds (no fingerprint to file
		// them under)
		let revision = rur::revision(&rur);
		let folder = images_folder(&revision);
		let guis_dir = crate::output::state_dir().join("guis");
		let directory = guis_dir.join(&folder);
		let from_disk: Vec<Option<GuiImage>> = missing
			.par_iter()
			.map(|(_, key, _, _)| {
				if revision == UNBUILT {
					return None;
				}
				let file = directory.join(key.file_name());
				if let Some(image) = std::fs::read(&file).ok().and_then(|bytes| GuiImage::from_file(&bytes)) {
					mark_used(&file);
					return Some(image);
				}
				kept_elsewhere(&folder, &key.file_name(), &file)
			})
			.collect();
		let mut left = Vec::new();
		for ((frame, key, gui, content), image) in missing.into_iter().zip(from_disk) {
			match image {
				Some(image) => {
					images.insert(key, Arc::new(image));
				}
				None => left.push((frame, key, gui, content)),
			}
		}
		if profile && !images.is_empty() {
			eprintln!("profile: {} GUI images from disk {:.1} ms", images.len(), started.elapsed().as_secs_f64() * 1000.0);
		}
		// recent failures are not tried again
		{
			let mut failures = self.gui_failures.keys.lock();
			failures.retain(|_, (when, _)| when.elapsed() < FAILURE_TTL);
			left.retain(|(_, key, gui, _)| match failures.get(key) {
				Some((_, reason)) => {
					warnings.push(format!("{}: not drawn ({reason})", gui.path));
					false
				}
				None => true,
			});
		}
		if left.is_empty() {
			return (textures, images);
		}
		let rur_start = Instant::now();
		let batches = frame_batches(left);
		let offline = self.assets.config().offline;
		let calls = batches.len();
		let next = std::sync::atomic::AtomicUsize::new(0);
		let finished: Mutex<Vec<(GuiKey, GuiImage)>> = Mutex::new(Vec::new());
		let problems: Mutex<Vec<String>> = Mutex::new(Vec::new());
		std::thread::scope(|scope| {
			for _ in 0..RUR_WORKERS.min(calls) {
				scope.spawn(|| {
					loop {
						let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
						let Some((scale, members, _)) = batches.get(index) else {
							break;
						};
						let requests: Vec<rur::Request> = members.iter().map(|(key, gui, content)| rur::Request { content: (*content).clone(), canvas: key.canvas(), margin: key.margin(), z_index_behavior: gui.z_index_behavior }).collect();
						match rur::render(&rur, &requests, *scale as f32 / 1024.0, &directory, offline) {
							Ok((pieces, report)) => {
								// rur may have rebuilt itself for these: its images go with its new build's (a folder
								// that may not be there yet)
								let revision = rur::revision(&rur);
								let directory = guis_dir.join(images_folder(&revision));
								let mut found = report.warnings;
								if let Err(error) = std::fs::create_dir_all(&directory) {
									found.push(format!("caching GUI textures in {}: {error}", directory.display()));
								}
								// made ready for the GPU and written to the disk cache on every core
								let done: Vec<(GuiKey, GuiImage, Option<String>)> = members
									.par_iter()
									.zip(pieces.into_par_iter())
									.filter_map(|((key, gui, _), piece)| match GuiImage::from_rgba(piece.width, piece.height, piece.rgba) {
										Ok(image) => {
											let file = directory.join(key.file_name());
											let written = if revision == UNBUILT { Ok(()) } else { crate::output::write_atomic(&file, &image.file_bytes()).inspect(|_| publish(&images_folder(&revision), &file)) };
											let problem = written.err().map(|error| format!("{}: caching its GUI texture: {error:#}", gui.path));
											Some((*key, image, problem))
										}
										Err(error) => {
											problems.lock().push(format!("{}: encoding its GUI texture: {error:#}", gui.path));
											None
										}
									})
									.collect();
								let mut finished = finished.lock();
								for (key, image, problem) in done {
									found.extend(problem);
									finished.push((key, image));
								}
								problems.lock().extend(found);
							}
							Err(error) => {
								let reason = format!("{error:#}");
								let mut failures = self.gui_failures.keys.lock();
								let mut problems = problems.lock();
								for (key, gui, _) in members {
									problems.push(format!("{}: not drawn ({reason})", gui.path));
									failures.insert(*key, (Instant::now(), reason.clone()));
								}
							}
						}
					}
				});
			}
		});
		warnings.extend(problems.into_inner());
		let finished = finished.into_inner();
		let drawn = finished.len();
		for (key, image) in finished {
			images.insert(key, Arc::new(image));
		}
		prune_disk_cache(&guis_dir.join(images_folder(&rur::revision(&rur))));
		if profile {
			eprintln!("profile: {drawn} GUI images from rur in {calls} calls {:.1} ms", rur_start.elapsed().as_secs_f64() * 1000.0);
		}
		(textures, images)
	}

	/// Uploads GUI images, decoded on every core, adding their textures to `textures`: into the GPU
	/// cache those marked so (each once, however many renders ask for it), the rest for the render that
	/// asks alone.
	fn upload_guis(&self, images: Vec<(GuiKey, Arc<GuiImage>, bool)>, textures: &mut GuiTextures, warnings: &mut Vec<String>) {
		let mut wanted = Vec::new();
		for (key, image, cached) in images {
			match self.gui_textures.get(&key).filter(|_| cached) {
				Some(texture) => {
					textures.insert(key, texture);
				}
				None => wanted.push((key, image, cached)),
			}
		}
		if wanted.is_empty() {
			return;
		}
		// one staging buffer for them all, every image's pieces decompressed into their places on every
		// core; then one submission copies them into their textures
		let device = &self.gpu.device;
		let mut offsets = Vec::with_capacity(wanted.len());
		let mut total = 0u64;
		for (_, image, _) in &wanted {
			offsets.push(total);
			total += image.size().next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
		}
		// the next of the two staging buffers kept between frames, mapped again once its last copies are
		// done (made larger when this frame's images need more); another upload using them meanwhile
		// (two clips with GUIs) takes a buffer of its own
		let mut pair = self.gui_uploads.staging.try_lock();
		let staging = match pair.as_deref_mut() {
			Some(pair) => {
				let slot = pair.turn % 2;
				pair.turn += 1;
				pair.used = Some(Instant::now());
				match pair.buffers[slot].take().filter(|(buffer, _)| buffer.size() >= total.max(4)) {
					Some((buffer, copied)) => {
						let (sender, mapped) = std::sync::mpsc::channel();
						buffer.slice(..).map_async(wgpu::MapMode::Write, move |result| {
							let _ = sender.send(result);
						});
						match self.gpu.wait_mapped(copied, &mapped) {
							Ok(Ok(())) => buffer,
							_ => device.create_buffer(&wgpu::BufferDescriptor { label: Some("gui staging"), size: total.max(4).next_power_of_two(), usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: true }),
						}
					}
					None => device.create_buffer(&wgpu::BufferDescriptor { label: Some("gui staging"), size: total.max(4).next_power_of_two(), usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: true }),
				}
			}
			None => device.create_buffer(&wgpu::BufferDescriptor { label: Some("gui staging"), size: total.max(4), usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: true }),
		};
		let unpacked: Vec<Result<()>> = {
			let mut mapped = staging.slice(..total.max(4)).get_mapped_range_mut();
			// every piece's place: the mapped bytes split in order
			let mut places: Vec<(usize, &[u8], &mut [u8])> = Vec::new();
			let mut rest: &mut [u8] = &mut mapped[..];
			let mut at = 0u64;
			for (index, ((_, image, _), offset)) in wanted.iter().zip(&offsets).enumerate() {
				let skip = (offset - at) as usize;
				rest = &mut std::mem::take(&mut rest)[skip..];
				at = *offset;
				let size = image.size() as usize;
				let (mut target, after) = std::mem::take(&mut rest).split_at_mut(size);
				rest = after;
				at += size as u64;
				for (piece, start) in image.pieces() {
					let length = GUI_CHUNK.min(size - start);
					let (place, more) = std::mem::take(&mut target).split_at_mut(length);
					target = more;
					places.push((index, piece, place));
				}
			}
			let failed: Vec<(usize, anyhow::Error)> = places.into_par_iter().filter_map(|(index, piece, place)| GuiImage::unpack(piece, place).err().map(|error| (index, error))).collect();
			let mut results: Vec<Result<()>> = (0..wanted.len()).map(|_| Ok(())).collect();
			for (index, error) in failed {
				results[index] = Err(error);
			}
			results
		};
		staging.unmap();
		let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gui uploads") });
		let mut made = Vec::new();
		for (((key, image, cached), offset), unpacked) in wanted.into_iter().zip(offsets).zip(unpacked) {
			// a frame's own image's texture goes back to the uploads after its frame (the GPU cache's stay)
			let recycle = (!cached).then_some(&self.gui_uploads);
			match unpacked.and_then(|()| self.staged_gui(&key, &image, &staging, offset, &mut encoder, recycle)) {
				Ok(texture) => made.push((key, cached, texture)),
				Err(error) => warnings.push(format!("GUI texture: {error:#}")),
			}
		}
		let copied = self.gpu.submit([encoder.finish()]);
		// the staging buffer back in its place, mapped again once these copies are done
		if let Some(pair) = pair.as_deref_mut() {
			let slot = (pair.turn + 1) % 2;
			pair.buffers[slot] = Some((staging, copied));
		}
		drop(pair);
		// into the GPU cache only once their copies are submitted: another render may draw with what it
		// finds there (one that got there first is kept)
		for (key, cached, texture) in made {
			let texture = if cached { self.gui_textures.get_or_load(&key, || Ok(texture)) } else { Ok(Arc::new(texture)) };
			match texture {
				Ok(texture) => {
					textures.insert(key, texture);
				}
				Err(error) => warnings.push(format!("GUI texture: {error:#}")),
			}
		}
	}

	/// The textures `draws` need: from memory, from disk, else drawn by rur. Problems go into
	/// `warnings`.
	pub(super) fn gui_textures(&self, scene: &Scene, draws: &[&GuiDraw], warnings: &mut Vec<String>) -> GuiTextures {
		let (mut textures, images) = self.resolve_guis(&[wanted(scene, draws)], warnings);
		// a still's are kept: the render after the service's prefetch finds them in the GPU cache
		self.upload_guis(images.into_iter().map(|(key, image)| (key, image, true)).collect(), &mut textures, warnings);
		textures
	}

	/// Loads (or has rur draw) every GUI texture `views` of `scene` need, so the render that follows
	/// finds them cached; the render service does this before taking a GPU slot. Returns the warnings.
	pub fn prefetch_guis(&self, scene: &Scene, views: &[ViewRequest]) -> Vec<String> {
		let mut warnings = Vec::new();
		if !drawn(scene) {
			return warnings;
		}
		let bounds = self.framing_bounds(scene);
		let plans: Vec<Vec<GuiDraw>> = views.iter().map(|view| plan(scene, &view.camera.place(bounds, view.width.max(1), view.height.max(1)), view)).collect();
		let draws: Vec<&GuiDraw> = plans.iter().flatten().collect();
		self.gui_textures(scene, &draws, &mut warnings);
		warnings
	}

	/// The GUIs of many frames, each a scene and the view drawing it, found together: one set of rur
	/// calls for every GUI image they miss (an animated GUI is a new image every frame: 13 telegraphs
	/// cost two rur calls a frame), before any of them takes a GPU slot. One [`FrameGuis`] per frame,
	/// the problems in the first's.
	pub fn resolve_frame_guis(&self, frames: &[(&Scene, &ViewRequest)]) -> Vec<FrameGuis> {
		let plans: Vec<Vec<GuiDraw>> = frames
			.par_iter()
			.map(|(scene, view)| {
				if !drawn(scene) {
					return Vec::new();
				}
				let (width, height) = (view.width.max(1), view.height.max(1));
				// an orbit with its target and distance fixed (a clip's) is placed without the scene's bounds
				let bounds = match view.camera {
					super::camera::CameraSpec::Orbit { distance: Some(_), target: Some(_), .. } | super::camera::CameraSpec::LookAt { .. } | super::camera::CameraSpec::CFrame { .. } => None,
					_ => self.framing_bounds(scene),
				};
				plan(scene, &view.camera.place(bounds, width, height), view)
			})
			.collect();
		let wanted: Vec<Members> = frames.iter().zip(&plans).map(|((scene, _), plan)| wanted(scene, &plan.iter().collect::<Vec<_>>())).collect();
		let mut warnings = Vec::new();
		let (textures, images) = if wanted.iter().all(Vec::is_empty) { Default::default() } else { self.resolve_guis(&wanted, &mut warnings) };
		// the GPU cache keeps an image other frames of these draw too; an animated GUI's image, a
		// frame's own, leaves the GPU once its frame has drawn
		let mut frames_drawing: HashMap<GuiKey, usize> = HashMap::new();
		for plan in &plans {
			let keys: std::collections::HashSet<GuiKey> = plan.iter().flat_map(GuiDraw::keys).collect();
			for key in keys {
				*frames_drawing.entry(key).or_default() += 1;
			}
		}
		let mut resolved: Vec<FrameGuis> = plans
			.iter()
			.map(|plan| {
				let mut frame = FrameGuis::default();
				for key in plan.iter().flat_map(GuiDraw::keys) {
					if let Some(texture) = textures.get(&key) {
						frame.textures.insert(key, texture.clone());
					} else if let Some(image) = images.get(&key)
						&& !frame.images.iter().any(|(kept, _, _)| *kept == key)
					{
						frame.images.push((key, image.clone(), frames_drawing.get(&key).is_some_and(|&frames| frames > 1)));
					}
				}
				frame
			})
			.collect();
		if let Some(first) = resolved.first_mut() {
			first.warnings = warnings;
		}
		resolved
	}

	/// A frame's GUI images uploaded (into the GPU cache, each once), with the textures it already had:
	/// what its render draws with ([`Renderer::submit_video_with`]). Problems go into `warnings`.
	pub fn upload_frame_guis(&self, guis: FrameGuis, warnings: &mut Vec<String>) -> GuiTextures {
		let FrameGuis { mut textures, images, warnings: found } = guis;
		warnings.extend(found);
		self.upload_guis(images, &mut textures, warnings);
		textures
	}

	/// The worlds a view's GUIs show (scene::guis::Viewport), by the draw's index: a window's drawn by
	/// this view's camera, at the view's size, transparent where it draws nothing; each framed
	/// ViewportFrame's by its own camera, at the size the view shows it. Problems go into `warnings`.
	pub(super) fn render_worlds(&self, scene: &Scene, draws: &[GuiDraw], request: &ViewRequest, bounds: Option<(Vec3, Vec3)>, warnings: &mut Vec<String>) -> HashMap<usize, Worlds> {
		let mut worlds: HashMap<usize, Worlds> = HashMap::new();
		for (index, draw) in draws.iter().enumerate() {
			let Some(gui) = scene.guis.get(draw.gui) else {
				continue;
			};
			for (at, framed) in draw.framed.iter().enumerate() {
				let Some(viewport) = gui.viewports.get(framed.viewport) else {
					continue;
				};
				let ViewportKind::Framed(frame) = &viewport.kind else {
					continue;
				};
				let picture = ViewRequest {
					width: framed.size.0,
					height: framed.size.1,
					camera: super::camera::CameraSpec::CFrame { components: frame.camera, fov: frame.fov, near: None },
					background: super::Background::Transparent,
					msaa: request.msaa,
					supersample: 1,
					bloom: false,
					ids: false,
					gui_scale: None,
					linear_scale: None,
					texture: true,
					opaque_share: false,
					// a ViewportFrame's world at full detail (not measured)
					lod: None,
				};
				match self.render(&viewport.scene, std::slice::from_ref(&picture)) {
					Ok((mut images, stats)) => {
						warnings.extend(stats.warnings);
						if let Some(texture) = images.pop().and_then(|image| image.texture) {
							let view = texture.create_view(&Default::default());
							worlds.entry(index).or_default().pictures.push(Picture { texture, view, framed: at });
						}
					}
					Err(error) => warnings.push(format!("{}: a ViewportFrame's world not drawn ({error:#})", gui.path)),
				}
			}
			let windows = gui.viewports.iter().filter(|viewport| viewport.kind == ViewportKind::Window).count();
			let (Some(viewport), GuiKind::Surface(surface)) = (gui.viewports.iter().find(|viewport| viewport.kind == ViewportKind::Window), &gui.kind) else {
				continue;
			};
			if draw.overlay {
				warnings.push(format!("{}: an AlwaysOnTop window (its world not drawn)", gui.path));
				continue;
			}
			if windows > 1 {
				warnings.push(format!("{}: {windows} window ViewportFrames (the first drawn)", gui.path));
			}
			let (width, height) = (request.width.max(1), request.height.max(1));
			let camera = request.camera.place(bounds, width, height);
			let window = ViewRequest {
				width,
				height,
				camera: camera.spec(),
				background: super::Background::Transparent,
				msaa: request.msaa,
				supersample: 1,
				bloom: false,
				ids: false,
				gui_scale: None,
				linear_scale: None,
				texture: true,
				opaque_share: true,
				// a ViewportFrame's world at full detail (not measured)
				lod: None,
			};
			// ROR_WINDOWS_DEBUG=<dir>: each window's world as an image too
			if let Some(directory) = std::env::var_os("ROR_WINDOWS_DEBUG") {
				let readback = ViewRequest { texture: false, ..window.clone() };
				if let Ok((images, _)) = self.render(&viewport.scene, std::slice::from_ref(&readback))
					&& let Some(image) = images.first()
				{
					let name = gui.path.replace(['.', '/', '\\', ' ', '"', '[', ']'], "_");
					let _ = crate::output::write_png(&std::path::Path::new(&directory).join(format!("{name}.png")), image.width, image.height, &image.rgba, true);
				}
			}
			match self.render(&viewport.scene, std::slice::from_ref(&window)) {
				Ok((mut images, stats)) => {
					warnings.extend(stats.warnings);
					if let Some(image) = images.pop()
						&& let Some(texture) = image.texture
					{
						let view = texture.create_view(&Default::default());
						let share_view = image.opaque_share.as_ref().map(|share| share.create_view(&Default::default()));
						worlds.entry(index).or_default().window = Some(Window { texture, view, share: image.opaque_share, share_view, rect: viewport.rect(surface.canvas), tint: viewport.tint, corner: viewport.corner(surface.canvas) });
					}
				}
				Err(error) => warnings.push(format!("{}: its window's world not drawn ({error:#})", gui.path)),
			}
		}
		worlds
	}

	/// A view's GUI quads with their textures (draws without a texture are left out), and the worlds they
	/// show ([`Renderer::render_worlds`], by the draw's index).
	pub(super) fn gui_geometry(&self, scene: &Scene, draws: &[GuiDraw], textures: &HashMap<GuiKey, Arc<GuiTexture>>, worlds: &HashMap<usize, Worlds>, lease: &super::reuse::Lease) -> ViewGuis {
		let mut vertices = Vec::new();
		let mut out = Vec::new();
		for (index, draw) in draws.iter().enumerate() {
			let Some(texture) = textures.get(&draw.key) else {
				continue;
			};
			let [u, v] = texture.uv;
			let uvs = [[0.0, 0.0], [u, 0.0], [0.0, v], [u, v]];
			let vertex = |index: usize| GuiVertex { position: draw.corners[index].into(), uv: uvs[index], normal: draw.normal.into(), params: draw.params };
			let start = vertices.len() as u32;
			vertices.extend_from_slice(&[vertex(0), vertex(2), vertex(1), vertex(1), vertex(2), vertex(3)]);
			// a frame's rect from canvas fractions to the quad's uv (the quad spans the texture's extent,
			// margins included, over 0..u and 0..v)
			let [left, top, right, bottom] = draw.key.extent();
			let to_uv = |[x0, y0, x1, y1]: [f32; 4]| {
				let to_u = |fraction: f32| (fraction - left) / (right - left).max(1e-6) * u;
				let to_v = |fraction: f32| (fraction - top) / (bottom - top).max(1e-6) * v;
				[to_u(x0), to_v(y0), to_u(x1), to_v(y1)]
			};
			let corner_uv = |[across, down]: [f32; 2]| [across / (right - left).max(1e-6) * u, down / (bottom - top).max(1e-6) * v, 0.0, 0.0];
			let world_group = |label: &str, uniform: &WindowUniform, view: &wgpu::TextureView, share: Option<&wgpu::TextureView>| {
				let buffer = lease.buffer("gui window", wgpu::BufferUsages::UNIFORM, bytemuck::bytes_of(uniform));
				self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
					label: Some(label),
					layout: &self.guis.window_layout,
					entries: &[
						wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
						wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.guis.window_sampler) },
						wgpu::BindGroupEntry { binding: 2, resource: buffer.as_entire_binding() },
						wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(share.unwrap_or(&self.guis.blank)) },
					],
				})
			};
			let shown = worlds.get(&index);
			let window = shown.and_then(|worlds| worlds.window.as_ref()).map(|window| {
				let uniform = WindowUniform { rect: to_uv(window.rect), tint: window.tint, corner: corner_uv(window.corner), ..Default::default() };				world_group("gui window", &uniform, &window.view, window.share_view.as_ref())
			});
			// each framed ViewportFrame's picture with what draws over it, in the order the GUI draws them
			let mut layers = Vec::new();
			for picture in shown.map(|worlds| worlds.pictures.as_slice()).unwrap_or_default() {
				let framed = &draw.framed[picture.framed];
				let Some(viewport) = scene.guis.get(draw.gui).and_then(|gui| gui.viewports.get(framed.viewport)) else {
					continue;
				};
				let corner = corner_uv(viewport.corner(draw.key.canvas()));
				let mut uniform = WindowUniform { rect: to_uv(framed.rect), tint: viewport.tint, plane: [framed.plane[0], framed.plane[1], framed.plane[2], 0.0], mode: [1.0, 0.0, 0.0, 0.0], corner, ..Default::default() };
				if let ViewportKind::Framed(crate::scene::guis::Framed { gradient: Some(gradient), .. }) = &viewport.kind {
					uniform.mode = [1.0, gradient.colors.len() as f32, gradient.transparency.len() as f32, 0.0];
					for (slot, (time, color)) in uniform.colors.iter_mut().zip(&gradient.colors) {
						*slot = [*time, color[0], color[1], color[2]];
					}
					for (slot, (time, value)) in uniform.transparency.iter_mut().zip(&gradient.transparency) {
						*slot = [*time, *value, 0.0, 0.0];
					}
				}
				// what draws over it: its own image (a missing one, rur having failed, as nothing)
				layers.push((framed.over.and_then(|key| textures.get(&key)).cloned(), world_group("gui framed", &uniform, &picture.view, None)));
			}
			out.push(ViewGui { texture: texture.clone(), window, layers, vertices: start..vertices.len() as u32, center: draw.center, z_offset: draw.z_offset, overlay: draw.overlay, screen: draw.screen });
		}
		let buffer = (!vertices.is_empty()).then(|| lease.buffer("guis", wgpu::BufferUsages::VERTEX, bytemuck::cast_slice(&vertices)));
		ViewGuis { buffer, draws: out }
	}

	/// Draws one GUI of `view` into the scene pass (the frame bind group is set).
	pub(super) fn draw_gui(&self, pass: &mut wgpu::RenderPass<'_>, view: &ViewGuis, draw: &ViewGui, samples: u32) {
		let Some(buffer) = &view.buffer else {
			return;
		};
		pass.set_pipeline(self.guis.scene(samples));
		pass.set_vertex_buffer(0, buffer.slice(..));
		pass.set_bind_group(1, &draw.texture.group, &[]);
		pass.set_bind_group(2, draw.window.as_ref().unwrap_or(&self.guis.no_window), &[]);
		pass.draw(draw.vertices.clone(), 0..1);
		self.draw_layers(pass, draw);
	}

	/// A GUI draw's framed ViewportFrames, each over what is drawn of it so far (the same quad again).
	fn draw_layers(&self, pass: &mut wgpu::RenderPass<'_>, draw: &ViewGui) {
		for (over, picture) in &draw.layers {
			pass.set_bind_group(1, over.as_ref().map_or(&self.guis.no_texture, |over| &over.group), &[]);
			pass.set_bind_group(2, picture, &[]);
			pass.draw(draw.vertices.clone(), 0..1);
		}
	}

	/// Encodes the AlwaysOnTop GUIs and ScreenGuis of `view` into a layer of their own (premultiplied,
	/// the image's size, [`overlay_order`]) and its copy into a readback buffer: [`composite_overlays`]
	/// lays it over the finished image after the Highlights. `None` without such GUIs.
	#[allow(clippy::too_many_arguments)]
	pub(super) fn encode_gui_overlays(&self, encoder: &mut wgpu::CommandEncoder, prepared: &Prepared, view: &ViewGuis, camera: &Camera, width: u32, height: u32, profile: &super::profile::Profile) -> Option<Readback> {
		let buffer = view.buffer.as_ref()?;
		let mut overlays: Vec<&ViewGui> = view.draws.iter().filter(|draw| draw.overlay).collect();
		if overlays.is_empty() {
			return None;
		}
		overlays.sort_by(|a, b| overlay_order((a.screen, a.center.distance_squared(camera.eye)), (b.screen, b.center.distance_squared(camera.eye))));
		let device = &self.gpu.device;
		let frame = self.frame_uniform(Some(camera), width, height, 0);
		let frame_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("gui overlay frame"), contents: bytemuck::bytes_of(&frame), usage: wgpu::BufferUsages::UNIFORM });
		let frame_group = self.frame_bind_group(prepared, &frame_buffer, &self.no_light_bindings(prepared), &self.defaults.empty_moments, &self.defaults.empty_shadow, &self.defaults.white.view, &self.defaults.black.view);
		let texture = device.create_texture(&wgpu::TextureDescriptor {
			label: Some("gui overlay"),
			size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: OUTPUT_FORMAT,
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
			view_formats: &[],
		});
		{
			let target = texture.create_view(&Default::default());
			let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("gui overlay"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: &target,
					depth_slice: None,
					resolve_target: None,
					ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
				})],
				depth_stencil_attachment: None,
				timestamp_writes: profile.pass("gui overlay"),
				occlusion_query_set: None,
			});
			pass.set_pipeline(&self.guis.overlay);
			pass.set_bind_group(0, &frame_group, &[]);
			pass.set_bind_group(2, &self.guis.no_window, &[]);
			pass.set_vertex_buffer(0, buffer.slice(..));
			for draw in overlays {
				pass.set_bind_group(1, &draw.texture.group, &[]);
				pass.set_bind_group(2, &self.guis.no_window, &[]);
				pass.draw(draw.vertices.clone(), 0..1);
				self.draw_layers(&mut pass, draw);
			}
		}
		let bytes_per_row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
		let readback = Readback {
			buffer: device.create_buffer(&wgpu::BufferDescriptor {
				label: Some("gui overlay"),
				size: bytes_per_row as u64 * height as u64,
				usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
				mapped_at_creation: false,
			}),
			bytes_per_row,
		};
		copy_out(encoder, &texture, &readback, width, height);
		Some(readback)
	}
}

/// The order of the overlay layer's GUIs (each its ScreenGui DisplayOrder, its squared distance from
/// the camera), first drawn first: the world's AlwaysOnTop GUIs far to near, then the ScreenGuis by
/// DisplayOrder, the higher over (screen-gui-probes: DisplayOrder 2 over 1); a tie keeps the scene's order.
fn overlay_order(a: (Option<i32>, f32), b: (Option<i32>, f32)) -> std::cmp::Ordering {
	match (a.0, b.0) {
		(None, None) => b.1.total_cmp(&a.1),
		(None, Some(_)) => std::cmp::Ordering::Less,
		(Some(_), None) => std::cmp::Ordering::Greater,
		(Some(a), Some(b)) => a.cmp(&b),
	}
}

/// Lays the AlwaysOnTop GUIs' and ScreenGuis' layer (premultiplied RGBA8) over the finished
/// straight-alpha image, in display values as the GPU would blend them (screen-gui-probes: black at
/// BackgroundTransparency 0.5 over 255 / 200, 100, 50 / 128 reads 127 / 100, 50, 25 / 64, over the post
/// effects).
pub(super) fn composite_overlays(rgba: &mut [u8], layer: &[u8]) {
	// in pieces on every core (each pixel its own)
	const PIECE: usize = 64 << 10;
	rgba.par_chunks_mut(PIECE).zip(layer.par_chunks(PIECE)).for_each(|(rgba, layer)| composite_overlay_piece(rgba, layer));
}

fn composite_overlay_piece(rgba: &mut [u8], layer: &[u8]) {
	for (pixel, over) in rgba.chunks_exact_mut(4).zip(layer.chunks_exact(4)) {
		if over[3] == 0 {
			continue;
		}
		let coverage = over[3] as f32 / 255.0;
		let below = pixel[3] as f32 / 255.0 * (1.0 - coverage);
		let alpha = coverage + below;
		for channel in 0..3 {
			let premultiplied = over[channel] as f32 / 255.0 + pixel[channel] as f32 / 255.0 * below;
			pixel[channel] = (premultiplied / alpha * 255.0).round().clamp(0.0, 255.0) as u8;
		}
		pixel[3] = (alpha * 255.0).round().clamp(0.0, 255.0) as u8;
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::render::gui_files::gui_image;

	#[test]
	fn gui_images_keep_their_levels_through_files_and_pieces() {
		// odd sides (the float mip path) and several pieces
		let (width, height) = (601u32, 377u32);
		let rgba: Vec<u8> = (0..width * height * 4).map(|index| (index.wrapping_mul(2654435761) >> 13) as u8).collect();
		let expected = gui_image(width, height, rgba.clone());
		let image = GuiImage::from_rgba(width, height, rgba).unwrap();
		assert!(image.ends.len() > 1);
		let read = GuiImage::from_file(&image.file_bytes()).unwrap();
		assert!(GuiImage::from_file(&image.file_bytes()[..30]).is_none(), "a cut file is no image");
		let whole = image.file_bytes();
		assert!(GuiImage::from_file(&whole[..whole.len() - 1]).is_none(), "nor one a byte short");
		assert!(GuiImage::from_file(&[&whole[..], &[0]].concat()).is_none(), "nor one a byte long");
		let mut unpacked = vec![0u8; read.size() as usize];
		let mut rest = &mut unpacked[..];
		for (piece, start) in read.pieces() {
			let length = GUI_CHUNK.min(read.size() as usize - start);
			let (place, more) = rest.split_at_mut(length);
			GuiImage::unpack(piece, place).unwrap();
			rest = more;
		}
		let (layout, _) = GuiImage::layout(width, height, read.levels);
		assert_eq!(layout.len(), expected.levels.len());
		for ((w, h, row, offset), level) in layout.into_iter().zip(&expected.levels) {
			for y in 0..h as usize {
				let start = offset as usize + y * row as usize;
				assert_eq!(&unpacked[start..start + w as usize * 4], &level[y * w as usize * 4..(y + 1) * w as usize * 4], "level {w}x{h} row {y}");
			}
		}
	}

	#[test]
	fn keys_quantize_and_cap() {
		let key = GuiKey::new(7, [800.0, 600.0], 1.5, true);
		assert_eq!(key.canvas(), [800.0, 600.0]);
		assert_eq!(key.scale(), 1.5);
		assert_eq!(key.margin(), [0, 0]);
		assert_eq!(key.extent(), [0.0, 0.0, 1.0, 1.0]);
		// big canvases fit the largest texture, margins included
		for (canvas, clips) in [([4000.0, 4000.0], true), ([3000.0, 500.0], false), ([1400.0, 1400.0], false)] {
			let key = GuiKey::new(7, canvas, 1.5, clips);
			let (width, height) = rur::image_size(key.canvas(), key.scale(), key.margin());
			assert!(width.max(height) as f32 <= MAX_TEXTURE, "{canvas:?}: {width} x {height}");
		}
		assert_ne!(GuiKey::new(7, [100.0, 50.0], 1.5, true), GuiKey::new(7, [100.5, 50.0], 1.5, true));
	}

	#[test]
	fn batches_are_made_frame_by_frame() {
		let gui = |path: &str| Gui {
			path: path.into(),
			kind: GuiKind::Billboard(crate::scene::guis::BillboardGui { anchor: Vec3::ZERO, extents: Vec3::ZERO, size: [0.0; 4], size_offset: [0.0; 2], camera_offset: Vec3::ZERO, world_offset: Vec3::ZERO }),
			content: Arc::new(Vec::new()),
			content_hash: 0,
			light_influence: 0.0,
			brightness: 1.0,
			always_on_top: false,
			max_distance: 0.0,
			clips: true,
			z_index_behavior: 1,
			viewports: Vec::new(),
		};
		let guis: Vec<Gui> = (0..9).map(|index| gui(&format!("Gui{index}"))).collect();
		let content = Arc::new(Vec::new());
		// 1000 x 1000 canvases at 1.5: 2.25 M device pixels, one to a batch of BATCH_PIXELS (4.2 M)
		let big = |content| GuiKey::new(content, [1000.0, 1000.0], 1.5, true);
		let small = |content| GuiKey::new(content, [100.0, 100.0], 1.5, true);
		let other_scale = |content| GuiKey::new(content, [100.0, 100.0], 2.0, true);
		let images = vec![
			(0, small(1), &guis[0], &content),
			(0, other_scale(2), &guis[1], &content),
			(0, small(3), &guis[2], &content),
			(1, small(4), &guis[3], &content),
			(1, big(5), &guis[4], &content),
			(1, big(6), &guis[5], &content),
			(3, small(7), &guis[6], &content),
			// a GUI's own image and what draws over its ViewportFrame: one batch while they fit two
			(4, big(8), &guis[7], &content),
			(4, big(9), &guis[7], &content),
			(4, big(10), &guis[8], &content),
		];
		let batches = frame_batches(images);
		let contents: Vec<Vec<u64>> = batches.iter().map(|(_, members, _)| members.iter().map(|(key, _, _)| key.content).collect()).collect();
		// a frame's GUIs never share a batch with another frame's, however much room is left
		assert_eq!(contents, vec![vec![1, 3], vec![2], vec![4, 5], vec![6], vec![7], vec![8, 9], vec![10]]);
		assert!(batches.iter().all(|(_, members, pixels)| *pixels <= BATCH_PIXELS || (*pixels <= 2 * BATCH_PIXELS && members.iter().all(|(_, gui, _)| std::ptr::eq(*gui, members[0].1)))));
	}

	#[test]
	fn a_framed_viewport_frame_is_planned_at_its_size_on_the_screen_under_what_draws_over_it() {
		// an emote's portrait: a BillboardGui 8 x 6 studs whose ViewportFrame fills 0.9 x 0.8 of it,
		// centred, a label over it
		use crate::render::camera::CameraSpec;
		use crate::scene::guis::{BillboardGui, Framed, Placement, Viewport};

		let billboard = BillboardGui { anchor: Vec3::ZERO, extents: Vec3::ZERO, size: [8.0, 0.0, 6.0, 0.0], size_offset: [0.0; 2], camera_offset: Vec3::ZERO, world_offset: Vec3::ZERO };
		let frame = Placement { size: [0.9, 0.0, 0.8, 0.0], position: [0.5, 0.0, 0.5, 0.0], anchor: [0.5, 0.5] };
		let framed = Framed { camera: [0.0, 0.0, 8.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], fov: 40.0, over: Some((Arc::new(vec![1]), 99)), gradient: None };
		let gui = Gui {
			path: "Billboard".into(),
			kind: GuiKind::Billboard(billboard),
			content: Arc::new(Vec::new()),
			content_hash: 7,
			light_influence: 0.0,
			brightness: 1.0,
			always_on_top: false,
			max_distance: 0.0,
			clips: false,
			z_index_behavior: 1,
			viewports: vec![Viewport { placement: vec![frame], tint: [1.0; 4], scene: Arc::new(Scene::default()), kind: ViewportKind::Framed(framed), corner_radius: None }],
		};
		let scene = Scene { guis: vec![gui], ..Default::default() };
		let request = ViewRequest { width: 1280, height: 720, camera: CameraSpec::LookAt { eye: [0.0, 0.0, -20.0], target: [0.0; 3], fov: 40.0 }, ..Default::default() };
		let camera = request.camera.place(None, request.width, request.height);
		let draws = plan(&scene, &camera, &request);
		let [framed] = draws[0].framed.as_slice() else {
			panic!("one framed ViewportFrame");
		};
		assert!(framed.rect.iter().zip([0.05, 0.1, 0.95, 0.9]).all(|(a, b)| (a - b).abs() < 1e-5), "{:?}", framed.rect);
		// as tall as the frame shows (0.8 of 6 studs 20 studs off at a 40-degree field: 0.8 of 297 pixels),
		// at its own proportions (0.9 x 8 by 0.8 x 6 studs)
		let tall = 0.8 * (6.0 * 720.0 * 0.5 / 20f32.to_radians().tan() / 20.0).round();
		assert_eq!(framed.size, ((tall.round() * 7.2 / 4.8).round() as u32, tall.round() as u32));
		// what draws over it: an image of its own, the GUI's canvas and scale
		assert_eq!(framed.over, Some(GuiKey { content: 99, ..draws[0].key }));
		assert_eq!(draws[0].keys().collect::<Vec<_>>(), [draws[0].key, GuiKey { content: 99, ..draws[0].key }]);
	}

	#[test]
	fn overflow_margins() {
		// a quarter of the canvas and 8 GUI pixels on each side, in whole device pixels
		let key = GuiKey::new(7, [200.0, 100.0], 1.5, false);
		assert_eq!(key.margin(), [87, 50]);
		let [left, top, right, bottom] = key.extent();
		assert!((left + 87.0 / 300.0).abs() < 1e-6 && (top + 50.0 / 150.0).abs() < 1e-6);
		assert!((right - 1.0 - 87.0 / 300.0).abs() < 1e-6 && (bottom - 1.0 - 50.0 / 150.0).abs() < 1e-6);
		assert_eq!(rur::image_size(key.canvas(), key.scale(), key.margin()), (300 + 174, 150 + 100));
	}

	#[test]
	fn big_billboards_keep_their_proportions() {
		// a billboard too big for a texture at the image's scale: its texture is drawn at a smaller
		// scale, and the quad still gives the canvas its whole pixels and the margins their share
		use crate::render::camera::CameraSpec;
		use crate::scene::guis::BillboardGui;

		let billboard = BillboardGui { anchor: Vec3::ZERO, extents: Vec3::ZERO, size: [24.0, 0.0, 12.0, 0.0], size_offset: [0.0; 2], camera_offset: Vec3::ZERO, world_offset: Vec3::ZERO };
		let gui = Gui {
			path: "Billboard".into(),
			kind: GuiKind::Billboard(billboard),
			content: Arc::new(Vec::new()),
			content_hash: 7,
			light_influence: 0.0,
			brightness: 1.0,
			always_on_top: false,
			max_distance: 0.0,
			clips: false,
			z_index_behavior: 1,
			viewports: Vec::new(),
		};
		let scene = Scene { guis: vec![gui], ..Default::default() };
		let request = ViewRequest { width: 1644, height: 1116, camera: CameraSpec::LookAt { eye: [0.0, 0.0, -8.0], target: [0.0; 3], fov: 70.0 }, ..Default::default() };
		let camera = request.camera.place(None, request.width, request.height);
		let draws = plan(&scene, &camera, &request);
		let key = draws[0].key;
		assert!(key.scale() < 1.5, "{}", key.scale());
		let focal = 1116.0 * 0.5 / 35f32.to_radians().tan();
		let quad = (draws[0].corners[1] - draws[0].corners[0]).length() * focal / 8.0;
		let canvas = (24.0 * focal / 8.0).round();
		let texture_canvas = key.canvas()[0] * key.scale();
		let texture = texture_canvas + 2.0 * key.margin()[0] as f32;
		assert!((canvas / quad - texture_canvas / texture).abs() < 1e-3, "{canvas} of {quad} on screen, {texture_canvas} of {texture} in the texture");
	}

	#[test]
	fn screen_guis_cover_the_view_below_the_top_bar() {
		// a fade over the whole view and a HUD below the top bar, in a recording of play (58 GUI pixels)
		// and in Studio's edit viewport (none); drawn 1:1 into the image's pixels, in the overlay layer,
		// at the view's GUI scale (744 GUI pixels tall, as billboards)
		use crate::render::camera::CameraSpec;
		use crate::scene::guis::ScreenGui;

		let screen = |name: &str, display_order: i32, inset: bool| Gui {
			path: format!("PlayerGui.{name}"),
			kind: GuiKind::Screen(ScreenGui { display_order, inset }),
			content: Arc::new(Vec::new()),
			content_hash: display_order as u64,
			light_influence: 0.0,
			brightness: 1.0,
			always_on_top: false,
			max_distance: 0.0,
			clips: false,
			z_index_behavior: 1,
			viewports: Vec::new(),
		};
		let request = ViewRequest { width: 1572, height: 862, supersample: 2, camera: CameraSpec::LookAt { eye: [2040.0, 2050.0, 2000.0], target: [2048.0; 3], fov: 70.0 }, ..Default::default() };
		let camera = request.camera.place(None, request.width, request.height);
		let pixel = |point: Vec3| {
			let clip = camera.view_projection(request.width as f32 / request.height as f32) * point.extend(1.0);
			((clip.x / clip.w + 1.0) * 0.5 * request.width as f32, (1.0 - clip.y / clip.w) * 0.5 * request.height as f32)
		};
		for play in [true, false] {
			let scene = Scene { guis: vec![screen("Transition", 3, false), screen("Hud", 0, true)], play, ..Default::default() };
			let draws = plan(&scene, &camera, &request);
			assert_eq!(draws.len(), 2);
			let scale = draws[0].key.scale();
			assert!((scale - 862.0 / 744.0).abs() < 1e-3 && draws.iter().all(|draw| draw.overlay && draw.key.clips), "the view's scale, whatever the supersampling");
			assert_eq!((draws[0].screen, draws[1].screen), (Some(3), Some(0)));
			let top = if play { 58.0 * scale } else { 0.0 };
			for (draw, top) in [(&draws[0], 0.0), (&draws[1], top)] {
				let [width, height] = draw.key.canvas().map(|size| size * scale);
				assert!((width - 1572.0).abs() < 0.01 && (height - (862.0 - top)).abs() < 0.01, "canvas {width} x {height} at play {play}");
				let (left, upper) = pixel(draw.corners[0]);
				let (right, lower) = pixel(draw.corners[3]);
				assert!(left.abs() < 0.02 && (upper - top).abs() < 0.02 && (right - 1572.0).abs() < 0.02 && (lower - 862.0).abs() < 0.02, "{left},{upper} - {right},{lower}");
			}
		}
		// the world's AlwaysOnTop GUIs far to near, then the ScreenGuis by DisplayOrder (a tie keeps the scene's order)
		let mut layer = vec![(Some(2), 0.0), (None, 4.0), (Some(1), 0.0), (None, 9.0), (Some(1), 1.0)];
		layer.sort_by(|a, b| overlay_order(*a, *b));
		assert_eq!(layer, vec![(None, 9.0), (None, 4.0), (Some(1), 0.0), (Some(1), 1.0), (Some(2), 0.0)]);
	}
}
