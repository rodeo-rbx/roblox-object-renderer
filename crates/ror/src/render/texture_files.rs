//! Textures kept on disk as the GPU takes them (`<state dir>/textures/<code>/<key>.tex`): a downloaded
//! asset's texture (its transcode, or its upload decoded, mip-mapped and block-compressed as ror does
//! it) with every level's rows padded to wgpu's copy alignment. One found there goes up with one read
//! into a staging buffer and one copy per level; made, it went through the asset cache's pack files
//! (a transcode is 3-6 of them), their zstd and KTX2 decoding and a staging allocation per level: a
//! fight recording's 145 particle and beam textures took 290 ms of CPU that way, a map's 67 materials
//! 80 ms of their first frame.
//!
//! Only textures whose source never changes are kept: an asset id's download is cached for good
//! (ror-assets), a transcode's too; Studio's own content, local files and composites are made every
//! time. `<code>` is a hash of the code that makes a texture's texels (build.rs: render::texels, the
//! transcodes' code, ror-image and its locked dependencies), so another build's textures never stand
//! in for this one's, and a rebuild that changed none of it finds them all. The least recently used
//! go past [`budget`]; another code's folder a day after it was last used.
//!
//! The content store (ror_assets::store) keeps them for every state dir: a texture this state dir
//! has not kept is taken from the store's `textures/<code>/` (hard-linked in, its header checked
//! against its length first), or from another state dir that used the store, before it is made; one
//! made here goes to the store too.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
use parking_lot::Mutex;

use super::gpu::Gpu;
use super::resources::{GpuTexture, TextureUse};
use super::texels::{FLAG_ALPHA, FLAG_DXT5NM, FLAG_NORMAL_RG, HEADER, header, layout};

/// The code that makes a texture's texels (build.rs).
const CODE: &str = env!("ROR_TEXTURE_CODE");

/// The disk the textures may take (`ROR_TEXTURE_FILES_MB`); the least recently used go past it.
fn budget() -> u64 {
	std::env::var("ROR_TEXTURE_FILES_MB").ok().and_then(|value| value.parse::<u64>().ok()).unwrap_or(4096) << 20
}
/// The folder is pruned at most this often.
const PRUNE_EVERY: Duration = Duration::from_secs(60);
/// Another code's textures go this long after they were last used.
const OLD_CODES: Duration = Duration::from_secs(24 * 3600);
/// A code's folder marked used (its `.used` file's time) at most this often by a process.
const MARK_USED_EVERY: Duration = Duration::from_secs(3600);

static LAST_PRUNE: Mutex<Option<Instant>> = Mutex::new(None);

/// Whether the texture of `key` (a texture cache key: a URI or a transcode layer's) is made from a
/// downloaded asset, which never changes.
pub(super) fn cacheable(key: &str) -> bool {
	let source = super::transcode::parse_layer_key(key).map_or(key, |(_, asset)| asset);
	!key.contains('#') && matches!(ror_assets::ContentId::parse(source), Ok(ror_assets::ContentId::Asset { .. }))
}

fn directory() -> PathBuf {
	crate::output::state_dir().join("textures").join(CODE)
}

/// The content store's folder of this code's textures.
fn store_directory() -> Option<PathBuf> {
	crate::output::store_dir().map(|store| store.join("textures").join(CODE))
}

/// The file name the texture of `key` drawn as `usage` is kept under (`bc`: the device takes
/// block-compressed formats).
fn name(key: &str, usage: TextureUse, bc: bool) -> String {
	let hash = blake3::hash(format!("{key}\n{usage:?}\n{bc}").as_bytes());
	format!("{}.tex", &hash.to_hex()[..32])
}

/// Keeps the texture made from `image` (as uploaded, `GpuTexture::uploadable`) for `usage` at
/// `path`; false when its levels are not whole (it is then made every time) or it could not be
/// written.
pub(super) fn write(path: &Path, image: &ror_image::Image, usage: TextureUse) -> bool {
	let Some(bytes) = super::texels::file_bytes(image, usage) else {
		return false;
	};
	let written = crate::output::write_atomic(path, &bytes).is_ok();
	prune();
	written
}

/// Gives the content store the texture kept at `path` (a name of the same file).
fn publish(path: &Path) {
	if let Some(store) = store_directory()
		&& let Some(name) = path.file_name()
		&& ror_assets::store::link(path, &store.join(name)).is_ok()
		&& let Some(root) = crate::output::store_dir()
	{
		ror_assets::store::maybe_prune(&root);
	}
}

/// The texture kept as `name` elsewhere, the content store first, then another state dir that used
/// it: a whole file (its header checked against its length) is made this state dir's too, at `own`
/// (hard-linked), and the store's.
fn take_kept(name: &str, own: &Path, label: &str) -> Option<GpuTexture> {
	let store = store_directory().map(|store| (store.join(name), true));
	let others = crate::output::other_state_dirs().iter().map(|dir| (dir.join("textures").join(CODE).join(name), false));
	take_from(store.into_iter().chain(others), own, label)
}

/// [`take_kept`] from `candidates` (a file, and whether it is the store's).
fn take_from(candidates: impl Iterator<Item = (PathBuf, bool)>, own: &Path, label: &str) -> Option<GpuTexture> {
	for (candidate, in_store) in candidates {
		if !candidate.is_file() {
			continue;
		}
		// whole (its header and length agree), and marked used; the store's goes when it is not
		let Some(found) = open(&candidate, label) else {
			if in_store {
				let _ = std::fs::remove_file(&candidate);
			}
			continue;
		};
		if ror_assets::store::link(&candidate, own).is_err() {
			return Some(found);
		}
		if !in_store {
			publish(own);
		}
		return open(own, label).or(Some(found));
	}
	None
}

/// The coarsest level a texture of `format`, `width` x `height` and `levels` levels may be held from:
/// its own sizes halve evenly down to it (so each level after it is the same size, and a view samples
/// its texels at the same coordinates and filter weights, as the full texture's: the shader picks
/// levels by the full size, mesh.wgsl MaterialInfo), in whole 4 x 4 blocks when block-compressed.
pub(super) fn max_base(format: ror_image::PixelFormat, width: u32, height: u32, levels: u32) -> u32 {
	let block = if format.block_bytes().is_some() { 4 } else { 1 };
	let mut base = 0;
	while base + 1 < levels && width.is_multiple_of(block << (base + 1)) && height.is_multiple_of(block << (base + 1)) {
		base += 1;
	}
	base
}

/// A kept texture's file, held open (its levels read from it as views need them, whatever prunes or
/// rewrites the folder meanwhile: Windows keeps a file open for reading readable once removed), and
/// what its header says.
pub(crate) struct Source {
	pub(super) file: Mutex<std::fs::File>,
	format: ror_image::PixelFormat,
	width: u32,
	height: u32,
	levels: u32,
	pub(super) max_base: u32,
	label: String,
}

impl Source {
	/// Its levels from `base` on: their bytes in the file as copied (rows padded) and as held (the
	/// GPU's).
	pub(super) fn span(&self, base: u32) -> (u64, u64) {
		let (levels, size) = layout(self.format, self.width, self.height, self.levels);
		let levels = &levels[base as usize..];
		(size - levels[0].offset, levels.iter().map(|level| level.row * level.rows).sum())
	}

	/// Reads its levels from `base` on into `staging` (as long as their span).
	pub(super) fn read_into(&self, file: &mut std::fs::File, base: u32, staging: &mut [u8]) -> Result<()> {
		use std::io::Seek;
		let (levels, _) = layout(self.format, self.width, self.height, self.levels);
		file.seek(std::io::SeekFrom::Start(HEADER as u64 + levels[base as usize].offset))
			.and_then(|_| file.read_exact(staging))
			.map_err(|error| anyhow::anyhow!("{}: reading its kept levels: {error}", self.label))
	}

	/// A texture of its levels from `base` on, copied by `encoder` from `staging` at `offset` (where
	/// [`Source::read_into`] put them).
	pub(super) fn make(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, staging: &wgpu::Buffer, offset: u64, base: u32) -> super::resources::Held {
		let (levels, _) = layout(self.format, self.width, self.height, self.levels);
		let levels = &levels[base as usize..];
		let start = levels[0].offset;
		let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
			label: Some(&self.label),
			size: wgpu::Extent3d { width: (self.width >> base).max(1), height: (self.height >> base).max(1), depth_or_array_layers: 1 },
			mip_level_count: self.levels - base,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: super::resources::texture_format(self.format),
			usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
			view_formats: &[],
		});
		for (index, level) in levels.iter().enumerate() {
			encoder.copy_buffer_to_texture(
				wgpu::TexelCopyBufferInfo { buffer: staging, layout: wgpu::TexelCopyBufferLayout { offset: offset + level.offset - start, bytes_per_row: Some(level.padded as u32), rows_per_image: Some(level.rows as u32) } },
				wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: index as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
				level.extent,
			);
		}
		let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
		super::resources::Held { texture, view, base, bytes: self.span(base).1 }
	}

	/// Its levels from `base` on, read straight into a staging buffer of their own and copied level
	/// by level into a texture of their own.
	pub(super) fn read(&self, gpu: &Gpu, file: &mut std::fs::File, base: u32, room: &dyn Fn(u64) -> Result<()>) -> Result<super::resources::Held> {
		let (span, bytes) = self.span(base);
		room(bytes)?;
		let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("texture staging"), size: span.max(4), usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: true });
		let read = self.read_into(file, base, &mut staging.slice(..span.max(4)).get_mapped_range_mut()[..span as usize]);
		staging.unmap();
		read?;
		let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("texture upload") });
		let held = self.make(gpu, &mut encoder, &staging, 0, base);
		gpu.submit([encoder.finish()]);
		Ok(held)
	}
}

/// The staging buffers kept texture levels go up through, a batch at a time ([`Staging::read`]): two
/// buffers taking turns, each mapped again once its last batch's copies are done (the files of one
/// read while the other's copies run), not a buffer of each texture's own: 32 MB of staging however
/// many go up (a map's 180 maps in 27 ms, as fast as each its own buffer).
pub(crate) struct Staging {
	buffers: Mutex<[Option<(wgpu::Buffer, super::gpu::Submitted)>; 2]>,
}

/// Its size: the most a batch carries (larger textures go up through buffers of their own).
const STAGING_BYTES: u64 = 16 << 20;

impl Staging {
	pub fn new() -> Self {
		Self { buffers: Mutex::new([None, None]) }
	}

	/// Lets go of the buffers (an idle renderer's).
	pub fn release(&self) -> u64 {
		self.buffers.lock().iter_mut().filter_map(Option::take).map(|(buffer, _)| buffer.size()).sum()
	}

	/// Each of `wanted`'s textures' levels from its base on, read and put on the GPU (those that
	/// still need them: another render may have read them meanwhile). `room` refuses levels the GPU
	/// has no room for.
	pub(super) fn read(&self, gpu: &Gpu, wanted: &[(&std::sync::Arc<GpuTexture>, u32)], room: &dyn Fn(u64) -> Result<()>) -> Result<()> {
		use rayon::prelude::*;
		let align = |bytes: u64| bytes.next_multiple_of(256);
		let mut small = Vec::new();
		for &(texture, base) in wanted {
			let Some(source) = texture.source() else { continue };
			let base = base.min(source.max_base);
			if align(source.span(base).0) > STAGING_BYTES {
				texture.levels_from(gpu, base, room)?;
			} else {
				small.push((texture, source, base));
			}
		}
		// as large as the batches need: a still of a few small textures takes a few MB of staging, not 32
		let size = small.iter().map(|(_, source, base)| align(source.span(*base).0)).sum::<u64>().clamp(4, STAGING_BYTES);
		let mut buffers = self.buffers.lock();
		let mut turn = 0;
		let mut rest = &small[..];
		while !rest.is_empty() {
			// a batch: as many as fit, their files held until their levels are in (Source::file)
			let mut count = 0;
			let mut end = 0;
			while count < rest.len() && end + align(rest[count].1.span(rest[count].2).0) <= STAGING_BYTES {
				end += align(rest[count].1.span(rest[count].2).0);
				count += 1;
			}
			let (batch, next) = rest.split_at(count);
			rest = next;
			let mut files: Vec<_> = batch.iter().map(|(_, source, _)| source.file.lock()).collect();
			let todo: Vec<usize> = (0..batch.len()).filter(|&index| batch[index].0.held().is_none_or(|held| held.base > batch[index].2)).collect();
			if todo.is_empty() {
				continue;
			}
			room(todo.iter().map(|&index| batch[index].1.span(batch[index].2).1).sum())?;
			let slot = turn % 2;
			turn += 1;
			let staging = match buffers[slot].take() {
				Some((staging, copied)) => {
					// mapped again once its last batch's copies are done
					let (sender, mapped) = std::sync::mpsc::channel();
					staging.slice(..).map_async(wgpu::MapMode::Write, move |result| {
						let _ = sender.send(result);
					});
					gpu.wait_mapped(copied, &mapped)?.map_err(|error| anyhow::anyhow!("texture staging: {error}"))?;
					staging
				}
				None => gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("texture staging"), size, usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: true }),
			};
			let mut offsets = Vec::with_capacity(todo.len());
			let mut offset = 0;
			for &index in &todo {
				offsets.push(offset);
				offset += align(batch[index].1.span(batch[index].2).0);
			}
			let read = {
				let mut mapped = staging.slice(..).get_mapped_range_mut();
				let mut slices = Vec::with_capacity(todo.len());
				let mut free: &mut [u8] = &mut mapped;
				for (position, &index) in todo.iter().enumerate() {
					let length = offsets.get(position + 1).copied().unwrap_or(offset) - offsets[position];
					let (slice, after) = std::mem::take(&mut free).split_at_mut(length as usize);
					let span = batch[index].1.span(batch[index].2).0 as usize;
					slices.push((index, &mut slice[..span]));
					free = after;
				}
				let mut jobs: Vec<(&mut std::fs::File, u32, &Source, &mut [u8])> = Vec::with_capacity(todo.len());
				let mut by_index: Vec<Option<&mut std::fs::File>> = files.iter_mut().map(|file| Some(&mut **file)).collect();
				for (index, slice) in slices {
					jobs.push((by_index[index].take().expect("one read per file"), batch[index].2, batch[index].1, slice));
				}
				jobs.into_par_iter().try_for_each(|(file, base, source, slice)| source.read_into(file, base, slice))
			};
			staging.unmap();
			read?;
			let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("texture upload") });
			let made: Vec<super::resources::Held> = todo.iter().zip(&offsets).map(|(&index, &offset)| batch[index].1.make(gpu, &mut encoder, &staging, offset, batch[index].2)).collect();
			let copied = gpu.submit([encoder.finish()]);
			// held once their copies are submitted: a render's submissions after this one find them filled
			for (&index, held) in todo.iter().zip(made) {
				batch[index].0.hold(held);
			}
			drop(files);
			buffers[slot] = Some((staging, copied));
		}
		// let go of once their copies are done: kept, they took 32 MB of a service between its jobs
		*buffers = [None, None];
		Ok(())
	}
}

/// The texture kept at `path`, its levels read once asked for ([`GpuTexture::levels_from`]); `None`
/// when there is none (or it cannot be read).
pub(super) fn open(path: &Path, label: &str) -> Option<GpuTexture> {
	let mut file = std::fs::File::options().read(true).write(true).open(path).ok()?;
	let mut head = [0u8; HEADER];
	file.read_exact(&mut head).ok()?;
	let header = header(&head)?;
	if file.metadata().ok()?.len() != HEADER as u64 + header.size {
		return None;
	}
	// used: the least recently used go first
	let _ = file.set_modified(SystemTime::now());
	let flag = |bit: u32| header.flags & bit != 0;
	let source = Source {
		file: Mutex::new(file),
		format: header.format,
		width: header.width,
		height: header.height,
		levels: header.levels,
		max_base: max_base(header.format, header.width, header.height, header.levels),
		label: label.to_string(),
	};
	Some(GpuTexture::on_disk(source, header.width, header.height, [flag(FLAG_ALPHA), flag(FLAG_DXT5NM), flag(FLAG_NORMAL_RG)], header.bytes))
}

impl super::Renderer {
	/// The texture of a texture cache key: a composite's, a transcode layer's (an image asset's, else
	/// its upload) or an image's with ror's own mips; one made from a downloaded asset is kept on disk
	/// as uploaded, found there the next time, and goes to the GPU from there once asked for
	/// ([`GpuTexture::levels_from`]).
	pub(super) fn make_texture(&self, uri: &str, usage: TextureUse) -> Result<GpuTexture> {
		if uri.starts_with(super::composite::KEY_PREFIX) {
			return self.composite_texture(uri);
		}
		let kept = cacheable(uri).then(|| name(uri, usage, self.gpu.supports_bc()));
		if let Some(name) = kept.as_deref().filter(|_| !self.assets.config().refresh) {
			let own = directory().join(name);
			if let Some(texture) = open(&own, uri).or_else(|| take_kept(name, &own, uri)) {
				mark_used();
				return Ok(texture);
			}
		}
		let image = self.texture_image(uri, usage)?;
		if let Some(name) = &kept {
			let own = directory().join(name);
			if write(&own, &GpuTexture::uploadable(&self.gpu, &image), usage)
				&& let Some(texture) = open(&own, uri)
			{
				publish(&own);
				return Ok(texture);
			}
		}
		self.gpu_room(image.levels.iter().map(|level| level.len() as u64).sum())?;
		GpuTexture::upload(&self.gpu, uri, &image, usage)
	}
}

/// Marks this code's folder used (its `.used` file's time, which another code's prune goes by: a
/// folder read from, not written to, keeps its own time), at most every [`MARK_USED_EVERY`].
fn mark_used() {
	static LAST: Mutex<Option<Instant>> = Mutex::new(None);
	let mut last = LAST.lock();
	if last.is_some_and(|when| when.elapsed() < MARK_USED_EVERY) {
		return;
	}
	*last = Some(Instant::now());
	let folder = directory();
	let _ = std::fs::create_dir_all(&folder);
	let _ = std::fs::write(folder.join(".used"), b"");
}

/// Keeps the folder within [`budget`] (the least recently used go) and removes other codes' folders
/// unused for [`OLD_CODES`], at most every [`PRUNE_EVERY`].
fn prune() {
	{
		let mut last = LAST_PRUNE.lock();
		if last.is_some_and(|when| when.elapsed() < PRUNE_EVERY) {
			return;
		}
		*last = Some(Instant::now());
	}
	let directory = directory();
	let modified = |path: &Path| std::fs::metadata(path).and_then(|metadata| metadata.modified()).ok();
	if let Some(parent) = directory.parent()
		&& let Ok(entries) = std::fs::read_dir(parent)
	{
		for entry in entries.flatten() {
			let path = entry.path();
			// last used: written to (the folder's time) or read from (its `.used` file's, mark_used)
			let used = modified(&path).max(modified(&path.join(".used")));
			if path != directory && path.is_dir() && used.and_then(|time| time.elapsed().ok()).is_some_and(|idle| idle > OLD_CODES) {
				let _ = std::fs::remove_dir_all(&path);
			}
		}
	}
	let Ok(entries) = std::fs::read_dir(&directory) else {
		return;
	};
	let mut files: Vec<(SystemTime, u64, PathBuf)> = entries
		.flatten()
		.filter(|entry| entry.path().extension().is_some_and(|extension| extension == "tex"))
		.filter_map(|entry| {
			let metadata = entry.metadata().ok()?;
			Some((metadata.modified().ok()?, metadata.len(), entry.path()))
		})
		.collect();
	let mut total: u64 = files.iter().map(|(_, size, _)| size).sum();
	let budget = budget();
	if total <= budget {
		return;
	}
	files.sort();
	// down to three quarters, so a full folder is not pruned again at once
	for (_, size, path) in files {
		if total <= budget / 4 * 3 {
			break;
		}
		if std::fs::remove_file(&path).is_ok() {
			total -= size;
		}
	}
}

/// A copy of what [`write`] writes, for its test: the file's bytes.
#[cfg(test)]
fn file_bytes(path: &Path) -> Vec<u8> {
	let mut bytes = Vec::new();
	std::fs::File::open(path).unwrap().read_to_end(&mut bytes).unwrap();
	bytes
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn only_downloaded_assets_are_kept() {
		assert!(cacheable("rbxassetid://123"));
		assert!(cacheable("http://www.roblox.com/asset/?id=12224218"));
		assert!(cacheable(&super::super::transcode::layer_key(ror_assets::Transcode::PackColor, "rbxassetid://5")));
		assert!(!cacheable("rbxasset://textures/face.png"));
		assert!(!cacheable("C:/models/skin.png"));
		assert!(!cacheable("rbxassetid://123#lod=1"));
	}

	#[test]
	fn a_kept_texture_reads_back_as_written() {
		let gpu = Gpu::new().expect("a GPU (or the software adapter)");
		let directory = std::env::temp_dir().join(format!("ror-texture-files-{}", std::process::id()));
		std::fs::create_dir_all(&directory).unwrap();
		let path = directory.join("a.tex");
		let rgba: Vec<u8> = (0..37 * 21 * 4).map(|index| (index * 7 % 251) as u8).collect();
		let mut image = ror_image::Image::from_rgba8(37, 21, rgba);
		ror_image::generate_mips(&mut image);
		let texture = GpuTexture::upload(&gpu, "test", &image, TextureUse::Data).unwrap();
		assert!(write(&path, &image, TextureUse::Data));
		let bytes = file_bytes(&path);
		let read = header(bytes[..HEADER].try_into().unwrap()).unwrap();
		assert_eq!((read.width, read.height, read.levels as usize, read.bytes), (37, 21, image.levels.len(), texture.bytes()));
		let (levels, _) = layout(read.format, read.width, read.height, read.levels);
		for (level, data) in levels.iter().zip(&image.levels) {
			for row in 0..level.rows {
				let at = HEADER + (level.offset + row * level.padded) as usize;
				assert_eq!(&bytes[at..at + level.row as usize], &data[(row * level.row) as usize..((row + 1) * level.row) as usize]);
			}
		}
		let loaded = open(&path, "test").unwrap();
		// nothing on the GPU until asked for; 37 x 21 halves evenly nowhere
		assert!(loaded.held().is_none());
		assert_eq!((loaded.width, loaded.height, loaded.full_bytes, loaded.has_alpha, loaded.max_base()), (37, 21, texture.bytes(), texture.has_alpha, 0));
		let held = loaded.levels_from(&gpu, 3, &|_| Ok(())).unwrap();
		assert_eq!((held.base, held.bytes, held.texture.mip_level_count()), (0, texture.bytes(), image.levels.len() as u32));
		let _ = std::fs::remove_dir_all(&directory);
	}

	#[test]
	fn a_kept_texture_goes_up_from_the_level_asked_for() {
		let gpu = Gpu::new().expect("a GPU (or the software adapter)");
		let directory = std::env::temp_dir().join(format!("ror-texture-levels-{}", std::process::id()));
		std::fs::create_dir_all(&directory).unwrap();
		let path = directory.join("b.tex");
		// 96 x 64 halves evenly 5 times (3 x 2), its RGBA8 levels down to 1 x 1
		let rgba: Vec<u8> = (0..96 * 64 * 4).map(|index| (index * 13 % 253) as u8).collect();
		let mut image = ror_image::Image::from_rgba8(96, 64, rgba);
		ror_image::generate_mips(&mut image);
		assert!(write(&path, &image, TextureUse::Data));
		let loaded = open(&path, "test").unwrap();
		assert_eq!(loaded.max_base(), 5);
		let level_bytes = |level: usize| image.levels[level].len() as u64;
		let held = loaded.levels_from(&gpu, 2, &|_| Ok(())).unwrap();
		assert_eq!((held.base, held.texture.width(), held.texture.height(), held.texture.mip_level_count()), (2, 24, 16, image.levels.len() as u32 - 2));
		assert_eq!(held.bytes, (2..image.levels.len()).map(level_bytes).sum::<u64>());
		// what it holds serves any coarser level; a finer one is read
		assert_eq!(loaded.levels_from(&gpu, 4, &|_| Ok(())).unwrap().base, 2);
		assert_eq!(loaded.levels_from(&gpu, 1, &|_| Ok(())).unwrap().base, 1);
		assert_eq!(loaded.bytes(), (1..image.levels.len()).map(level_bytes).sum::<u64>());
		// past its coarsest allowed level, from that one
		let again = open(&path, "test").unwrap();
		assert_eq!(again.levels_from(&gpu, 9, &|_| Ok(())).unwrap().base, 5);
		// block-compressed: whole 4 x 4 blocks at the base
		assert_eq!(max_base(ror_image::PixelFormat::Bc1, 1024, 512, 11), 7);
		assert_eq!(max_base(ror_image::PixelFormat::Bc1, 1000, 512, 10), 1);
		assert_eq!(max_base(ror_image::PixelFormat::Rgba8, 1024, 1024, 1), 0);
		let _ = std::fs::remove_dir_all(&directory);
	}

	#[test]
	fn a_texture_kept_elsewhere_is_taken_whole_or_not_at_all() {
		let directory = std::env::temp_dir().join(format!("ror-texture-take-{}", std::process::id()));
		let (store, other, own) = (directory.join("store"), directory.join("other"), directory.join("own"));
		for dir in [&store, &other, &own] {
			std::fs::create_dir_all(dir).unwrap();
		}
		let rgba: Vec<u8> = (0..16 * 8 * 4).map(|index| (index * 5 % 251) as u8).collect();
		let mut image = ror_image::Image::from_rgba8(16, 8, rgba);
		ror_image::generate_mips(&mut image);
		assert!(write(&other.join("a.tex"), &image, TextureUse::Data));
		// a store file cut short is not taken, and goes; another state dir's whole one is
		let bytes = file_bytes(&other.join("a.tex"));
		std::fs::write(store.join("a.tex"), &bytes[..bytes.len() - 1]).unwrap();
		let candidates = || vec![(store.join("a.tex"), true), (other.join("a.tex"), false)].into_iter();
		let taken = take_from(candidates(), &own.join("a.tex"), "test").expect("the other state dir's");
		assert_eq!((taken.width, taken.height), (16, 8));
		assert!(!store.join("a.tex").exists());
		assert_eq!(file_bytes(&own.join("a.tex")), bytes, "a name of the same file");
		assert!(take_from(vec![(store.join("b.tex"), true)].into_iter(), &own.join("b.tex"), "test").is_none());
		let _ = std::fs::remove_dir_all(&directory);
	}
}
