//! Roblox's light grid: OutdoorAmbient and the sky's light occluded by parts in 4-stud voxels
//! (notes/render.md "Light grid"; experiments lightgrid, lg-*).
//!
//! Each voxel holds how much sky light enters it from above: the mean over its eight horizontal
//! neighbours one layer up (not the one straight above) of the light entering them times their
//! transmission, 1 above everything and outside the grid; it stores 1 - (1 - L)^4. A part fills a
//! voxel by the share of the voxel's horizontal cross-section it covers inside the voxel's layer
//! (any thickness; an upright sheet across its thickness too) for blocks, by volume for balls,
//! cylinders and wedges, times 1 - Transparency;
//! parts side by side add up (their union, kept per stud of the cross-section). MeshParts, FileMesh
//! SpecialMeshes, unions and parts without CastShadow do not count. Surfaces read it trilinearly between voxel centres 6
//! studs along their normal (shaders/mesh.wgsl).

use glam::{Mat3A, Vec3, Vec3A};
use rayon::prelude::*;

use crate::scene::extract::mesh_type;
use crate::scene::{Geometry, Part, Scene};

/// The voxel's side in studs.
pub const VOXEL: f32 = 4.0;
/// The grid's largest extent in voxels, horizontally and vertically (larger scenes keep the voxels
/// around their middle and the top; surfaces outside read the nearest edge).
const MAX_XZ: usize = 384;
const MAX_Y: usize = 96;
/// A voxel's cross-section in cells of a stud: where parts share a voxel, their coverage combines
/// per cell (lg-occupancy "slats": bars 1 stud wide every 2 studs fill half of each voxel, as one
/// part covering half would).
const CELLS: usize = 4;
/// The volume shares of balls, cylinders and wedges sample each cell this many times finer along
/// each axis.
const SPLIT: usize = 2;

pub struct LightGrid {
	/// World position of the grid's minimum corner (a multiple of 4).
	pub origin: Vec3,
	/// Voxels along X, Y, Z.
	pub dims: [usize; 3],
	/// Visibility 0..255, X fastest, then Y, then Z (a 3D texture's layout).
	pub values: Vec<u8>,
}

/// How a part fills voxels.
#[derive(Clone, Copy)]
enum Fill {
	/// Horizontal coverage of the part's box inside the voxel's layer.
	Box,
	/// Volume share inside the shape, times a factor fitted per shape (lg-shapes).
	Volume(Shape, f32),
}

#[derive(Clone, Copy)]
enum Shape {
	Ball,
	/// A SpecialMesh sphere: the ellipsoid in the part's box.
	Ellipsoid,
	Cylinder,
	Wedge,
	CornerWedge,
}

fn fill(part: &Part) -> Option<Fill> {
	if !part.cast_shadow || part.transparency >= 1.0 {
		return None;
	}
	// A part whose CFrame or size is not a number (a script's look-at along nothing leaves one) fills nothing: its box
	// would make the grid's bounds NaN.
	if !part.cframe.is_finite() || !part.size.is_finite() {
		return None;
	}
	// lg-shapes: a 20-stud ball, a cylinder 12 across and a 24 x 12 x 24 wedge darken the floor
	// under them as 0.79, 0.93 and 1 of their volume share would (the wedge's fit 0.98 puts 18 % fewer
	// of its pixels within 2 of 255; corner wedges taken as wedges); a block turned 45 degrees by its
	// horizontal coverage; a head MeshPart 24 x 4 x 24 not at all (nor do a character's MeshParts darken
	// a character: ao-model)
	match &part.geometry {
		// lg-shapes: a 24 x 4 x 24 block drawn as the head FileMesh leaves the floor open, drawn as a
		// SpecialMesh sphere it closes it as 0.89 of the ellipsoid's volume share would; other
		// SpecialMesh types are taken as their box (not measured)
		Geometry::Special { mesh_type: mesh_type::FILE_MESH, .. } => None,
		// (the sphere's box is the mesh's Scale and Offset on the part: those must be numbers too)
		Geometry::Special { mesh_type: mesh_type::SPHERE, scale, offset, .. } if !scale.is_finite() || !offset.is_finite() => None,
		Geometry::Special { mesh_type: mesh_type::SPHERE, .. } => Some(Fill::Volume(Shape::Ellipsoid, 0.89)),
		Geometry::Block | Geometry::Truss { .. } | Geometry::Special { .. } => Some(Fill::Box),
		Geometry::Ball => Some(Fill::Volume(Shape::Ball, 0.79)),
		Geometry::Cylinder => Some(Fill::Volume(Shape::Cylinder, 0.93)),
		Geometry::Wedge => Some(Fill::Volume(Shape::Wedge, 1.0)),
		Geometry::CornerWedge => Some(Fill::Volume(Shape::CornerWedge, 1.0)),
		Geometry::MeshPart { .. } | Geometry::Union { .. } | Geometry::Collision { .. } => None,
	}
}

/// Inside a shape, in its normalized local frame (each coordinate in -1..1 spans the part).
fn inside(shape: Shape, u: Vec3A, size: Vec3) -> bool {
	if u.abs().max_element() > 1.0 {
		return false;
	}
	match shape {
		Shape::Ball => {
			let radius = size.min_element() * 0.5;
			(u * Vec3A::from(size) * 0.5).length_squared() <= radius * radius
		}
		Shape::Ellipsoid => u.length_squared() <= 1.0,
		Shape::Cylinder => {
			let radius = size.y.min(size.z) * 0.5;
			let y = u.y * size.y * 0.5;
			let z = u.z * size.z * 0.5;
			y * y + z * z <= radius * radius
		}
		// full height at +Z, the slope down to the front (-Z) edge
		Shape::Wedge => u.y <= u.z,
		// the apex above the +X, -Z corner
		Shape::CornerWedge => u.y <= u.x && u.y <= -u.z,
	}
}

/// A part as the grid sees it.
struct Filler {
	fill: Fill,
	opacity: f32,
	size: Vec3,
	center: Vec3A,
	half: Vec3A,
	/// World to the part's frame.
	inverse: Mat3A,
	/// Half the world box's extent.
	extent: Vec3A,
	/// A box whose axes lie along the world's: its world box is exact.
	aligned: bool,
}

impl Filler {
	fn new(part: &Part, fill: Fill) -> Self {
		let rotation = part.cframe.matrix3;
		// a SpecialMesh sphere is the mesh's box: the part's size times Scale, moved by Offset
		let (size, center) = match &part.geometry {
			Geometry::Special { mesh_type: mesh_type::SPHERE, scale, offset, .. } => (part.size * *scale, part.cframe.translation + rotation * Vec3A::from(*offset)),
			_ => (part.size, part.cframe.translation),
		};
		let half = Vec3A::from(size * 0.5);
		Filler {
			fill,
			opacity: 1.0 - part.transparency.clamp(0.0, 1.0),
			size,
			center,
			half,
			inverse: rotation.transpose(),
			extent: rotation.abs() * half,
			aligned: matches!(fill, Fill::Box) && (0..3).all(|column| rotation.col(column).abs().max_element() > 0.9999),
		}
	}

	/// The share of each stud cell of the voxel at `corner` (its minimum) the part fills, cell
	/// (i, k) at i + CELLS k.
	fn cells(&self, corner: Vec3A) -> [f32; CELLS * CELLS] {
		let mut cells = [0.0f32; CELLS * CELLS];
		let cell = VOXEL / CELLS as f32;
		match self.fill {
			Fill::Box if self.aligned => {
				let (mut lo, mut hi) = (self.center - self.extent, self.center + self.extent);
				// any thickness inside the layer counts in full (lg-occupancy t0.2-t4)
				if hi.y.min(corner.y + VOXEL) - lo.y.max(corner.y) <= 1e-3 {
					return cells;
				}
				// an upright sheet (thinner than a voxel across x or z, at least a voxel long along the
				// other, through the layer's whole height) fills the voxels it passes through across
				// its thickness, as any thickness does along y (lg-specular: 6 studs in front of an 80 x
				// 40 x 0.5-stud wall OutdoorAmbient and the sky specular fall to 0.87 of the open 40
				// studs down, as an opaque wall's voxels give; its 0.125 share left them open; lg-walls'
				// walls 2 studs long, lg-occupancy's slats 1 stud tall and lg-edge's roof edges keep their
				// share)
				if hi.y >= corner.y + VOXEL - 1e-3 && lo.y <= corner.y + 1e-3 {
					for (axis, other) in [(0, 2), (2, 0)] {
						let inside = hi[axis].min(corner[axis] + VOXEL) - lo[axis].max(corner[axis]) > 1e-3;
						if hi[other] - lo[other] < VOXEL || hi[axis] - lo[axis] >= VOXEL || !inside {
							continue;
						}
						lo[axis] = corner[axis];
						hi[axis] = corner[axis] + VOXEL;
					}
				}
				let overlap = |lo: f32, hi: f32, start: f32| (hi.min(start + cell) - lo.max(start)).clamp(0.0, cell) / cell;
				for k in 0..CELLS {
					let z = overlap(lo.z, hi.z, corner.z + k as f32 * cell);
					if z == 0.0 {
						continue;
					}
					for i in 0..CELLS {
						cells[i + CELLS * k] = overlap(lo.x, hi.x, corner.x + i as f32 * cell) * z;
					}
				}
			}
			Fill::Box => {
				// a vertical line through each cell's centre, across the layer, against the turned box
				let up = self.inverse * Vec3A::Y;
				for k in 0..CELLS {
					for i in 0..CELLS {
						let start = self.inverse * (corner + Vec3A::new((i as f32 + 0.5) * cell, 0.0, (k as f32 + 0.5) * cell) - self.center);
						let (mut t0, mut t1) = (0.0f32, VOXEL);
						for axis in 0..3 {
							if up[axis].abs() < 1e-6 {
								if start[axis].abs() > self.half[axis] {
									t1 = -1.0;
								}
							} else {
								let a = (-self.half[axis] - start[axis]) / up[axis];
								let b = (self.half[axis] - start[axis]) / up[axis];
								t0 = t0.max(a.min(b));
								t1 = t1.min(a.max(b));
							}
						}
						if t1 > t0 {
							cells[i + CELLS * k] = 1.0;
						}
					}
				}
			}
			Fill::Volume(shape, factor) => {
				// points half a stud apart through each cell (the factors' fit samples so)
				let step = VOXEL / (CELLS * SPLIT) as f32;
				for k in 0..CELLS {
					for i in 0..CELLS {
						let mut count = 0;
						for b in 0..SPLIT {
							for a in 0..SPLIT {
								for j in 0..CELLS * SPLIT {
									let offset = Vec3A::new((i * SPLIT + a) as f32 + 0.5, j as f32 + 0.5, (k * SPLIT + b) as f32 + 0.5) * step;
									if inside(shape, self.inverse * (corner + offset - self.center) / self.half, self.size) {
										count += 1;
									}
								}
							}
						}
						cells[i + CELLS * k] = count as f32 / (SPLIT * SPLIT * CELLS * SPLIT) as f32 * factor;
					}
				}
			}
		}
		cells
	}
}

/// Everything a filler's voxels read of its part: its fill, frame, size, transparency and a
/// SpecialMesh sphere's Scale and Offset.
fn encode(part: &Part, fill: Fill, key: &mut Vec<u8>) {
	let floats = |key: &mut Vec<u8>, values: &[f32]| key.extend_from_slice(bytemuck::cast_slice(values));
	let (tag, factor) = match fill {
		Fill::Box => (0u8, 0.0),
		Fill::Volume(shape, factor) => (1 + shape as u8, factor),
	};
	key.push(tag);
	floats(key, &[factor, part.transparency]);
	floats(key, &part.size.to_array());
	floats(key, &part.cframe.to_cols_array());
	if let Geometry::Special { mesh_type: mesh_type::SPHERE, scale, offset, .. } = &part.geometry {
		floats(key, &scale.to_array());
		floats(key, &offset.to_array());
	}
}

/// Everything [`build`] reads, in order: the parts that fill the grid ([`encode`]). Equal keys
/// build equal grids.
pub fn key(scene: &Scene) -> Vec<u8> {
	let mut key = Vec::new();
	for part in &scene.parts {
		if let Some(fill) = fill(part) {
			encode(part, fill, &mut key);
		}
	}
	key
}

/// How many parts fill the grid.
pub fn filler_count(scene: &Scene) -> usize {
	scene.parts.iter().filter(|part| fill(part).is_some()).count()
}

pub fn build(scene: &Scene) -> Option<LightGrid> {
	build_from(scene, None, false).map(|state| LightGrid { origin: state.origin, dims: state.dims, values: state.values })
}

/// What a build keeps for the next one's update: each filler's key and voxels, the voxels'
/// transmissions and the light entering every layer.
struct Detail {
	keys: Vec<u8>,
	ranges: Vec<std::ops::Range<usize>>,
	filled: Vec<Vec<(u32, f32)>>,
	transmission: Vec<f32>,
	entering: Vec<f32>,
}

/// A built grid, with what its next update needs when it was asked to keep that.
pub struct GridState {
	pub origin: Vec3,
	pub dims: [usize; 3],
	/// Its visibility per voxel, as [`LightGrid::values`]
	pub values: Vec<u8>,
	fillers: usize,
	detail: Option<Detail>,
}

impl GridState {
	pub fn fillers(&self) -> usize {
		self.fillers
	}

	pub fn kept(&self) -> bool {
		self.detail.is_some()
	}

	/// Lets go of what only an update reads.
	pub fn forget(&mut self) {
		self.detail = None;
		self.values = Vec::new();
	}

	pub fn bytes(&self) -> u64 {
		let detail = self.detail.as_ref().map_or(0, |detail| {
			detail.keys.len() + detail.ranges.len() * 16 + detail.filled.iter().map(|voxels| voxels.len() * 8 + 24).sum::<usize>() + (detail.transmission.len() + detail.entering.len()) * 4
		});
		(self.values.len() + detail) as u64
	}
}

/// The scene's grid, exactly as [`build`] makes it. From `previous` (a state kept from a scene with
/// as many fillers in the same grid) only the voxels the changed fillers filled or fill now are
/// combined again, and the light below them propagated again; otherwise it is built anew. `keep`
/// keeps what the next update needs.
pub fn build_from(scene: &Scene, previous: Option<GridState>, keep: bool) -> Option<GridState> {
	let mut keys = Vec::new();
	let mut ranges = Vec::new();
	let mut fillers = Vec::new();
	for part in &scene.parts {
		if let Some(fill) = fill(part) {
			let start = keys.len();
			encode(part, fill, &mut keys);
			ranges.push(start..keys.len());
			fillers.push(Filler::new(part, fill));
		}
	}
	if fillers.is_empty() {
		return None;
	}
	let (min, dims) = frame(&fillers);
	if let Some(GridState { origin, dims: kept_dims, values, detail: Some(detail), .. }) = previous
		&& origin == min
		&& kept_dims == dims
		&& detail.ranges.len() == ranges.len()
		&& let Some(state) = update(&fillers, (&keys, &ranges), (min, dims), values, detail, keep)
	{
		return Some(state);
	}
	Some(full(&fillers, (keys, ranges), (min, dims), keep))
}

/// The grid's minimum corner and its voxels along X, Y and Z around `fillers`.
fn frame(fillers: &[Filler]) -> (Vec3, [usize; 3]) {
	let mut lo = Vec3A::splat(f32::MAX);
	let mut hi = Vec3A::splat(f32::MIN);
	for filler in fillers {
		lo = lo.min(filler.center - filler.extent);
		hi = hi.max(filler.center + filler.extent);
	}
	// two open voxels around, one above (the light entering it is the open sky's)
	let mut min = ((Vec3::from(lo) - Vec3::new(8.0, 8.0, 8.0)) / VOXEL).floor() * VOXEL;
	let mut max = ((Vec3::from(hi) + Vec3::new(8.0, 4.0, 8.0)) / VOXEL).ceil() * VOXEL;
	let limits = Vec3::new(MAX_XZ as f32, MAX_Y as f32, MAX_XZ as f32) * VOXEL;
	for axis in 0..3 {
		if max[axis] - min[axis] > limits[axis] {
			let middle = ((min[axis] + max[axis]) * 0.5 / VOXEL).round() * VOXEL;
			// keep the top (where the light enters) when cutting vertically
			if axis == 1 {
				min[axis] = max[axis] - limits[axis];
			} else {
				min[axis] = middle - limits[axis] * 0.5;
				max[axis] = middle + limits[axis] * 0.5;
			}
		}
	}
	// (at least a voxel along each axis: past about 1e11 studs a float cannot hold the cut's span around the middle, and
	// the cut came out empty)
	let dims = [((max.x - min.x) / VOXEL) as usize, ((max.y - min.y) / VOXEL) as usize, ((max.z - min.z) / VOXEL) as usize].map(|n| n.max(1));
	(min, dims)
}

/// A voxel's minimum corner (voxel x + nx (z + nz y)).
fn corner(voxel: usize, min: Vec3, [nx, _, nz]: [usize; 3]) -> Vec3A {
	let (x, rest) = (voxel % nx, voxel / nx);
	let (z, y) = (rest % nz, rest / nz);
	Vec3A::from(min) + Vec3A::new(x as f32, y as f32, z as f32) * VOXEL
}

/// The voxels a filler covers (index x + nx (z + nz y), ascending) and the share of each.
fn voxels(filler: &Filler, min: Vec3, dims: [usize; 3]) -> Vec<(u32, f32)> {
	let [nx, ny, nz] = dims;
	let mut voxels = Vec::new();
	let first = ((Vec3::from(filler.center - filler.extent) - min) / VOXEL).floor().max(Vec3::ZERO);
	let last = ((Vec3::from(filler.center + filler.extent) - min) / VOXEL).floor().min(Vec3::new((nx - 1) as f32, (ny - 1) as f32, (nz - 1) as f32));
	if first.x > last.x || first.y > last.y || first.z > last.z {
		return voxels;
	}
	for y in first.y as usize..=last.y as usize {
		for z in first.z as usize..=last.z as usize {
			for x in first.x as usize..=last.x as usize {
				let voxel = x + nx * (z + nz * y);
				let share = filler.cells(corner(voxel, min, dims)).iter().sum::<f32>() / (CELLS * CELLS) as f32;
				if share > 0.0 {
					voxels.push((voxel as u32, share));
				}
			}
		}
	}
	voxels
}

/// Several parts covering a voxel in part, combined per cell: its transmission's factor.
fn partial_cover(voxel: usize, group: impl Iterator<Item = usize>, fillers: &[Filler], min: Vec3, dims: [usize; 3]) -> f32 {
	let mut cells = [1.0f32; CELLS * CELLS];
	for number in group {
		let filler = &fillers[number];
		for (cell, share) in cells.iter_mut().zip(filler.cells(corner(voxel, min, dims))) {
			*cell *= 1.0 - (share * filler.opacity).min(1.0);
		}
	}
	cells.iter().sum::<f32>() / (CELLS * CELLS) as f32
}

fn full(fillers: &[Filler], (keys, ranges): (Vec<u8>, Vec<std::ops::Range<usize>>), (min, dims): (Vec3, [usize; 3]), keep: bool) -> GridState {
	let [nx, ny, nz] = dims;
	// transmission per voxel, 1 open; index x + nx (z + nz y) (layer-major for the propagation)
	let mut transmission = vec![1.0f32; nx * ny * nz];
	// a part covering a voxel's whole cross-section multiplies its transmission alone; the voxels
	// parts cover in part are combined per cell once every part is known
	let filled: Vec<Vec<(u32, f32)>> = fillers.par_iter().map(|filler| voxels(filler, min, dims)).collect();
	let mut partial: Vec<(u32, u32, f32)> = Vec::new();
	for (number, voxels) in filled.iter().enumerate() {
		let opacity = fillers[number].opacity;
		for &(voxel, share) in voxels {
			if share >= 1.0 - 1e-6 {
				transmission[voxel as usize] *= 1.0 - opacity;
			} else {
				partial.push((voxel, number as u32, share));
			}
		}
	}
	let filled = keep.then_some(filled);
	partial.par_sort_unstable_by_key(|&(voxel, number, _)| (voxel, number));
	for group in partial.chunk_by(|a, b| a.0 == b.0) {
		let voxel = group[0].0 as usize;
		if let [(_, number, share)] = group {
			transmission[voxel] *= 1.0 - (share * fillers[*number as usize].opacity).min(1.0);
			continue;
		}
		transmission[voxel] *= partial_cover(voxel, group.iter().map(|&(_, number, _)| number as usize), fillers, min, dims);
	}
	let mut entering = keep.then(|| vec![0.0f32; nx * ny * nz]);
	let values = propagate(&transmission, dims, entering.as_deref_mut());
	let detail = filled.zip(entering).map(|(filled, entering)| Detail { keys, ranges, filled, transmission, entering });
	GridState { origin: min, dims, values, fillers: fillers.len(), detail }
}

/// The transmission [`full`] gives one voxel: the parts covering all of it in their order, then
/// those covering some of it together.
fn voxel_transmission(voxel: u32, fillers: &[Filler], filled: &[Vec<(u32, f32)>], min: Vec3, dims: [usize; 3]) -> f32 {
	let mut transmission = 1.0f32;
	let mut partial: Vec<(usize, f32)> = Vec::new();
	for (number, voxels) in filled.iter().enumerate() {
		let (Some(first), Some(last)) = (voxels.first(), voxels.last()) else {
			continue;
		};
		if voxel < first.0 || voxel > last.0 {
			continue;
		}
		if let Ok(at) = voxels.binary_search_by_key(&voxel, |&(voxel, _)| voxel) {
			let share = voxels[at].1;
			if share >= 1.0 - 1e-6 {
				transmission *= 1.0 - fillers[number].opacity;
			} else {
				partial.push((number, share));
			}
		}
	}
	match partial.as_slice() {
		[] => {}
		[(number, share)] => transmission *= 1.0 - (share * fillers[*number].opacity).min(1.0),
		group => transmission *= partial_cover(voxel as usize, group.iter().map(|&(number, _)| number), fillers, min, dims),
	}
	transmission
}

/// A rectangle of voxel columns, x0..x1 by z0..z1.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Columns {
	x0: usize,
	x1: usize,
	z0: usize,
	z1: usize,
}

impl Columns {
	fn add(columns: &mut Option<Columns>, x: usize, z: usize) {
		*columns = Some(match *columns {
			Some(c) => Columns { x0: c.x0.min(x), x1: c.x1.max(x + 1), z0: c.z0.min(z), z1: c.z1.max(z + 1) },
			None => Columns { x0: x, x1: x + 1, z0: z, z1: z + 1 },
		});
	}

	fn union(a: Option<Columns>, b: Option<Columns>) -> Option<Columns> {
		match (a, b) {
			(Some(a), Some(b)) => Some(Columns { x0: a.x0.min(b.x0), x1: a.x1.max(b.x1), z0: a.z0.min(b.z0), z1: a.z1.max(b.z1) }),
			(a, None) => a,
			(None, b) => b,
		}
	}

	/// One column wider on every side, within the grid.
	fn grown(self, nx: usize, nz: usize) -> Columns {
		Columns { x0: self.x0.saturating_sub(1), x1: (self.x1 + 1).min(nx), z0: self.z0.saturating_sub(1), z1: (self.z1 + 1).min(nz) }
	}
}

/// [`full`]'s grid from a kept one whose fillers are as many and fill the same grid: the voxels
/// the changed fillers covered or cover now get their transmission again ([`voxel_transmission`]),
/// and where one changed, the light below it is propagated again over the columns it can reach (a
/// voxel's light reads the 3 x 3 columns above it: one more column on each side per layer down).
/// `None` when the changes cover more than a layer's voxels: building anew is then as quick.
fn update(fillers: &[Filler], (keys, ranges): (&[u8], &[std::ops::Range<usize>]), (min, dims): (Vec3, [usize; 3]), mut values: Vec<u8>, mut detail: Detail, keep: bool) -> Option<GridState> {
	let [nx, ny, nz] = dims;
	let changed: Vec<usize> = (0..fillers.len()).filter(|&number| keys[ranges[number].clone()] != detail.keys[detail.ranges[number].clone()]).collect();
	let fresh: Vec<Vec<(u32, f32)>> = changed.par_iter().map(|&number| voxels(&fillers[number], min, dims)).collect();
	let covered: usize = changed.iter().zip(&fresh).map(|(&number, voxels)| detail.filled[number].len() + voxels.len()).sum();
	if covered > nx * nz {
		return None;
	}
	let mut affected: Vec<u32> = changed.iter().zip(&fresh).flat_map(|(&number, voxels)| detail.filled[number].iter().chain(voxels).map(|&(voxel, _)| voxel)).collect();
	affected.sort_unstable();
	affected.dedup();
	for (&number, voxels) in changed.iter().zip(fresh) {
		detail.filled[number] = voxels;
	}
	let transmissions: Vec<f32> = affected.par_iter().map(|&voxel| voxel_transmission(voxel, fillers, &detail.filled, min, dims)).collect();
	let mut columns: Vec<Option<Columns>> = vec![None; ny];
	for (&voxel, transmission) in affected.iter().zip(transmissions) {
		let old = &mut detail.transmission[voxel as usize];
		if old.to_bits() != transmission.to_bits() {
			*old = transmission;
			let (x, rest) = (voxel as usize % nx, voxel as usize / nx);
			Columns::add(&mut columns[rest / nz], x, rest % nz);
		}
	}
	// a layer's light reads the layer above's: from the highest changed layer down
	let mut reached: Option<Columns> = None;
	for y in (0..ny.saturating_sub(1)).rev() {
		reached = Columns::union(reached, columns[y + 1]).map(|columns| columns.grown(nx, nz));
		let Some(region) = reached else {
			continue;
		};
		let (lower, upper) = detail.entering.split_at_mut(nx * nz * (y + 1));
		let here = &mut lower[nx * nz * y..];
		let above = &upper[..nx * nz];
		let layer = &detail.transmission[nx * nz * (y + 1)..nx * nz * (y + 2)];
		for z in region.z0..region.z1 {
			for x in region.x0..region.x1 {
				let light = entering_at(above, layer, x, z, nx, nz);
				here[x + nx * z] = light;
				values[x + nx * (y + ny * z)] = visibility(light);
			}
		}
	}
	if keep {
		detail.keys = keys.to_vec();
		detail.ranges = ranges.to_vec();
	}
	Some(GridState { origin: min, dims, values, fillers: fillers.len(), detail: keep.then_some(detail) })
}

/// The light entering voxel column (x, z) of a layer from the one above it (its entering light and
/// transmission), as [`propagate`] sums it: each row of three, then the three rows, less the
/// centre, over 8; outside the grid 1 (the rows beyond it 3).
fn entering_at(above: &[f32], layer: &[f32], x: usize, z: usize, nx: usize, nz: usize) -> f32 {
	let passed = |x: isize, z: usize| if x < 0 || x >= nx as isize { 1.0 } else { above[x as usize + nx * z] * layer[x as usize + nx * z] };
	let row = |z: isize| {
		if z < 0 || z >= nz as isize {
			3.0
		} else {
			let x = x as isize;
			passed(x - 1, z as usize) + passed(x, z as usize) + passed(x + 1, z as usize)
		}
	};
	let z = z as isize;
	let sum = row(z - 1) + row(z) + row(z + 1) - passed(x as isize, z as usize);
	sum / 8.0
}

/// The visibility stored for the light entering a voxel: 1 - (1 - L)^4, of 255.
#[inline]
fn visibility(entering: f32) -> u8 {
	let visibility = 1.0 - (1.0 - entering.clamp(0.0, 1.0)).powi(4);
	(visibility * 255.0).round() as u8
}

/// Visibility per voxel (0..255, X fastest, then Y, then Z) from transmissions (X fastest, then Z,
/// then Y): the light entering each voxel from above, layer by layer from the top, is the 3 x 3 sum
/// over the layer above (light times transmission, 1 outside: a border of open voxels) less its
/// centre, over 8. `kept` (X fastest, then Z, then Y) takes each layer's entering light.
fn propagate(transmission: &[f32], dims: [usize; 3], mut kept: Option<&mut [f32]>) -> Vec<u8> {
	let [nx, ny, nz] = dims;
	let mut values = vec![255u8; nx * ny * nz];
	let mut entering = vec![1.0f32; nx * nz];
	let px = nx + 2;
	let mut passed = vec![1.0f32; px * (nz + 2)];
	let mut rows = vec![3.0f32; nx * (nz + 2)];
	for y in (0..ny).rev() {
		if y + 1 < ny {
			let layer = &transmission[nx * nz * (y + 1)..nx * nz * (y + 2)];
			passed[px..px * (nz + 1)].par_chunks_mut(px).enumerate().for_each(|(z, row)| {
				for x in 0..nx {
					row[x + 1] = entering[x + nx * z] * layer[x + nx * z];
				}
			});
			rows[nx..nx * (nz + 1)].par_chunks_mut(nx).enumerate().for_each(|(z, row)| {
				let source = &passed[px * (z + 1)..px * (z + 2)];
				for x in 0..nx {
					row[x] = source[x] + source[x + 1] + source[x + 2];
				}
			});
			entering.par_chunks_mut(nx).enumerate().for_each(|(z, row)| {
				for x in 0..nx {
					let sum = rows[x + nx * z] + rows[x + nx * (z + 1)] + rows[x + nx * (z + 2)] - passed[x + 1 + px * (z + 1)];
					row[x] = sum / 8.0;
				}
			});
		}
		values.par_chunks_mut(nx * ny).enumerate().for_each(|(z, slice)| {
			for x in 0..nx {
				slice[x + nx * y] = visibility(entering[x + nx * z]);
			}
		});
		if let Some(kept) = kept.as_deref_mut() {
			kept[nx * nz * y..nx * nz * (y + 1)].copy_from_slice(&entering);
		}
	}
	values
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use glam::Affine3A;

	use super::*;

	fn block(min: Vec3, max: Vec3) -> Part {
		Part {
			path: "Part".into(),
			class: "Part".into(),
			cframe: Affine3A::from_translation((min + max) * 0.5),
			size: max - min,
			geometry: Geometry::Block,
			color: Vec3::ONE,
			material: 256,
			material_variant: String::new(),
			transparency: 0.0,
			reflectance: 0.0,
			cast_shadow: true,
			double_sided: false,
			texture: None,
			vertex_color: Vec3::ONE,
			surface_appearance: None,
			decals: Vec::new(),
			surfaces: [0; 6],
			skeleton: None,
			body_part: None,
			composite: None,
			seed: 0,
			collision: None,
			referent: None,
		}
	}

	/// The stored visibility at a world point's voxel.
	fn at(grid: &LightGrid, point: Vec3) -> f32 {
		let v = ((point - grid.origin) / VOXEL).floor();
		let [nx, ny, _] = grid.dims;
		grid.values[v.x as usize + nx * (v.y as usize + ny * v.z as usize)] as f32 / 255.0
	}

	#[test]
	fn roof_closes_the_layer_below_its_own() {
		// lg-height: a 48-stud roof at 8-9; the voxel under it (4-8) is closed, its own layer open
		let scene = Scene { parts: vec![Arc::new(block(Vec3::new(-24.0, 8.0, -24.0), Vec3::new(24.0, 9.0, 24.0)))], ..Default::default() };
		let grid = build(&scene).unwrap();
		assert_eq!(at(&grid, Vec3::new(2.0, 6.0, 2.0)), 0.0);
		assert_eq!(at(&grid, Vec3::new(2.0, 10.0, 2.0)), 1.0);
	}

	/// A part whose CFrame is not a number (a script's look-at along nothing leaves one) fills no voxel, wherever it comes
	/// among the parts: the grid is the other parts'. It made the grid's bounds NaN and every dimension 0 (a 0.86 x 0.43 x
	/// 0.43 part among five), and `corner` divided by zero.
	#[test]
	fn a_part_that_is_not_a_number_fills_nothing() {
		let roof = block(Vec3::new(-24.0, 8.0, -24.0), Vec3::new(24.0, 9.0, 24.0));
		let floor = block(Vec3::new(-30.0, -1.0, -30.0), Vec3::new(30.0, 0.0, 30.0));
		let mut lost = block(Vec3::ZERO, Vec3::new(0.86, 0.43, 0.43));
		lost.cframe = Affine3A { matrix3: Mat3A::from_cols_array(&[f32::NAN; 9]), translation: Vec3A::splat(f32::NAN) };
		let alone = build(&Scene { parts: vec![Arc::new(roof.clone()), Arc::new(floor.clone())], ..Default::default() }).unwrap();
		for parts in [vec![roof.clone(), floor.clone(), lost.clone()], vec![lost.clone(), roof.clone(), floor.clone()]] {
			let scene = Scene { parts: parts.into_iter().map(Arc::new).collect(), ..Default::default() };
			assert_eq!(filler_count(&scene), 2);
			let grid = build(&scene).unwrap();
			assert_eq!((grid.origin, grid.dims), (alone.origin, alone.dims));
			assert!(grid.values == alone.values);
		}
	}

	/// A part parked very far off (1e11 studs) leaves the grid at least a voxel along each axis: a float cannot hold the
	/// cut's span around the middle of such a scene, the cut came out empty, and `corner` divided by zero.
	#[test]
	fn a_part_parked_very_far_off_leaves_the_grid_a_voxel_at_least() {
		let roof = block(Vec3::new(-24.0, 8.0, -24.0), Vec3::new(24.0, 9.0, 24.0));
		let parked = block(Vec3::new(1.0e11, 0.0, 0.0), Vec3::new(1.0e11 + 4.0, 4.0, 4.0));
		let grid = build(&Scene { parts: vec![Arc::new(roof), Arc::new(parked)], ..Default::default() }).unwrap();
		assert!(grid.dims.iter().all(|&n| n >= 1), "{:?}", grid.dims);
		assert_eq!(grid.values.len(), grid.dims.iter().product::<usize>());
	}

	#[test]
	fn parts_side_by_side_fill_a_voxel_together() {
		// two halves of every voxel from two parts close the layer below as one roof would
		let left = block(Vec3::new(-24.0, 8.0, -24.0), Vec3::new(0.0, 9.0, 24.0));
		let halves: Vec<Part> = (0..12).map(|k| block(Vec3::new(-24.0 + 4.0 * k as f32, 8.0, -24.0), Vec3::new(-22.0 + 4.0 * k as f32, 9.0, 24.0))).collect();
		let mut parts = vec![left];
		parts.extend(halves.iter().map(|part| {
			let mut part = part.clone();
			part.cframe.translation.x += 2.0;
			part
		}));
		parts.extend(halves);
		let grid = build(&Scene { parts: parts.into_iter().map(Arc::new).collect(), ..Default::default() }).unwrap();
		assert_eq!(at(&grid, Vec3::new(-10.0, 6.0, 2.0)), 0.0);
		assert_eq!(at(&grid, Vec3::new(10.0, 6.0, 2.0)), 0.0);
		// slats 1 wide every 2 studs: half of each voxel, 1 - 0.5^4 under them
		let slats = (0..24).map(|k| Arc::new(block(Vec3::new(-24.0 + 2.0 * k as f32, 8.0, -24.0), Vec3::new(-23.0 + 2.0 * k as f32, 9.0, 24.0)))).collect();
		let grid = build(&Scene { parts: slats, ..Default::default() }).unwrap();
		assert!((at(&grid, Vec3::new(2.0, 6.0, 2.0)) - 0.9375).abs() < 0.003);
	}

	#[test]
	fn an_upright_sheet_closes_its_voxels() {
		// lg-specular: an 80 x 40 x 0.5-stud wall darkens the open grid in front of it, as a wall
		// filling its voxels does; slats (parts_side_by_side_fill_a_voxel_together) and a wall only 2
		// studs long keep their share
		let wall = block(Vec3::new(-40.0, -20.0, 6.25), Vec3::new(40.0, 20.0, 6.75));
		let grid = build(&Scene { parts: vec![Arc::new(wall)], ..Default::default() }).unwrap();
		assert!(at(&grid, Vec3::new(0.0, -10.0, 2.0)) < 0.95);
		let short = block(Vec3::new(-1.0, -20.0, 6.25), Vec3::new(1.0, 20.0, 6.75));
		let grid = build(&Scene { parts: vec![Arc::new(short)], ..Default::default() }).unwrap();
		assert!(at(&grid, Vec3::new(0.0, -10.0, 2.0)) > 0.999);
	}

	#[test]
	fn propagation_matches_the_plain_loop() {
		let dims = [7, 5, 6];
		let [nx, ny, nz] = dims;
		let mut seed = 7u32;
		let transmission: Vec<f32> = (0..nx * ny * nz)
			.map(|_| {
				seed ^= seed << 13;
				seed ^= seed >> 17;
				seed ^= seed << 5;
				(seed % 1000) as f32 / 999.0
			})
			.collect();
		let mut expected = vec![255u8; nx * ny * nz];
		let mut entering = vec![1.0f32; nx * nz];
		for y in (0..ny).rev() {
			if y + 1 < ny {
				let passed: Vec<f32> = (0..nx * nz).map(|i| entering[i] * transmission[i + nx * nz * (y + 1)]).collect();
				for z in 0..nz {
					for x in 0..nx {
						let mut sum = 0.0;
						for dz in -1i32..=1 {
							for dx in -1i32..=1 {
								if dx == 0 && dz == 0 {
									continue;
								}
								let (sx, sz) = (x as i32 + dx, z as i32 + dz);
								sum += if sx < 0 || sz < 0 || sx >= nx as i32 || sz >= nz as i32 { 1.0 } else { passed[sx as usize + nx * sz as usize] };
							}
						}
						entering[x + nx * z] = sum / 8.0;
					}
				}
			}
			for z in 0..nz {
				for x in 0..nx {
					expected[x + nx * (y + ny * z)] = ((1.0 - (1.0 - entering[x + nx * z].clamp(0.0, 1.0)).powi(4)) * 255.0).round() as u8;
				}
			}
		}
		assert_eq!(propagate(&transmission, dims, None), expected);
	}

	#[test]
	fn updates_match_a_grid_built_anew() {
		let mut seed = 99u32;
		let mut random = move || {
			seed ^= seed << 13;
			seed ^= seed >> 17;
			seed ^= seed << 5;
			seed as f32 / u32::MAX as f32
		};
		// a floor and a tall pillar fix the grid's frame; blocks (some turned, some see-through),
		// balls, cylinders and wedges over it share voxels whole and in part
		let mut parts = vec![block(Vec3::new(-64.0, -1.0, -64.0), Vec3::new(64.0, 0.0, 64.0)), block(Vec3::new(-2.0, 0.0, -2.0), Vec3::new(2.0, 60.0, 2.0))];
		for index in 0..60 {
			let position = Vec3::new(random() * 100.0 - 50.0, random() * 40.0, random() * 100.0 - 50.0);
			let size = Vec3::new(0.5 + random() * 9.0, 0.5 + random() * 5.0, 0.5 + random() * 9.0);
			let mut part = block(position, position + size);
			part.geometry = [Geometry::Block, Geometry::Ball, Geometry::Cylinder, Geometry::Wedge, Geometry::Block][index % 5].clone();
			if index % 3 == 0 {
				part.cframe = Affine3A::from_rotation_translation(glam::Quat::from_rotation_y(random() * 6.0), part.cframe.translation.into());
			}
			if index % 4 == 0 {
				part.transparency = 0.5;
			}
			parts.push(part);
		}
		let mut scene = Scene { parts: parts.into_iter().map(Arc::new).collect(), ..Default::default() };
		let mut state = build_from(&scene, None, true).unwrap();
		let mut updated = 0;
		for step in 0..40 {
			// limbs swinging: a few parts move a little, turn, fade or stop filling (CastShadow)
			for index in [2 + step % 7, 9, 20 + step % 11, 33] {
				let part = Arc::make_mut(&mut scene.parts[index]);
				part.cframe.translation += glam::Vec3A::new(random() * 3.0 - 1.5, random() * 2.0 - 1.0, random() * 3.0 - 1.5);
				if step % 3 == 0 {
					part.cframe.matrix3 = glam::Mat3A::from_rotation_y(random() * 6.0) * part.cframe.matrix3;
				}
				if step % 5 == 0 {
					part.transparency = random();
				}
			}
			if step % 10 == 9 {
				let part = Arc::make_mut(&mut scene.parts[40]);
				part.cast_shadow = !part.cast_shadow;
			}
			let expected = build(&scene).unwrap();
			let fillers = filler_count(&scene);
			let previous = Some(state).filter(|state| state.fillers() == fillers);
			updated += usize::from(previous.as_ref().is_some_and(|previous| previous.origin == expected.origin && previous.dims == expected.dims));
			state = build_from(&scene, previous, true).unwrap();
			assert_eq!((state.origin, state.dims), (expected.origin, expected.dims), "step {step}");
			assert!(state.values == expected.values, "step {step}: {} voxels differ", state.values.iter().zip(&expected.values).filter(|(a, b)| a != b).count());
		}
		assert!(updated >= 15, "only {updated} steps updated a kept grid");
	}

	/// The build's cost on a place of many blocks: `cargo test --release lightgrid_cost -- --ignored --nocapture`
	#[test]
	#[ignore]
	fn lightgrid_cost() {
		let mut parts = vec![block(Vec3::new(-1024.0, -1.0, -1024.0), Vec3::new(1024.0, 0.0, 1024.0))];
		let mut seed = 12345u32;
		let mut random = move || {
			seed ^= seed << 13;
			seed ^= seed >> 17;
			seed ^= seed << 5;
			seed as f32 / u32::MAX as f32
		};
		for _ in 0..30000 {
			let position = Vec3::new(random() * 1500.0 - 750.0, random() * 300.0, random() * 1500.0 - 750.0);
			let size = Vec3::new(1.0 + random() * 12.0, 1.0 + random() * 6.0, 1.0 + random() * 12.0);
			let mut part = block(position, position + size);
			part.cframe = Affine3A::from_rotation_translation(glam::Quat::from_rotation_y(random() * 6.0), part.cframe.translation.into());
			parts.push(part);
		}
		let scene = Scene { parts: parts.into_iter().map(Arc::new).collect(), ..Default::default() };
		let start = std::time::Instant::now();
		let grid = build(&scene).unwrap();
		eprintln!("{} parts: {:?} voxels in {:.1} ms", scene.parts.len(), grid.dims, start.elapsed().as_secs_f64() * 1000.0);
	}
}
