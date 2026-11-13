// The forward pass for parts: PBR shading with the sun (shadow-mapped), sky ambient (SH) and
// specular (prefiltered cube), local lights, and Roblox's material shading models.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
// bindings 2 and 3, the sun's shadow map: shadow.wgsl
@group(0) @binding(4) var environment_map: texture_cube<f32>;
// the sky chain the Atmosphere's fog fades into (the skybox unturned by SkyboxOrientation)
@group(0) @binding(19) var faded_map: texture_cube<f32>;
@group(0) @binding(5) var linear_sampler: sampler;
@group(0) @binding(6) var brdf_lut: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> lights: array<Light>;
// the opaque parts' depth (ForceField's contact glow; a 1x1 placeholder without ForceField)
@group(0) @binding(8) var scene_depth: texture_depth_2d;
@group(0) @binding(9) var organic_noise: texture_2d<f32>;
// Roblox's surface atlas (Studs, Inlet, ...): albedo (DXT1) and normal (DXT5nm) tiles
@group(0) @binding(10) var surface_albedo: texture_2d<f32>;
@group(0) @binding(11) var surface_normal: texture_2d<f32>;
// Studio's wang index (pre-2022 Wang-tiled materials)
@group(0) @binding(12) var wang_index: texture_2d<f32>;
// local lights with Shadows: six depth faces each (+X, -X, +Y, -Y, +Z, -Z) and their view-projections
@group(0) @binding(13) var light_shadow_maps: texture_depth_2d_array;
@group(0) @binding(14) var<storage, read> light_shadow_matrices: array<mat4x4<f32>>;
// the view's lights by cluster (render::light_clusters): a bit per light, frame.light_clusters.w u32s a
// cluster
@group(0) @binding(20) var<storage, read> light_clusters: array<u32>;
// the light grid's sky visibility, one texel per 4-stud voxel (render::lightgrid)
@group(0) @binding(15) var light_grid: texture_3d<f32>;
// the screen-space ambient occlusion of the opaque surfaces (render::ao; 1 x 1 white without)
@group(0) @binding(16) var ambient_occlusion: texture_2d<f32>;
// the opaque scene (√x) before the transparent parts, single-sampled: what Glass shows through it
// (1 x 1 black without Glass)
@group(0) @binding(17) var scene_color: texture_2d<f32>;

// Glass's shading normal (its normal map in), n·v, the sky it reflects (the prefiltered sky at the
// sky specular's level, without its Fresnel), the sun's and local lights' glint and the sky's diffuse
// light on its surface, left by shade_surface for fs_glass
var<private> glass_normal: vec3<f32>;
var<private> glass_facing: f32;
var<private> glass_sky: vec3<f32>;
var<private> glass_glint: vec3<f32>;
var<private> glass_sky_diffuse: vec3<f32>;

// what the sky's diffuse light falls to where the grid is closed: a fixed warm grey whatever the sky
// (lg-sky: on a white floor under a 48-stud roof the same for skies dark, white, default, white in
// the Up or the Dn image only and at midnight, half at EnvironmentDiffuseScale 0.5)
const LIGHT_GRID_SKY_CLOSED: vec3<f32> = vec3<f32>(0.104, 0.100, 0.092);
// what of the sky specular the closed grid leaves (lg-terms: 0.13; Studio's own reading under the roof
// changes between capture sessions, lg-sky)
const LIGHT_GRID_SPECULAR_FLOOR: f32 = 0.12;
// how far along the normal a surface reads the grid (lg-height, lg-walls: 6 studs)
const LIGHT_GRID_OFFSET: f32 = 6.0;

// the light grid at a surface: trilinear between voxel centres, 6 studs along the normal; 1 without
// a grid
fn light_grid_visibility(position: vec3<f32>, normal: vec3<f32>) -> f32 {
	if (frame.light_grid.w < 0.5) {
		return 1.0;
	}
	let uvw = (position + normal * LIGHT_GRID_OFFSET - frame.light_grid.xyz) * frame.light_grid_scale.xyz;
	return textureSampleLevel(light_grid, linear_sampler, uvw, 0.0).r;
}

@group(1) @binding(0) var color_map: texture_2d<f32>;
@group(1) @binding(1) var normal_map: texture_2d<f32>;
@group(1) @binding(2) var metalness_map: texture_2d<f32>;
@group(1) @binding(3) var roughness_map: texture_2d<f32>;
@group(1) @binding(4) var material_sampler: sampler;
@group(1) @binding(5) var emissive_map: texture_2d<f32>;
// The material maps' full sizes (texels): the mips are chosen by these, whatever levels of a map the
// GPU holds (render::texture_files: a map whose finest levels no view samples is held without them)
struct MaterialInfo {
	// xy colour, zw normal; xy metalness, zw roughness; xy emissive
	sizes: array<vec4<f32>, 3>,
	// their slots in the residency feedback (fs_residency): colour, normal, metalness, roughness;
	// emissive
	ids: array<vec4<u32>, 2>,
}
@group(1) @binding(6) var<uniform> material_info: MaterialInfo;

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) normal: vec3<f32>,
	@location(2) uv: vec2<f32>,
	@location(3) tangent: vec4<f32>,
	@location(4) color: vec4<f32>,
}

struct VertexOutput {
	@builtin(position) clip: vec4<f32>,
	@location(0) world_position: vec3<f32>,
	@location(1) world_normal: vec3<f32>,
	@location(2) uv: vec2<f32>,
	@location(3) world_tangent: vec4<f32>,
	@location(4) color: vec4<f32>,
	@location(5) object_position: vec3<f32>,
	@location(6) object_normal: vec3<f32>,
	@location(7) @interpolate(flat) instance: u32,
}

fn vertex_output(input: VertexInput, instance_index: u32) -> VertexOutput {
	let instance = instances[instance_index];
	let world = instance.model * vec4<f32>(input.position, 1.0);
	var output: VertexOutput;
	output.clip = frame.view_proj * world;
	output.world_position = world.xyz;
	output.world_normal = normalize((instance.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz);
	output.world_tangent = vec4<f32>(normalize((instance.model * vec4<f32>(input.tangent.xyz, 0.0)).xyz), input.tangent.w);
	output.uv = input.uv;
	output.color = input.color;
	output.object_position = input.position;
	output.object_normal = input.normal;
	output.instance = instance_index;
	return output;
}

@vertex
fn vs_main(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
	return vertex_output(input, instance_index);
}

// A posed skinned mesh's bind pose: the same vertex of its unposed copy (render::skinning)
struct BindInput {
	@location(5) position: vec3<f32>,
	@location(6) normal: vec3<f32>,
}

// A decal on a posed skinned mesh (TEX_DECAL_BIND): drawn on the posed surface, projected from where
// the bind pose puts it, on the faces whose bind-pose normal points most nearly the decal's way in
// the part, so it sticks to the surface as the bones move it (notes/render.md "Decals": skin-pose's
// posed character with a glow and blockUpperLeft as Front, Right, Left and Back decals, orange pixels'
// IoU 0.55 / 0.44 projected as posed, 0.52 / 0.49 from the bind pose on every face turned that way,
// 0.83 / 0.86 so)
@vertex
fn vs_decal_bind(input: VertexInput, bind: BindInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
	var output = vertex_output(input, instance_index);
	output.object_position = bind.position;
	output.object_normal = bind.normal;
	return output;
}

struct Surface {
	albedo: vec3<f32>,
	alpha: f32,
	normal: vec3<f32>,
	roughness: f32,
	metalness: f32,
	emissive: vec3<f32>,
	// dielectric F0 and how much of Schlick's Fresnel applies (instance.specular)
	f0: f32,
	fresnel: f32,
	// specular visibility term (instance.specular.z): 0 correlated Smith, 1 Schlick-GGX, 2 implicit, 3 Kelemen
	visibility_mode: f32,
	// how much of the albedo the sun and local lights' diffuse keeps
	diffuse_weight: vec3<f32>,
}

// Image-based lighting (notes/render.md "Sky", "Metals"; ibl-spheres, ibl-scale): the sky specular is
// its Fresnel times the prefiltered sky (whose levels lose what Studio's storage cuts,
// render::environment::stored), metalness and the sky specular both fading with the view depth
// (`specular_fade`), and the prefiltered sky's level for a roughness
const SPECULAR_LOD_BIAS: f32 = 0.375;
const SPECULAR_LOD_SCALE: f32 = 4.5;
// the sky specular's Fresnel table is read 0.125 levels past the prefiltered level, at roughness
// (lod + 0.125) / 5 = 0.1 + 0.9 roughness
const SKY_FRESNEL_LOD_OFFSET: f32 = 0.125;

// Material texture coordinates in studs (notes/materials.md "Mapping"). Every face counts from its
// top-left corner as seen from outside, u to the right and v down.
struct MaterialFrame {
	studs: vec2<f32>,
	// the material face (box faces in NormalId order, 6 the wedge's slope, 7 / 8 the corner wedge's)
	face: u32,
	// part-space directions of +u and of up the texture (-v) for normal mapping; zero: the vertex
	// tangent frame follows the uv already
	u_axis: vec3<f32>,
	up_axis: vec3<f32>,
}

// The box face (NormalId order) of a direction's dominant axis.
fn box_face(direction: vec3<f32>) -> u32 {
	let a = abs(direction);
	if (a.x >= a.y && a.x >= a.z) {
		return select(3u, 0u, direction.x > 0.0);
	}
	if (a.y >= a.z) {
		return select(4u, 1u, direction.y > 0.0);
	}
	return select(5u, 2u, direction.z > 0.0);
}

fn material_frame(input: VertexOutput, instance: Instance) -> MaterialFrame {
	let size = instance.size.xyz;
	var frame: MaterialFrame;
	let mode = instance.flags.w;
	if (mode == UV_PLANAR) {
		let code = min(u32(round(input.color.a * 255.0)), 8u);
		frame.face = code;
		let u_edge = MATERIAL_FACE_U[code] * size;
		let v_edge = MATERIAL_FACE_V[code] * size;
		frame.studs = input.uv * vec2<f32>(length(u_edge), length(v_edge));
		if ((instance.flags.y & TEX_ORGANIC) != 0u && code >= 6u) {
			// Organic slopes count v along the box axes: 0.99 (|y| + |z|) of the edge, not its length
			// (organic-slopes: 0.9897-0.9900 on four wedges, from the top edge)
			frame.studs.y = input.uv.y * ORGANIC_SLOPE_SCALE * dot(abs(v_edge), vec3<f32>(1.0));
		}
		return frame;
	}
	// a part-space point relative to the box centre (studs) and the face it projects onto
	var p: vec3<f32>;
	var face: u32;
	if (mode == UV_BALL) {
		// the unit sphere's point pushed out along its ray onto the cube face
		let o = input.object_position;
		face = box_face(o);
		let g = o * (0.5 / max(abs(dot(o, FACE_NORMAL[face])), 1e-4));
		p = g * size / BALL_UV_SCALE;
	} else {
		p = (instance.part_inverse * vec4<f32>(input.world_position, 1.0)).xyz * size;
		let n = (instance.part_inverse * vec4<f32>(normalize(input.world_normal), 0.0)).xyz * size;
		face = box_face(n);
	}
	frame.face = face;
	let u_axis = MATERIAL_FACE_U[face];
	let v_axis = MATERIAL_FACE_V[face];
	// box faces count from their top-left corner, meshes from the face's centre (meshpart-relief:
	// R15's torso mesh at 4x and 1x, NCC 0.93-0.98 from the centre against -0.01 from the corner)
	let corner = select(0.5 * vec2<f32>(abs(dot(size, u_axis)), abs(dot(size, v_axis))), vec2<f32>(0.0), mode == UV_BOX);
	frame.studs = vec2<f32>(dot(p, u_axis), dot(p, v_axis)) + corner;
	frame.u_axis = u_axis;
	frame.up_axis = -v_axis;
	return frame;
}

// The Organic pattern (notes/materials.md "Organic pattern"): Roblox's noise texture, read at 0.03
// noise uv per tile on MaterialVariants (per material on built-in ones), gives k; level
// i = floor(8 k) picks two copies of the maps, shifted by sin(3 i, 7 i) and sin(3 (i + 1), 7 (i + 1))
// tiles, cross-faded linearly by fract(8 k) (organic-variant).
struct OrganicLayers {
	on: bool,
	a: vec2<f32>,
	b: vec2<f32>,
	fade: f32,
	// derivatives of the unshifted coordinate (the shifts jump where the level changes)
	dx: vec2<f32>,
	dy: vec2<f32>,
}

// `uv` already carries the part's texture offset (instance.tiling.xy); the noise reads the face's own
// tiles, shifted by the part's noise offset (instance.tiling.zw; both random per part on built-in
// materials, none on MaterialVariants).
fn organic_layers(uv: vec2<f32>, tiles: vec2<f32>, instance: Instance) -> OrganicLayers {
	var layers: OrganicLayers;
	layers.dx = dpdx(uv);
	layers.dy = dpdy(uv);
	layers.on = (instance.flags.y & TEX_ORGANIC) != 0u;
	if (layers.on) {
		let k = textureSampleLevel(organic_noise, material_sampler, tiles * instance.material.w + instance.tiling.zw, 0.0).r;
		let level = k * ORGANIC_LEVELS;
		let i = floor(level);
		layers.fade = level - i;
		layers.a = uv + sin(vec2<f32>(3.0, 7.0) * i);
		layers.b = uv + sin(vec2<f32>(3.0, 7.0) * (i + 1.0));
	}
	return layers;
}

// A pre-2022 Wang-tiled material's texture coordinate: the texture is a 4 x 4 atlas of quarter-tile
// cells, and the wang index (128 x 128, wrapping) names the atlas cell of every quarter-tile cell of
// the surface: column in red, row in alpha, 0..3 stored as multiples of 64 (material-swatches:
// Brick 280 of 280 indices, Cobblestone 68 of 70; whole-frame NCC 0.83-0.97).
fn wang_uv(uv: vec2<f32>) -> vec2<f32> {
	let cells = uv * 4.0;
	let cell = floor(cells);
	let texel = vec2<i32>(cell - 128.0 * floor(cell / 128.0));
	let index = textureLoad(wang_index, texel, 0);
	let atlas = round(vec2<f32>(index.r, index.a) * (255.0 / 64.0));
	return (atlas + cells - cell) * 0.25;
}

// Mip filtering as Studio's renders show it: the two levels around the LOD, the upper weighted
// 2 (f - 0.25) clamped to 0..1 for the LOD's fraction f (a narrowed trilinear blend). material-zoom
// (slabs straight down from 6 to 96 studs) against ror's own levels: Studio shows level 0.60-0.71 at
// LOD 0.56, 0.96-1.07 at 1.06, 0.97-1.09 at 1.25 (material-swatches), 1.52-1.71 at 1.56 and
// 2.50-2.82 at 2.56 (Brick, Carpet, CeramicTiles, RoofShingles, WoodPlanks, ClayRoofTiles), with no
// LOD bias. The GPU's own blend here sits a quarter level higher (0.12 of the upper level at f =
// 0.06, all of it from 0.5 on), which the colour maps' former -0.5 bias half made up for; so each
// level is read at its whole LOD (gradients scaled to it, anisotropic filtering kept) and mixed by
// hand.
fn studio_mips(map: texture_2d<f32>, size: vec2<f32>, uv: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> vec4<f32> {
	let x = length(dx * size);
	let y = length(dy * size);
	let major = max(x, y);
	let minor = max(min(x, y), 1e-8);
	// the anisotropic LOD: the major axis over the anisotropy (at most 16, the sampler's clamp)
	let lod = log2(max(major / min(major / minor, 16.0), 1e-8));
	if (lod <= 0.0) {
		return textureSampleGrad(map, material_sampler, uv, dx, dy);
	}
	let level = floor(lod);
	let scale = exp2(level - lod);
	let lower = textureSampleGrad(map, material_sampler, uv, dx * scale, dy * scale);
	let weight = saturate((lod - level - 0.25) * 2.0);
	// the upper level weighs nothing in the first quarter of a level: not read
	if (weight == 0.0) {
		return lower;
	}
	let upper = textureSampleGrad(map, material_sampler, uv, dx * scale * 2.0, dy * scale * 2.0);
	return mix(lower, upper, weight);
}

// A material map at `uv` (through the Wang atlas on Wang-tiled materials), or its two Organic layers
// cross-faded, filtered with the gradients of the unshifted coordinate: the Organic shifts and the
// Wang cells jump.
fn material_texel(map: texture_2d<f32>, size: vec2<f32>, uv: vec2<f32>, layers: OrganicLayers, wang: bool) -> vec4<f32> {
	if (!layers.on) {
		return studio_mips(map, size, select(uv, wang_uv(uv), wang), layers.dx, layers.dy);
	}
	let a = studio_mips(map, size, layers.a, layers.dx, layers.dy);
	let b = studio_mips(map, size, layers.b, layers.dx, layers.dy);
	return mix(a, b, layers.fade);
}

// GGX of width α
fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
	let a2 = alpha * alpha;
	let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
	return a2 / (PI * d * d);
}


fn visibility_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
	let a = roughness * roughness;
	let gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a * a) + a * a);
	let gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a * a) + a * a);
	return 0.5 / max(gv + gl, 1e-5);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
	return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

// What is left of a surface's sky specular (x) and metalness (y) at its view depth (notes/render.md
// "Specular fade"): Studio fades both linearly to none, at 1 / frame.specular_fade.x and .y studs
// (render::materials SPECULAR_FADE; ibl-fade: spheres as large on screen from 10 to 1000 studs, a
// white metal 1.26 x its 150-stud reflection at 10 studs, 0.09 x at 1000; ibl-distance; ibl-fade's row
// 40 deep across a 100-degree view reads as 40 at its ends, 72 studs away). The direct specular does
// not fade (sa-sun: black spheres alike from 60 and 150 studs).
fn specular_fade(world_position: vec3<f32>) -> vec2<f32> {
	let depth = (frame.view_proj * vec4<f32>(world_position, 1.0)).w;
	return saturate(vec2<f32>(1.0) - depth * frame.specular_fade.xy) * frame.specular_fade.zw;
}

// The direct specular (the sun's and local lights'; notes/render.md "Direct specular"): GGX of width
// α = width (w + (1 - w) roughness²) seen through Schlick-GGX with k = k0 + k1 roughness
// (frame.direct_specular, render::materials DIRECT_SPECULAR); the reflectance rises by Schlick's
// (1 - v·h)⁵ of weight `fresnel` from the dielectric's F0 and from the metal's albedo, the
// dielectric's then times g (1 + c α²) (frame.direct_reflectance = [g, c, fresnel, compensation],
// DIRECT_REFLECTANCE). Each share keeps its own multiple scattering's energy: the dielectric's x
// (1 + F0 (1 / E - 1)) with E the split-sum table's directional albedo at the prefiltered level's
// roughness, the metal's x (1 + compensation (1 / E_m - 1)) with E_m read at roughness a + b r, its
// lobe `width` times as wide (frame.direct_metal = [a, b, width, separate], DIRECT_METAL; separate 0
// is the law before, the mixed reflectance times one compensation). Fitted to the sun's term of
// metal-sun's metals (albedo 0.24-0.97, metalness 0.66 and 1, roughness 0.30-0.79, four azimuths,
// 30-150 studs), sun-grazing's tiles seen 5-35 degrees up (black dielectrics near the mirror
// direction at n·v 0.09: 1.5-2.5 x the law before, Schlick's rise), sa-sun's and ibl-scale's spheres
// and a ship close-up's parts. `wrap` raises the diffuse term's n·l (at most 1; a SurfaceLight's
// face, `local_lights`).
fn direct_light(surface: Surface, n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>, wrap: f32) -> vec3<f32> {
	let diffuse_cos = min(dot(n, l) + wrap, 1.0);
	// no light (in the sun's shadow, outside a cone): its terms times 0
	if (diffuse_cos <= 0.0 || all(radiance == vec3<f32>(0.0))) {
		return vec3<f32>(0.0);
	}
	// plain Lambert: Studio's sun-lit tiles show no Fresnel loss on the diffuse term, and metals keep
	// it (material-swatches: a sun-lit Metal swatch, metalness 1, is 0.975 x Lambert of its albedo)
	let diffuse = surface.diffuse_weight * surface.albedo / PI * diffuse_cos;
	let n_dot_l = max(dot(n, l), 0.0);
	if (n_dot_l <= 0.0) {
		return diffuse * radiance;
	}
	let h = normalize(v + l);
	let n_dot_v = max(dot(n, v), 1e-4);
	let n_dot_h = max(dot(n, h), 0.0);
	let v_dot_h = max(dot(v, h), 0.0);
	let lobe = frame.direct_specular;
	let reflectance = frame.direct_reflectance;
	var visibility = visibility_smith(n_dot_v, n_dot_l, surface.roughness);
	let mode = u32(surface.visibility_mode);
	if (mode == 1u) {
		// uncorrelated Smith with Schlick-GGX, k = α / 2
		let k = surface.roughness * surface.roughness * 0.5;
		visibility = 0.25 / ((n_dot_v * (1.0 - k) + k) * (n_dot_l * (1.0 - k) + k));
	} else if (mode == 2u) {
		// implicit (G = n·l n·v)
		visibility = 0.25;
	} else if (mode == 3u) {
		// Kelemen
		visibility = 0.25 / max(v_dot_h * v_dot_h, 1e-4);
	} else if (mode == 4u) {
		// uncorrelated Smith with Schlick-GGX, k = k0 + k1 roughness
		let k = min(lobe.z + lobe.w * surface.roughness, 1.0);
		visibility = 0.25 / ((n_dot_v * (1.0 - k) + k) * (n_dot_l * (1.0 - k) + k));
	}
	let alpha = (lobe.x + (1.0 - lobe.x) * surface.roughness * surface.roughness) * lobe.y;
	// Schlick's rise towards 1 at grazing v·h (the material's own weight on top): on the dielectric's
	// F0 before its gain g (1 + c α²), on the metal's albedo
	let rise = pow(1.0 - v_dot_h, 5.0) * (surface.fresnel + reflectance.z);
	let gain = reflectance.x * (1.0 + reflectance.y * alpha * alpha);
	// the table at the prefiltered sky level's roughness, lod / 5 (the sky specular's Fresnel reads it
	// 0.125 levels further on)
	let lod = min(SPECULAR_LOD_BIAS + SPECULAR_LOD_SCALE * surface.roughness, frame.environment.w);
	let table = textureSampleLevel(brdf_lut, linear_sampler, vec2<f32>(lod / frame.environment.w, n_dot_v), 0.0).rg;
	let metal = frame.direct_metal;
	let scatter = 1.0 / max(table.x + table.y, 0.05) - 1.0;
	let f_d = vec3<f32>(gain * (surface.f0 + (1.0 - surface.f0) * rise));
	let d_d = distribution_ggx(n_dot_h, alpha);
	var specular = d_d * f_d * (1.0 + surface.f0 * scatter) * (1.0 - surface.metalness);
	if (surface.metalness > 0.0 || metal.w < 0.5) {
		let table_m = textureSampleLevel(brdf_lut, linear_sampler, vec2<f32>(clamp(metal.x + metal.y * surface.roughness, 0.0, 1.0), n_dot_v), 0.0).rg;
		let scatter_m = 1.0 / max(table_m.x + table_m.y, 0.05) - 1.0;
		let f_m = surface.albedo + (vec3<f32>(1.0) - surface.albedo) * rise;
		let d_m = distribution_ggx(n_dot_h, alpha * metal.z);
		if (metal.w > 0.5) {
			specular += d_m * f_m * (1.0 + reflectance.w * scatter_m) * surface.metalness;
		} else {
			specular = mix(d_d * f_d, d_m * f_m, surface.metalness) * (1.0 + mix(surface.f0, reflectance.w, surface.metalness) * scatter_m);
		}
	}
	specular *= visibility;
	return (diffuse + specular * n_dot_l) * radiance;
}

// Local lights (notes/render.md "Local lights"). No inverse square: full strength out to 0.71 ·
// Range, then (1 - smoothstep(0.71 R, R, d)) / (1 + ((d - 0.71 R) / 10.5)²), steeper the longer the
// Range (light-range: Range 4-120 within 0.005 of the plateau).
const LIGHT_FADE_START: f32 = 0.71;
const LIGHT_FADE_LENGTH: f32 = 10.5;

fn light_fade(distance: f32, range: f32) -> f32 {
	let start = LIGHT_FADE_START * range;
	let beyond = max(distance - start, 0.0) / LIGHT_FADE_LENGTH;
	return (1.0 - smoothstep(start, range, distance)) / (1.0 + beyond * beyond);
}

// SpotLight and SurfaceLight cones: t², t = (cos θ - cos(Angle / 2)) / (1 - cos(Angle / 2)) (light-spot:
// Angle 30-180)
fn light_cone(cos_angle: f32, cos_half: f32) -> f32 {
	if (cos_half >= 1.0) {
		return 0.0;
	}
	let t = saturate((cos_angle - cos_half) / (1.0 - cos_half));
	return t * t;
}

// The share of a shadowed light reaching `world_position`: its shadow face by the major axis from
// the light (a SurfaceLight's: from light.shadow.z behind its face, render::light_data), the position
// pushed off the surface by a texel and a half, 16 compares over a disk of LIGHT_SHADOW_FILTER of a
// face's 1024 texels whatever its size (light.shadow.w; light-materials g2: Studio's edge on the floor
// 8 studs from the light is ~0.22 studs from 10 to 90 %; 10 texels gave 0.34).
const LIGHT_SHADOW_TEXELS: f32 = 1024.0;
const LIGHT_SHADOW_FILTER: f32 = 6.5;

fn light_shadow(light: Light, world_position: vec3<f32>, geometric_normal: vec3<f32>) -> f32 {
	let from_light = world_position - (light.position.xyz - light.direction.xyz * light.shadow.z);
	let a = abs(from_light);
	var face = 4u + select(1u, 0u, from_light.z > 0.0);
	if (a.x >= a.y && a.x >= a.z) {
		face = select(1u, 0u, from_light.x > 0.0);
	} else if (a.y >= a.z) {
		face = 2u + select(1u, 0u, from_light.y > 0.0);
	}
	// a texel at this distance: 2 tan(50°) / size (the faces' 100° field of view), and one of 1024;
	// a narrow SurfaceLight's one face aimed down its normal, 2 tan(its half-angle) (render::light_data)
	var reach = max(a.x, max(a.y, a.z)) * 2.3835;
	if (light.extra.z > 0.0) {
		face = 0u;
		reach = max(dot(from_light, light.direction.xyz), 1e-3) * 2.0 * light.extra.z;
	}
	let texel = reach / light.shadow.w;
	let disk = reach / LIGHT_SHADOW_TEXELS * LIGHT_SHADOW_FILTER;
	let layer = u32(light.shadow.x) + face;
	// off the surface, and towards the light by as much as the surface recedes across the filter
	// disk (the slope of a surface the light grazes would shadow itself otherwise), up to tan θ 50
	// (light-graze-kinds: capped at 10, lights 0.05-0.2 studs over a floor shadowed it to 0.53-0.95
	// of itself where Studio keeps 0.66-1.03 as `light_graze` gives)
	let l = -from_light / max(length(from_light), 1e-4);
	let n_dot_l = clamp(dot(geometric_normal, l), 0.02, 1.0);
	let slope = sqrt(1.0 - n_dot_l * n_dot_l) / n_dot_l;
	let lifted = world_position + geometric_normal * (texel * 1.5) + l * (disk * slope);
	let clip = light_shadow_matrices[layer] * vec4<f32>(lifted, 1.0);
	let ndc = clip.xyz / clip.w;
	let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
	let taps = array<vec2<f32>, 16>(
		vec2<f32>(-0.94201624, -0.39906216), vec2<f32>(0.94558609, -0.76890725),
		vec2<f32>(-0.094184101, -0.92938870), vec2<f32>(0.34495938, 0.29387760),
		vec2<f32>(-0.91588581, 0.45771432), vec2<f32>(-0.81544232, -0.87912464),
		vec2<f32>(-0.38277543, 0.27676845), vec2<f32>(0.97484398, 0.75648379),
		vec2<f32>(0.44323325, -0.97511554), vec2<f32>(0.53742981, -0.47373420),
		vec2<f32>(-0.26496911, -0.41893023), vec2<f32>(0.79197514, 0.19090188),
		vec2<f32>(-0.24188840, 0.99706507), vec2<f32>(-0.81409955, 0.91437590),
		vec2<f32>(0.19984126, 0.78641367), vec2<f32>(0.14383161, -0.14100790),
	);
	var sum = 0.0;
	for (var i = 0; i < 16; i++) {
		sum += textureSampleCompareLevel(light_shadow_maps, shadow_sampler, uv + taps[i] * (LIGHT_SHADOW_FILTER / LIGHT_SHADOW_TEXELS), layer, ndc.z);
	}
	return sum / 16.0;
}

// What a shadowed light keeps of itself on a surface it grazes, Studio's own self-shadowing that the
// lifted lookup above avoids (light-graze, -graze-more, -graze-kinds, -graze-pairs: PointLights
// 0.05-4 studs over floors at Range 4-120, shadowed over unshadowed, rings at least a fifth of their
// radius in from the floor's edge, where the darkening thins): a function of u = (d / Range)^1.5 tan θ
// alone, θ from the surface's normal to the light, 1 - 0.258 x^1.84 / (1 + x^1.84), x = max(u -
// 0.59, 0) / 7.96 (rms 0.008 over 660 rings, Brightness 1-100 alike; whole to u 0.6, 0.95 at 4.4,
// 0.86 at 9, 0.78 at 20). `range` is the longest of the shadowed lights at the light's position
// (render::shared_graze_ranges). A SpotLight's u is 1.63 times a PointLight's (light-graze-kinds:
// Angle 90 and 150 along a floor 0.5 and 2 studs up, rms 0.037 -> 0.014); a SurfaceLight darkens
// nothing so (2-stud faces of Angle 180 along the floor: 0.98-0.99 of the unshadowed out to 14 studs
// 2 studs up, 0.86-0.98 from 8 to 14 with the face half under the floor, where the rule at its
// nearest point gave 0.74).
const LIGHT_GRAZE_SPOT: f32 = 1.63;

fn light_graze(range: f32, distance: f32, n_dot_l: f32, scale: f32) -> f32 {
	let c = max(n_dot_l, 1e-3);
	let u = scale * pow(distance / range, 1.5) * sqrt(max(1.0 - c * c, 0.0)) / c;
	let x = pow(max(u - 0.59, 0.0) / 7.96, 1.84);
	return 1.0 - 0.258 * x / (1.0 + x);
}

// render::light_clusters' depth slices (SLICES, FIRST_DEPTH 1 stud)
const LIGHT_CLUSTER_SLICES: f32 = 32.0;

// The cluster of the view's pixel `pixel` (in the target's pixels) at `world_position`: its tile, and
// its slice by the logarithm of its view depth, as render::light_clusters finds it (that marks a light
// one cluster further each way, for the rounding)
fn light_cluster(world_position: vec3<f32>, pixel: vec2<f32>) -> u32 {
	let grid = frame.light_clusters;
	let depth = dot(world_position - frame.camera_position.xyz, frame.light_cluster_view.xyz);
	let slice = u32(clamp(floor(log(max(depth, 1e-6)) * frame.light_cluster_view.w), 0.0, LIGHT_CLUSTER_SLICES - 1.0));
	let tile = vec2<u32>(min(floor(pixel / grid.x), grid.yz - 1.0));
	return (slice * u32(grid.z) + tile.y) * u32(grid.y) + tile.x;
}

// What one light adds at a surface point, into the light's own share (`sum`) and the far surfaces'
// voxels (`voxels`); nothing at all beyond its Range and its voxels' reach
fn local_light(index: u32, surface: Surface, world_position: vec3<f32>, n: vec3<f32>, v: vec3<f32>, geometric_normal: vec3<f32>, far: f32, read_at: vec3<f32>, sum: ptr<function, vec3<f32>>, voxels: ptr<function, vec3<f32>>) {
	let light = lights[index];
	if (far > 0.0 && light.extra.x > 0.0) {
		*voxels += voxel_light(light, read_at);
	}
	if (far >= 1.0) {
		return;
	}
	let kind = u32(light.position.w);
	var source = light.position.xyz;
	var straight = false;
	if (kind == 2u) {
		// a SurfaceLight shines from the point of its face nearest the surface (light-surface)
		let offset = world_position - light.position.xyz;
		let across = vec2<f32>(dot(offset, light.right.xyz), dot(offset, light.up.xyz));
		source += light.right.xyz * clamp(across.x, -light.right.w, light.right.w);
		source += light.up.xyz * clamp(across.y, -light.up.w, light.up.w);
		straight = abs(across.x) <= light.right.w && abs(across.y) <= light.up.w && dot(offset, light.direction.xyz) > 0.0;
	}
	let to_light = source - world_position;
	let distance = length(to_light);
	let range = light.color.w;
	if (distance >= range || distance < 1e-4) {
		return;
	}
	let l = to_light / distance;
	var strength = light_fade(distance, range);
	if (kind != 0u) {
		// straight ahead of a SurfaceLight's face the cone is whole at any Angle, 0 included (light-angle-small)
		strength *= select(light_cone(dot(-l, light.direction.xyz), light.direction.w), 1.0, straight);
	}
	if (light.shadow.x >= 0.0 && strength > 0.0) {
		strength *= light_shadow(light, world_position, geometric_normal);
		if (kind != 2u) {
			strength *= light_graze(light.extra.y, distance, dot(geometric_normal, l), select(1.0, LIGHT_GRAZE_SPOT, kind == 1u));
		}
	}
	// a SurfaceLight's diffuse n·l (to the nearest point) is raised by the face's size over the
	// distance, at most 1: r / d, r half the square root of its area (light-surface, -floor,
	// -rect: per case r 1.02-1.07 times that for faces of 1, 6 and 12 studs, 2.50-2.53 for 12 x 2
	// faces whichever way round, against 2.45; walls parallel to large faces show no n·l, small
	// ones far away nearly all of it, a floor at right angles reads n·l times the face centre's
	// height over the nearest edge's, and Angle 180 falls off beyond 55 degrees)
	var wrap = 0.0;
	if (kind == 2u) {
		wrap = sqrt(light.right.w * light.up.w) / distance;
	}
	*sum += direct_light(surface, n, v, l, light.color.xyz * strength, wrap);
}

fn local_lights(surface: Surface, world_position: vec3<f32>, n: vec3<f32>, v: vec3<f32>, geometric_normal: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
	// far from the camera the lights' voxels light surfaces instead, read 6 studs out along the
	// geometric normal whichever way the surface faces, no shadows, the diffuse alone (light-far: walls
	// and a ball lit by PointLights of Range 8-60 1,000-2,400 studs off, rms 0.015 of full strength;
	// SurfaceLights from their face's nearest point), taking over linearly from 580 to 600 studs from
	// the camera (light-far switch, straddle: halfway at 590 whatever the light's own distance). A
	// light placed with a playback never gets there (light-latency: a wall 1,000 studs off stays dark
	// 0.1-4 s after its light is placed, while particles near the camera take theirs at once; the
	// benchmark's sky dome as drawn with no light)
	let far = saturate((distance(world_position, frame.camera_position.xyz) - LOCAL_LIGHT_FAR_START) / (LOCAL_LIGHT_FAR_END - LOCAL_LIGHT_FAR_START));
	let read_at = world_position + geometric_normal * 6.0;
	var sum = vec3<f32>(0.0);
	var voxels = vec3<f32>(0.0);
	// the cluster's lights, lowest first: the view's own order; a view without clusters walks every
	// light the same way (one copy of the light's code in the shader)
	let count = frame.counts.x;
	let clustered = frame.light_clusters.w > 0.0;
	let words = select((count + 31u) / 32u, u32(frame.light_clusters.w), clustered);
	let first = select(0u, light_cluster(world_position, pixel) * words, clustered);
	for (var word = 0u; word < words; word++) {
		let left = count - word * 32u;
		var bits = select(select(0xffffffffu, (1u << left) - 1u, left < 32u), light_clusters[first + word], clustered);
		while (bits != 0u) {
			let i = word * 32u + firstTrailingBit(bits);
			bits &= bits - 1u;
			local_light(i, surface, world_position, n, v, geometric_normal, far, read_at, &sum, &voxels);
		}
	}
	return sum * (1.0 - far) + surface.diffuse_weight * surface.albedo * voxels * (VOXEL_LIGHT_SURFACES * far);
}

// ForceField (notes/render.md "ForceField"): unlit, radiance `radiance` · c² (albedo is c²), blended
// like other transparency in Roblox's gamma-2 space with an alpha of (1 - Transparency) times a glow,
// capped at 1. The glow is the larger of a rim, rising below n·v 0.42 (forcefield-angles: single
// quads turned 0-87°; the interior 0.052 also fits forcefield-blend's six backgrounds), and a
// contact glow of the same shape within 0.27 studs (view depth) of the opaque surface behind
// (forcefield-depth, -contact; forcefield-neon: a ship model's green quads at Transparency -0.5 and
// 0.5 parallel to black, gray and green Neon at gaps of 0.03-3 studs, each quad within 0.8 of 255 on
// average, where 0.3 left them 3.0 off, 14 at 0.1 studs: over bright Neon the glow's colour darkens
// what shows through). TextureIDs only show as an animated band (forcefield-time), so the rest state
// drawn here is untextured.
const FORCEFIELD_ALPHA: f32 = 0.052;
const FORCEFIELD_RIM_EDGE: f32 = 0.42;
const FORCEFIELD_CONTACT_REACH: f32 = 0.27;
const FORCEFIELD_GLOW_POWER: f32 = 1.7;

fn forcefield(albedo: vec3<f32>, radiance: f32, opacity: f32, n_dot_v: f32, gap: f32) -> vec4<f32> {
	let rim = saturate(1.0 - n_dot_v / FORCEFIELD_RIM_EDGE);
	let contact = saturate(1.0 - gap / FORCEFIELD_CONTACT_REACH);
	let glow = pow(max(rim, contact), FORCEFIELD_GLOW_POWER);
	let alpha = min(opacity * mix(FORCEFIELD_ALPHA, 1.0, glow), 1.0);
	return vec4<f32>(albedo * radiance, max(alpha, 0.0));
}

// The view-depth distance (studs) from a fragment to the opaque surface behind it: the projection is
// reversed-Z infinite, depth = near / view depth (camera_position.w is near; nothing behind reads 0)
fn depth_gap(position: vec4<f32>) -> f32 {
	let behind = textureLoad(scene_depth, vec2<i32>(position.xy), 0);
	let near = frame.camera_position.w;
	return near / max(behind, 1e-9) - near / max(position.z, 1e-9);
}

// Where a surface reads its material maps (shade_surface; the residency feedback, fs_residency): its
// material frame, the coordinate with its Organic layers (and their derivatives) and whether it goes
// through the Wang atlas, and a decal's face and own coordinate on it.
struct MaterialCoordinates {
	material: MaterialFrame,
	uv: vec2<f32>,
	organic: OrganicLayers,
	wang: bool,
	face: u32,
	bind_pose: bool,
	mesh_uv_decal: bool,
	face_only: bool,
	decal_uv: vec2<f32>,
}

fn material_coordinates(input: VertexOutput, instance: Instance, is_decal: bool) -> MaterialCoordinates {
	let flags = instance.flags.y;
	let material = material_frame(input, instance);
	let tiles = material.studs / max(instance.size.w, 1e-3);
	// plus the material's per-part offset (instance.tiling.xy on parts)
	let projected = tiles + select(vec2<f32>(0.0), instance.tiling.xy, (flags & TEX_DECAL) == 0u);
	var uv = select(input.uv, projected, (flags & TEX_MATERIAL) != 0u);
	// decals: project the part's box face onto the surface (a posed skinned mesh's from its bind pose:
	// `vs_decal_bind`)
	let face = min(instance.flags.z, 5u);
	let bind_pose = (flags & TEX_DECAL_BIND) != 0u;
	let decal_point = select(input.world_position, (instance.model * vec4<f32>(input.object_position, 1.0)).xyz, bind_pose);
	let box_position = (instance.part_inverse * vec4<f32>(decal_point, 1.0)).xyz;
	let face_u = dot(box_position, FACE_RIGHT[face]) + 0.5;
	let face_v = dot(box_position, FACE_DOWN[face]) + 0.5;
	let mesh_uv_decal = (flags & TEX_DECAL_MESH_UV) != 0u;
	let face_only = (flags & TEX_DECAL_FACE_ONLY) != 0u;
	var decal_uv = select(vec2<f32>(face_u, face_v), input.uv, mesh_uv_decal);
	if ((flags & TEX_DECAL_TILED) != 0u) {
		// studs across the face from its top-left corner (a MeshPart's: from its centre; a classic
		// head's: uv units) plus the offsets, per tile (decal-meshes: the offsets move the image up and
		// to the left)
		let width = abs(dot(instance.size.xyz, FACE_RIGHT[face]));
		let height = abs(dot(instance.size.xyz, FACE_DOWN[face]));
		var position = select(vec2<f32>(face_u * width, face_v * height), input.uv, mesh_uv_decal);
		if (face_only) {
			// a Ball's or Cylinder's face counts its studs before the push onto the round shape
			// (decal-shapes: 1-stud tiles 0.5 studs a square across a ball's middle, 0.3 at its face's rim)
			position = input.uv * vec2<f32>(width, height);
		}
		if ((flags & TEX_DECAL_CENTERED) != 0u) {
			position -= vec2<f32>(width, height) * 0.5;
		}
		decal_uv = (position + instance.tiling.zw) / max(instance.tiling.xy, vec2<f32>(1e-3));
	}
	// a Decal's UVOffset and UVScale map the face's uv to its image's: uv * UVScale + UVOffset (decals on
	// 4-stud blocks: at scale 2 the image shows at half size in the face's top-left quarter, at offset
	// 0.25 it moves left by a quarter of the face); the face's own extent still bounds it, the image's
	// edges held past its own
	var image_uv = decal_uv;
	if ((flags & TEX_DECAL_UV) != 0u) {
		image_uv = decal_uv * instance.tiling.xy + instance.tiling.zw;
	}
	uv = select(uv, image_uv, is_decal);
	let organic = organic_layers(uv, tiles, instance);
	let wang = (flags & TEX_WANG) != 0u;
	var coordinates: MaterialCoordinates;
	coordinates.material = material;
	coordinates.uv = uv;
	coordinates.organic = organic;
	coordinates.wang = wang;
	coordinates.face = face;
	coordinates.bind_pose = bind_pose;
	coordinates.mesh_uv_decal = mesh_uv_decal;
	coordinates.face_only = face_only;
	coordinates.decal_uv = decal_uv;
	return coordinates;
}

// Whether a decal leaves out this fragment of its part's surface (it draws nothing there): on a
// Ball's or Cylinder's face only where the shape's face is its own; else only where the surface
// faces the decal's side (the box face itself on blocks), and inside the face for a Decal; on a
// MeshPart only where its normal points most nearly that way in the part (its bind pose's on a posed
// skinned mesh: `vs_decal_bind`). Its image's texels decide none of it.
fn decal_left_out(input: VertexOutput, instance: Instance, coordinates: MaterialCoordinates) -> bool {
	let flags = instance.flags.y;
	let face = coordinates.face;
	if (coordinates.face_only && min(u32(round(input.color.a * 255.0)), 8u) != face) {
		return true;
	}
	if (!coordinates.mesh_uv_decal) {
		let part_normal = normalize((instance.part_inverse * vec4<f32>(normalize(input.world_normal), 0.0)).xyz);
		let outside = (flags & TEX_DECAL_TILED) == 0u && (any(coordinates.decal_uv < vec2<f32>(0.0)) || any(coordinates.decal_uv > vec2<f32>(1.0)));
		var turned_away = dot(part_normal, FACE_NORMAL[face]) <= 1e-3;
		if ((flags & TEX_DECAL_DOMINANT) != 0u) {
			let surface_normal = select(normalize(input.world_normal), (instance.normal_matrix * vec4<f32>(input.object_normal, 0.0)).xyz, coordinates.bind_pose);
			// the box's inverse divides each axis by the box's size: times it, the part's own axes
			turned_away = box_face((instance.part_inverse * vec4<f32>(surface_normal, 0.0)).xyz * instance.size.xyz) != face;
		}
		return turned_away || outside;
	}
	return false;
}

// `occluded`: opaque surfaces take the ambient occlusion (render::ao); transparent ones are not in it.
// The entry point fixes `is_decal`, so ordinary surfaces have no decal discard path and can
// reject hidden fragments before material sampling and lighting.
fn shade_surface(input: VertexOutput, front_facing: bool, occluded: bool, is_decal: bool) -> vec4<f32> {
	let instance = instances[input.instance];
	let flags = instance.flags.y;
	let shading = instance.flags.x;
	var surface: Surface;
	var base = instance.color.rgb;
	var alpha = instance.color.a;
	// in a ViewportFrame a MeshPart's vertex alpha is the image's (fs_opaque)
	if ((flags & TEX_VERTEX_ALPHA) != 0u) {
		alpha *= input.color.a;
	}
	// a MeshPart's FileMesh vertex colours multiply its colour before the decode, so its albedo by
	// their square (mesh-vertex-colors: gray 117 vertices darken lit Plastic to 0.21-0.23, Neon to
	// 0.044-0.054 = 0.21²)
	if ((flags & TEX_VERTEX_TINT) != 0u) {
		base *= input.color.rgb * input.color.rgb;
	}
	// Keep the coordinate derivatives above the material branches; maps whose values are unused
	// need no reads, even though their bind slots still hold the defaults.
	let coordinates = material_coordinates(input, instance, is_decal);
	let material = coordinates.material;
	let uv = coordinates.uv;
	let organic = coordinates.organic;
	let wang = coordinates.wang;
	// where a decal leaves its part's surface out, before any map is read there (the reads take
	// explicit gradients, so the fragments left as helpers keep their quads' derivatives)
	if (is_decal && decal_left_out(input, instance, coordinates)) {
		discard;
	}
	var raw_texel = vec4<f32>(1.0);
	if ((flags & TEX_COLOR) != 0u) {
		raw_texel = material_texel(color_map, material_info.sizes[0].xy, uv, organic, wang);
	}
	// far tiling (pre-2022 Slate, Concrete, Granite, CorrodedMetal; 2022 Granite): the colour map
	// blends into larger tiles linearly with view depth (material-far: the same weight through 15, 30
	// and 60 degree fields of view and across the frame), in stored values before decoding (it darkens
	// the swatches as measured: Granite 0.932, predicted 0.945; CorrodedMetal 0.977 / 0.975)
	let far = select(0.0, instance.texture.x, (flags & TEX_MATERIAL) != 0u);
	if (far > 0.0 && (flags & TEX_COLOR) != 0u) {
		let far_texel = studio_mips(color_map, material_info.sizes[0].xy, uv * far, organic.dx * far, organic.dy * far);
		let depth = (frame.view_proj * vec4<f32>(input.world_position, 1.0)).w;
		raw_texel = vec4<f32>(mix(raw_texel.rgb, far_texel.rgb, saturate(depth * instance.texture.y)), raw_texel.a);
	}
	// colour textures are stored raw; Roblox decodes them with a square (gamma 2.0), material colour
	// maps with the sRGB curve (material-swatches: flat-lit swatches against their packs)
	// and every colour texture lands 2.9 % darker than its decode (texture-decode: Decal, TextureID
	// and SpecialMesh textures 0.968 / 0.974 / 0.973 of ror's radiance per channel, constant over the
	// range; material-swatches: material maps 0.966-0.972)
	let texel = vec4<f32>(select(raw_texel.rgb * raw_texel.rgb, srgb_to_linear(raw_texel.rgb), (flags & TEX_SRGB) != 0u) * TEXTURE_ALBEDO_SCALE, raw_texel.a);
	var normal_sample = vec4<f32>(0.5, 0.5, 1.0, 1.0);
	if ((flags & TEX_NORMAL) != 0u) {
		normal_sample = material_texel(normal_map, material_info.sizes[0].zw, uv, organic, wang);
	}
	var normal_texel = normal_sample.xyz;
	if ((flags & (TEX_NORMAL_DXT5NM | TEX_NORMAL_RG)) != 0u) {
		let xy = select(normal_sample.rg, vec2<f32>(normal_sample.a, normal_sample.g), (flags & TEX_NORMAL_DXT5NM) != 0u) * 2.0 - 1.0;
		normal_texel = vec3<f32>(xy, sqrt(saturate(1.0 - dot(xy, xy)))) * 0.5 + 0.5;
	}
	// surface types (Plastic primitives): the face's group of the atlas, a tile per 2 x 2 studs from
	// the face's top-left corner (surfaces: map-odd2, map-rotated)
	let surface_group = select((instance.flags.z >> (4u * SURFACE_FACE[min(material.face, 8u)])) & 15u, 0u, (flags & TEX_DECAL) != 0u);
	let surface_tiles = material.studs * 0.5;
	let surface_uv = vec2<f32>(fract(surface_tiles.x), (f32(max(surface_group, 1u) - 1u) * 4.0 + 1.0 + fract(surface_tiles.y)) / SURFACE_ATLAS_TILES);
	let surface_scale = vec2<f32>(1.0, 1.0 / SURFACE_ATLAS_TILES);
	let surface_dx = dpdx(surface_tiles) * surface_scale;
	let surface_dy = dpdy(surface_tiles) * surface_scale;
	var surface_albedo_texel = 1.0;
	var surface_normal_texel = vec4<f32>(0.5);
	if (surface_group != 0u) {
		surface_albedo_texel = textureSampleGrad(surface_albedo, material_sampler, surface_uv, surface_dx, surface_dy).r;
		surface_normal_texel = textureSampleGrad(surface_normal, material_sampler, surface_uv, surface_dx, surface_dy);
	}
	var metalness_texel = 0.0;
	if ((flags & TEX_METALNESS) != 0u) {
		metalness_texel = material_texel(metalness_map, material_info.sizes[1].xy, uv, organic, wang).r;
	}
	var roughness_sample = vec4<f32>(1.0);
	if ((flags & TEX_ROUGHNESS) != 0u) {
		roughness_sample = material_texel(roughness_map, material_info.sizes[1].zw, uv, organic, wang);
	}
	let roughness_texel = select(roughness_sample.r, roughness_sample.g, (flags & TEX_ROUGHNESS_GREEN) != 0u);
	var emissive_sample = vec4<f32>(0.0);
	if ((flags & TEX_EMISSIVE) != 0u) {
		emissive_sample = material_texel(emissive_map, material_info.sizes[2].xy, uv, organic, wang);
	}
	var emissive_texel = select(emissive_sample.rgb, emissive_sample.bbb, (flags & TEX_EMISSIVE_BLUE) != 0u);
	if ((flags & TEX_EMISSIVE_RED) != 0u) {
		emissive_texel = emissive_sample.rrr;
	}
	// a union with UsePartColor off shows its source parts' colours, decoded with a square, instead of
	// the part's (union-sources "chair-partcolor-off-red", "suntemple")
	if ((flags & TEX_VERTEX_COLOR) != 0u) {
		base = input.color.rgb * input.color.rgb;
	}
	if ((flags & TEX_COLOR) != 0u) {
		if ((flags & TEX_ALPHA_OVERLAY) != 0u) {
			// the part colour and the tinted map mixed by the map's alpha in Roblox's gamma-2 colour
			// space, then squared (particle-sa-blend neon-overlay: a dark gray part under the cones'
			// orange-tinted map, every tenth of alpha within 1 of 255 of Studio's, where the linear mix read
			// up to 15 over; overlay-light: a light gray part under it tinted dark blue)
			let mixed = mix(sqrt(base), sqrt(texel.rgb * instance.tint.rgb), texel.a);
			base = mixed * mixed;
		} else if ((flags & TEX_ALPHA_TINTMASK) != 0u && (flags & TEX_MATERIAL) != 0u) {
			// built-in packs: the part colour mixed towards white by the alpha in Roblox's gamma-2
			// colour space, then squared (material-tint: black, gray and blue Brick, CeramicTiles,
			// Carpet and WoodPlanks per pixel, rms 0.0095 against 0.071 for a linear mix)
			let tint = mix(vec3<f32>(1.0), sqrt(base), texel.a);
			base = texel.rgb * tint * tint;
		} else if ((flags & TEX_ALPHA_TINTMASK) != 0u) {
			base = texel.rgb * mix(vec3<f32>(1.0), instance.color.rgb, texel.a);
		} else {
			base *= texel.rgb;
			// a built-in pack's alpha is never coverage (Glass's colour map is 0.002 on average, and
			// Studio's glass panes show at every Transparency; material-glass)
			alpha *= select(texel.a, 1.0, (flags & TEX_MATERIAL) != 0u);
		}
	}
	if (surface_group != 0u) {
		// decoded with a square, relative to the tile's flat background (128): stud tops 1.14x,
		// outlines 0.69x, Inlet floors 0.41x, whatever the part colour (surfaces: flat runs)
		let tile = surface_albedo_texel * (255.0 / 128.0);
		base *= tile * tile;
	}
	surface.albedo = base;
	surface.alpha = alpha;
	surface.roughness = instance.params.x;
	surface.metalness = instance.params.y;
	if ((flags & TEX_ROUGHNESS) != 0u) {
		surface.roughness = roughness_texel;
	}
	if ((flags & TEX_METALNESS) != 0u) {
		surface.metalness = metalness_texel;
	}
	surface.roughness = clamp(surface.roughness, 0.03, 1.0);
	surface.f0 = instance.specular.x;
	surface.fresnel = instance.specular.y;
	surface.visibility_mode = instance.specular.z;
	// EnvironmentSpecularScale s makes metals metallic: a surface is metalness · s metallic times the
	// fade, for the sun's and the sky's specular and for its diffuse (ibl-scale at 150 studs, where the
	// fade leaves 0.88: a white metal's sky specular grows as s (0.04 + 0.96 · 0.88 s), metals keep 12 %
	// of their diffuse at s = 1; ibl-fade: none at 10 studs, 45 % at 700)
	let specular_scale = frame.environment.y;
	let fade = specular_fade(input.world_position);
	let map_metalness = surface.metalness;
	surface.metalness = fade.y * map_metalness * specular_scale;
	// the sky specular's level: s and the fade
	let sky_specular = fade.x * specular_scale;
	let f0 = mix(vec3<f32>(surface.f0), surface.albedo, surface.metalness);
	// the sun's and local lights' diffuse gives up what the sky specular reflects head-on (ibl-scale:
	// 1 - 0.034 s on dielectrics at every n·v; the sky's diffuse below follows its Fresnel instead)
	surface.diffuse_weight = (1.0 - surface.metalness) * (vec3<f32>(1.0) - sky_specular * f0);

	var n = normalize(input.world_normal);
	if (!front_facing) {
		n = -n;
	}
	if ((flags & TEX_NORMAL) != 0u || surface_group != 0u) {
		var t = normalize(input.world_tangent.xyz - n * dot(n, input.world_tangent.xyz));
		var b = cross(n, t) * input.world_tangent.w;
		if (((flags & TEX_MATERIAL) != 0u || surface_group != 0u) && dot(material.u_axis, material.u_axis) > 0.5) {
			// projected material textures and surface tiles (laid in the same face frame, also on a Plastic
			// without textures): the projection face's axes, onto the surface
			let u_world = (instance.model * vec4<f32>(material.u_axis, 0.0)).xyz;
			let up_world = (instance.model * vec4<f32>(material.up_axis, 0.0)).xyz;
			t = normalize(u_world - n * dot(n, u_world) + vec3<f32>(1e-6, 0.0, 0.0));
			b = cross(n, t) * select(-1.0, 1.0, dot(cross(n, t), up_world) >= 0.0);
		}
		var sample = select(vec3<f32>(0.0, 0.0, 1.0), normal_texel * 2.0 - 1.0, (flags & TEX_NORMAL) != 0u);
		if ((flags & TEX_MATERIAL) != 0u) {
			sample = vec3<f32>(sample.xy * instance.specular.w, sample.z);
		}
		if (surface_group != 0u) {
			// the tile's relief over the material's (whiteout blend)
			let relief = vec2<f32>(surface_normal_texel.a, surface_normal_texel.g) * 2.0 - 1.0;
			sample = vec3<f32>(sample.xy + relief, sample.z * sqrt(saturate(1.0 - dot(relief, relief))));
		}
		n = normalize(t * sample.x + b * sample.y + n * sample.z);
	}
	surface.normal = n;

	if (shading == SHADING_NEON) {
		// 25 · c⁴ per channel (albedo is c²; neon-colors), capped with every surface in `shade`
		return vec4<f32>(surface.albedo * surface.albedo * instance.params.w, alpha);
	}
	if (shading == SHADING_UNLIT) {
		return vec4<f32>(surface.albedo, alpha);
	}
	if (shading == SHADING_FORCEFIELD) {
		// both faces draw; the interpolated normal against the view ray (instance.color.a holds 1 -
		// Transparency unclamped)
		let view = normalize(frame.camera_position.xyz - input.world_position);
		return forcefield(surface.albedo, instance.params.w, alpha, abs(dot(normalize(input.world_normal), view)), depth_gap(input.clip));
	}

	let v = normalize(frame.camera_position.xyz - input.world_position);
	// the geometric normal of the side drawn (double-sided meshes show both)
	let geometric_normal = normalize(input.world_normal) * select(-1.0, 1.0, front_facing);
	let shadow = shadow_factor(input.world_position, geometric_normal) * cloud_sun(input.world_position);
	var color = direct_light(surface, n, v, frame.sun_direction.xyz, frame.sun_color.rgb * shadow, 0.0);
	color += local_lights(surface, input.world_position, n, v, geometric_normal, input.clip.xy);

	// ambient: the flat Ambient / OutdoorAmbient terms plus the sky's irradiance
	let n_dot_v = max(dot(n, v), 1e-4);
	// flat ambient: no Fresnel or view dependence (ambient-cubes), the full albedo on metals whatever
	// the sky specular (material-swatches; ibl-scale: what it adds is the same at every s).
	// OutdoorAmbient only as far as the light grid sees the sky (Ambient everywhere: lightgrid)
	let sky_open = light_grid_visibility(input.world_position, geometric_normal);
	let flat_ambient = max(frame.ambient.rgb, frame.outdoor_ambient.rgb * (sky_open * frame.light_grid_scale.w));
	// Studio's screen-space ambient occlusion darkens the flat ambient and the sky's diffuse light,
	// not the sun or local lights (ao-gap: identical profiles under Ambient, OutdoorAmbient and the
	// sky's light, none under the sun or a PointLight), nor Glass, which occludes what is behind it at
	// Transparency 0 but takes none itself (ao-sees: a bar half a stud in front of a Glass tile leaves
	// it at 1.01 of its unoccluded light where ror's halo dimmed it to 0.76)
	var occlusion = 1.0;
	if (occluded && shading != SHADING_GLASS) {
		occlusion = textureSampleLevel(ambient_occlusion, linear_sampler, input.clip.xy * frame.viewport.zw, 0.0).r;
	}
	color += surface.albedo * flat_ambient * occlusion;
	// the sky specular keeps 0.12 of itself where the grid is closed; with GlobalShadows on the open grid
	// leaves it what it leaves OutdoorAmbient (ibl-near: spheres of every kind from 3 and 150 studs read
	// 0.989 of their sky specular with GlobalShadows on, with and without a Sky; 0.12 + 0.88 · 0.986 =
	// 0.988)
	let sky_reach = LIGHT_GRID_SPECULAR_FLOOR + (1.0 - LIGHT_GRID_SPECULAR_FLOOR) * sky_open * frame.light_grid_scale.w;
	// the sky specular's Fresnel (ibl-spheres): the split sum with Studio's table normalised by its
	// own energy, (F0 A + B) / (A + B): no loss with roughness (a white metal reflects the same at every
	// n·v and roughness), the Fresnel rise of dielectrics fading with roughness. The table is read at
	// 0.1 + 0.9 roughness, a little past the prefiltered level's roughness lod / 5 (ibl-spheres' white
	// dielectric spheres by n·v, Studio / ror at grazing angles (n·v 0.05-0.2): 0.92-0.98 at lod / 5,
	// 0.975-1.005 here, head-on 1.01-1.03 either way; Plastic and SmoothPlastic balls 1.00-1.035; the
	// map's own roughness rises 14-17 % more at grazing angles still)
	let lod = min(SPECULAR_LOD_BIAS + SPECULAR_LOD_SCALE * surface.roughness, frame.environment.w);
	let brdf = textureSampleLevel(brdf_lut, linear_sampler, vec2<f32>((lod + SKY_FRESNEL_LOD_OFFSET) / frame.environment.w, n_dot_v), 0.0).rg;
	let sky_fresnel = (f0 * brdf.x + brdf.y) / max(brdf.x + brdf.y, 1e-4);
	// sky light: the sky's ambient cube (common.wgsl `sky_light`), giving up what the sky specular
	// reflects (sky-light; ibl-scale, per n·v); towards the closed grid's grey linearly in the grid's
	// visibility (lg-edge: Studio's light across a roof's edge is grey + (sky - grey) v, rms 0.002-0.005)
	let sky_irradiance = mix(LIGHT_GRID_SKY_CLOSED, sky_light(frame, n), sky_open);
	let sky_kd = (1.0 - surface.metalness) * (vec3<f32>(1.0) - sky_specular * sky_fresnel);
	let sky_diffuse = sky_kd * surface.albedo * sky_irradiance * (frame.environment.x * occlusion);
	color += sky_diffuse;

	let r = reflect(-v, n);
	// Part.Reflectance (reflectance): a mirror image of the sky lit by the diffuse light there, over
	// the rest of the shading; it does not need the sky specular, and metals show none
	let reflectance = instance.params.z * (1.0 - map_metalness);
	if (reflectance > 0.0) {
		let mirror = textureSampleLevel(environment_map, linear_sampler, r, SPECULAR_LOD_BIAS).rgb;
		let sun_light = frame.sun_color.rgb * shadow * max(dot(n, frame.sun_direction.xyz), 0.0) / PI;
		let light = 0.5 * flat_ambient + 0.6 * sky_irradiance * frame.environment.x + 0.36 * sun_light;
		color = mix(color, mirror * light, reflectance);
	}
	// the sky specular: the prefiltered sky (render::environment::prefilter_specular) times its Fresnel,
	// s and the fade (ibl-spheres at 150 studs: a white mirror metal shows the default sky at 0.76, a
	// dielectric head-on at 0.034), a little less blue than the sky (frame.sky_specular_colour); the
	// closed grid leaves 0.12 of it (lg-terms)
	let prefiltered = textureSampleLevel(environment_map, linear_sampler, r, lod).rgb;
	color += prefiltered * frame.sky_specular_colour.rgb * sky_fresnel * sky_specular * sky_reach;
	if (shading == SHADING_GLASS) {
		var bare = surface;
		bare.albedo = vec3<f32>(0.0);
		glass_normal = n;
		glass_facing = n_dot_v;
		glass_sky = prefiltered * sky_specular * sky_reach;
		glass_sky_diffuse = sky_diffuse;
		glass_glint = direct_light(bare, n, v, frame.sun_direction.xyz, frame.sun_color.rgb * shadow, 0.0) + local_lights(bare, input.world_position, n, v, geometric_normal, input.clip.xy);
	}
	color += surface.albedo * instance.params.w;
	if ((flags & TEX_EMISSIVE) != 0u) {
		// an emissive mask glows in the part colour (instance.material, with the strength) times the
		// colour map (as decoded, before the textures' 2.9 % loss) times the mask as stored
		// (variant-emission: uniform masks of 76, 130 and 255 emit 0.299, 0.506 and 1 of the full mask; a
		// colour map of 130 cuts it to 0.223; CrackedLava's glow in material-glass); in AlphaMode
		// Transparency the map's alpha fades it with the surface (sa-emission-sweep)
		var colour_map = select(vec3<f32>(1.0), texel.rgb / TEXTURE_ALBEDO_SCALE, (flags & TEX_COLOR) != 0u);
		// in AlphaMode Overlay the glow takes the overlaid colour: the part colour mixed with the map
		// in Color by the map's alpha (sa-emission-sweep: the portal atlas over black, gray and white
		// parts glows 12 / 42 / 67 of 255 in red, where ror's map alone read 18-21)
		if ((flags & (TEX_COLOR | TEX_ALPHA_OVERLAY)) == (TEX_COLOR | TEX_ALPHA_OVERLAY)) {
			let overlaid = mix(sqrt(instance.color.rgb), sqrt(colour_map * instance.tint.rgb), texel.a);
			colour_map = overlaid * overlaid;
		}
		color += instance.material.rgb * colour_map * emissive_texel;
	}
	return vec4<f32>(color, alpha);
}

// A surface's colour as seen through the Atmosphere or the fog (Neon and unlit surfaces too:
// atmosphere-explore's Neon tiles fade like any part), at most EXPOSED_CAP once exposed, before it
// blends or the MSAA resolve averages it (transparent Neon at 7.84: neon-transparency)
fn shade(input: VertexOutput, front_facing: bool, occluded: bool, is_decal: bool) -> vec4<f32> {
	var shaded = shade_surface(input, front_facing, occluded, is_decal);
	if (frame.atmosphere.w > 0.0) {
		shaded = vec4<f32>(atmosphere_fog(frame, faded_map, linear_sampler, input.world_position, shaded.rgb, 0.0), shaded.a);
	} else if (frame.fog.z > 0.5) {
		shaded = vec4<f32>(linear_fog(frame, input.world_position, shaded.rgb, 0.0), shaded.a);
	}
	return vec4<f32>(min(shaded.rgb, vec3<f32>(EXPOSED_CAP / frame.environment.z)), shaded.a);
}

// The scene target holds √x (Roblox blends in its gamma-2 space: a transparent surface over a
// background gives (a·√src + (1-a)·√dst)², neon-transparency); the resolve squares it back.
// In a ViewportFrame a MeshPart's FileMesh vertex alpha is the image's, premultiplied (render::guis
// windows; an effect's masks, Studio captures of its rift: geometry of vertex alpha 0 is a
// hole through the frame's own background that still hides what lies behind it there, its decals
// drawn over it).
@fragment
fn fs_opaque(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
	let shaded = shade(input, front_facing, true, false);
	let alpha = select(1.0, input.color.a, (instances[input.instance].flags.y & TEX_VERTEX_ALPHA) != 0u);
	return vec4<f32>(sqrt(max(shaded.rgb, vec3<f32>(0.0))) * alpha, alpha);
}

// a decal's opacity, 1 - Transparency times its texel's alpha, stops at 1 (vfx-decal-negative)
@fragment
fn fs_decal(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
	let shaded = shade(input, front_facing, true, true);
	let alpha = saturate(shaded.a);
	return vec4<f32>(sqrt(max(shaded.rgb, vec3<f32>(0.0))) * alpha, alpha);
}

@fragment
fn fs_transparent(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
	let shaded = shade(input, front_facing, false, false);
	// premultiplied alpha, in √x
	return vec4<f32>(sqrt(max(shaded.rgb, vec3<f32>(0.0))) * shaded.a, shaded.a);
}

// Glass shows the opaque scene through it, moved against its normal: sampled at P - k cos^p(θ) n⊥,
// n⊥ the shading normal's part across the view's axis and θ its angle to that axis, at the glass's
// depth (glass-refraction-law: Balls 3, 6 and 12 studs across magnify what is behind them alike
// whatever their size, distance and how far behind the background is; per ring of the ball k 0.49
// at its centre falling to 0.40 at n⊥ 0.93 at Transparency 0.7, fitted k = 1.7 (1 - T), p = 0.25
// over T 0.6 / 0.7 / 0.8; panes turned 15, 35 and 55 degrees shift it by that times the sine of the
// turn; an inside-out sphere mesh draws its far half, whose inward normals shrink it). Past
// Transparency 1 (a Highlight keeps such Glass drawn, scene::extract) it pushes what is behind
// outward, k = -2.36 (T - 1) min(1 + 0.32 (T - 1), sqrt(1 + 0.95 (T - 1))), p = 0.3 + 0.3 (T - 1)
// up to T 4 and 0.15 more per unit of T beyond, at most 3.5, θ then the normal's angle to the
// direction to the camera (glass-highlight: k -0.59 / -1.30 / -3.03 / -7.74 / -21, p 0.35 / 0.40 /
// 0.55 / 0.95 / 1.5 at T 1.25 / 1.5 / 2 / 3 / 5; glass-bubble-law, -mesh, -geometry and -scale,
// Balls and an inside-out bubble sphere before coded stripes: k 6.8 / 20 / 43 / 58 / 200 / 370 /
// 930 at T 3 / 5 / 7 / 10 / 15 / 30 / 55, a 6- and a 26-stud sphere's centres magnifying -20 and
// -4.1 times at T 10, k 65 both; the 26-stud one's push still falls past 90 degrees from the axis,
// to nothing at its silhouette, 17.6 / 13.0 / 7.7 / 2.5 studs at n·v 0.51 / 0.36 / 0.23 / 0.07).
// Each screen coordinate of the sample on its own: one off the screen takes the pixel's own; under
// the bubbles' law (past T 4) one on it is held within GLASS_REACH of the point GLASS_REACH_CENTRE
// of the way from the screen's centre to the pixel (glass-bubble-law and -mesh: the runs across
// fall back where those up and down do, turned by 90 degrees; glass-quirks' balls at T 3 closer
// without it; the inside-out sphere at T 5 lands 75 px of its middle on the one stripe 248-293 px
// out, 158-203 up and down; glass-bubble-scale's 26-stud sphere at T 10 at most 238-279 px out at
// 240 px from the centre, 279-320 at 140): so such bubbles show Studio's rings, the held samples
// sweeping to the window's edge between the rim and the middle, whose samples leave the screen. A
// sample on an opaque surface in front of the glass takes the pixel's own (glass-sides: the black
// ball inside an inside-out sphere never shows through its far half; the bubble attacks: the floor
// nearer than the far half where its samples pushed down land). Transparent parts, particles, beams
// and other Glass behind it do not show (glass-compose: each hidden on the half behind the pane, an
// opaque Neon block seen whole), so it replaces what is behind it.
const GLASS_REFRACTION: f32 = 1.7;
const GLASS_REFRACTION_FALLOFF: f32 = 0.25;
const GLASS_PAST: f32 = 2.36;
const GLASS_PAST_LOW_GROWTH: f32 = 0.32;
const GLASS_PAST_GROWTH: f32 = 0.95;
const GLASS_PAST_FALLOFF_FROM: f32 = 0.3;
const GLASS_PAST_LOW_FALLOFF: f32 = 0.3;
const GLASS_PAST_JOIN: f32 = 3.0;
const GLASS_PAST_FALLOFF: f32 = 0.15;
const GLASS_PAST_FALLOFF_MAX: f32 = 3.5;
const GLASS_REACH: f32 = 0.18;
const GLASS_REACH_CENTRE: f32 = 0.25;

// From T = 0.5 up Glass multiplies the radiance behind by mix(c, 1, 2T - 1) per channel, c its
// colour (0..1 as stored), and adds no surface (material-glass: 5-stud panes over coloured stripes,
// the stripes seen through each pane against the same stripes beside it; white 0.99 x at 0.5, 0.7
// and 0.9; red's green 0.00 / 0.39 / 0.78 x; blue (13, 105, 172) at 0.5 0.05 / 0.40 / 0.66 x), and
// reflects the sky over that, mixed in by F = (1 - T²) 0.97 (1 - n·v)², none head-on
// (glass-fresnel: balls over a gray Neon panel pass 0.96 / 0.91 / 0.83 / 0.72 / 0.59 of it at r / R
// 0.6 / 0.75 / 0.85 / 0.925 / 0.97 at T 0.5, losses 1 - T² scales to T 0.8, fit rms 0.005 over T
// 0.5-0.9; over a black wall they add what that share of the sky gives), with the sun's and local
// lights' glint at full strength whatever T (the glint the same size at T 0.5 to 1; an effect's rings
// are a PointLight's glint on eight nested spheres). Below 0.5 it blends linearly: out = (1 - 2T)²
// s + 2T (1 - F) c d + 4T (1 - T) F sky + (1 - (1 - 2T)²) glint, s the surface as if opaque with
// the sky's diffuse light on it cut by a further 1 - 2T, d what is behind (glass-low: facing panes
// over gray Neon and black pass 0.200 / 0.407 / 0.604 / 0.806 / 0.903 of what is behind at T 0.1 /
// 0.2 / 0.3 / 0.4 / 0.45, 2T, balls 2T (1 - F) ring by ring, reflecting 0.4-1.2 F of the sky at
// their rims; glass-low-light: the opaque pane's light shows by 0.641 / 0.360 / 0.161 / 0.040 /
// 0.012 under flat Ambient, the sun or OutdoorAmbient alone, (1 - 2T)², and by 0.513 / 0.217 /
// 0.064 / 0.007 / 0.003 under the sky's diffuse light alone, (1 - 2T)³; material-glass's stripes
// through its T 0.3 pane in flat light 0.597 behind + 0.161 s, rms 0.0006 over six stripe colours,
// linear and not in √ space). Past Transparency 1 the tint stays white and the sky's reflection off
// (glass-quirks: balls at T 1 / 1.5 / 3 with disabled Highlights show only the glint over a black
// wall). The target holds √x.
const GLASS_FRESNEL: f32 = 0.97;

@fragment
fn fs_glass(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
	let instance = instances[input.instance];
	let t = 1.0 - instance.color.a;
	// shading the surface leaves its normal, reflection and glint (glass_normal and the rest)
	let surface = shade(input, front_facing, false, false);
	// the view's axis: the w row of the view-projection
	let forward = normalize(vec3<f32>(frame.view_proj[0].w, frame.view_proj[1].w, frame.view_proj[2].w));
	let across = glass_normal - forward * dot(glass_normal, forward);
	let past = max(t - 1.0, 0.0);
	var strength = GLASS_REFRACTION * (1.0 - t);
	var falloff = GLASS_REFRACTION_FALLOFF;
	if (t > 1.0) {
		// glass-highlight's law up to T 4, where the bubbles' meets it
		strength = -GLASS_PAST * past * min(1.0 + GLASS_PAST_LOW_GROWTH * past, sqrt(1.0 + GLASS_PAST_GROWTH * past));
		let low = GLASS_PAST_FALLOFF_FROM + GLASS_PAST_LOW_FALLOFF * min(past, GLASS_PAST_JOIN);
		falloff = min(low + GLASS_PAST_FALLOFF * max(past - GLASS_PAST_JOIN, 0.0), GLASS_PAST_FALLOFF_MAX);
	}
	// past Transparency 1 the falloff follows n·v, the normal against the direction to the camera
	let facing = select(abs(dot(glass_normal, forward)), glass_facing, t > 1.0);
	let moved = frame.view_proj * vec4<f32>(input.world_position - across * (strength * pow(max(facing, 1e-4), falloff)), 1.0);
	let size = vec2<f32>(textureDimensions(scene_color));
	let own = input.clip.xy / size;
	var uv = vec2<f32>(0.5, -0.5) * (moved.xy / max(moved.w, 1e-6)) + vec2<f32>(0.5);
	// each screen coordinate that leaves the screen takes the pixel's own, the other kept; under the
	// bubbles' law (past T 4) one on it is held within GLASS_REACH of the point a quarter of the way
	// from the screen's centre to the pixel
	let off = (uv < vec2<f32>(0.0)) | (uv > vec2<f32>(1.0));
	if (past > GLASS_PAST_JOIN) {
		let centre = vec2<f32>(0.5) + GLASS_REACH_CENTRE * (own - vec2<f32>(0.5));
		uv = clamp(uv, centre - vec2<f32>(GLASS_REACH), centre + vec2<f32>(GLASS_REACH));
	}
	uv = select(uv, own, off);
	if (moved.w <= 0.0 || textureLoad(scene_depth, vec2<i32>(min(uv * size, size - 1.0)), 0) > input.clip.z) {
		uv = own;
	}
	let behind = textureSampleLevel(scene_color, linear_sampler, uv, 0.0);
	let d = behind.rgb * behind.rgb;
	let c = sqrt(instance.color.rgb);
	let grazing = 1.0 - glass_facing;
	let fresnel = GLASS_FRESNEL * grazing * grazing * max(1.0 - t * t, 0.0);
	var color: vec3<f32>;
	if (t >= 0.5) {
		let tint = mix(c, vec3<f32>(1.0), saturate(2.0 * t - 1.0));
		color = mix(tint * d, glass_sky, fresnel) + glass_glint;
	} else {
		let cover = (1.0 - 2.0 * t) * (1.0 - 2.0 * t);
		let shown = cover * max(surface.rgb - 2.0 * t * glass_sky_diffuse, vec3<f32>(0.0));
		color = shown + 2.0 * t * (1.0 - fresnel) * c * d + 4.0 * t * (1.0 - t) * fresnel * glass_sky + (1.0 - cover) * glass_glint;
	}
	// at most EXPOSED_CAP once exposed, as every surface (`shade`)
	color = clamp(color, vec3<f32>(0.0), vec3<f32>(EXPOSED_CAP / frame.environment.z));
	return vec4<f32>(sqrt(color), max(behind.a, saturate(1.0 - 2.0 * t)));
}

// A part at Transparency 0 whose SurfaceAppearance is in AlphaMode Transparency covers round(4a) of
// the 4 samples of each pixel its texel of alpha a falls in, opaque and writing depth there, so what
// lies behind shows in the other samples whatever the order it is drawn in (particle-sa-coverage: the
// light through such a quad steps at alpha 0.125, 0.375, 0.625 and 0.875 and is flat in between, no
// dither as the hardware's alpha to coverage gives)
struct CoverageOutput {
	@location(0) color: vec4<f32>,
	@builtin(sample_mask) mask: u32,
}

fn coverage_samples(alpha: f32) -> u32 {
	return u32(clamp(floor(alpha * 4.0 + 0.5), 0.0, 4.0));
}

@fragment
fn fs_coverage(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> CoverageOutput {
	let shaded = shade(input, front_facing, false, false);
	let samples = coverage_samples(shaded.a);
	if (samples == 0u) {
		discard;
	}
	var output: CoverageOutput;
	output.color = vec4<f32>(sqrt(max(shaded.rgb, vec3<f32>(0.0))), 1.0);
	output.mask = (1u << samples) - 1u;
	return output;
}

// The scene depth prepass for those parts, where soft particles fade: where they cover 2 or more of
// the 4 samples (particle-sa-coverage: behind a quad's texels on 1 sample a particle fades by the
// panel behind it as it does alone, on 2 or 3 by its distance to the quad)
@fragment
fn fs_coverage_prepass(input: VertexOutput, @builtin(front_facing) front_facing: bool) {
	let shaded = shade(input, front_facing, false, false);
	if (coverage_samples(shaded.a) < 2u) {
		discard;
	}
}

// Depth alone (no colour) where a transparent part is mostly opaque, drawn right after its colour:
// what is drawn after it stays behind those texels (notes/render.md "Particles", occlusion)
// The scene depth prepass for parts that write depth only where their texels are opaque
@fragment
fn fs_cutout_prepass(input: VertexOutput, @builtin(front_facing) front_facing: bool) {
	let shaded = shade(input, front_facing, false, false);
	if (shaded.a < 0.5) {
		discard;
	}
}

@fragment
fn fs_cutout_depth(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
	let shaded = shade(input, front_facing, false, false);
	if (shaded.a < 0.5) {
		discard;
	}
	return vec4<f32>(0.0);
}

// The ambient occlusion's input: opaque surfaces' world position and normal at the AO resolution
// (render::ao)
// the surface's depth (ao.wgsl makes its position from it) and normal
struct AoPrepassOutput {
	@location(0) depth: vec4<f32>,
	@location(1) normal: vec4<f32>,
}

fn ao_prepass_output(input: VertexOutput, front_facing: bool) -> AoPrepassOutput {
	var output: AoPrepassOutput;
	output.depth = vec4<f32>(input.clip.z, 0.0, 0.0, 0.0);
	var normal = normalize(input.world_normal) * select(-1.0, 1.0, front_facing);
	if (AO_FACE_NORMALS) {
		// the triangle's own normal, towards the camera
		let face = normalize(cross(dpdx(input.world_position), dpdy(input.world_position)));
		normal = face * select(-1.0, 1.0, dot(face, frame.camera_position.xyz - input.world_position) > 0.0);
	}
	output.normal = vec4<f32>(normal, 1.0);
	return output;
}

@fragment
fn fs_ao_prepass(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> AoPrepassOutput {
	return ao_prepass_output(input, front_facing);
}

// The same into multisampled targets: the normal (the positions come from the depth, ao.wgsl
// `fs_ao_resolve`)
@fragment
fn fs_ao_prepass_samples(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
	return ao_prepass_output(input, front_facing).normal;
}

@fragment
fn fs_ao_prepass_coverage_samples(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
	let output = ao_prepass_output(input, front_facing);
	let shaded = shade(input, front_facing, false, false);
	if (coverage_samples(shaded.a) < 4u) {
		discard;
	}
	return output.normal;
}

const AO_FACE_NORMALS: bool = true;

// The parts drawn by coverage, where they cover all 4 samples: Studio's occlusion leaves out texels
// covering 3 or fewer (particle-sa-coverage cover-white: around the cone map's blobs of alpha 0.44-0.51,
// none of it above 0.875, the panel 3 studs behind darkens 0.0-0.2 of 255 beyond Studio's panel offset,
// where taking the texels on 3 samples darkens it 3-5; ror's rings beside the dome map's opaque blobs
// then darken 10-14 against Studio's 13-18)
@fragment
fn fs_ao_prepass_coverage(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> AoPrepassOutput {
	let output = ao_prepass_output(input, front_facing);
	let shaded = shade(input, front_facing, false, false);
	if (coverage_samples(shaded.a) < 4u) {
		discard;
	}
	return output;
}

// Texture residency (render::residency): before a view draws, its surfaces drawn once more as its
// passes rasterise them, each fragment keeping per material map the finest whole level studio_mips
// picks there (its LOD's floor, 0 at or below 0), as 32 minus it, the most of all fragments (0: no
// fragment samples the map), at the map's slot in this buffer (MaterialInfo.ids; 0: a map always held
// whole)
@group(2) @binding(0) var<storage, read_write> residency: array<atomic<u32>>;

fn record_level(slot: u32, size: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) {
	if (slot == 0u) {
		return;
	}
	// studio_mips' LOD
	let x = length(dx * size);
	let y = length(dy * size);
	let major = max(x, y);
	let minor = max(min(x, y), 1e-8);
	let lod = log2(max(major / min(major / minor, 16.0), 1e-8));
	// level 0 too where the LOD is no number
	let level = select(0u, u32(min(floor(lod), 31.0)), lod > 0.0);
	let code = 32u - level;
	if (atomicLoad(&residency[slot]) < code) {
		atomicMax(&residency[slot], code);
	}
}

// What shade_surface reads of each map, under the same flags: the five maps through material_texel
// with the unshifted coordinate's derivatives, the colour map also at its far tiling; none where a
// decal leaves its part's surface out (what it reads there is discarded)
fn record_surface(input: VertexOutput, is_decal: bool) {
	let instance = instances[input.instance];
	let flags = instance.flags.y;
	let coordinates = material_coordinates(input, instance, is_decal);
	if (is_decal && decal_left_out(input, instance, coordinates)) {
		return;
	}
	let dx = coordinates.organic.dx;
	let dy = coordinates.organic.dy;
	if ((flags & TEX_COLOR) != 0u) {
		record_level(material_info.ids[0].x, material_info.sizes[0].xy, dx, dy);
		let far = select(0.0, instance.texture.x, (flags & TEX_MATERIAL) != 0u);
		if (far > 0.0) {
			record_level(material_info.ids[0].x, material_info.sizes[0].xy, dx * far, dy * far);
		}
	}
	if ((flags & TEX_NORMAL) != 0u) {
		record_level(material_info.ids[0].y, material_info.sizes[0].zw, dx, dy);
	}
	if ((flags & TEX_METALNESS) != 0u) {
		record_level(material_info.ids[0].z, material_info.sizes[1].xy, dx, dy);
	}
	if ((flags & TEX_ROUGHNESS) != 0u) {
		record_level(material_info.ids[0].w, material_info.sizes[1].zw, dx, dy);
	}
	if ((flags & TEX_EMISSIVE) != 0u) {
		record_level(material_info.ids[1].x, material_info.sizes[2].xy, dx, dy);
	}
}

@fragment
fn fs_residency(input: VertexOutput) {
	record_surface(input, false);
}

@fragment
fn fs_residency_decal(input: VertexOutput) {
	record_surface(input, true);
}

// Shadow map pass: depth only, from the sun.
@vertex
fn vs_shadow(input: VertexInput, @builtin(instance_index) instance_index: u32) -> @builtin(position) vec4<f32> {
	let instance = instances[instance_index];
	let world = instance.model * vec4<f32>(input.position, 1.0);
	return frame.shadow_view_proj * world;
}
