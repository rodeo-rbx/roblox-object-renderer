// Shared declarations: the per-view frame uniform, per-draw instance data, lights, colour helpers.

struct Frame {
	view_proj: mat4x4<f32>,
	inv_view_proj: mat4x4<f32>,
	shadow_view_proj: mat4x4<f32>,
	// xyz camera position, w unused
	camera_position: vec4<f32>,
	// xyz direction towards the sun, w = 1 when sun shadows are on
	sun_direction: vec4<f32>,
	// linear rgb sun radiance
	sun_color: vec4<f32>,
	// linear rgb Lighting.Ambient
	ambient: vec4<f32>,
	// linear rgb Lighting.OutdoorAmbient
	outdoor_ambient: vec4<f32>,
	// sky irradiance, 9 SH coefficients (rgb in xyz)
	sky_sh: array<vec4<f32>, 9>,
	// sky light (irradiance / π, scaled as Studio lights) facing +X, -X, +Y, -Y, +Z, -Z (rgb in xyz)
	sky_cube: array<vec4<f32>, 6>,
	// x EnvironmentDiffuseScale, y EnvironmentSpecularScale, z exposure multiplier, w specular cube max mip
	environment: vec4<f32>,
	// the sun's shadow (shadow.wgsl): x cascades, y the moments' warp exponent, z their least variance
	// (of the warped depth squared), w the darkening of what the bound lets through
	shadow: vec4<f32>,
	// xyz fog colour (linear), w unused
	fog_color: vec4<f32>,
	// x fog start, y fog end, z fog enabled, w unused
	fog: vec4<f32>,
	// x light count, y debug mode, z output alpha mode, w unused
	counts: vec4<u32>,
	// viewport size in pixels, 1 / size
	viewport: vec4<f32>,
	// the light grid (notes/render.md "Light grid"): xyz its minimum corner, w 1 when there is one
	light_grid: vec4<f32>,
	// xyz 1 / its size in studs, w OutdoorAmbient's factor where the sky is open
	light_grid_scale: vec4<f32>,
	// the sun's shadow cascades (shadow.wgsl): each one's depth pass view-projection, its sphere (xyz
	// centre, w radius, 0 unused) and its square in the moment atlas (x column, y side, in texels)
	shadow_cascades: array<mat4x4<f32>, 4>,
	shadow_spheres: array<vec4<f32>, 4>,
	shadow_regions: array<vec4<f32>, 4>,
	// the Atmosphere's fog: x extinction per stud, y the sky chain level the nearest surfaces fade
	// into, z the sharpest level Offset allows, w the level's power of the transmission (0 without an
	// Atmosphere, which replaces the fog)
	atmosphere: vec4<f32>,
	// the direct specular's lobe (mesh.wgsl `direct_light`): x the GGX width at roughness 0, y the
	// width's scale, z and w the visibility's k at roughness 0 and its growth with the roughness
	direct_specular: vec4<f32>,
	// the sky specular and the metalness fading with the view depth (mesh.wgsl `specular_fade`): x and y
	// 1 / the depths where each is gone, z the sky specular's level and w the metalness's scale at
	// depth 0
	specular_fade: vec4<f32>,
	// the direct specular's reflectance (mesh.wgsl `direct_light`): x the dielectric's gain, y its growth
	// with the width squared, z the weight of Schlick's rise, w the reflectance the metals' multiple
	// scattering is compensated with
	direct_reflectance: vec4<f32>,
	// the sky specular's colour (mesh.wgsl `shade_surface`, render::materials SKY_SPECULAR_COLOUR): rgb
	sky_specular_colour: vec4<f32>,
	// the direct specular's metallic share (mesh.wgsl `direct_light`): x and y the split-sum table's
	// roughness coordinate for its multiple scattering (x + y roughness), z its lobe width's scale, w 1
	// when each share is compensated by its own reflectance (0: the mixed reflectance times one)
	direct_metal: vec4<f32>,
	// what the haze surfaces fade into adds to the sky chain's (`atmosphere_fog`): per cube face +X,
	// -X, +Y, -Y, +Z, -Z (rgb), and w of haze_shape 1 when there is any (the chain's alpha then holds
	// the haze's share, blurred level by level as its colour)
	haze_light: array<vec4<f32>, 6>,
	haze_shape: vec4<f32>,
	// the lights' voxels baked for the view (render::voxel_field, voxel_read.wgsl): xyz its first
	// voxel's index, w 1 when there is one; xyz its voxels along each axis
	voxel_field: vec4<f32>,
	voxel_field_size: vec4<f32>,
	// Terrain Clouds' dimming of the sun (render::clouds::sun_cover): x the share they take off, yz the
	// camera's X and Z, w unused
	cloud_sun: vec4<f32>,
	// the view's lights by cluster (render::light_clusters, mesh.wgsl `light_cluster`): x the tile side
	// in pixels, y tiles across, z tiles down, w u32s a cluster (0: no clusters, every light everywhere);
	// xyz the camera's forward axis, w the depth slices per unit of ln(depth)
	light_clusters: vec4<f32>,
	light_cluster_view: vec4<f32>,
}



struct Instance {
	model: mat4x4<f32>,
	// inverse-transpose of the model's 3x3, in the upper-left
	normal_matrix: mat4x4<f32>,
	// linear base colour, a = opacity (1 - Transparency)
	color: vec4<f32>,
	// x roughness, y metalness, z reflectance, w emissive multiplier
	params: vec4<f32>,
	// xyz the drawn shape's size in studs (material texture coordinates), w studs per material tile
	size: vec4<f32>,
	// x shading model, y texture flags, z decal face (decals) or surface types (parts: 4 bits per
	// NormalId face, 0 none or 1 + the surface atlas group), w material texture mapping (UV_*)
	flags: vec4<u32>,
	// decal / texture tiling: x studs per tile u, y studs per tile v, z offset u, w offset v; on parts xy
	// the material texture's offset (tiles), zw the Organic pattern's noise offset (noise uv)
	tiling: vec4<f32>,
	// world → the part's box, normalised to -0.5..0.5 on every axis (decal projection)
	part_inverse: mat4x4<f32>,
	// x dielectric F0, y Fresnel strength (0 constant F0, 1 Schlick), z specular visibility term,
	// w the material normal map's strength (parts; 1 as stored)
	specular: vec4<f32>,
	// xyz emission of the material's emissive map at full mask (linear tint · strength), w the Organic
	// pattern's noise frequency (noise uv per tile)
	material: vec4<f32>,
	// material textures: x the colour map's far tiling (far tiles per tile, 0: none), y 1 / the view
	// depth where it has taken over (pre-2022), zw unused
	texture: vec4<f32>,
	// rgb the colour map's tint where it overlays the part colour (SurfaceAppearance.Color in
	// AlphaMode Overlay, linear), else 1; a unused
	tint: vec4<f32>,
}

struct Light {
	// xyz position (a surface light's face centre), w kind (0 point, 1 spot, 2 surface)
	position: vec4<f32>,
	// xyz direction the light faces, w cos(half angle)
	direction: vec4<f32>,
	// radiance (π · 1.98 · Brightness · Color^2.2), w range
	color: vec4<f32>,
	// surface lights: xyz the face's first in-plane axis, w its half extent
	right: vec4<f32>,
	// surface lights: xyz the face's second in-plane axis, w its half extent
	up: vec4<f32>,
	// x the first of the light's six shadow map layers, -1 without Shadows, y 1 with Shadows, z how
	// far behind a SurfaceLight's face its shadow is cast from, w its faces' texels a side
	shadow: vec4<f32>,
	// what its voxels give (`voxel_light`): rgb 1.98 · Brightness · Color², w how far they reach
	voxel_color: vec4<f32>,
	// x 1 when far surfaces take its voxels, 0 for a light placed with a playback; y the Range its
	// shadow's grazing darkening goes by (the longest of the shadowed lights at its position)
	extra: vec4<f32>,
}

// Studio keeps the lights' voxels within this many studs of the camera: particles reading them
// further away get none (light-particle-reach: a lattice read 474 studs away straight ahead is lit,
// 514 dark; 400 to the side and 296 ahead, lit to 491.6 and dark from 494.0)
const VOXEL_LIGHT_REACH: f32 = 493.0;

// Surfaces from these distances to the camera take their local light from the lights' voxels
// instead of per pixel, all of it from the second (mesh.wgsl `local_lights`, light-far; mirrored by
// render::LOCAL_LIGHT_FAR_END)
const LOCAL_LIGHT_FAR_START: f32 = 580.0;
const LOCAL_LIGHT_FAR_END: f32 = 600.0;

// Studio's voxels of local light (notes/render.md "Local lights"), which light particles, beams,
// trails and surfaces far from the camera: each light is written into 4-stud voxels (centres at 4i +
// 2) by a fade of its own, (1 - smoothstep(0.713 R, 1.007 R, d)) / (1 + ((d - 0.713 R) / 4.35)²), d
// from the voxel's centre to the light (a SurfaceLight's face's nearest point), times its cone t,
// not t² (a voxel centred on a SpotLight takes none); every voxel is averaged with its 26
// neighbours and read trilinearly, per axis four voxels weighted (1 - t, 1, 1, t) / 3. light-far's
// walls 1,000 studs off and light-particle-voxels' particles fitted together: rms 0.0043 of the full
// light for each (the parts' fade under a [1, 2, 1] blur: 0.015); light-particle-cones' SpotLights
// and SurfaceLights 0.002-0.003, beams under a SpotLight 0.0016.
const VOXEL_FADE_START: f32 = 0.713;
const VOXEL_FADE_END: f32 = 1.007;
const VOXEL_FADE_LENGTH: f32 = 4.35;
// what of it particles, beams and trails take (light-particle-voxels; beams 0.9655 in
// light-particle-field) and surfaces (light-far)
const VOXEL_LIGHT_PARTICLES: f32 = 0.966;
const VOXEL_LIGHT_SURFACES: f32 = 0.981;

// what one voxel centred at `centre` holds of `light` (its kind passed along)
fn voxel_inject(light: Light, kind: u32, centre: vec3<f32>) -> f32 {
	var source = light.position.xyz;
	var straight = false;
	if (kind == 2u) {
		let offset = centre - light.position.xyz;
		let across = vec2<f32>(dot(offset, light.right.xyz), dot(offset, light.up.xyz));
		source += light.right.xyz * clamp(across.x, -light.right.w, light.right.w);
		source += light.up.xyz * clamp(across.y, -light.up.w, light.up.w);
		straight = abs(across.x) <= light.right.w && abs(across.y) <= light.up.w && dot(offset, light.direction.xyz) > 0.0;
	}
	let to_centre = centre - source;
	let distance = length(to_centre);
	let range = light.color.w;
	let start = VOXEL_FADE_START * range;
	let past = max(distance - start, 0.0) / VOXEL_FADE_LENGTH;
	var value = (1.0 - smoothstep(start, VOXEL_FADE_END * range, distance)) / (1.0 + past * past);
	if (kind != 0u) {
		let cos_half = light.direction.w;
		let along = dot(to_centre / max(distance, 1e-6), light.direction.xyz);
		value *= select(select(saturate((along - cos_half) / max(1.0 - cos_half, 1e-6)), 0.0, cos_half >= 1.0), voxel_straight(cos_half), straight);
	}
	return value;
}

// What a SurfaceLight's voxels hold straight ahead of its face by its Angle: none at Angle 0, where
// surfaces are lit whole (light-particle-narrow: 4-stud faces at Angle 0 / 0.5 / 1 / 1.5 / 2 / 3+
// give particles 0 / 0.42 / 0.62 / 0.88 / 0.88 / 1 of it, an 8-stud face at Angle 0 none)
fn voxel_straight(cos_half: f32) -> f32 {
	let angle = degrees(2.0 * acos(clamp(cos_half, -1.0, 1.0)));
	return sqrt(saturate(angle / 3.0));
}

// the trilinear weight of the k-th of four voxels along an axis, t of the way between the middle two
fn voxel_weight(k: i32, t: f32) -> f32 {
	return select(select(1.0, t, k == 3), 1.0 - t, k == 0);
}

// The light a local light's voxels give at `position` (scaled by the reader: VOXEL_LIGHT_*)
fn voxel_light(light: Light, position: vec3<f32>) -> vec3<f32> {
	let kind = u32(light.position.w);
	var nearest = light.position.xyz;
	if (kind == 2u) {
		let offset = position - light.position.xyz;
		nearest += light.right.xyz * clamp(dot(offset, light.right.xyz), -light.right.w, light.right.w);
		nearest += light.up.xyz * clamp(dot(offset, light.up.xyz), -light.up.w, light.up.w);
	}
	if (distance(position, nearest) >= light.voxel_color.w) {
		return vec3<f32>(0.0);
	}
	let grid = (position - 2.0) * 0.25;
	let base = floor(grid);
	let t = grid - base;
	var sum = 0.0;
	for (var k = 0; k < 4; k++) {
		let wk = voxel_weight(k, t.z);
		for (var j = 0; j < 4; j++) {
			let wjk = voxel_weight(j, t.y) * wk;
			for (var i = 0; i < 4; i++) {
				let centre = (base + vec3<f32>(f32(i - 1), f32(j - 1), f32(k - 1))) * 4.0 + 2.0;
				sum += voxel_weight(i, t.x) * wjk * voxel_inject(light, kind, centre);
			}
		}
	}
	return light.voxel_color.rgb * (sum / 27.0);
}

const SHADING_STANDARD: u32 = 0u;
const SHADING_NEON: u32 = 1u;
const SHADING_GLASS: u32 = 2u;
const SHADING_FORCEFIELD: u32 = 3u;
const SHADING_UNLIT: u32 = 4u;

const TEX_COLOR: u32 = 1u;
const TEX_NORMAL: u32 = 2u;
const TEX_METALNESS: u32 = 4u;
const TEX_ROUGHNESS: u32 = 8u;
// material textures (Plastic, Wood, ...) projected on the part's box
const TEX_MATERIAL: u32 = 16u;
// colour map alpha cuts out the surface (SurfaceAppearance AlphaMode Transparency)
const TEX_ALPHA_TRANSPARENCY: u32 = 32u;
// colour map alpha blends between the part colour and the map (AlphaMode Overlay)
const TEX_ALPHA_OVERLAY: u32 = 64u;
// colour map alpha masks where the tint applies (AlphaMode TintMask)
const TEX_ALPHA_TINTMASK: u32 = 128u;
// vertex colours multiply the base colour
const TEX_VERTEX_COLOR: u32 = 256u;
// the normal map is DXT5nm: X in alpha, Y in green, Z reconstructed
const TEX_NORMAL_DXT5NM: u32 = 512u;
// a Decal / Texture drawn over its part: the colour map is projected onto the face flags.z
const TEX_DECAL: u32 = 1024u;
// a Texture: the projection tiles (studs per tile and offsets in `tiling`)
const TEX_DECAL_TILED: u32 = 2048u;
// material textures tile with the Organic pattern
const TEX_ORGANIC: u32 = 4096u;
// the Organic pattern's levels per unit of noise (organic-variant), and its slopes' v per stud of
// |y| + |z| along the edge (organic-slopes)
const ORGANIC_LEVELS: f32 = 8.0;
const ORGANIC_SLOPE_SCALE: f32 = 0.99;
// an emissive map (mask) lights the surface
const TEX_EMISSIVE: u32 = 8192u;
// the colour map decodes with the sRGB curve (material and SurfaceAppearance maps) instead of a square
const TEX_SRGB: u32 = 16384u;
// a decal mapped through the mesh's own uv (a classic head's face: its uv covers only the front)
const TEX_DECAL_MESH_UV: u32 = 32768u;
// a Texture tiling from the face's centre instead of its top-left corner (MeshParts)
const TEX_DECAL_CENTERED: u32 = 65536u;
// the normal map holds X and Y in red and green (BC5), Z reconstructed
const TEX_NORMAL_RG: u32 = 131072u;
// material textures are a 4 x 4 atlas of Wang cells laid out by the wang index (pre-2022)
const TEX_WANG: u32 = 262144u;
// a FileMesh's vertex colours tint the part's colour (MeshParts)
const TEX_VERTEX_TINT: u32 = 524288u;
// the roughness map is a TexturePack's metalness / roughness layer: roughness in green
const TEX_ROUGHNESS_GREEN: u32 = 1048576u;
// the emissive map is a material pack's third layer: the mask in blue
const TEX_EMISSIVE_BLUE: u32 = 2097152u;
// a decal on a Ball or Cylinder: through the uv of the primitive's face it names, on that face alone
const TEX_DECAL_FACE_ONLY: u32 = 4194304u;
// a MeshPart in a ViewportFrame: its FileMesh's vertex alpha is the frame image's alpha (render::guis
// windows)
const TEX_VERTEX_ALPHA: u32 = 8388608u;
// a decal on a posed skinned mesh: projected from the bind pose (`vs_decal_bind`)
const TEX_DECAL_BIND: u32 = 16777216u;
// a Decal's UVOffset / UVScale (in `tiling`: scale, offset): its image at uv * scale + offset
const TEX_DECAL_UV: u32 = 33554432u;
// a MeshPart's decal: on the faces whose normal's dominant axis in the part is its face's
const TEX_DECAL_DOMINANT: u32 = 67108864u;
// the emissive map is a SurfaceAppearance's own mask image: the mask in red (sa-emission-sweep)
const TEX_EMISSIVE_RED: u32 = 134217728u;

// How material textures map onto the drawn shape (instance.flags.w; notes/materials.md "Mapping")
// the mesh's own uv (SurfaceAppearance, MeshPart.TextureID)
const UV_MESH: u32 = 0u;
// primitives: the vertex uv spans its flat face (code in color.a, MATERIAL_FACE_U / V)
const UV_PLANAR: u32 = 1u;
// a Ball: the surface point projected from the centre onto the cube face of its dominant axis
const UV_BALL: u32 = 2u;
// meshes and unions: projected along the dominant axis of the part-space normal
const UV_BOX: u32 = 3u;
// a Ball's texture is 1 / 0.9575 larger than its cube's face coordinates (material-mapping)
const BALL_UV_SCALE: f32 = 0.9575;

// material texture faces: the unit-space edges along which u and v grow across the face (box faces
// in NormalId order, the wedge's slope, the corner wedge's two slopes); render::geometry::MATERIAL_FACES
const MATERIAL_FACE_U = array<vec3<f32>, 9>(
	vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
	vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0),
	vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0),
);
const MATERIAL_FACE_V = array<vec3<f32>, 9>(
	vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, -1.0, 0.0),
	vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, -1.0, 0.0),
	vec3<f32>(0.0, -1.0, -1.0), vec3<f32>(0.0, -1.0, 1.0), vec3<f32>(-1.0, -1.0, 0.0),
);

// The surface property (NormalId) each material face shows: box faces their own, the wedge's slope
// Front, the corner wedge's slopes Back and Left (surfaces-shapes)
const SURFACE_FACE = array<u32, 9>(0u, 1u, 2u, 3u, 4u, 5u, 5u, 2u, 3u);
// the surface atlas: 16 tiles of 2 x 2 studs stacked vertically, four copies of each group (Studs,
// Glue / Weld, Inlet, Universal); reading the second copy keeps filtering inside identical tiles
const SURFACE_ATLAS_TILES: f32 = 16.0;

// per Enum.NormalId: face normal, decal +u (right) and +v (down) in the part's space; matches
// render::geometry::FACE_BASIS
const FACE_NORMAL = array<vec3<f32>, 6>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 0.0, -1.0));
const FACE_RIGHT = array<vec3<f32>, 6>(vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0));
const FACE_DOWN = array<vec3<f32>, 6>(vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, -1.0, 0.0));

const PI: f32 = 3.14159265359;
// colour textures' albedo relative to their decode (texture-decode, material-swatches)
const TEXTURE_ALBEDO_SCALE: f32 = 0.971;
// transparent Neon's emission limit (√ 7.84 = 2.8; neon-transparency)
const NEON_TRANSPARENT_CAP: f32 = 7.84;
// Studio clamps every surface's exposed radiance at the same 7.84 per channel before it blends or
// the MSAA resolve averages it: a Neon line's quarter-covered edge row reads 0.50-0.53 (7.84 / 16)
// at EV -1, 0 and +1 alike, 0.40 (6.44 / 16) at EV -2 (bloom-lines), and the bloom sees the clamped
// edges (bloom-phase: 2.5 and 3.5 px lines glow alike at EV 0 and -1)
const EXPOSED_CAP: f32 = 7.84;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
	let low = c / 12.92;
	let high = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
	return select(high, low, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
	let low = c * 12.92;
	let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
	return select(high, low, c <= vec3<f32>(0.0031308));
}

// The Atmosphere's fog (notes/render.md "Atmosphere"). How much of a surface `distance` studs away
// shows through: exp(-0.1 · Density⁴ · distance) (atmosphere-distance: Density 0.2-1 within 0.3 %,
// whatever the haze, glare and Offset); the rest is the sky seen through the haze, blurred: the
// sky's specular chain at a level that grows with the transmission, the nearest surfaces taking the
// widest blur and the farthest the sky right behind them (atmosphere-mips), Offset keeping the
// level from getting sharper than it allows (atmosphere-objects2).
fn atmosphere_transmission(frame: Frame, distance: f32) -> f32 {
	return exp(-frame.atmosphere.x * distance);
}

fn atmosphere_fog(frame: Frame, sky: texture_cube<f32>, sky_sampler: sampler, position: vec3<f32>, color: vec3<f32>, additive: f32) -> vec3<f32> {
	let offset = position - frame.camera_position.xyz;
	let distance = length(offset);
	let direction = offset / max(distance, 1e-4);
	let transmission = atmosphere_transmission(frame, distance);
	let level = max(frame.atmosphere.y * pow(transmission, frame.atmosphere.w), frame.atmosphere.z);
	let sky_sample = textureSampleLevel(sky, sky_sampler, direction, level);
	var haze = sky_sample.rgb;
	if (frame.haze_shape.w > 0.0) {
		haze += haze_over_reflected(frame, direction) * sky_sample.a;
	}
	// additive light (beams, particles with LightEmission) fades without taking the haze's colour
	return color * transmission + haze * (1.0 - transmission) * (1.0 - additive);
}

// By the time of day's tint the sky chain holds the haze reflections show, lit by the tinted sky
// light; surfaces fade into the haze drawn, lit by the untinted light (a place at night): what that
// adds along `direction` where the haze is whole, the six faces' difference blended by the
// direction's squared components (`render::environment::AtmosphereSky::sky`), to be taken by the
// haze's share blurred as the chain's level is (a place at night: the share unblurred left a floor
// 55-70 studs off 3-4 of 255 too blue)
fn haze_over_reflected(frame: Frame, direction: vec3<f32>) -> vec3<f32> {
	let n2 = direction * direction;
	let x = select(frame.haze_light[1].xyz, frame.haze_light[0].xyz, direction.x > 0.0);
	let y = select(frame.haze_light[3].xyz, frame.haze_light[2].xyz, direction.y > 0.0);
	let z = select(frame.haze_light[5].xyz, frame.haze_light[4].xyz, direction.z > 0.0);
	return (n2.x * x + n2.y * y + n2.z * z) / max(n2.x + n2.y + n2.z, 1e-8);
}

// Lighting's fog (FogStart, FogEnd, FogColor), unless an Atmosphere with Density replaces it: with u
// the distance's share of the way from FogStart to FogEnd, the fog takes u (1 + 0.325 (1 - u)) of
// the colour (fog: tile rows at 5-150 studs, rms 0.006; linear in u misses by 0.06), its colour the
// radiance 1.935 · FogColor² (the sky's scale; frame.fog_color). Additive light fades.
fn linear_fog_share(frame: Frame, distance: f32) -> f32 {
	let u = saturate((distance - frame.fog.x) / max(frame.fog.y - frame.fog.x, 1e-3));
	return u * (1.0 + 0.325 * (1.0 - u));
}

fn linear_fog(frame: Frame, position: vec3<f32>, color: vec3<f32>, additive: f32) -> vec3<f32> {
	let fog = linear_fog_share(frame, length(frame.camera_position.xyz - position));
	return color * (1.0 - fog) + frame.fog_color.rgb * fog * (1.0 - additive);
}

// What the Atmosphere or Lighting's fog leaves of a particle, beam or trail: what it shows and covers
// in the √ target alike fade by what a surface at its distance keeps, taking nothing of the haze's or
// the fog's colour (vfx-atmosphere-fade: gray squares at LightEmission 1 and 0 and beams at
// LightEmission 1, 25 to 1,600 studs off over a dark sky: through Density 0.375 Studio keeps 0.96 /
// 0.86 / 0.49 / 0.04 of their light at 25 / 100 / 400 / 1,600 studs, the transmission 0.95 / 0.82 /
// 0.45 / 0.04, where ror's light faded before the √ kept its square root; in a black fog to 1,600
// studs the squares at its end vanish into the sky behind them, not the fog's black;
// vfx-atmosphere-panel: before a gray panel in a red fog a LightEmission 0 square keeps its gray as it
// fades into the reddened panel; beam-misc fog 94.99 → 99.92 % within 2)
fn fog_kept(frame: Frame, position: vec3<f32>) -> f32 {
	let distance = length(position - frame.camera_position.xyz);
	if (frame.atmosphere.w > 0.0) {
		return atmosphere_transmission(frame, distance);
	}
	if (frame.fog.z > 0.5) {
		return 1.0 - linear_fog_share(frame, distance);
	}
	return 1.0;
}

// Studio's sky light on a surface of albedo 1 facing n: the ambient cube's six values blended by
// the normal's squared components (sky-kernel: skies white in one image only light a ball as the
// cube of their exact cosine irradiance, rms 0.004; the 9-coefficient SH irradiance misses by 0.06,
// with a glow facing away from the image that Studio does not have)
fn sky_light(frame: Frame, n: vec3<f32>) -> vec3<f32> {
	let n2 = n * n;
	let x = select(frame.sky_cube[1].xyz, frame.sky_cube[0].xyz, n.x > 0.0);
	let y = select(frame.sky_cube[3].xyz, frame.sky_cube[2].xyz, n.y > 0.0);
	let z = select(frame.sky_cube[5].xyz, frame.sky_cube[4].xyz, n.z > 0.0);
	return (n2.x * x + n2.y * y + n2.z * z) / max(n2.x + n2.y + n2.z, 1e-8);
}

fn sh_irradiance(frame: Frame, n: vec3<f32>) -> vec3<f32> {
	// Ramamoorthi & Hanrahan's irradiance from 9 radiance SH coefficients
	let c1 = 0.429043;
	let c2 = 0.511664;
	let c3 = 0.743125;
	let c4 = 0.886227;
	let c5 = 0.247708;
	let L00 = frame.sky_sh[0].xyz;
	let L1m1 = frame.sky_sh[1].xyz;
	let L10 = frame.sky_sh[2].xyz;
	let L11 = frame.sky_sh[3].xyz;
	let L2m2 = frame.sky_sh[4].xyz;
	let L2m1 = frame.sky_sh[5].xyz;
	let L20 = frame.sky_sh[6].xyz;
	let L21 = frame.sky_sh[7].xyz;
	let L22 = frame.sky_sh[8].xyz;
	let x = n.x;
	let y = n.y;
	let z = n.z;
	return c1 * L22 * (x * x - y * y) + c3 * L20 * z * z + c4 * L00 - c5 * L20
		+ 2.0 * c1 * (L2m2 * x * y + L21 * x * z + L2m1 * y * z)
		+ 2.0 * c2 * (L11 * x + L1m1 * y + L10 * z);
}
