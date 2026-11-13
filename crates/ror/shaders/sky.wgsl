// The skybox behind the scene (or the flat / transparent background), with the sun and the moon.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var sky_cube: texture_cube<f32>;
@group(0) @binding(2) var sky_sampler: sampler;

struct SkyParams {
	// x radiance multiplier, y draw sky (1) or transparent background (0), z 1 when the background is a
	// ViewportFrame's clear colour, written as it is (not premultiplied: Background::Frame)
	params: vec4<f32>,
	// background colour when the sky is off
	background: vec4<f32>,
	// the sun's direction and the tangent of half its angular size (the texture's square)
	sun: vec4<f32>,
	// the moon's
	moon: vec4<f32>,
	// x the sun's weight, y the moon's: what a texel of value 1 adds to the √ radiance (0: not drawn)
	bodies: vec4<f32>,
	// the skybox's own images: x 1 when drawn from them (else the cube), z the radiance of a texel
	// of 1
	images: vec4<f32>,
	// SkyboxOrientation's rotation by columns: a direction looks along rotation · direction in them
	rotation: array<vec4<f32>, 3>,
	// the time of day's display tints above and below the horizon, w 1 when tinted
	// (render::environment::tint_at)
	tint_above: vec4<f32>,
	tint_below: vec4<f32>,
	// the Atmosphere (render::environment::AtmosphereSky): haze (τ0, b, floor, 1 when on), its colour,
	// its light along +X, -X, +Y, -Y, +Z, -Z, the glare, its exponents, towards the sun
	haze: vec4<f32>,
	haze_color: vec4<f32>,
	haze_light: array<vec4<f32>, 6>,
	glare: vec4<f32>,
	decay: vec4<f32>,
	glare_sun: vec4<f32>,
	// the camera's rays: its rotation and projection inverted, the eye at the origin
	// (render::camera::Camera::ray_unprojection)
	rays: mat4x4<f32>,
	// Terrain Clouds (render::clouds::uniform): x 1 when drawn, y the shape's threshold (Cover), z
	// Density, w the march's steps; Color (linear, w the detail's amplitude); the detail's scale, the
	// march's start within the layer, the coverage under which a pixel shows the sky alone, w 1 for
	// plain alpha blending; their light (render::clouds::Light): the sun's, the sky's and the flat
	// ambient's weights, the sun's optical depth's scale; the sky light's gray floor and its weight
	// by the sun's height, the multiple scattering's weight
	clouds: vec4<f32>,
	cloud_color: vec4<f32>,
	cloud_shape_params: vec4<f32>,
	cloud_light: vec4<f32>,
	cloud_sky: vec4<f32>,
	cloud_spare: vec4<f32>,
}
@group(0) @binding(3) var<uniform> sky: SkyParams;
@group(0) @binding(4) var sun_texture: texture_2d<f32>;
@group(0) @binding(5) var moon_texture: texture_2d<f32>;
@group(0) @binding(6) var sky_images: texture_2d_array<f32>;
@group(0) @binding(7) var cloud_shape: texture_3d<f32>;
@group(0) @binding(8) var cloud_sampler: sampler;
@group(0) @binding(9) var cloud_detail: texture_3d<f32>;


// A celestial body's texture along `direction`: a square of the body's angular size facing the
// viewer, upright and mirrored left to right (sky-elements: a white square sun texture covers 10.5
// degrees each way at SunAngularSize 21; moon-black-size30 shows the moon texture's right edge on
// the left)
fn body(texture: texture_2d<f32>, centre: vec4<f32>, direction: vec3<f32>) -> vec3<f32> {
	let facing = dot(direction, centre.xyz);
	if (facing <= 0.0 || centre.w <= 0.0) {
		return vec3<f32>(0.0);
	}
	let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(centre.y) > 0.999);
	let right = normalize(cross(centre.xyz, helper));
	let up = cross(right, centre.xyz);
	let uv = vec2<f32>(-dot(direction, right), -dot(direction, up)) / (facing * centre.w);
	if (any(abs(uv) > vec2<f32>(1.0))) {
		return vec3<f32>(0.0);
	}
	return textureSampleLevel(texture, sky_sampler, uv * 0.5 + 0.5, 0.0).rgb;
}

// Roblox's skybox layout (render::environment::roblox_sky_direction): (u, v, image), images Bk, Dn,
// Ft, Lf, Rt, Up = 0..5
fn roblox_sky(direction: vec3<f32>) -> vec3<f32> {
	let a = abs(direction);
	var at: vec3<f32>;
	if (a.x >= a.y && a.x >= a.z) {
		if (direction.x > 0.0) {
			at = vec3<f32>(direction.z / a.x, -direction.y / a.x, 3.0);
		} else {
			at = vec3<f32>(-direction.z / a.x, -direction.y / a.x, 4.0);
		}
	} else if (a.y >= a.z) {
		if (direction.y > 0.0) {
			at = vec3<f32>(-direction.z / a.y, -direction.x / a.y, 5.0);
		} else {
			at = vec3<f32>(-direction.z / a.y, direction.x / a.y, 1.0);
		}
	} else if (direction.z > 0.0) {
		at = vec3<f32>(-direction.x / a.z, -direction.y / a.z, 0.0);
	} else {
		at = vec3<f32>(direction.x / a.z, -direction.y / a.z, 2.0);
	}
	return vec3<f32>(at.xy * 0.5 + 0.5, at.z);
}

// `direction`'s (u, v) on the given image of the layout, as `roblox_sky` maps the directions it
// assigns to that image
fn roblox_sky_on(direction: vec3<f32>, image: f32) -> vec2<f32> {
	var at: vec2<f32>;
	switch (i32(image)) {
		case 3: {
			at = vec2<f32>(direction.z, -direction.y) / direction.x;
		}
		case 4: {
			at = vec2<f32>(direction.z, direction.y) / direction.x;
		}
		case 5: {
			at = vec2<f32>(-direction.z, -direction.x) / direction.y;
		}
		case 1: {
			at = vec2<f32>(direction.z, -direction.x) / direction.y;
		}
		case 0: {
			at = vec2<f32>(-direction.x, -direction.y) / direction.z;
		}
		default: {
			at = vec2<f32>(-direction.x, direction.y) / direction.z;
		}
	}
	return at * 0.5 + 0.5;
}

// The time of day's radiance tint along `direction` (render::environment::tint_at)
fn tint_at(direction: vec3<f32>) -> vec3<f32> {
	let side = max(abs(direction.x), abs(direction.z));
	var blend = select(1.0, 0.0, direction.y > 0.0);
	if (side > 0.0) {
		blend = saturate(0.75 * (1.0 - direction.y / side));
	}
	let shown = mix(sky.tint_above.rgb, sky.tint_below.rgb, blend);
	return shown * shown;
}

// The skybox seen through the Atmosphere (render::environment::AtmosphereSky::sky)
fn through_atmosphere(direction: vec3<f32>, skybox: vec3<f32>) -> vec3<f32> {
	var haze = 1.0;
	if (direction.y > 0.0) {
		haze = 1.0 - sky.haze.z * exp(-sky.haze.x * pow(direction.y, -sky.haze.y));
	}
	if (haze <= 0.0) {
		return skybox;
	}
	let n2 = direction * direction;
	let light_x = select(sky.haze_light[1].rgb, sky.haze_light[0].rgb, direction.x > 0.0);
	let light_y = select(sky.haze_light[3].rgb, sky.haze_light[2].rgb, direction.y > 0.0);
	let light_z = select(sky.haze_light[5].rgb, sky.haze_light[4].rgb, direction.z > 0.0);
	let light = (light_x * n2.x + light_y * n2.y + light_z * n2.z) / max(n2.x + n2.y + n2.z, 1e-8);
	let lobe = max(0.5 + 0.5 * dot(direction, sky.glare_sun.xyz), 1e-6);
	let glare = sky.glare.rgb * pow(vec3<f32>(lobe), sky.decay.rgb);
	return skybox * (1.0 - haze) + sky.haze_color.rgb * light * haze + glare * (haze * haze);
}

// The skybox drawn from its own images, as the cube would hold it with all their detail. Each
// pixel's mip from its own footprint on the image it samples, as a GPU picks it: the direction's
// screen derivatives `ddx` and `ddy` projected onto that image (continuous across the images' seams,
// where the (u, v)'s own derivatives jump). A view at an angle to the images minifies them in parts
// and Studio blends in coarser mips there (a skybox survey: 99.26 → 99.44 % of pixels within 2
// of 255 against one level for the whole view)
fn skybox(direction: vec3<f32>, ddx: vec3<f32>, ddy: vec3<f32>) -> vec3<f32> {
	let rotation = mat3x3<f32>(sky.rotation[0].xyz, sky.rotation[1].xyz, sky.rotation[2].xyz);
	let turned = rotation * direction;
	let at = roblox_sky(turned);
	let du = roblox_sky_on(turned + rotation * ddx, at.z) - at.xy;
	let dv = roblox_sky_on(turned + rotation * ddy, at.z) - at.xy;
	let texel = textureSampleGrad(sky_images, sky_sampler, at.xy, i32(at.z), du, dv).rgb;
	var radiance = texel * texel * sky.images.z;
	if (sky.tint_above.w > 0.5) {
		radiance *= tint_at(direction);
	}
	if (sky.haze.w > 0.5) {
		radiance = through_atmosphere(direction, radiance);
	}
	return radiance;
}

@fragment
fn fs_sky(input: FullscreenOutput) -> @location(0) vec4<f32> {
	// the scene target holds √x (mesh.wgsl)
	if (sky.params.z > 0.5) {
		return vec4<f32>(sqrt(sky.background.rgb), sky.background.a);
	}
	if (sky.params.y < 0.5) {
		return vec4<f32>(sqrt(sky.background.rgb) * sky.background.a, sky.background.a);
	}
	// reversed infinite projection: depth 0.5 is a finite point on the pixel's ray
	let point = sky.rays * vec4<f32>(input.ndc, 0.5, 1.0);
	let direction = normalize(point.xyz / point.w);
	let ddx = dpdx(direction);
	let ddy = dpdy(direction);
	let cloud = clouds(direction);
	var color: vec3<f32>;
	if (sky.images.x > 0.5) {
		color = skybox(direction, ddx, ddy);
	} else {
		color = textureSampleLevel(sky_cube, sky_sampler, direction, 0.0).rgb;
	}
	// the sun and the moon add to the √ radiance (sky-sun: over skyboxes of gray 0 to 255 the sun's
	// centre is (√sky + 1.11)², its texture's value times the weight, at any exposure), the clouds over
	// them (the skybox and its haze behind: clouds stay white over an Atmosphere)
	var shown = sqrt(max(color * sky.params.x, vec3<f32>(0.0)));
	shown += body(sun_texture, sky.sun, direction) * sky.bodies.x;
	shown += body(moon_texture, sky.moon, direction) * sky.bodies.y;
	return vec4<f32>(over_clouds(shown, cloud), 1.0);
}
