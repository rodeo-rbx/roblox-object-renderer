// HDR → display: supersample downscale, exposure, tonemap, colour correction, sRGB encode.

struct ResolveParams {
	// x exposure multiplier, y tonemapper (0 none, 1 default, 2 retro), z supersample factor, w alpha mode (0 opaque, 1 straight alpha, 2 a ViewportFrame's target: Background::Frame)
	params: vec4<f32>,
	// colour correction: x brightness, y contrast, z saturation, w enabled
	correction: vec4<f32>,
	tint: vec4<f32>,
	// the bloom: x its weight, y render pixels per texel of its first level, z 1 with the bloom on
	bloom: vec4<f32>,
	// x 1 with SunRaysEffect's rays to add, y 1 to show non-finite light in magenta (ROR_NAN_CHECK)
	rays: vec4<f32>,
}

// Whether every component is a finite number, by its exponent bits (WGSL has no isnan, and a compiler
// may take x != x as false)
fn finite(value: vec4<f32>) -> bool {
	let exponent = bitcast<vec4<u32>>(value) & vec4<u32>(0x7f800000u);
	return all(exponent != vec4<u32>(0x7f800000u));
}

@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<uniform> resolve: ResolveParams;
// the bloom's first level with the coarser ones added (shaders/bloom.wgsl), sampled bilinearly
@group(0) @binding(2) var bloom_texture: texture_2d<f32>;
@group(0) @binding(4) var bloom_sampler: sampler;
// SunRaysEffect's radiance at a quarter of the resolution (shaders/sun_rays.wgsl)
@group(0) @binding(6) var sun_rays: texture_2d<f32>;

// Studio's default tone curve, per channel (notes/render.md "Tone curve"); its input is the scene
// radiance in Roblox's units (a white surface facing a Brightness-1 sun is 1). Up to x = 2.13 the
// rational fit of sun-lit tiles (sun-board), its linear term (the toe) and the input's scale refitted
// to tone-toe's gray tiles under flat light without fog (sun-board's tiles stood in 190 studs of the
// default fog, which had raised the toe to 0.022270; the scale keeps its bright end within 0.2 of
// 255); above, continuous with it, a shoulder approaching its ceiling as a power of the distance,
// T = C - (C - start) (1 + k (x - 2.1313))^-q: 253 of 255 from x 7.1, 254 from 8.4, 255 from 10.7 and
// past 1 from 12.7 (tone-retro: Neon tiles under a ColorCorrection of Brightness -0.4 and -0.8, which
// acts before the shown value clips, and inside glowing tiles, x 2-21).
const TONE_TOE: f32 = 0.00329;
const TONE_SCALE: f32 = 1.0098;
const TONE_SHOULDER_X: f32 = 2.1313;
const TONE_SHOULDER_START: f32 = 0.795199;
const TONE_CEILING: f32 = 1.00999;
const TONE_SHOULDER_K: f32 = 0.32012;
const TONE_SHOULDER_POWER: f32 = 2.07968;

// ColorGradingEffect's Retro preset (TonemapperPreset 1; tone-retro: gray and coloured Neon at -3, 0
// and +2 EV, darkened by ColorCorrection to see above 1, inside glowing tiles): log2 of the shown value
// at log2 x = -10, -9.5, ..., 4, linear in between and past the ends. Per channel; nearly √(0.52 x)
// from x 0.03 to 1.8 (255 from x 1.9), a toe below and a shoulder above that goes on past 1 (1.65 at
// x 7.84, 1.73 at 9.5).
const RETRO: array<f32, 29> = array<f32, 29>(
	-7.2414, -6.7848, -6.3280, -5.8712, -5.4162, -4.9665, -4.5287, -4.1098, -3.7172, -3.3548,
	-3.0230, -2.7207, -2.4448, -2.1915, -1.9547, -1.7230, -1.4883, -1.2476, -0.9922, -0.7298,
	-0.4651, -0.2097, 0.0362, 0.2533, 0.4418, 0.6035, 0.7312, 0.8572, 0.9833
);

fn retro(x: f32) -> f32 {
	let t = (log2(max(x, 1e-9)) + 10.0) * 2.0;
	let index = clamp(i32(floor(t)), 0, 27);
	let shown = exp2(mix(RETRO[index], RETRO[index + 1], t - f32(index)));
	return shown * shown;
}

fn tonemap(color: vec3<f32>, mode: f32) -> vec3<f32> {
	let x = max(color, vec3<f32>(0.0));
	if (mode < 0.5) {
		return saturate(x);
	}
	if (mode > 1.5) {
		return vec3<f32>(retro(x.r), retro(x.g), retro(x.b));
	}
	let scaled = x * TONE_SCALE;
	let rational = (scaled * (2.725891 * scaled + TONE_TOE)) / (scaled * (2.307040 * scaled + 1.951868) + 1.0);
	let shoulder = TONE_CEILING - (TONE_CEILING - TONE_SHOULDER_START) * pow(1.0 + TONE_SHOULDER_K * max(x - TONE_SHOULDER_X, vec3<f32>(0.0)), vec3<f32>(-TONE_SHOULDER_POWER));
	return select(rational, shoulder, x > vec3<f32>(TONE_SHOULDER_X));
}

@fragment
fn fs_resolve(input: FullscreenOutput) -> @location(0) vec4<f32> {
	let factor = i32(resolve.params.z);
	let origin = vec2<i32>(input.clip.xy) * factor;
	var sum = vec4<f32>(0.0);
	for (var y = 0; y < factor; y++) {
		for (var x = 0; x < factor; x++) {
			sum += textureLoad(hdr, origin + vec2<i32>(x, y), 0);
		}
	}
	let premultiplied = sum / f32(factor * factor);
	var glow = vec3<f32>(0.0);
	if (resolve.bloom.z > 0.5) {
		// the output pixel's centre in render pixels, on the bloom's grid
		let position = input.clip.xy * f32(factor);
		let uv = position / (vec2<f32>(textureDimensions(bloom_texture)) * resolve.bloom.y);
		glow = textureSampleLevel(bloom_texture, bloom_sampler, uv, 0.0).rgb * resolve.bloom.x;
	}
	// SunRaysEffect adds to the radiance, under the exposure (post-sunrays-sky ev-2: a quarter at -2 EV)
	var rays = vec3<f32>(0.0);
	if (resolve.rays.x > 0.5) {
		let size = vec2<f32>(textureDimensions(hdr)) / f32(factor);
		rays = textureSampleLevel(sun_rays, bloom_sampler, input.clip.xy / size, 0.0).rgb;
	}
	if (resolve.rays.y > 0.5 && !(finite(premultiplied) && finite(vec4<f32>(glow, 1.0)) && finite(vec4<f32>(rays, 1.0)))) {
		return vec4<f32>(1.0, 0.0, 1.0, 1.0);
	}
	// A ViewportFrame's target (Background::Frame): its clear colour and what covers it averaged as they
	// are, the colour clipped at 1 and the alpha the mean opacity
	if (resolve.params.w > 1.5) {
		return vec4<f32>(sqrt(tonemap(premultiplied.rgb * premultiplied.rgb * resolve.params.x, resolve.params.y)), saturate(premultiplied.a));
	}
	if (resolve.params.w > 0.5) {
		return transparent(premultiplied, rays, glow);
	}
	// the scene target holds premultiplied √x: back to straight radiance (with an opaque background
	// the coverage is 1)
	let alpha = premultiplied.a;
	let straight = select(vec3<f32>(0.0), premultiplied.rgb / alpha, alpha > 1e-5);
	var radiance = straight * straight * alpha;
	radiance += rays;
	// exposure, then the bloom (computed from exposed values), then the tone curve
	let color = radiance * resolve.params.x + glow;
	// Roblox encodes with a square root (its colours decode with a square)
	var shown = sqrt(tonemap(color, resolve.params.y));
	if (resolve.correction.w > 0.5) {
		shown = color_correction(shown);
	}
	return vec4<f32>(shown, 1.0);
}

// A transparent background (notes/render.md "Transparent backgrounds"): straight alpha. The scene
// target's colour is premultiplied √x, light that covers nothing (LightEmission's share) added to it
// and not to the alpha, which is the share of the background covered: over black the scene shows the
// colour squared, as on an opaque black background. A pixel shows that light, its glow added: exactly
// the opaque black background's image over black, light that covers nothing a tint over other
// colours (its alpha its brightest channel). Where the covered share shows brighter as a straight
// colour (a surface at the coverage, at most EXPOSED_CAP: what is past it is light), the light is
// raised to that brightness, its hue kept: an anti-aliased edge or a see-through surface keeps its
// own colour over light backgrounds, not the share of it the tone curve's toe leaves over black.
// Sun rays are light that covers nothing too.
fn transparent(premultiplied: vec4<f32>, rays: vec3<f32>, glow: vec3<f32>) -> vec4<f32> {
	let coverage = saturate(premultiplied.a);
	let light = sqrt(tonemap((premultiplied.rgb * premultiplied.rgb + rays) * resolve.params.x + glow, resolve.params.y));
	let surface = select(vec3<f32>(0.0), premultiplied.rgb / coverage, coverage > 1e-5);
	let covered = coverage * sqrt(tonemap(min(surface * surface * resolve.params.x, vec3<f32>(EXPOSED_CAP)), resolve.params.y));
	let brightest = max(light.r, max(light.g, light.b));
	let shown = light * max(max(covered.r, max(covered.g, covered.b)) / max(brightest, 1e-6), 1.0);
	let alpha = max(coverage, max(shown.r, max(shown.g, shown.b)));
	var straight = select(vec3<f32>(0.0), shown / alpha, alpha > 1e-5);
	if (resolve.correction.w > 0.5) {
		straight = color_correction(straight);
	}
	return vec4<f32>(straight, alpha);
}

// ColorCorrectionEffect on the shown values (0-1), after the bloom (post-color: grays, colours and
// Neon under flat light, each property alone, within 0.3 of 255 on average): Contrast scales about
// 0.5, Saturation about a luma of weights (0.340, 0.550, 0.110), Brightness adds, TintColor
// multiplies, then the result clips. Several effects add their Brightness, Contrast and Saturation
// and multiply their tints (two effects of Brightness 0.2 and Contrast 0.3 fit as contrast first).
const CORRECTION_LUMA: vec3<f32> = vec3<f32>(0.3395, 0.5500, 0.1105);

fn color_correction(shown: vec3<f32>) -> vec3<f32> {
	var color = (shown - 0.5) * (1.0 + resolve.correction.y) + 0.5;
	let luma = dot(color, CORRECTION_LUMA);
	color = luma + (color - luma) * (1.0 + resolve.correction.z);
	color += resolve.correction.x;
	return saturate(color * resolve.tint.rgb);
}
