# Rendering

ror draws what Roblox Studio shows at the highest graphics quality (quality level 21, `Future`
lighting) with wgpu: Vulkan or Direct3D 12 on Windows, Vulkan on Linux, Metal on macOS. The code is
in `crates/ror/src/scene` (reading files into a scene) and `crates/ror/src/render` (drawing it), the
shaders in `crates/ror/shaders`.

## From a file to a scene

- `scene::load` reads `.rbxm`, `.rbxmx`, `.rbxl` and `.rbxlx` files with rbx-dom (a place draws its
  Workspace), an asset id through `ror-assets`, or a recording ree wrote from a `.luau` script.
- `scene::extract` turns the instances into a `Scene`: parts with their geometry, material,
  colour, transparency and textures, characters (body parts, clothing composited with the body
  colours), emitters, beams, trails, lights, Highlights, GUIs and the environment (Lighting, its
  Sky, Atmosphere, Clouds and post effects). A part's world transform comes from its CFrame; a model
  pivots as Roblox's `GetPivot` does.
- `scene::playback` applies a playback (what is emitted, switched, set, tweened and moved when) to
  the document at a moment, so a scene is that moment's state ([playback.md](playback.md)).
- Content (`rbxasset://`, `rbxassetid://`, asset URLs, local files) is resolved by `ror-assets`:
  built-in files from the local Roblox install (Studio's, else the Player's), the rest downloaded once and cached.

## What draws

Parts of every shape (surface types such as Studs included), MeshParts, SpecialMeshes, unions,
SurfaceAppearance and the built-in materials of both generations (with MaterialVariants), decals and
textures, Neon, ForceField, transparency, Glass, beams, particles, the legacy effects (Fire, Smoke,
Sparkles, the ForceField instance's shield, Explosion), trails, skinned meshes, characters (R6 and R15
body parts, Shirts, Pants and ShirtGraphics, BodyColors, CharacterMeshes), Highlights, SurfaceGuis,
BillboardGuis and ScreenGuis, local lights with their shadows, bloom, the sky with its sun and moon,
the Atmosphere, fog, ColorCorrection, Blur, DepthOfField and SunRays, sun shadows, Roblox's light
grid, screen-space ambient occlusion, ViewportFrames and Terrain's Clouds.

Not drawn: Terrain itself (smooth terrain, water, grass), stars; layered clothing draws as its
authored mesh, not fitted to the body.

## Colour pipeline

The rules every shader shares (`shaders/common.wgsl`):

- Colours decode with a square: albedo = c² (c in 0..1), not the sRGB curve.
- Light is linear in Lighting.Brightness. Diffuse is plain Lambert: x = Brightness · albedo · n·l +
  specular, with x the radiance before exposure.
- ExposureCompensation multiplies the radiance by 2^EV before the tone curve.
- The display value is √T(x) per channel (a square-root encode), with the rational curve
  T(x) = y (2.725891 y + 0.00329) / (y (2.307040 y + 1.951868) + 1), y = 1.0098 x, up to x = 2.1313,
  and above it a shoulder approaching its ceiling, T(x) = 1.00999 - 0.214791 (1 + 0.32012 (x -
  2.1313))^-2.07968. A white surface facing a Brightness-1 sun has x = 1 (184 of 255).
- Every surface's exposed radiance is clamped at 7.84 per channel before it blends and before the
  MSAA resolve averages its samples.
- The scene target holds √x, so blending happens in that space, as Studio's does.

## Light

- Ambient: flat, every face alike: x = 2 · max(Ambient², OutdoorAmbient²) · albedo per channel.
  OutdoorAmbient needs the sky's visibility (the light grid), and both are darkened in creases
  (ambient occlusion).
- The light grid: voxels of 4 studs on world multiples of 4. Each voxel holds the light entering it
  from above, the mean over its 8 horizontal neighbours one layer up of their light times their
  transmission, and stores v = 1 - (1 - L)⁴. A surface reads v trilinearly 6 studs out along its
  geometric normal. Blocks fill a voxel by their horizontal coverage of it, at any thickness, times
  1 - Transparency⁴; balls, cylinders and wedges by their volume share times 0.79, 0.93 and 1.0;
  MeshParts fill nothing. OutdoorAmbient is multiplied by v; Ambient, local lights and the sun are not.
- Sky light (EnvironmentDiffuseScale): albedo times an ambient cube of the sky, its six directions
  weighed by the normal's squares; sky specular (EnvironmentSpecularScale) from prefiltered levels of
  the sky (`render::environment`).
- The sun and moon follow Lighting's ClockTime and GeographicLatitude (`render::environment::
  sun_direction`). Sun shadows are cascaded exponential variance shadow maps blurred by a separable
  Gaussian by ShadowSoftness, as Roblox has described its renderer (GDC 2020); the filter lies in
  the shadow map's plane (a block's shadow edges are 0.31 studs wide at ShadowSoftness 0.2).
- Local lights (PointLight, SpotLight, SurfaceLight): Range, Brightness, Angle and Face as Studio
  draws them, with shadows; many at once go through light clusters (`render::light_clusters`), and
  particles, beams and trails read them from a voxel field (`render::voxel_field`).
- Ambient occlusion: screen space, from the opaque parts' depths and normals; a still camera's
  clip keeps it from frame to frame and makes it again only round what moved.

## Sky, atmosphere and fog

- The skybox's six images: Ft at -Z and Bk at +Z, upright seen from inside; Lf at +X and Rt at -X;
  Up and Dn a quarter turn (`render::environment::roblox_sky_direction`). Shown as the tone curve of
  radiance 1.935 · texel², drawn from the images themselves in full detail; SkyboxOrientation turns it.
- The Atmosphere: haze and glare over the sky, and distant surfaces fading into it
  (`render::environment::AtmosphereSky`).
- Fog: with u the distance's share of the way from FogStart to FogEnd (0-1), a surface takes
  u (1 + 0.325 (1 - u)) of the fog colour, shown like a sky texel, 1.935 · FogColor².

## Clouds

Terrain Clouds are marched through the cloud volumes of the user's Roblox install
(`content/sky/clouds.dds` for their shape, `cloudDetail3D.dds` eating into its edges), cut where
Cover puts its threshold, lit by the sun, the sky and the flat ambient, and laid over the sky. They
also dim the sun on what stands near the camera (`render::clouds::sun_cover`).

This repository draws them with a generic model (`shaders/clouds.wgsl`,
`render/clouds/generic.rs`): a single-scattering ray march through a spherical shell round a planet
the camera stands on, Beer-Lambert extinction, a two-lobe Henyey-Greenstein phase function for the
sun with a short march towards it for self-shadowing, and a share of light standing in for multiple
scattering. Cover's thresholds (how much of the sky each Cover clouds) and Color's exponent are
measured against Studio; the rest is chosen by eye, so these clouds look like clouds but are not
Studio's. `ROR_TUNE_CLOUDS="key=value,..."` sets the model's figures (the `Tuning` fields' names).
Comparisons with Studio leave clouds out of both sides: Studio's reshape slowly and never settle.

## Post effects and bloom

- Studio has a built-in bloom without any BloomEffect; ror draws it and BloomEffect from a fitted
  model (`render::post`, `shaders/bloom.wgsl`): sources clamped at 7.84, a grid of half-resolution
  texels, and a kernel fitted to Studio's halos.
- ColorCorrection, ColorGrading (TonemapperPreset), Blur, DepthOfField and SunRays follow Lighting
  and Camera children as Studio applies them (`shaders/post.wgsl`).

## GUIs

SurfaceGuis, BillboardGuis and ScreenGuis are drawn by [rur](https://github.com/Anime-Vanguards/roblox-ui-renderer),
the offline Roblox UI renderer: ror sends each GUI's subtree, rur answers with its image, and ror
places it on its face, over its adornee, or over the finished frame (`render::rur`,
`render::guis`). A ViewportFrame's world is drawn by ror: through its own camera, lit by its own
Ambient, LightColor and LightDirection with no sky light, shadows or tone curve; or, for a window
whose camera looks through the face, as the view's camera sees it.

## Where to start in the code

- `render::Renderer::render_view` draws one view; `render/mod.rs` holds the passes in order.
- `render::pipelines` builds the pipelines from the WGSL files.
- `render::prepared` caches what a part needs on the GPU between frames; `render::residency` and
  `render::texture_files` keep textures on disk and on the GPU.
