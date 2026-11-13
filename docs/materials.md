# Materials and lighting rules

The material code is `crates/ror/src/render/materials.rs`, `render/texels.rs` (textures as the GPU
takes them) and `shaders/mesh.wgsl`. The general light and colour rules are in
[rendering.md](rendering.md).

## Built-in materials

- Two generations: the 2022 materials and the pre-2022 ones. A place chooses with
  `MaterialService.Use2022Materials` (`--materials-from <place>` takes a place's choice and its
  MaterialVariants; without it ror draws the 2022 materials).
- Each material's textures are a texture pack: a small XML asset (`texturepack_version`, `usage`,
  `alphamode`, `tiling`, and asset ids for `color`, `normal`, `metalness`, `roughness`, `emissive`,
  any of them missing). SurfaceAppearance and MaterialVariant `TexturePack` properties point at the
  same kind of asset. `render::materials::PACKS_2022` lists the built-in materials' packs.
- Studio draws Roblox's KTX2 transcodes of these packs (BC1 / BC3 with Roblox's mips), not the
  uploaded PNGs, and so does ror (`ror-image`, `render::transcode`; `ROR_TRANSCODES=0` draws the
  uploads).
- SmoothPlastic, Neon and ForceField have no textures; Plastic and Foil only a normal map.
- How a material's texture maps onto a part (studs per tile, the Organic pattern's per-part offsets,
  the Wang tiling of some materials) is measured per material and kept in the table with its pack.

## SurfaceAppearance and MaterialVariant

- A SurfaceAppearance's ColorMap, NormalMap, MetalnessMap and RoughnessMap replace the part's
  material entirely (its Material and TextureID no longer matter), tinted by its Color; AlphaMode
  Overlay, Transparency and TintMask decide what its colour map's alpha does.
- A MaterialVariant from the place's MaterialService replaces a base material's maps by name
  (`MaterialVariant` on the part, or MaterialService's override for that material), with its own
  StudsPerTile, pattern and emissive mask.
- Metals keep (1 - metalness)(1 - k s F) of the sky's diffuse light with EnvironmentSpecularScale s
  on, so a white cube's diffuse plus specular equals diffuse alone.

## Neon

- Unlit: the sun and ambient light do not change it; ExposureCompensation does.
- Emission x = 25.75 · c⁴ per channel (c the colour in 0..1), clamped like every surface at 7.84
  before it blends, so saturated colours clip to their channel ratios (orange shows yellow).
- It feeds the bloom ([rendering.md](rendering.md)).

## Glass

- What shows through Glass is the opaque scene only, moved against the surface's normal: sampled at
  P - k cos^p(θ) n⊥, n⊥ the shading normal's part across the view's axis and θ its angle to that
  axis, at the glass's depth; k falls as Transparency rises. Particles, other transparent parts and
  beams behind it do not show through it.
- Over the bent scene: its tint, its sky reflection by Fresnel, and the sun's glint.
- Past Transparency 1 Glass draws nothing, except under a Highlight (even a disabled one), where
  Studio draws it as glass at its rim: faint rings.

## ForceField

- The ForceField material draws unlit, as Studio's pattern over the part's shape, stronger at
  grazing angles; its colour and Transparency set its strength, and its pattern moves with time
  (`shaders/mesh.wgsl`, the `forcefield-*` experiments).
- The ForceField instance on a character draws Studio's own shield particles instead (a legacy
  effect, [particles.md](particles.md)).

## Decals and textures

- Decals and Textures draw on their face with the part's lighting; a Texture tiles by its
  StudsPerTileU / V and offsets. Transparency and Color3 tint them; ZIndex orders decals on one face.
- A Decal's UVOffset and UVScale map the face's uv to its image's as uv * UVScale + UVOffset: at
  UVScale (2, 2) the image fills the face's top-left quarter, at UVOffset (0.25, 0) it moves left by a
  quarter of the face. The face still bounds it.
- On MeshParts and SpecialMeshes a decal follows rules measured mesh by mesh (`fs_decal` in
  `shaders/mesh.wgsl` and the decal code in `render/mod.rs` say which).

## Characters

- R6 and R15 body parts from Studio's rigs (`rbxasset://avatar/...`), CharacterMeshes, and the
  body colours composited under Shirts, Pants and ShirtGraphics on the classic clothing templates,
  as Studio composites them (`render::skinning`, `scene::extract`).
- Skinned meshes are posed by their Bones on the CPU and drawn with their parts' materials.

## Highlights

- Fill and outline drawn over everything under the Highlight's adornee (or parent), DepthMode
  AlwaysOnTop or Occluded, FillTransparency and OutlineTransparency (`render::highlights`).
