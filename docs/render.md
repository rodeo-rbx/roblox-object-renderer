# `ror render`: inputs, cameras and output

`ror render` draws one or more views of a model, a place, an asset or a script into PNGs. This page
is the command's reference; what draws and the rules each thing follows are in
[rendering.md](rendering.md), and where content comes from and how it is cached in
[assets.md](assets.md).

`<state dir>` is the [state folder](building.md#standalone-commands-and-state); a checkout defaults
to `<checkout>/.cache`.

```sh
ror render path/to/Model.rbxm                  # <state dir>/out/Model/render.png
ror render Model.rbxm --views front,iso,turntable:8 --size 1024x1024 --out out/model
ror render 1234567890                          # a model asset id
ror render ReplicatedStorage.Models.Character --studio "My Place"   # export (read-only), render
ror render Build.luau --views iso              # a script that builds into the Workspace
```

## Inputs

`.rbxm`, `.rbxmx`, `.rbxl`, `.rbxlx` files (a place draws its Workspace), asset ids /
`rbxassetid://` URIs, or with `--studio <name>` a dotted instance path in an open Studio (exported
read-only through `SerializationService`, with the place's MaterialVariants; needs a Studio bridge,
[parity.md](parity.md)), or a `.luau` script run in the engine emulator ([scripts.md](scripts.md)).
`--select <path>` draws one instance of a file.

## Views

`--views` presets `front back left right top bottom iso default turntable:N`, or one orbit
(`--yaw`, `--pitch`, `--fov`), or any camera as JSON (`--camera`), e.g.
`{"mode":"lookAt","eye":[0,5,-10],"target":[0,2,0],"fov":40}` or
`{"mode":"orbit","yaw":25,"pitch":10,"fov":35}` (orbits frame the model's bounds as its parts
draw; a model drawing no part, an effect whose parts are invisible, by what its effects draw at
the moment drawn). A JSON list of cameras draws a view each (`camera1`, `camera2`, ...) from one
load of the input: many candidate cameras of one scene in one command.

- `--focus <instance>` frames what is at or under one instance (a dotted path, or its last names:
  `Head`), the rest drawn around it.
- `--frame minX,minY,minZ,maxX,maxY,maxZ` frames a box (a timeline's union of `stats.framed`: one
  camera for every moment).
- `--camera-from <earlier output>` gives each view the camera its namesake used there (a before /
  after pair from one camera, whatever changed); from a `ror cost` run, `--views` may name its
  gameplay views, and left at its default draws every one of them.
- A mesh URI ending in `#lod=N` draws that level of the file's chain.

## What a game draws

`--lod roblox[:QUALITY]` (opt-in) draws each MeshPart at the level of its chain Roblox draws for each
view's camera and its RenderFidelity, and leaves out the parts the graphics quality (1-21, default
21) does not draw: too small on screen, or past its view distance (300 studs at quality 1-3, 700 at
10; [cost.md](cost.md) "Levels of detail" and `crate::lod`).

- Levels go by the distance to the part's box, scaled by the field of view (full detail to 250 studs
  at 70 degrees, then levels past 500, 2,000 and 3,000); Performance takes the whole chain, Automatic
  the levels the file marks high quality, Precise none; shadows stay at full detail, as Studio draws
  them.
- `--lod levels` changes the levels alone and culls nothing (an A/B against a full-detail render).
- Each view's `lod` in `summary.json` counts the MeshParts at each level (`levels`, and per mesh
  `meshes`: copies at levels 0-4 and left out), the parts left out (`culled`) and Performance copies
  without a chain (`unchained`: Roblox makes their levels at run time, ror draws them at full detail).
- `video` and `compare` take it too; a job sets `"lod": "roblox:21"` per view. Without it ror draws
  every mesh at full detail and every part.

## Image

`--size WxH` (default 1536x1536), `--supersample N` (render N times larger and average down),
`--msaa 1|4`, `--background sky|transparent|#rrggbb`, `--no-bloom`.

## The place and the world around it

- The place an object belongs to: `--materials-from <place>` takes that place's MaterialService (2022
  or pre-2022 materials, MaterialVariants); `--lighting-from <place>` its Lighting (properties, sky,
  atmosphere, post effects). Without them the object renders under Roblox's default lighting with the
  2022 materials.
- The world it stands in (opt-in): `--map <place or model file>` draws that file's Workspace (a
  model's top-level instances) around the input, in a Folder at the root named `Map` (`Map
  (setting)` when the input has a root `Map`), lit by the file's Lighting and MaterialService unless
  `--lighting-from` / `--materials-from` name others. Orbit views, a video's framing and the authoring
  lints read the input alone (`Scene::framing_subject`); `stats.framed` is the input's box.
  `--map-origin x,y,z[,yaw]` moves the map so that point of it lands on the input's origin, turned
  by the yaw about it (an effect drawn where a character stands in a match). `render` and `video`
  take both.

## What it collides with

`--collision [collide|query|touch|all]` draws each part as Roblox's physics sees it
(`scene/collision.rs`): a MeshPart or union as its saved collision data (its convex hulls, each its
own colour; Box fidelity a warm grey box), a Part as its shape (a SpecialMesh changes what it draws,
not what it collides with), invisible parts too (a wall at Transparency 1 draws as the same wall
seen), effects and textures left out. The filter keeps the parts with CanCollide (the default),
CanQuery, CanTouch, or all, whatever their Transparency. Decoding saved collision data needs
a decoder that is not part of this repository: a public build refuses the view.

## Output

One PNG per view plus `summary.json` (camera of each view, part count, bounds, the box the views
framed (`stats.framed`), the particles drawn and the emitters, beams and trails there, missing assets
with reasons, timings). `--json` prints the same on stdout.

`--ids` also writes `<view>.ids.png`, which part is in front at every pixel (its index + 1 in red,
green and blue, low byte first; 0 for none), `parts.json` (each part's path, class, material,
transparency, position) and `visible.json` (per view, each part index and the pixels it owns: with
`--no-png` the only output, for visibility sweeps over thousands of views); a job sets `"ids": true`
per view.

`--alone <instance>` (with `--ids`; a job's `alone`) says how much of one instance each view shows:
the view's `alone` in the result holds the pixels its parts own in the view (`shown`) and the pixels
they own drawn from the same camera with nothing else (`whole`), so `shown / whole` is the share of
it that nothing hides. Choosing among cameras by what hides a subject takes one command so, where
drawing the instance by itself in a second job per camera took 20 of them for 10 cameras.

## Authoring lint

In `summary.json` (and `--json`) of `render` and `video` alike: `weightless`, the ParticleEmitters
whose particles fly (Speed's larger end over 8 studs/s) with neither Drag (0 or less) nor
Acceleration (every axis under 0.5 studs/s²), at a constant velocity as if weightless:
`{"count", "flying", "share", "emitters": [{"path", "copies", "speed", "particles"}], "line"}`.

- Only emitters that emit in what is drawn count (enabled with a Rate, or emitted by the playback by
  the moment drawn; over a video, in any frame), not dormant ones or the legacy effects; their Speed,
  Drag and Acceleration are as the playback's sets and tweens leave them.
- `count` of the `flying` are weightless, `share` their ratio; `emitters` names them by path
  (`copies` at it), up to eight paths, those that put out the most particles first, with Speed's
  larger end; `line` (when any is weightless), which ror also prints to stderr: `motion: 3 of 21
  flying emitters are weightless (no Drag or Acceleration): Meteor.Core.Sparks x2 (40 studs/s), ...`.
- Hand-made effects seldom fly weightless: 4.7 % of 57,141 emitters faster than 8 studs/s in one
  game's attack effects (`crates/ror/src/lint.rs`).
- A video's summary adds each frame's brightness, linear fades, restraint within a form, frozen
  clouds, refused images, rigid beams and each effect's counts ([video.md](video.md)).

## Inspect and diff

What a render shows, as text: `ror inspect <out>` lists each view's parts by the share of the image
they cover, with class, material and transparency; `--at X,Y` names the part at a pixel (both read a
render made with `--ids`). `ror diff <before> <after>` compares two renders of the same views: the
share of pixels changed, where, the parts that gained or lost pixels (both rendered with `--ids`),
and `<view>.diff.png` (changed pixels in orange over the render in gray). `--json` for either.

## GUIs

- GUIs in the world: SurfaceGuis on their part's face and BillboardGuis over their part or
  attachment, with their content (Frames, text, images, UICorner / UIStroke / UIGradient, layouts)
  drawn by [rur](https://github.com/Anime-Vanguards/roblox-ui-renderer), the offline Roblox UI
  renderer. ror finds rur through `ROR_RUR` (a checkout) or beside this checkout
  (`../roblox-ui-renderer`), and Node through `ROR_NODE` or the PATH; without them GUIs are skipped
  with a warning in the job's `warnings`.
- A ViewportFrame's world, framed or a window, is clipped to the frame's rect and rounded by its
  UICorner as rur rounds the frame's background.
- A view's `guiScale` sets the device pixels per GUI pixel for BillboardGui offsets and text
  (default: the image height / 744, Studio's 1.5 at a 1116 px tall image).
- GUI images are cached by content in memory and in `<state dir>/guis` (per rur build);
  `ROR_NO_GUIS=1` renders without them. ror sends rur its GUI atlases through rur's render service,
  straight to it once a `rur render` through Node has started it (`ROR_RUR_DIRECT=0`: always through
  Node), and a video finds its frames' GUI images ahead of their draws, outside the GPU slots: an
  animated GUI is a new image every frame.
- A SurfaceGui's ViewportFrame whose camera looks through its face (a window, as EgoMoose's
  ViewportWindow builds one: a rift, a portal) shows its world as the view's camera sees it, its
  meshes' vertex alpha cutting holes through the frame's background as in Studio; any other
  ViewportFrame of a SurfaceGui or BillboardGui shows its world through its own camera (an emote's
  portrait of the player behind them), fitted and clipped to the frame, lit by its Ambient,
  LightColor and LightDirection, tinted by ImageColor3, ImageTransparency and its UIGradient, under
  what the GUI draws over it.
- GUIs on the screen: the ScreenGuis of a `PlayerGui` in the file (a recording's: ree keeps the ones
  the player's PlayerGui shows there, a cutscene's fade among them) over the finished image, post
  effects included, by DisplayOrder, at the view's size (744 GUI pixels tall unless `guiScale`),
  below the top bar unless they ignore it. A ScreenGui anywhere else (a place's StarterGui, the
  Workspace) draws nothing.

## Not drawn

Terrain is not drawn. A place's Workspace.Terrain (smooth terrain, water, grass) is skipped; only its
Clouds are drawn (`ror compare` leaves them out of both sides). Layered clothing (WrapLayer /
WrapTarget) is not fitted to the body it is worn on: it draws as its authored mesh.
