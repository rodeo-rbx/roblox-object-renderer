# Playback: effects over time

Effects draw at a moment of their timeline with a playback: what is emitted, switched, set, tweened
and moved when, and the moment drawn (`time`, seconds after the effect is placed in the world, as a
game clones its effects in). The code is `crates/ror/src/scene/playback.rs` and `scene/particles.rs`;
how particles, beams and trails draw is in [particles.md](particles.md), and clips of a playback in
[video.md](video.md).

Without one a model draws as authored: each enabled emitter's steady state, beams at rest and no
trails (they need motion). With one, particles are where and when they were emitted (each particle's
position, size, colour, transparency, rotation and flipbook frame from its own age and random
streams), trails follow the points their attachments left, and beams scroll by TextureSpeed from the
frame they were enabled in. The world updates every 1/60 s unless the playback gives `frames`.

```sh
ror render Effect.rbxm --emit VfxAttack.Shoot --time 0.3        # EmitCount attributes at 0
ror render Effect.rbxm --emit VfxAttack.Hit=20@0.1 --enable VfxAttack.Aura --disable VfxAttack.Aura@0.4 --time 0.5
ror render Effect.rbxm --playback effect.json --time 0.25
# the Lighting while it plays: an exposure flash, an impact frame the effect carries, its lights fading
ror video Effect.rbxm --set "Lighting.ExposureCompensation+=4@0.2" \
  --tween "Lighting.ExposureCompensation=initial@0.2+0.5" \
  --parent "VfxAttack.ImpactFrame=Lighting@0.3" --parent "VfxAttack.ImpactFrame=nil@0.38" \
  --tween "VfxAttack.Brightness=0@0.4+0.9:class=Light"
```

## Flags

- `--emit <path>[=<count>|=<mode>][@<seconds>]` emits from every ParticleEmitter at or under the
  dotted path from the file's root (empty: everything drawn): `count` particles, or by each emitter's
  EmitCount / EmitDelay / EmitDuration attributes (`=attributes`, the default, or another mode,
  [below](#emit-modes)).
- `--enable` / `--disable <path>[@<seconds>]` switch the emitters, trails, beams, lights and
  Highlights there (a post effect, Sky or Atmosphere only when the path or class names it), and win
  over the playback from then on: its own switches, sets and tweens of Enabled no longer reach them,
  wherever it parents them, and what it parents under the path is switched too; disabled, they emit
  nothing more. So `--disable` takes a part of a recorded effect out of the picture however the
  recording brings it in (bisecting a frame); a later forced switch takes over.
- `--time` is the moment drawn. `--playback` takes the JSON ([below](#the-playback-json)), inline or
  a file; the flags add to it. A job takes it as `"playback"`.
- `--set <path>.<Property>[+|-]=<value>[@<seconds>]` sets a property from then on, `+=` / `-=`
  relative to its value then.
- `--tween <path>.<Property>[+|-]=<value>[@<start>[+<duration>]][:<EasingStyle>][:<In|Out|InOut>]
  [:reverses][:repeat=<n>][:delay=<s>][:class=<Class>]` plays it as
  `TweenService:Create(...):Play()` at the start (TweenInfo's 1 s, Quad, Out by default), from the
  value it has then.
- `--parent <path>=<new parent>[@<seconds>]` moves an instance (`""` or `nil`: out of the world).
- Values: numbers, `true` / `false`, `#rrggbb` or `r,g,b` (0-1), a JSON array as the playback JSON
  writes one (`[0.3,0,0.3,0]` a UDim2, `[[0,1,0],[1,0,0]]` a NumberSequence), text, `initial` (as in
  the file).
- Paths are dotted from the file's root; `*` is every child (`Lighting.*`); a name a dot, a bracket
  or an edge space would split is quoted (`Impact.FX03["Exp4 [5]"]`), as ror writes paths and ree
  writes a recording's.

## The Lighting

The Lighting is a target like any instance: `Lighting.<Property>`, `Lighting.<child>.<Property>`.

- A place's own; a model's is a copy of `--lighting-from`'s, else one holding ror's default
  environment, given to the model only when a playback names it.
- Post effects, a Sky or an Atmosphere an effect carries take effect once parented into it, as an
  effect's code parents them (a Sky or Atmosphere whose Enabled is false stays out, a playback's way
  to hold one back).
- A `parent` action on `Lighting` with a `class` moves the Lighting's own of that class (the lent
  place's) out (`""`) or back (`"Lighting"`): ree's recordings write it when the run parks the
  place's look, as cutscene scripts often do.
- The Terrain's Clouds go with it, at their path in a place: `Workspace.Terrain.Clouds.<Property>`
  (Color, Cover, Density, Enabled). They are the clouds ror draws: a copy of `--lighting-from`'s,
  else the file's own, else a disabled Clouds (no clouds, as ror's default environment draws) that a
  playback can switch on. ree's recordings write them when the run changes the Terrain's clouds, as
  a weather script does; every frame of a clip reads them again.

## The playback JSON

```json
{
  "time": 0.3,
  "actions": [
    {"at": 0, "target": "Effect.Shoot", "emit": "attributes"},
    {"at": 0, "target": "Effect.Arrows", "enabled": false, "class": "Trail"},
    {"at": 0.1, "target": "Effect.Charge", "emit": 12, "timeScale": 0.5},
    {"at": 0.2, "target": "Effect.Ring", "set": {"Transparency": 0.5, "Color": [1, 0.5, 0]}}
  ],
  "tweens": [
    {"target": "Effect.Sphere", "property": "Size", "keys": [
      {"at": 0, "value": [0.001, 0.001, 0.001]},
      {"at": 0.35, "value": [20, 20, 20], "easing": "Quad", "direction": "Out"}]}
  ],
  "motion": [
    {"target": "Effect.Arrow", "relative": true, "keys": [
      {"at": 0, "cframe": [0, 0, 0]},
      {"at": 0.15, "cframe": [0, 0, -30], "easing": "Quad", "direction": "Out"}]}
  ],
  "prewarm": 0
}
```

- `actions`, at `at` seconds on the `target` path and the ParticleEmitters, Trails, Beams, lights,
  Highlights and post effects under it (only those of `class`, or a class inheriting it, if given):
  `emit` (a count or a mode, below; a count may carry `cframe`, the 12-number world CFrame of the
  emitter's part or attachment at the call, and `partSize`: the burst leaves from there, as a script
  moving an attachment between Emit calls in one frame leaves a burst at each place), `enabled`,
  `timeScale` (emitters), `clear` (emitters' and trails' Clear()); `set` gives the target (with
  `class`: each instance of it at or under the target) properties from then on; `parent` moves it
  under another path (`""`: out of the world).
- Values: numbers, enums by their item's number, booleans, `[x, y, z]` vectors and Color3s,
  12-number CFrames, strings and content URIs, `[[t, v, envelope], ...]` NumberSequences,
  `[[t, r, g, b], ...]` ColorSequences, `[min, max]` NumberRanges, a GUI's `[xScale, xOffset,
  yScale, yOffset]` UDim2s, `[scale, offset]` UDims, `[x, y]` Vector2s, `[minX, minY, maxX, maxY]`
  Rects, an instance reference as the dotted path of the instance it names, `null` for none;
  `{"add": 4}` relative to the value then, `{"initial": true}` the value in the file. A property keeps
  its type, as in Roblox: a value it cannot take stops the render with the property and the forms it
  takes.
- A content ID and the Content property that replaced it are one value (`Image` / `ImageContent`,
  `Texture` / `TextureContent`, `MeshId` / `MeshContent`, ...), as in Roblox.
- `tweens`: a property over time, with `keys` (between keyframes, each segment eased like TweenService
  by its end key's `easing` and `direction`) or with `to` (as TweenService plays one: `"at"`,
  `"duration"`, `"easing"`, `"direction"`, `"repeatCount"`, `"reverses"`, `"delay"`, from the value
  the property has at `at`).
- `motion`: a part's CFrame, a model's pivot or an attachment's world CFrame over time, everything
  under it moving along; `relative` keys are in the target's own frame at rest.
- `prewarm`: emitters and trails enabled in the file have run this long at time 0.

ree's recordings of a `.luau` script are playbacks in this format, with the frame times the run saw
([scripts.md](scripts.md)).

## Emit modes

Many effects carry their timing as attributes on each ParticleEmitter: `EmitCount`, `EmitDelay` and
`EmitDuration`. An `emit` action (and `--emit` on the command line) can play them four ways
(`scene::playback::ByAttributes`); each wait ends at the first frame at or after it:

- `attributes`: EmitCount (5 without one) is emitted EmitDelay after the call (at once without one),
  and the emitter is enabled from the call until EmitDuration after it, 0 without one: an emitter
  enabled in the file is switched off at once. The default of `render --emit <path>` and of `video`.
- `immediate`: the same without the delay: EmitCount at once whatever EmitDelay.
- `cutscene`: after EmitDelay, an emitter with an EmitDuration above 0 is enabled for it, with no
  burst; one without is emitted EmitCount (`ParticleEmitter:Emit`'s 16 without one). What is enabled in
  the file runs on.
- `delayed`: the timing the Studio calibration runs were captured with: EmitCount (5 without one)
  after EmitDelay, then enabled from there for an EmitDuration above 0.

## Against Studio

`ror studio calibrate vfx-attack:<bundle.rbxm>|auto:<effect path>[|move|still][|<moment>,...]` plays
any effect of a bundle (a model file of effects) by its emit attributes in the capture Studio frame
by frame against ror's playback of the same frame, `vfx-emitter:...` copies one emitter per variant
with one property varied, and `exposure:<timeline experiment>` averages a long exposure of many plays
([parity.md](parity.md)).

