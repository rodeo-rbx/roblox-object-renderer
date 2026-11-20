# Effect playback and timelines

Playbacks let you render particles, beams, trails, tweens, property changes, and moving parts at any timestamp `t` (seconds after the effect enters the world).

Implementation: `crates/ror/src/scene/playback.rs` and `scene/particles.rs`. See [particles.md](particles.md) for how effects render and [video.md](video.md) for rendering full clips.

Without a playback, enabled emitters render in steady state, beams sit at rest, and trails draw nothing. With a playback, `ror` steps the simulation at 60 Hz (1/60 s per frame, unless `frames` are specified explicitly in JSON), tracking every emitted particle's age and random streams, trail attachment history, and beam texture scroll from the frame it was enabled.

```sh
ror render Effect.rbxm --emit VfxAttack.Shoot --time 0.3
ror render Effect.rbxm --emit VfxAttack.Hit=20@0.1 --enable VfxAttack.Aura --disable VfxAttack.Aura@0.4 --time 0.5
ror render Effect.rbxm --playback effect.json --time 0.25

# Combine with Lighting flashes, impact frames, and fading lights
ror video Effect.rbxm \
  --set "Lighting.ExposureCompensation+=4@0.2" \
  --tween "Lighting.ExposureCompensation=initial@0.2+0.5" \
  --parent "VfxAttack.ImpactFrame=Lighting@0.3" \
  --parent "VfxAttack.ImpactFrame=nil@0.38" \
  --tween "VfxAttack.Brightness=0@0.4+0.9:class=Light"
```

## CLI flags

| Flag | Syntax and behavior |
|---|---|
| `--time` | Timestamp in seconds to render for `ror render`. |
| `--playback` | Playback definition as an inline JSON string or path to a `.json` file ([schema below](#the-playback-json)). CLI flags below layer on top of it. In job JSON, pass `"playback"`. |
| `--emit` | `<path>[=<count>\|=<mode>][@<seconds>]` — Emit from every `ParticleEmitter` at or under `<path>` (empty path targets the whole scene). Pass an integer count or an [emit mode](#emit-modes) (defaults to `=attributes`). |
| `--enable` / `--disable` | `<path>[@<seconds>]` — Toggle emitters, trails, beams, lights, and Highlights under `<path>` (post effects, `Sky`, and `Atmosphere` toggle only when targeted explicitly by path or class). **CLI force-overrides win** over playback JSON switches/sets/tweens from that timestamp onward, even if reparented, until a later forced flag overrides them. Handy for bisecting parts of a recorded effect. |
| `--set` | `<path>.<Property>[+\|-]=<value>[@<seconds>]` — Set a property at `<seconds>`. Use `+=` / `-=` for values relative to the current value at that moment. |
| `--tween` | `<path>.<Property>[+\|-]=<value>[@<start>[+<duration>]][:<EasingStyle>][:<In\|Out\|InOut>][:reverses][:repeat=<n>][:delay=<s>][:class=<Class>]` — Play a `TweenService`-style tween starting at `<start>` from the current value (defaults to `1s`, `Quad`, `Out`). |
| `--parent` | `<path>=<new parent>[@<seconds>]` — Reparent an instance (`""` or `nil` removes it from the world). |

**CLI value syntax:** numbers, `true`/`false`, `#rrggbb`, `r,g,b` (0–1), JSON arrays matching playback JSON (`[0.3,0,0.3,0]` for `UDim2`, `[[0,1,0],[1,0,0]]` for `NumberSequence`), plain strings, or `initial` (reset to the value saved in the file).

**Instance paths:** dotted paths from the file root. `*` matches all children (`Lighting.*`). Quote names containing dots, brackets, or leading/trailing spaces (e.g., `Impact.FX03["Exp4 [5]"]`), matching `ror` and `ree` path formatting.

## Animating Lighting and Clouds

`Lighting` and its children can be targeted like any other instance (`Lighting.<Property>`, `Lighting.<child>.<Property>`):

- Places use their own `Lighting`. Models copy `--lighting-from` if supplied, or create a default `Lighting` environment whenever a playback references `Lighting`.
- Post effects, `Sky`, and `Atmosphere` instances carried inside an effect model take effect as soon as they are parented into `Lighting` (a `Sky` or `Atmosphere` with `Enabled = false` remains inactive).
- A `parent` action targeting `Lighting` with a `class` filter moves the scene's original instances of that class out (`""`) or back (`"Lighting"`). `ree` recordings emit this when a cutscene temporarily swaps out a map's lighting.
- Terrain clouds are targeted at `Workspace.Terrain.Clouds.<Property>` (`Color`, `Cover`, `Density`, `Enabled`). They initialize from `--lighting-from`, then the input file, else a disabled `Clouds` instance that a playback can enable. Video clips re-read clouds every frame.

## The playback JSON

```json
{
  "time": 0.3,
  "actions": [
    { "at": 0, "target": "Effect.Shoot", "emit": "attributes" },
    { "at": 0, "target": "Effect.Arrows", "enabled": false, "class": "Trail" },
    { "at": 0.1, "target": "Effect.Charge", "emit": 12, "timeScale": 0.5 },
    { "at": 0.2, "target": "Effect.Ring", "set": { "Transparency": 0.5, "Color": [1, 0.5, 0] } }
  ],
  "tweens": [
    {
      "target": "Effect.Sphere",
      "property": "Size",
      "keys": [
        { "at": 0, "value": [0.001, 0.001, 0.001] },
        { "at": 0.35, "value": [20, 20, 20], "easing": "Quad", "direction": "Out" }
      ]
    }
  ],
  "motion": [
    {
      "target": "Effect.Arrow",
      "relative": true,
      "keys": [
        { "at": 0, "cframe": [0, 0, 0] },
        { "at": 0.15, "cframe": [0, 0, -30], "easing": "Quad", "direction": "Out" }
      ]
    }
  ],
  "prewarm": 0
}
```

### Top-level fields

- **`time`:** Target render timestamp in seconds (for `ror render`).
- **`prewarm`:** Seconds to pre-simulate emitters and trails that start enabled in the file before `t = 0`.
- **`actions`:** Events firing at `at` seconds on `target` and matching descendants (optionally filtered by `class`, including subclasses):
  - `emit`: integer burst count or an [emit mode](#emit-modes). Numeric bursts may also include `cframe` (12-component world CFrame the burst leaves from) and `partSize` (the emitting part's size then), which place the burst whatever the `motion` has for its emitter. Roblox starts an `Emit`'s particles when the frame is next drawn, where the emitter's part or attachment is then and over the part as it is then, not where it was at the call (measured, Studio 0.748: an emitter moved to three places in one render step, emitting at the first two, drew every particle at the third), so a recording gives all of one frame's emits of an emitter the same place.
  - `enabled`: boolean toggle.
  - `timeScale`: emitter `TimeScale`.
  - `clear`: mimics `:Clear()` on emitters and trails.
  - `set`: map of properties to apply to `target` (or matching `class` descendants).
  - `parent`: destination path (`""` removes from world).
- **`tweens`:** Property animations over time, defined either as `keys` (keyframe segments eased by the end key's `easing` and `direction`) or `to` (TweenService-style: `at`, `duration`, `easing`, `direction`, `repeatCount`, `reverses`, `delay`, interpolating from the property's value at `at`).
- **`motion`:** Animates a part's `CFrame`, a model's pivot, or an attachment's world `CFrame` (moving all descendants along with it). Set `"relative": true` to interpret keyframes relative to the target's rest transform.

### JSON value encodings

| Roblox type | JSON form |
|---|---|
| Number / Bool / String / Content URI | JSON number / boolean / string |
| Enum | Numeric enum item value |
| `Vector3` / `Color3` | `[x, y, z]` or `[r, g, b]` |
| `CFrame` | 12-number array `[x, y, z, R00..R22]` (or `[x, y, z]` translation) |
| `NumberSequence` | `[[t, v, envelope], ...]` |
| `ColorSequence` | `[[t, r, g, b], ...]` |
| `NumberRange` | `[min, max]` |
| `UDim2` / `UDim` | `[xScale, xOffset, yScale, yOffset]` / `[scale, offset]` |
| `Vector2` / `Rect` | `[x, y]` / `[minX, minY, maxX, maxY]` |
| Instance Ref | Dotted path string, or `null` |
| Relative / Reset | `{"add": 4}` (offset from current value), `{"initial": true}` (file default) |

Properties are strictly type-checked: passing an incompatible value fails the render with an error listing accepted forms. Legacy content string properties and newer `Content` properties alias to the same underlying value (`Image`/`ImageContent`, `Texture`/`TextureContent`, `MeshId`/`MeshContent`, etc.).

Recordings produced by `ree` from `.luau` scripts use this exact JSON format (plus recorded frame timestamps); see [scripts.md](scripts.md).

## Emit modes

Many Roblox VFX models store timing in attributes on each `ParticleEmitter`: `EmitCount`, `EmitDelay`, and `EmitDuration`. An `emit` action or `--emit` flag supports four modes (`scene::playback::ByAttributes`). All delays snap to the first frame at or after the delay elapses:

| Mode | Behavior |
|---|---|
| `attributes` *(default)* | Emits `EmitCount` (default `5`) after `EmitDelay` (default `0`). Enables the emitter from the call until `EmitDuration` (default `0`, immediately turning off emitters that started enabled in the file). Default for `render --emit <path>` and `video`. |
| `immediate` | Same as `attributes`, but ignores `EmitDelay` and fires `EmitCount` immediately. |
| `cutscene` | After `EmitDelay`, if `EmitDuration > 0` enables the emitter for that duration (no burst); otherwise bursts `EmitCount` (default `16`, matching `ParticleEmitter:Emit()`). Leaves already-enabled file emitters running. |
| `delayed` | Matches Studio calibration capture timing: waits `EmitDelay`, emits `EmitCount` (default `5`), then enables for `EmitDuration` if `> 0`. |

## Calibrating playbacks against Studio

Use `ror studio calibrate` to verify effect playback frame by frame against Studio ([parity.md](parity.md)):
- `vfx-attack:<bundle.rbxm>|auto:<effect path>[|move|still][|<moment>,...]`
- `vfx-emitter:<bundle>|<emitter path>|<moment>|<variants>`
- `exposure:<timeline experiment>`

