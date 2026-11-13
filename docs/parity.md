# Measuring parity with Studio

ror's rules are measured, not guessed: a dedicated capture Studio renders the same fixtures, and ror
is compared with it pixel by pixel, or with Studio's own counters for `ror cost`. The code is in
`crates/ror/src/studio`: `mod.rs` (the capture Studio), `place.rs` (the capture place), `capture.rs`
(capturing its viewport, Windows only), `compare.rs`, `export.rs`, `mcp.rs` (the bridge client) and
`calibrate.rs` with `calibrate/*` (the experiments).

`<state dir>` is the [state folder](building.md#standalone-commands-and-state); a checkout defaults
to `<checkout>/.cache`.

## What you need

- Roblox Studio on Windows (captures read the viewport's pixels from its window).
- A Studio MCP bridge: an MCP server reached over TCP (JSON-RPC, one message a line) that lists the
  open Studios and runs Luau in one of them by id, through the tools `list_roblox_studios` and
  `execute_luau` (`studio_id`, `datamodel_type`, `code`). The bridge is not part of this repository;
  set `ROR_STUDIO_MCP` to its address (`host:port`, or a port on 127.0.0.1). It only ever needs to
  listen on localhost (see [SECURITY.md](../SECURITY.md)).
- Nothing else for the offline half: stored captures replay without a Studio (below).

## The capture Studio

`ror studio open` writes a local place, `ror-capture.rbxl`, and opens it in Studio; captures only ever
run in the Studio that has this place open, and every other Studio gets read-only queries
(`SerializationService` exports, property reads). The place is never a game place: models, lighting
and materials are loaded into it per capture. One process at a time drives a capture Studio (a lock
file in `<state dir>/studio/`); several can run side by side under other names
(`ROR_CAPTURE_PLACE=<name>`, e.g. one with `--pre2022` materials). `--technology` writes the place
with another Lighting.Technology (scripts cannot set it), `--mesh-lods` its
`Workspace.MeshStreamingAndImprovedLods`.

```sh
ror studio open                                   # the capture Studio (--pre2022 for the old materials)
ror studio status                                 # its windows and viewport
ror studio luau -e "return workspace.CurrentCamera.ViewportSize"
ror studio export "My Place" Workspace.Map --out map.rbxm   # read-only export from any open Studio
ror studio close
```

## Comparing

```sh
ror compare Model.rbxm --views default,iso        # Studio vs ror: report.json + diffs
ror compare Model.rbxm --camera '{"mode":"orbit","yaw":25,"pitch":10,"fov":35}'
ror compare Map.rbxm --camera-from out/map-cost --views path1-0-far --lod roblox  # as a game draws, beside ror-full.png
ror studio calibrate material-swatches          # an experiment; see calibrate.rs
```

`ror compare <model>` loads the model into the capture Studio under the same lighting and camera,
captures Studio's viewport, renders ror's view at the viewport's size, and writes `report.json`
(per view the share of pixels within 2 and within 8 of 255), `studio.png`, `ror.png`, `diff.png` and
`side-by-side.png`. Comparisons hold every Beam still (TextureSpeed 0) on both sides and leave out
Terrain's clouds, whose frames never settle in Studio. `tools/parity.py` reruns a fixed set (Studio's
R15 and R6 rigs and four classic catalog items, by asset id); `tools/survey.py` compares many instances
of one place; `ror render --ids` plus `tools/analysis/part_errors.py` ranks a compare's parts by error.

Captures are cached in `<state dir>/captures`, keyed by model, lighting, camera, size, Studio version
and material generation: a rerun only renders ror's side. `ROR_RECAPTURE=1` captures again;
`ROR_CAPTURE_OFFLINE="<version>|<materials>"` replays stored captures without a Studio.
Several capture Studios can run side by side (`ROR_CAPTURE_PLACE=<name>`); each is driven by one
process at a time (a lock file). Analysis scripts: `tools/analysis/`.

## Agreement

Agreement is the share of pixels within 2 / within 8 of 255 of Studio's: R15 97.7 / 99.6 %, R6
98.3 / 99.7 %, catalog gear and hats 99.0-99.9 / 99.7-100.0 %, game characters, props and places
98.0-99.99 / 99.4-100.0 %.


## Calibration experiments

`ror studio calibrate <experiment>` builds a fixture model in code (tiles, balls, panels, emitters,
lights), captures it in Studio under controlled lighting from known cameras, renders ror's side, and
writes `out/calibrate/<experiment>/<run>/studio.png` and `ror.png` with `samples.json`: pixel values
at known world points on both sides, with the inputs a fit needs. The experiment names are listed in
`crates/ror/src/studio/calibrate.rs`, `experiment()`; families include `material-*`, `ibl-*`,
`sky-*`, `atmosphere-*`, `fog*`, `post-*`, `bloom-*`, `neon-*`, `glass-*`, `light-*`, `lg-*`
(the light grid), `ao-*`, `shadow-*`, `particle-*`, `beam-*`, `trail-*`, `legacy-*`, `gui-*` and
`forcefield-*`.

- Timeline experiments (particles over time, effects) run a playback in Studio frame by frame and
  compare each captured frame with ror's playback of the same frame time (`calibrate/timeline.rs`).
- `vfx-attack:<bundle.rbxm>|auto:<effect path>` plays any effect of a bundle by its emit attributes;
  `vfx-emitter:<bundle>|<emitter path>|<moment>|<variants>` copies one emitter per variant with one
  property varied; `exposure:<timeline experiment>` averages a long exposure of many plays, since a
  single frame of random particles matches Studio's only by chance; `vfx-sweep:<spec.json>` plays many
  effects a few to a view.
- `tools/analysis/` holds the fits and metrics (numpy and scipy): each script's docstring names the
  experiments it reads and the constants it fits.

Checking a change without a Studio: `ROR_RERENDER=1 ror studio calibrate <experiment>` draws ror's side
again against each run's stored `studio.png`, and `python tools/analysis/scorecard.py snapshot
<before.json> <prefixes...>` / `compare <before.json> <after.json>` scores every stored run before and
after. The stored captures are not in Git (`out/` and `.cache/` are ignored): run the experiments in
your own Studio first.

`cargo run -p ror --release --example tools -- vfx_fuzz 500 1 out/vfx-fuzz extreme` draws random
effect scenes (every property of particles, beams, trails, lights, decals, Glass and Highlights, at
their usual values and their edges) and reports panics, slow renders, drawings that differ twice,
effects that draw nothing (`sane`) and non-finite light (`ROR_NAN_CHECK=1` shows it magenta in any
render).

Some of the maintainers' experiments are not in this repository: those built on test images, meshes
and models that are not published (uniform gray maps, effect bundles and models from a game). The
rules they measured are in the code and in these docs.

## Reporting a difference

Use the "Looks different from Studio" issue form: the Studio version and graphics settings, a minimal
scene you built yourself, the command, and Studio's image next to ror's. See
[CONTRIBUTING.md](../CONTRIBUTING.md) "Measuring against Studio".
