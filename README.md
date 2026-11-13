# ror: an offline renderer for Roblox objects

ror renders Roblox 3D objects (a saved `.rbxm` / `.rbxmx` model, a place, an asset id, or an
instance exported from a running Studio) offline, without opening Studio or a game. It aims at what
Studio shows at the highest graphics quality (quality level 21, `Future` lighting), measured against
captures from a dedicated Studio instance, and it is built to be called by many processes at once: a
warm render service keeps the GPU device, pipelines, meshes, textures and parsed models loaded and
renders requests in parallel.

> [!NOTE]
> Almost all of the code here was written by AI: Claude, Anthropic's model, at its maintainers' direction. A lot of
> work went into making it correct, usable and fast for real projects (its renders are measured against Roblox Studio
> captures and checked by tests), but in the end Claude wrote nearly all of it.

[![A unit's attack effect playing in a match map, rendered by ror](docs/media/effect-in-a-map.webp)](docs/media/effect-in-a-map.mp4)

A unit's attack in a match map, drawn by ror from a recording of the game's own attack code run in
the engine emulator [ree](https://github.com/Anime-Vanguards/roblox-engine-emulator): the charge,
the orb, the flash and the burst's dust, with its particles, beams and trails frame by frame and no
Studio open. [Full-size MP4](docs/media/effect-in-a-map.mp4) (1920x1080, 3 s, loops).

![Studio and ror side by side for three units and part of a scene](docs/media/studio-and-ror.jpg)

Studio (left of each pair) and ror (right) from the same camera and lighting: three units and part
of a scene. Each label is the share of pixels within 2 and within 8 of 255 of Studio's, over the
whole 1644x1116 frame. Studio's side is a stored capture from Studio 0.741; ror's side is drawn by
the current build (`ror compare`, [docs/parity.md](docs/parity.md)).

[![A unit on a glowing ring, the camera turning once round it](docs/media/unit-turntable.webp)](docs/media/unit-turntable.mp4)

A unit turned once round in 8 s with `ror video --orbit 360`: the sun's glints, the sky light and
the Neon ring's bloom follow the camera. [MP4](docs/media/unit-turntable.mp4).


## What it does

- **Renders** models, places, asset ids, instances from an open Studio and `.luau` scripts to PNGs
  from any camera: parts, meshes, both generations of materials, SurfaceAppearance, characters,
  lights and shadows, the sky, atmosphere, fog, post effects, and GUIs through
  [rur](https://github.com/Anime-Vanguards/roblox-ui-renderer).
- **Plays effects over time**: particles, beams and trails at any moment of a playback (what is
  emitted, switched, tweened and moved when), as stills or as MP4, WebM, GIF or PNG frames.
- **Runs scripts**: a `.luau` file that builds a model or an effect runs in ree, and ror draws what
  it puts in the Workspace.
- **Counts what a scene costs Roblox to draw**: draw calls, triangles, levels of detail and
  occlusion culling per view, and what would cut them most (`ror cost`).
- **Measures itself against Studio**: `ror compare` and the calibration experiments capture the same
  scene in a dedicated Studio and compare pixel by pixel.
- **Serves many callers at once**: one warm render service per state dir, shared by every `ror`
  process.

## Quick start

Download the zip for your platform from
[Releases](https://github.com/Anime-Vanguards/roblox-object-renderer/releases) and extract `ror.exe`
(Windows) or `ror` (Linux and macOS). Then run `ror --version` and
`ror render path/to/Model.rbxm --views iso --out out/model`.

Or add ror to `rokit.toml`:

```toml
[tools]
ror = "Anime-Vanguards/roblox-object-renderer@0.1.0"
```

Use rokit 1.0 or later on Linux and macOS: rokit 0.2.x cannot pick `ror` there because its name
differs from the repository's. Aftman and foreman work with the same zip. A release binary keeps its
state dir (`<state dir>` below) in your user cache; a build in a checkout uses `<checkout>/.cache`.
[docs/building.md](docs/building.md#standalone-commands-and-state) lists the state paths, the
platforms and how to verify a download.

To build from source you need Rust 1.90 or later. Rendering needs a GPU (Vulkan, Direct3D 12 or
Metal). You don't need Roblox Studio to render: ror reads Roblox's built-in textures, sky, avatar meshes and fonts from an installed Roblox, the
Player or Studio, without running it (`ROR_ROBLOX_CONTENT` names a `content` folder anywhere else).
With neither it still renders, without those built-ins: no default sky or sun, studs or default particle
texture. Studio itself is only for exporting from an open place (`--studio`) and for comparing ror with
Studio (`ror compare`, `ror studio`). ffmpeg, rur, ree and a Roblox cookie are optional;
[docs/building.md](docs/building.md) has the full list and the Cargo features.

```sh
cargo build --release                                         # target/release/ror
target/release/ror render path/to/Model.rbxm                  # <state dir>/out/Model/render.png
target/release/ror render Model.rbxm --views front,iso,turntable:8 --size 1024x1024 --out out/model
target/release/ror render 1234567890                          # a model asset id
target/release/ror render ReplicatedStorage.Models.Character --studio "My Place"   # export (read-only), render
target/release/ror render Build.luau --views iso              # a script that builds into the Workspace
target/release/ror video Effect.rbxm --duration 2             # a clip of an effect's timeline
target/release/ror cost Map.rbxm --sweep 5                    # what a map costs Roblox to draw
```

## Documentation

| page | what it covers |
|---|---|
| [docs/building.md](docs/building.md) | release installs, verification, state paths, building, Cargo features and the x264 encoder |
| [docs/render.md](docs/render.md) | `ror render`: inputs, cameras, levels of detail, maps, collision view, output, lints, inspect and diff, GUIs |
| [docs/playback.md](docs/playback.md) | effects over time: the playback flags and JSON, emit modes, the Lighting while an effect plays |
| [docs/video.md](docs/video.md) | `ror video`: cameras, formats, encoders, the contact sheet, copies, summary metrics and lints |
| [docs/scripts.md](docs/scripts.md) | `.luau` scripts as input, run in the engine emulator |
| [docs/cost.md](docs/cost.md) | `ror cost`: draw calls, triangles, levels of detail and occlusion as Roblox counts them |
| [docs/service.md](docs/service.md) | the render service, many callers at once, its HTTP API, the job JSON, memory admission, profiling |
| [docs/assets.md](docs/assets.md) | downloads, texture caches, refused assets, the content store and disk use |
| [docs/parity.md](docs/parity.md) | how agreement with Studio is measured: the capture Studio, compares, agreement figures, calibration experiments |
| [docs/rendering.md](docs/rendering.md) | the rules behind the pictures: scene model, colour pipeline, light, shadows, sky, clouds, post effects, GUIs |
| [docs/materials.md](docs/materials.md) | materials, SurfaceAppearance, MaterialVariants, Neon, Glass, ForceField, decals, characters |
| [docs/particles.md](docs/particles.md) | particles, beams, trails and legacy effects |
| [docs/layout.md](docs/layout.md) | where things are in this repository |

## Status and limitations

ror is used every day by its maintainers, and its rules are measured against Studio captures (see
[docs/parity.md](docs/parity.md)). It is not a published crate. Release builds target Windows x86_64,
Linux x86_64 and macOS arm64 ([docs/building.md](docs/building.md)); Windows with MSVC is the main
development platform. Interfaces (flags, the job JSON, `summary.json`) change as the rules do; read
the docs of the commit you build.

- Terrain is not drawn: a place's smooth terrain, water and grass are skipped, only its Clouds are
  drawn. Layered clothing (WrapLayer / WrapTarget) draws as its authored mesh, not fitted to the body
  it is worn on.
- What a public build leaves out: the collisions-only view (`render --collision`) needs a decoder of
  saved collision data that is not part of this repository, so a public build says it is built
  without collision support; Terrain Clouds are drawn with a generic model that is not calibrated
  against Studio ([docs/rendering.md](docs/rendering.md) "Clouds"); and the calibration experiments
  built on test assets that are not published are left out ([docs/parity.md](docs/parity.md)).
- Parity work (`ror compare`, `ror studio`, `ror cost --check`) needs Roblox Studio on Windows and a
  Studio MCP bridge, which is not part of this repository.

## Contributing, security and license

- [CONTRIBUTING.md](CONTRIBUTING.md): how changes land (this repository is published from the
  maintainers' internal one), development setup, tests and what we can't accept.
- [SECURITY.md](SECURITY.md): reporting a vulnerability, and how ror handles the Roblox cookie and
  its local listeners.
- [SUPPORT.md](SUPPORT.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), [AGENTS.md](AGENTS.md).

ror is licensed under the GNU General Public License, version 2 or (at your option) any later version
([LICENSE](LICENSE)). It links third-party code under its own licenses, some of it Apache-2.0 only, so
a ror binary is distributed under GPL-3.0-or-later: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Not affiliated with or endorsed by Roblox Corporation. Roblox and Roblox Studio are trademarks of
Roblox Corporation. Luau is a trademark of Roblox Corporation.
