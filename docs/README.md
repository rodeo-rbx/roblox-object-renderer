# ror documentation

Guides to ror's commands and to the rules and formats it follows, for using it and for reading and
extending the code. The [README](../README.md) is the short tour.

Using ror:

| page | what it covers |
|---|---|
| [building.md](building.md) | release installs, verification, state paths, building, Cargo features and the x264 encoder |
| [render.md](render.md) | `ror render`: inputs, cameras, levels of detail, maps, the collision view, output, lints, inspect and diff, GUIs |
| [playback.md](playback.md) | effects over time: the playback flags and JSON, emit modes, the Lighting while an effect plays |
| [video.md](video.md) | `ror video`: cameras, formats, encoders, the x264 feature, the contact sheet, copies, summary metrics and lints |
| [scripts.md](scripts.md) | `.luau` scripts as input, run in the engine emulator |
| [cost.md](cost.md) | `ror cost`: how Roblox batches draws, counts triangles, picks levels of detail and culls by occlusion |
| [service.md](service.md) | the render service, many callers at once, its HTTP API, the job JSON, caches, memory admission, profiling |
| [assets.md](assets.md) | downloads, texture caches, refused assets, the content store and disk use |
| [parity.md](parity.md) | how agreement with Studio is measured: the capture Studio, compares, agreement figures, calibration experiments |

The rules behind the pictures, and the code:

| page | what it covers |
|---|---|
| [rendering.md](rendering.md) | the scene model, what draws, the colour pipeline, light, shadows, sky, atmosphere, fog, clouds, post effects, GUIs |
| [materials.md](materials.md) | built-in materials of both generations, SurfaceAppearance, MaterialVariants, Neon, Glass, ForceField, decals |
| [particles.md](particles.md) | particles, beams, trails and legacy effects |
| [layout.md](layout.md) | where things are in this repository |

How the rules were found: each one comes from captures of a dedicated Studio instance, compared
with ror pixel by pixel or read through Studio's own counters, and from Roblox's public
documentation and talks. Constants in the code are fitted to those measurements; the experiment that
measured a rule is named next to it in the code (`ror studio calibrate <experiment>`, see
[parity.md](parity.md)). Code comments also cite the maintainers' lab notes (`notes/<area>.md`), which
are not part of this repository; these pages summarize the rules those notes established.
