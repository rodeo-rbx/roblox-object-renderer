# Requirements and building

ror is built from source; it is not a published crate and has no releases. Windows with the MSVC
toolchain is the main platform; macOS builds render through Metal and read built-in content from
`RobloxStudio.app` (system-memory admission and GPU budget checks are off there); Linux is not tested.

## Requirements

- Rust 1.90 or later. On Windows, the MSVC toolchain; `.cargo/config.toml` links with rust-lld,
  which comes with the toolchain.
- A GPU with Vulkan, Direct3D 12 or Metal, or a software adapter for tests.
- A local Roblox Studio install: built-in textures, sky and avatar meshes and fonts are read from it
  at run time (`ROR_ROBLOX_CONTENT` names its `content` folder if it is not found).
- Optional: ffmpeg for WebM, GIF and MP4 without the `x264` feature (`ROR_FFMPEG`, else the PATH,
  else winget's `Gyan.FFmpeg`).
- Optional, for SurfaceGuis, BillboardGuis and ScreenGuis: a checkout of
  [rur](https://github.com/Anime-Vanguards/roblox-ui-renderer) (`ROR_RUR`, or `../roblox-ui-renderer`
  beside this checkout) and Node 18+; rur builds its native app on first use. Without it GUIs are
  skipped with a warning.
- Optional, for `.luau` inputs: [ree](https://github.com/Anime-Vanguards/roblox-engine-emulator):
  `ROR_REE` (its binary, or a checkout built with `cargo build --release`), a checkout at
  `../roblox-engine-emulator` built the same way, or `ree` on the PATH ([scripts.md](scripts.md)).
- Optional: a Roblox account cookie for assets that need authentication, in this checkout's `.env`
  (`ROBLOX_ASSET_DELIVERY_COOKIE=...`, ignored by Git) or a file named by `ROR_COOKIE_ENV`. It is
  only ever sent to assetdelivery.roblox.com ([SECURITY.md](../SECURITY.md)).
- Optional, for parity work: Roblox Studio running on Windows and a Studio MCP bridge
  ([parity.md](parity.md)).

## Building

```sh
cargo build --release                      # target/release/ror
cargo test --release --workspace           # tests; some need a GPU or a Studio install, see CONTRIBUTING.md
cargo build --profile dist                 # whole-program build (thin LTO, stripped)
```

`cargo build --release` is the everyday build: incremental, ror's crates with cargo's ThinLTO across
their 256 codegen units (`lto = false`) and every dependency built as one unit. A clean build took
141 s instead of 178 s with renders byte-identical and as fast or faster. `lto = "off"` would rebuild
a one-line edit in `crates/ror` in 10-15 s instead of 24 s (ThinLTO re-optimises every unit importing
from an edited one), but a 1536x1536 still then took 10-15 % longer, so the profile keeps ThinLTO.

## Cargo features

Cargo features of `ror`:

- `x264` (off by default): MP4s are encoded by x264 in the process instead of through ffmpeg
  ([video.md](video.md)). It links `ror_x264.lib`, which you build with
  `python vendor/x264/build.py <x264 checkout> <nasm.exe> <folder>` (Windows, Visual Studio 2022's C
  tools, Git for Windows' bash and NASM 2.16.03; the commands are in its header), and finds it
  through `ROR_X264_LIB_DIR`:

  ```sh
  python vendor/x264/build.py ../x264 C:/tools/nasm-2.16.03/nasm.exe ../x264-lib
  ROR_X264_LIB_DIR=../x264-lib cargo build --release --features x264
  ```

  x264 is GPL-2.0-or-later, like ror: see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) before
  you distribute a binary built with it. Only x86_64-pc-windows-msvc is supported.
- `dhat-heap`: heap profiling with dhat.

