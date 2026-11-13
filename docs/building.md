# Requirements and building

ror is available as a standalone binary from
[Releases](https://github.com/Anime-Vanguards/roblox-object-renderer/releases), or you can build it
from source. Windows with MSVC is the main development platform.

## Release installs

Extract `ror.exe` or `ror` from the zip for your platform, put it on your PATH and run
`ror --version`. Only the executable is needed at run time. The archive also holds the licence texts,
the notices and `SOURCE.md`, which gives the source tag, the exact commit, the pinned dependencies and
the rebuild command. A binary is distributed under GPL-3.0-or-later; the source stays
GPL-2.0-or-later.

On Linux and macOS use rokit 1.0 or later: rokit 0.2.x cannot select `ror` there because its name
differs from the repository's. Aftman and foreman also work, because the binary is the zip's first
entry and its only executable, stored with Unix mode 0755.

### Platforms

| platform | runner | notes |
|---|---|---|
| Windows x86_64 | `windows-latest` | the C runtime is built in, so no Visual C++ redistributable is needed |
| Linux x86_64 | `ubuntu-22.04` | glibc 2.35 or later |
| macOS arm64 | `macos-15` | renders through Metal and reads built-in content from `RobloxStudio.app` |

Other architectures have no release job. A release is published only after every platform builds and
passes its smoke test. The maintainers develop on Windows. Studio parity and system-memory admission
need Windows. The GPU budget checks need a Vulkan driver with `VK_EXT_memory_budget`, so they run on
Windows and Linux but not on Direct3D 12 or Metal.

### How releases are built

The [release workflow](../.github/workflows/release.yml) builds the public tag on clean GitHub
runners, with no build cache, using Rust 1.90.0, the public default features and
`cargo build --locked --profile dist --bin ror --target <triple>`. On Windows it sets
`RUSTFLAGS="-C target-feature=+crt-static"` to include the C runtime.

Each zip is then tested on a fresh runner of its platform. The smoke test extracts it outside any
checkout, checks `--version` and renders a hand-made shape fixture directly and through the service.
Windows uses a software adapter when needed, Linux installs Mesa's software Vulkan driver and macOS
uses Metal. Studio and rur are not installed, so parity and GUI rendering are not smoke-tested.

A push to main that changes the release tooling, and a manual run, build, test and upload artifacts
without publishing (a dry run). Only a tag push publishes: the zips of the dry run that passed on main
at the tagged commit, or, when there is none, zips the tag run builds and tests itself first.

### Verifying a download

Download `SHA256SUMS` next to the zip and compare the zip's SHA-256 with its line:

```sh
sha256sum --check --ignore-missing SHA256SUMS    # Linux
shasum -a 256 ror-0.1.0-macos-aarch64.zip        # macOS: compare with the line in SHA256SUMS
gh attestation verify <zip> -R Anime-Vanguards/roblox-object-renderer
```

On Windows, `Get-FileHash <zip> -Algorithm SHA256` prints the digest to compare. The attestation
shows that GitHub built the zip for this repository. Each release also includes GitHub's source archive
of the tag, and `SOURCE.md` says how to get the crates pinned in `Cargo.lock`.

## Standalone commands and state

`render`, `video`, `cost`, `inspect`, `diff`, `clean`, `serve` and `service` work outside a
checkout. Name the input file, output folder or asset id as usual; ror has no default project.
Rendering needs a GPU or a software Vulkan or Direct3D adapter. The built-in Roblox content and the
optional helpers described below come from your own installation, and a file of plain shapes renders
without Roblox installed.

A standalone executable keeps its files in these folders by default:

| platform | state | shared content store |
|---|---|---|
| Windows | `%LOCALAPPDATA%\ror\state` | `%LOCALAPPDATA%\ror\store` |
| Linux | `$XDG_CACHE_HOME/ror/state`, else `~/.cache/ror/state` | the same cache's `ror/store` |
| macOS | `~/Library/Caches/ror/state` | `~/Library/Caches/ror/store` |

Set `ROR_STATE_DIR` to move the state, and `ROR_STORE` to move the shared store (`ROR_STORE=0`
turns it off). `serve --state-dir` sets the service's state folder. Default render and video output
goes under `<state dir>/out/`; `--out` names another folder. The service runs a copy of ror in
`<state dir>/bin/` and uses the same checkout, or lack of one, as the process that started it.

### What needs a checkout

- A ror checkout is the nearest folder above the executable that holds `Cargo.lock` and
  `crates/ror/Cargo.toml`. A build there keeps `<checkout>/.cache`, the checkout's `.env`, rur and
  ree beside it, and the checkout's `out/` as the default for compare, calibration and clean.
  Its content store is where a host's is (listed below), so a store that already exists stays in
  place. On macOS that is `~/.cache/ror/store`, not the `~/Library/Caches` folder in the table above.
- A standalone binary never loads a `.env` or looks for helper checkouts beside its executable or
  in the current directory. Name them instead: `ROR_COOKIE_ENV` for a cookie file, `ROR_RUR` for a
  rur checkout and `ROR_REE` for a built ree checkout.
- Without rur, GUIs are skipped with a warning. Without ree, `.luau` input reports that a built ree
  checkout is needed. Installing their native binaries is not enough: ror also uses rur's Node
  scripts and ree's build output.
- Building, the example tools and the development fixtures need a source checkout. Studio comparison
  and calibration need the Windows Studio setup described in [parity.md](parity.md).

ror's own executable chooses these defaults when it starts (`output::init_binary`), and a service
copy chooses them from the ror that started it. A program that links the Rust library as a host
does not, and keeps the library's original defaults:

- Its checkout is the nearest folder above its executable that holds a `Cargo.lock` (any Cargo
  project), else `.`. `output::set_repo_dir()` sets it instead, before it is first read.
- Its state dir is `ROR_STATE_DIR`, else `.cache` in that checkout. `output::set_state_dir()` sets
  it instead.
- Its content store is in `%LOCALAPPDATA%\ror\store` on Windows, else `$XDG_CACHE_HOME/ror/store` or
  `~/.cache/ror/store`, also on macOS.
- `output::repo_dir()` and `output::state_dir()` return a `PathBuf` and cannot fail.

## Requirements

- Rust 1.90 or later. On Windows, the MSVC toolchain; `.cargo/config.toml` links with rust-lld,
  which comes with the toolchain.
- A GPU with Vulkan, Direct3D 12 or Metal, or a software adapter for tests.
- An installed Roblox, the Player or Studio: ror reads the built-in textures, sky, avatar meshes and
  fonts from it at run time without running it, looking for Studio's newest install, then the Player's
  (`ROR_ROBLOX_CONTENT` names a `content` folder anywhere else). Without one, renders lack those
  built-ins: no default sky or sun, studs or default particle texture.
- Roblox Studio only to export instances from an open place (`--studio`) and for the parity work
  (`ror compare`, `ror studio`).
- Optional: ffmpeg for WebM, GIF and MP4 without the `x264` feature (`ROR_FFMPEG`, else the PATH,
  else winget's `Gyan.FFmpeg`).
- Optional, for SurfaceGuis, BillboardGuis and ScreenGuis: a checkout of
  [rur](https://github.com/Anime-Vanguards/roblox-ui-renderer) (`ROR_RUR`, or `../roblox-ui-renderer`
  beside this checkout) and Node 18+; rur builds its native app on first use. Without it GUIs are
  skipped with a warning.
- Optional, for `.luau` inputs: [ree](https://github.com/Anime-Vanguards/roblox-engine-emulator):
  `ROR_REE` (its binary, or a checkout built with `cargo build --release`), a checkout at
  `../roblox-engine-emulator` built the same way, or `ree` on the PATH ([scripts.md](scripts.md)).
- Optional: a Roblox account cookie for assets that need authentication, in a source checkout's `.env`
  (`ROBLOX_ASSET_DELIVERY_COOKIE=...`, ignored by Git) or a file named by `ROR_COOKIE_ENV`; without
  one, ror uses the account signed in to Roblox Studio on this machine (`ROR_STUDIO_COOKIE=0` turns
  that off). It is only ever sent to assetdelivery.roblox.com ([SECURITY.md](../SECURITY.md)).
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

