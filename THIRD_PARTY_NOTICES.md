# Third-party notices

ror is licensed under the GNU General Public License, version 2 or (at your option) any later version
([LICENSE](LICENSE)), GPL-2.0-or-later for short. It builds on the third-party components below, each under its own license. Nothing from Roblox is bundled: Roblox's built-in
textures, meshes, skies, cloud volumes and fonts are read from the user's own Roblox install (the Player or
Studio) at run time, and assets are downloaded from Roblox on demand into the user's cache.

## x264 (only with the `x264` feature)

- What: the H.264 encoder ror links in the process when built with `--features x264`
  ([docs/video.md](docs/video.md)). It is not in this repository: `vendor/x264/build.py` builds it
  from upstream x264 (https://code.videolan.org/videolan/x264, commit
  c24e06c2e184345ceb33eb20a15d1024d9fd3497) with one change to `common/win32thread.c` (each thread
  starts at a priority ror sets), together with `vendor/x264/ror_x264.c`, ror's entry points into it.
- License: GPL-2.0-or-later. x264's license text is [vendor/x264/COPYING](vendor/x264/COPYING), the
  same file x264 ships.
- What it means: x264's license is the same as ror's. If you distribute a binary built with the
  feature, the corresponding source you provide includes x264's (with `build.py` and `ror_x264.c`),
  and the rest of "Distributing a binary" below applies. A build without the feature contains no
  x264 code.
- H.264 itself is covered by patents in some countries; distributing an H.264 encoder may need a
  patent license there. That is separate from the copyright licenses here.

## Code adapted from other projects

- `crates/ror-mesh/src/union.rs`: the union mesh data layouts are adapted from
  [rbx_mesh](https://github.com/krakow10/rbx_mesh) by krakow10, licensed MIT OR Apache-2.0; used here
  under its MIT license, below (the license names no holder; the crate's author is krakow10).

> Permission is hereby granted, free of charge, to any
> person obtaining a copy of this software and associated
> documentation files (the "Software"), to deal in the
> Software without restriction, including without
> limitation the rights to use, copy, modify, merge,
> publish, distribute, sublicense, and/or sell copies of
> the Software, and to permit persons to whom the Software
> is furnished to do so, subject to the following
> conditions:
>
> The above copyright notice and this permission notice
> shall be included in all copies or substantial portions
> of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
> ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
> TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
> PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
> SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
> CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
> OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
> IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
> DEALINGS IN THE SOFTWARE.

- `crates/ror/src/nvenc.rs`: the structure layouts, constants and identifiers of NVIDIA's encoder
  interface follow `nvEncodeAPI.h` (NVENC API 12.0, as
  [nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers) n12.0.16.1 carries it), Copyright (c)
  2010-2022 NVIDIA Corporation, under the MIT license above. The encoder itself is the installed
  driver's; none of it is distributed with ror.

- `crates/ror-image/src/bc.rs`: the BCn decoders follow the arithmetic of
  [bcdec](https://github.com/iOrange/bcdec) by iOrange, licensed MIT OR Unlicense; used here
  under the Unlicense. The tests compare against the `bcdec_rs` crate (MIT), a dev-dependency.

## Distributing a binary

ror's own code is GPL-2.0-or-later. A built ror also links Rust crates that are licensed under
Apache-2.0 alone (among them draco-core, ring, and wgpu's codespan-reporting and spirv). The Free
Software Foundation considers Apache-2.0 compatible with version 3 of the GPL but not version 2, so
distribute a binary under GPL-3.0-or-later, which the "or later" allows. Include its text
(https://www.gnu.org/licenses/gpl-3.0.txt), the corresponding source of the whole work and the
crates' license notices. The source itself can be used under version 2 or later.

## Rust dependencies

The crates in [Cargo.lock](Cargo.lock), fetched from crates.io when you build, each carry their own
license text in their package. All are permissive or weak copyleft:

| licenses | crates |
|---|---|
| MIT and/or Apache-2.0 (most, including wgpu, naga, rbx-dom, image, serde, clap, rayon) | most of the 270 crates |
| Apache-2.0 | draco-core (the FileMesh 7.00 Draco decoder), codespan-reporting, mintex, spirv |
| BSD-3-Clause | zstd-sys (zstd's C sources), zstd-safe, subtle; moxcms and pxfm (or Apache-2.0) |
| ISC | libloading, rustls-webpki, untrusted; ring (Apache-2.0 AND ISC), rustls (or MIT / Apache-2.0) |
| Zlib | foldhash, zlib-rs |
| BSD-2-Clause-Patent | ustr |
| CDLA-Permissive-2.0 | webpki-roots (the Mozilla root certificates' data) |
| MPL-2.0 | option-ext (file-level copyleft: changes to its own files stay MPL-2.0) |
| CC0-1.0 or Unicode-3.0 options | hexf-parse, blake3, constant_time_eq, unicode-ident |

Development only, not in a release binary: rusqlite (MIT) with SQLite (public domain) bundled, used by
`ror-mesh`'s cache scan tool and test; bcdec_rs (MIT); dhat (MIT OR Apache-2.0, the `dhat-heap`
feature). A full inventory with every license text can be generated with
[cargo-about](https://github.com/EmbarkStudios/cargo-about) or checked with
[cargo-deny](https://github.com/EmbarkStudios/cargo-deny) (`cargo deny check licenses`).

## Build tools (not distributed)

NASM 2.16.03 (BSD-2-Clause) and Visual Studio's C compiler build x264 when you use the `x264`
feature; they are not part of any ror binary.

## Trademarks

Roblox and Roblox Studio are trademarks of Roblox Corporation. Luau is a trademark of Roblox
Corporation. This project is not affiliated with, sponsored by, or endorsed by Roblox Corporation.
