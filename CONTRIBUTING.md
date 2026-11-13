# Contributing to ror

Thanks for helping out! This guide covers how changes get in, what we need from you, and what we can't accept.

## How changes land here

This repository is published from the maintainers' internal repository by an automated sync, so `main` only moves through sync commits.

Pull requests are still the way in, and they're reviewed here like anywhere else. Once a maintainer accepts yours, they apply it internally. It then comes back in the next sync commit with you as its author, and the PR is closed with a link to that commit. So a closed PR with a "Landed in ..." comment means it was merged.

## Before you start

- **Small fixes** (typos, an obvious bug with a test): open a PR straight away.
- **Anything bigger** (new features, new flags, behavior changes, refactors): open an issue first and wait for a maintainer to agree on the approach. PRs that skip this step may be closed so nobody's time is wasted.
- **Differences from Roblox Studio**: use the "Looks different from Studio" issue form, and include Studio's output next to ours. Parity claims need evidence; see [Measuring against Studio](#measuring-against-studio).
- **Security problems**: never in a public issue. See [SECURITY.md](SECURITY.md).

Search existing issues before filing. A thumbs-up on an existing issue helps more than a duplicate.

## Development setup

1. Install Rust 1.90 or later (on Windows the MSVC toolchain; rust-lld, which comes with it, links
   the Windows target) and clone this repository.
2. `cargo build --release` builds `target/release/ror`. `cargo test --release --workspace` runs every
   test; the team runs ror's own with `cargo test --release -p ror -- --test-threads=1`.
3. What the tests need:
   - nothing extra: ror-image, ror-mesh's synthetic tests, ror-assets' unit and mock-server tests,
     and most of ror's scene, playback, lint, budget and cost tests;
   - a GPU, or a software adapter (WARP on Windows, which ror falls back to): ror's render, video,
     service and job tests;
   - an installed Roblox (the Player or Studio): one ror-assets test, which skips without it, and renders that use
     built-in content (skies, avatar meshes, fonts);
   - the Roblox client's local cache: ror-mesh's `local_cache` test, which skips without it;
   - Roblox Studio running with a Studio MCP bridge: nothing in `cargo test`, only parity work
     ([docs/parity.md](docs/parity.md)).
4. Optional: [rur](https://github.com/Anime-Vanguards/roblox-ui-renderer) (GUIs) and
   [ree](https://github.com/Anime-Vanguards/roblox-engine-emulator) (`.luau` inputs) checked out
   beside this checkout, ffmpeg on the PATH (WebM, GIF, MP4), and the `x264` feature
   ([docs/building.md](docs/building.md)).
5. The scripts in `tools/` use Python 3.10+ with numpy, scipy, Pillow and opencv-python
   (`tools/perf_suite.py` also psutil and pywin32).

Before you open a PR, run:

```sh
cargo build --release --workspace --all-targets
cargo test --release --workspace -- --test-threads=1
cargo clippy --release --workspace --all-targets
```

## Making a change

- Keep a PR to one change. Unrelated fixes go in their own PRs.
- Add or update tests for every change in behavior. Run the whole test command above before you push, and say in the PR what you ran.
- Follow the style of the surrounding code (tabs for indentation, the existing naming). Rust uses hard tabs and long lines, and there is no rustfmt configuration: don't run `cargo fmt` over the tree, it would rewrite every file. Clippy's existing warnings are tolerated; don't add new ones. Doc comments say what a thing is and the rule it follows, with the experiment that measured it where there is one.
- Update the docs (README, `docs/`) when flags, outputs or behavior change.
- Write commit messages and PR titles as one plain sentence saying what changed ("Particles with ZOffset move by view depth").
- Don't commit generated output, build folders, renders or logs.

### Measuring against Studio

These tools aim to match Roblox Studio exactly, so a change that claims parity needs:

- the Studio version and graphics settings you compared against;
- a minimal scene or script that shows the difference, built from scratch (parts, a script, a small .rbxm you made yourself);
- the numbers or images before and after, side by side with Studio's.

Never copy code, shaders, tables or other data out of Roblox's binaries or installed files. Fit values from measurements instead.

## Content and code we can't accept

- Models, places, images, meshes, audio, fonts or text you don't own or have the right to share. That includes assets from Roblox games (ours included) and Roblox's own fonts and textures. Tests and fixtures build their own scenes.
- Code copied from Roblox, decompiled or reverse-engineered from Roblox's binaries, or taken from projects whose license isn't compatible with GPL-2.0-or-later.
- Network calls to anything other than the Roblox endpoints the tools already use, telemetry, or anything that sends user data elsewhere.
- Credentials of any kind in code, tests, logs or screenshots (never paste a `.ROBLOSECURITY` cookie anywhere).

## AI-assisted contributions

You're welcome to use AI tools. The maintainers use coding agents heavily too. The rules:

- **You're responsible for every line.** Read and understand the change before you open the PR, and be ready to explain it. Don't submit code you couldn't defend in review.
- **Say so in the PR.** The template asks which tool you used and how.
- **No unattended bulk PRs.** Agent-generated drive-by PRs across many files, or PRs opened in batches with no issue behind them, will be closed without review.
- **Agents follow [AGENTS.md](AGENTS.md).** Point your agent at it.
- **Tests are required.** Generated code with no test, or tests that don't fail without the fix, won't be merged.

## Review

Issues and pull requests are triaged and reviewed by the maintainers' automated review, so a first reply usually comes within the hour; anything it can't decide goes to a maintainer. Bugs and requests filed as issues are fixed by the maintainers: you don't need to send a pull request for them.

Every pull request is screened for security first, then reviewed for quality and fit with the project. You make the changes a review asks for: push them to the same PR, which is reviewed again. Changes to build scripts, CI, dependencies or agent instructions always need a maintainer's sign-off.

## License

ror is licensed under the GNU General Public License, version 2 or (at your option) any later version ([LICENSE](LICENSE)). By submitting a contribution you agree that it's licensed under the same terms, and you confirm that you have the right to submit it.

## Code of conduct

Everyone taking part follows our [Code of Conduct](CODE_OF_CONDUCT.md).
