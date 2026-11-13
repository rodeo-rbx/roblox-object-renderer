# Assets, caches and disk use

Where ror's content comes from, what it keeps, and what keeps renders, captures and downloads from
piling up. The code is in `crates/ror-assets` (content URIs, downloads, the disk cache, the content
store, the Studio install, the cookie), `crates/ror-image` and `crates/ror/src/render/texels.rs`
(textures as the GPU takes them).

`<state dir>` is the [state folder](building.md#standalone-commands-and-state); a checkout defaults
to `<checkout>/.cache`.

## Downloads

Assets download from Roblox on first use and are cached in `<state dir>/assets`: a render or clip
fetches what a scene lacks ahead of drawing, many at a time through assetdelivery's batch endpoint (a
19.7k-part map's 615 files in ~4 s against ~40 s one at a time). Built-in textures, sky and avatar
meshes come from the local Roblox install (Studio's, else the Player's). `--offline` uses only what is cached.

An asset Roblox refuses is listed as missing with the reason assetdelivery gives: `HTTP 403, rejected
by Roblox moderation`, `archived by its creator` and `not yet reviewed by Roblox moderation` are
refused to every account and place, and Studio, which cannot load them either, draws nothing for
them, as ror does; `not authorized for this account` and `authentication required` may need the
cookie ([building.md](building.md) "Requirements") or another account.

## Textures

- Image and SurfaceAppearance TexturePack textures draw from Roblox's KTX2 transcodes, as Studio does
  (BC1 / BC3 with Roblox's mips), the uploaded image where there is none; `ROR_TRANSCODES=0` draws the
  uploads.
- A texture made from a downloaded asset is kept as the GPU takes it in `<state dir>/textures/<code>`
  (4 GB, least recently used dropped; `ROR_TEXTURE_FILES_MB` sets it; `<code>` a hash of exactly the code
  that decides its bytes, render/texels.rs, the transcodes' joining, ror-image and its locked
  dependencies: builds never share one wrongly, and a rebuild that changed none of that keeps them
  all) and uploads from there with one read and one copy per level.
- Downloads and kept textures also go to the user's content store (below), so a fresh clone,
  worktree or build starts from what any other one on the machine already has.
- A material map read from there goes to the GPU from the finest level the views sample, the image
  unchanged (a large place seen whole reads 100 MB of its maps' 1.46 GB; `ROR_RESIDENCY=off` reads
  them whole); a clip's frames under a still camera wait for the views' feedback once, the parts it
  saw needing nothing more and the others' maps read whole.

## Disk use

What keeps it from piling up (renders, captures and downloads can reach tens of GB):

- Calibration runs write studio.png and ror.png. diff.png and side-by-side.png are made from those
  two: runs write them only with `ROR_DIFF_IMAGES=1` (`ror compare` always does) and remove an older
  pair, and `python tools/analysis/side_by_side.py <experiment or folder>` draws them again, the
  same pixels.
- Kept renders: `python tools/keep.py <tag> [experiment filter]` keeps each run's ror.png and
  background.png as `<tag>.png` / `<tag>-background.png`, hard links that cost nothing until ror
  draws the run again.
- A file written again with the same bytes is kept as it is (its time updated), so a run drawn again
  unchanged costs nothing and keeps its links. Fast PNGs (captures, calibration, renders) filter each
  row adaptively: 23-37 % smaller than with the Up filter, about 1 ms more per image.
- The content store (its default folder is in [building.md](building.md); `ROR_STORE=<dir>`
  another, `ROR_STORE=0` none): builds, checkouts and state dirs using the same store share their
  downloads (`assets/`), kept textures (`textures/<code>/`), GUI images (`guis/<rur build>-<code>/`),
  refusals, Vulkan pipeline cache and remembered GPU through it. A state dir missing one takes the
  store's as a hard link (a download only when its metadata names it and its size, a texture or GUI
  image only when its header agrees with its length), else that of another state dir that used the
  store (`state-dirs.txt`), and gives the store whatever it downloads or makes. Files are placed whole
  by a rename, never written into, so processes share it safely; it keeps `ROR_STORE_MB` (4096) MB,
  the least recently used going first (checked at most hourly). A fresh clone's first 4K map clip:
  49-89 s -> 4.4-4.8 s.
- Downloads can be shared between state dirs: `ROR_SHARED_STATE_DIRS=<dir>;<dir>` (another tool's
  `.cache`, say) makes a file missing here but cached there a hard link to it, not a download.
- `ror clean` makes identical files one file under all their names (hard links), each kind among
  its own: out/'s PNG files, the downloads (with the shared state dirs'), Studio captures, decoded
  material packs; never captures with out/. `--dry-run` reports, `--older-than <days>` also removes
  out/'s diff.png and side-by-side.png older than that, `--out <dir>` cleans another output folder.

Files under out/ and the state dir may be hard links: several names, one file. ror only ever
replaces a file by a rename, which leaves its other names as they were; copy over such files the same
way (`cp --remove-destination`, or delete first), since writing into one in place (`cp` onto an
existing file, `shutil.copyfile`, an editor saving) changes every name.
