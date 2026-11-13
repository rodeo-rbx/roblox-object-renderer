# Video output

`ror video` (and the render service's `POST /video`) draws a model or an effect over a stretch of its
timeline. The code is `crates/ror/src/video.rs` (the clip), `mp4.rs` (ror's own MP4 writer),
`x264.rs` (the in-process encoder) and `render/nv12.rs` (the GPU's colour conversion).

## Running it

`ror video` goes through the same warm render service as `ror render` ([service.md](service.md)).

```sh
ror video Effect.rbxm                                   # 2 s, every emitter by its emit attributes
ror video Cutscene.rbxm --emit =cutscene                # with cutscene timing
ror video Effect.rbxm --duration 1.5 --speed 0.25 --view iso   # a quarter speed: 6 s of video
ror video Model.rbxm --orbit 360 --duration 4 --view front     # once round
ror video Effect.rbxm --background transparent                 # video.webm, with alpha
ror video Effect.rbxm --playback effect.json --start 0.2 --fps 60
```

- What happens: the playback flags of `render` (`--emit`, `--enable`, `--disable`, `--playback`,
  and `--set`, `--tween`, `--parent` for properties, the Lighting among them; [playback.md](playback.md));
  without any of the first four, every emitter plays by its attributes at 0 (`--emit =attributes`:
  EmitCount after EmitDelay, enabled from 0 for EmitDuration; `--emit =cutscene` cutscene timing;
  `--no-emit` draws only what the file enables). `--start` is the first frame's moment, `--duration`
  the seconds of timeline drawn, `--fps` the frames per second, `--speed` how fast the video plays
  them (0.25: four times slower, the video four times longer).
- Camera: one view, `--view` (a preset), `--yaw` / `--pitch`, `--fov` or `--camera`. An orbit is
  fitted once to everything the clip draws (the parts and, at nine moments, the live particles
  without the farthest 3 % each way, the beams and the trails), so the camera holds still while the
  effect plays; `--frame` and `--focus` frame a box or an instance instead, and `--orbit <degrees>`
  turns the camera round it over the clip at one distance.
- `--camera-from <out>` takes an earlier video's framed box (the same `--view` and `--orbit` then
  frame alike: a before / after pair from one camera), or an earlier render's view of the same name
  as a fixed camera.
- `--camera-track <file>` moves the camera as a recording's game camera moved: a JSON list of keys
  `{"at": <s>, "cframe": [x, y, z, R00 ... R22], "fov": <deg>}` (ree's `camera.json`), each held from
  its moment until the next and the first before its own. Emitters that camera could not draw stand
  still meanwhile, neither emitting nor ageing, as Studio leaves them.
- Output, in `--out` (default `<state dir>/out/<input>-video`: the
  [state folder](building.md#standalone-commands-and-state), `<checkout>/.cache` in a checkout), at
  `--size WxH` (default 1920x1080, even for mp4 and webm): `video.mp4` (H.264), or another of the
  [formats](#formats) with `--format` (`webm`, `gif`, `apng`, `png`). The summary's `views` hold the
  first frame's camera: `ror render <input> --views <view> --camera-from <out>` draws a still from
  it.

### The sheet first

`--sheet-first` (a job's `sheetFirst`; through the service): the sheet's moments are drawn first and
the answer comes once `sheet.png` is written, the clip following in the background: a model's 4 s
orbit at 2560x1440 answers in 113 ms instead of 359. The answer is the summary so far: `pending`
names `clip.pending`, which stands in `--out` until `video.mp4` and `summary.json` are written
(`clip.failed`, holding the error, if they are not). The files are the same bytes as without it.
Clips into one `--out` are drawn one after another, so a second job there waits for the first to
finish. A caller that reads a clip folder waits for `clip.pending` to go first.

### Copies

`--copy '{"file": "small.mp4", "args": [...]}'` (a job's `copies`; repeatable, an MP4 clip's):
another file made from the clip's frames while they are drawn, by an ffmpeg fed the NV12 the clip's
encoder takes, with `args` as its output options between that input and the file (codec, rate,
`-vf`, `-movflags`); the extension names the container (mp4, m4v, mov, mkv, webm). The summary's
`copies` say which were `written`, why one was not (`error`, also a warning) and how long the clip
waited for each after its own file (`waitedMs`). A smaller copy to share made so (NVENC at a planned
bitrate) was ready 0.3 s after a 4K clip, where encoding it from the clip afterwards took 2.4 s with
NVENC, 5.9 s with x264's two passes. A clip wanted only at that size is cheaper encoded once, by
`--nvenc` ([Encoders](#encoders)).


## Formats

- `mp4` (the default): H.264, limited-range BT.709 NV12 converted from the finished display RGB on
  the GPU (1.5 bytes a pixel read back instead of 4), flattening a transparent background over
  black. The file tags BT.709 matrix and primaries, the sRGB transfer, limited range and centred
  chroma, and keeps its index ahead of the frames (as ffmpeg's `+faststart` does).
- `webm` (VP9, keeping a transparent background's alpha; the default with `--background
  transparent`) and `gif`, through ffmpeg.
- `apng` (written by ror, lossless and large) and `png` (`frames/0000.png`, ...).
- Every format also writes `sheet.png`, a contact sheet of 12 moments (`--sheet N`) labelled with
  their time and the particles drawn, and `summary.json`.

## Encoders

`ROR_VIDEO_ENCODER=auto|x264|libx264|h264_nvenc` chooses how an MP4 is encoded:

- `x264`: in ror's process, with the `x264` cargo feature (off by default). It links `ror_x264.lib`,
  which `vendor/x264/build.py` builds from x264 at commit c24e06c (the x264 ffmpeg 7.1 carries, core
  164 r3192) with its assembly, 8-bit, and `vendor/x264/ror_x264.c`, ror's entry points; set up as
  ffmpeg sets up libx264, the H.264 stream is byte for byte the one ffmpeg writes. Each frame goes
  from its mapped readback straight into the encoder; its threads run below normal priority. Point
  `ROR_X264_LIB_DIR` at the folder holding the library when you build with `--features x264`.
  x264 is GPL-2.0-or-later, like ror ([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).
  x86_64-pc-windows-msvc only.
- `libx264`: ffmpeg's libx264 through a pipe. Without the `x264` feature this is what `auto` uses.
- `h264_nvenc`: NVIDIA's encoder through ffmpeg (`p1`, VBR CQ 19, spatial AQ); ror checks that NVENC
  opens at the requested size and falls back to libx264 if it cannot.

`--nvenc <quality>[:<kbps>]` (a job's `nvenc`) encodes an MP4 on NVIDIA's hardware encoder inside
ror's process instead, whatever `ROR_VIDEO_ENCODER` says: constant quality (1-51, lower is better),
spending at most `kbps` kilobits a second, so a clip of a known length fits a size. The frames go
from their readback into the driver's buffer and the encoding is the GPU's encoder block's work: a
286-frame 1440p clip took 9.7 s of CPU where x264 and an ffmpeg copy took 20.5, and five such clips
at once 56 where they took 88. It encodes as ffmpeg's `h264_nvenc` does with `-preset p3 -tune hq
-rc vbr -cq <quality> -maxrate <kbps>k -bf 3 -rc-lookahead 8 -spatial-aq 1`: a 9.5 s clip came out
the size and PSNR of that ffmpeg copy (8.50 MB, 47.09 dB against the frames). B-frames reorder the
stream, so the MP4 carries composition offsets and an edit list. It needs an NVIDIA GPU and its
driver (`nvEncodeAPI64.dll`, `nvcuda.dll`; 522.25 or later): without one the clip fails naming it,
and the caller encodes another way.

The settings: x264's `ultrafast` preset at CRF 19, 6 frame threads below 4K and 8 from it, so a clip's
bytes are the same on any machine. Against the `veryfast` CRF 17 used before, one 1080p clip encodes
1.6-2x faster and eight at once up to 2.1x, at as high a PSNR against the rendered frames and SSIM
0.98 or better, in files 2-3 times as large (`ROR_VIDEO_X264=veryfast` gives the old arguments,
through ffmpeg). ffmpeg is `ROR_FFMPEG`, else the first on PATH, else winget's `Gyan.FFmpeg`; without
it a video that needs it is an animated PNG and the summary says so.

## How frames are made

- The model is read once. A playback that only emits has its scene extracted once, every frame then
  working out only its particles and trails; one that changes only the Lighting reads its Lighting
  alone each frame; otherwise one document makes the frames in order, the playback resolved against
  it once and each moment's changes undone after it, the roots and models the playback never changes
  reused.
- Scenes are built a few frames ahead while the GPU draws, their GUI images found ahead (16 frames at
  a time, from memory, disk or rur) and put on the GPU a frame or two before their draw.
- A frame whose scene is the one before's, value for value, under the same camera is that frame's
  image again (`ROR_FRAME_REUSE=off` draws it anyway). A still camera keeps its ambient occlusion
  from frame to frame and makes it again only round what moved (`ROR_AO_HISTORY=off`).
- A frame takes one of the service's GPU slots once it is recorded, until the GPU has drawn it, so
  several clips record side by side and other renders interleave.

## What the summary measures

`summary.json` holds the frames, timings, the box framed, the particles at each sheet moment and the
most at any frame, the parts, emitters, beams and trails in the world in any frame, the emitters
flying weightless over the clip, missing assets and warnings, and the first frame's camera as
`views`. For a VFX review (`crates/ror/src/lint.rs`):

- `frameStats`, one per encoded frame: `{"time", "frame", "nearWhite", "bright", "luma"}`, the pixels
  whose red, green and blue are all 230 or more, the pixels of Rec.709 luma 0.8 or more, and the
  mean luma (0-1), of the pixels the encoder takes.
- `weightless`: emitters whose particles fly with neither Drag nor Acceleration.
- `lint.fades` `{"count", "entries": [{"path", "property", "kind"}]}`: plain linear fades, `sequence`
  (a Transparency of two keypoints or several on one line) or `tween` (a tween whose keys are linear
  in time).
- `lint.restraint` `{"duplicateTextures", "mixedBlends"}`: within one form (the emitters of one
  parent), one texture on two emitters or more, and LightEmission values more than 1 apart or of both
  signs.
- `lint.frozen` `[{"path", "drag", "lifetime"}]`: emitters with Drag 15 or more whose Lifetime's
  larger end is a second or more: a cloud that stops dead and hangs.
- `lint.refused` `[{"path", "asset", "reason", "paths"}]`: images Roblox refuses to everyone
  (`moderated`, `archived`, `unreviewed`), which nothing draws.
- `lint.rigid` `{"count", "beams", "entries": [{"path", "copies", "textureSpeed", "curved"}]}`:
  beams drawn as a static straight streak (TextureSpeed 0 and no curve in every frame they draw).
- `counts.effects` `[{"root", "emitters", "textures", "beams", "trails"}]`: each effect (a child of
  the loaded root) over the clip.
- `lines`: what `ror video` prints to stderr about them.

`ROR_VIDEO_PROFILE=1` adds the output stages' times as `outputProfile`.
