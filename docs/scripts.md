# Scripts as input

A model can be its code: `render` and `video` take a `.luau` script, run it in the engine emulator
([roblox-engine-emulator](https://github.com/Anime-Vanguards/roblox-engine-emulator), ree) and draw
what it puts in the Workspace, without Studio. A model or an effect is built and polished as a
script, and the same script runs in Studio once it looks right. The code is
`crates/ror/src/script.rs`.

```luau
-- Build.luau
return function(Context)
	local Model = Instance.new("Model")
	for Index = 0, (Context.Floors or 5) - 1 do
		local Part = Instance.new("Part")
		Part.Anchored = true
		Part.Size = Vector3.new(4 - Index * 0.6, 2, 4 - Index * 0.6)
		Part.Position = Vector3.new(0, 1 + Index * 2, 0)
		Part.Parent = Model
	end
	Model.Parent = workspace
end
```

```sh
ror render Build.luau --views front,iso --context '{"Floors":3}'
ror render Build.luau --time 1                   # after a second of its tweens and task.waits
ror render Build.luau --mount ReplicatedStorage/Shared=src/shared   # its requires
ror video Build.luau --duration 2
```

- The script is a ree harness: a ModuleScript returning `function(Context)` (or `{ Run = ... }`) that
  parents what it builds to the Workspace, and may yield. `Context` holds `--context`'s fields. It
  runs in play mode at 60 frames a second: TweenService, `task`, RunService events, `:Emit` and
  AnimationTracks run; physics, the server and CSG (`UnionAsync`) do not (ree's
  [docs/cli.md](https://github.com/Anime-Vanguards/roblox-engine-emulator/blob/main/docs/cli.md#what-is-not-simulated)
  "What is not simulated").
- `render` records the run to `--time` (default 0, what the script built at once) and draws that
  moment; `video` records to the clip's end. The recording is the playback (`--playback` is refused;
  `--emit`, `--set`, `--tween` and the rest add to it, [playback.md](playback.md)), and nothing else
  is emitted.
- Requires reach other code through services. `--mount Service/Name=<dir|file>` (repeatable) places
  a source directory or model file at an instance path, as ree's `--mount` does. The run starts with
  a fresh world; unrelated default projects and Studio snapshots are never mounted. `--offline` also
  applies to the emulator.
- Output: the render's or video's files in `--out` (default `<state dir>/out/<script name>`, `-video`
  for a video), the recording (`scene.rbxm`, `playback.json`, `events.json`) in its `record/`, and a
  `script` entry in the CLI summary: the run's status, frames, instances and error lines (also printed
  as `ror: script error: ...`). This entry is saved in `record/script.json` too, including for
  `--sheet-first` videos. A run that puts nothing in the Workspace is an error naming why.
- ree is `ROR_REE` (its binary, or its checkout built with `cargo build --release`), else the
  release build of `../roblox-engine-emulator` beside this checkout when ror runs from its checkout,
  else `ree` on the PATH (a tool manager's install, pinned per project).
