//! Checks, without a GPU, that a clip's scenes made with the roots its playback never changes extracted
//! once (video.rs, `ROR_SCENE_REUSE`) are the scenes extracted whole: every recording (a folder holding
//! `scene.rbxm` and `playback.json`, as `ree record` and other recording tools write them) under the
//! folders given, at `--fps` frames a second over its playback, compared value for value.
//!
//!     cargo run --release --example tools -- scene_reuse <folder> [<folder> ...] [--fps 10] [--seconds 3]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ror::job::SceneCache;
use ror::render::ViewRequest;
use ror::scene::Playback;
use ror::video::{VideoFormat, VideoJob};

fn recordings(dir: &Path, found: &mut Vec<PathBuf>) {
	if dir.join("scene.rbxm").is_file() && dir.join("playback.json").is_file() {
		found.push(dir.to_path_buf());
	}
	for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
		if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
			recordings(&entry.path(), found);
		}
	}
}

pub fn main() -> Result<()> {
	ror::init_threads();
	let mut args: Vec<String> = std::env::args().skip(1).collect();
	let mut option = |name: &str, default: f32| -> Result<f32> {
		match args.iter().position(|arg| arg == name) {
			Some(index) => {
				let value = args.get(index + 1).context(name.to_string())?.parse()?;
				args.drain(index..index + 2);
				Ok(value)
			}
			None => Ok(default),
		}
	};
	let fps = option("--fps", 10.0)?;
	let seconds = option("--seconds", 3.0)?;
	let mut found = Vec::new();
	for dir in &args {
		recordings(Path::new(dir), &mut found);
	}
	found.sort();
	let scenes = SceneCache::default();
	let (mut checked, mut frames, mut skipped, mut failed) = (0, 0, 0, 0);
	let (mut whole_ms, mut reused_ms) = (0.0, 0.0);
	for dir in &found {
		let playback: Playback = match std::fs::read_to_string(dir.join("playback.json")).ok().and_then(|text| serde_json::from_str(&text).ok()) {
			Some(playback) => playback,
			None => {
				skipped += 1;
				continue;
			}
		};
		let duration = playback.frames.as_ref().and_then(|frames| frames.last().copied()).unwrap_or(seconds).min(seconds).max(0.1);
		let count = ((duration * fps).round() as usize).max(1);
		let times: Vec<f32> = (0..count).map(|index| index as f32 / fps).collect();
		let job = VideoJob {
			input: dir.join("scene.rbxm").display().to_string(),
			select: None,
			out: None,
			view: ViewRequest::default(),
			name: "video".into(),
			playback: Some(playback),
			playback_file: None,
			start: 0.0,
			duration,
			fps,
			speed: 1.0,
			times: None,
			orbit: 0.0,
			format: VideoFormat::Png,
			sheet: 0,
			materials_from: None,
			lighting_from: None,
			map: None,
			frame: None,
			focus: None,
			ffmpeg: None,
			camera_track: None,
			sheet_first: false,
			copies: Vec::new(),
			relative_to: None,
			nvenc: None,
		};
		match ror::video::check_scene_reuse(&job, &scenes, &times) {
			Ok((0, ..)) => skipped += 1,
			Ok((count, reused, roots, whole, reused_time)) => {
				checked += 1;
				frames += count;
				whole_ms += whole;
				reused_ms += reused_time;
				if whole > 200.0 || reused == 0 {
					println!("{}: {count} frames, {reused} of {roots} roots extracted once, {:.1} -> {:.1} ms a frame", dir.display(), whole / count as f64, reused_time / count as f64);
				}
			}
			Err(error) => {
				failed += 1;
				println!("FAILED {}: {error:#}", dir.display());
			}
		}
	}
	println!("{checked} recordings, {frames} frames compared, {failed} failed, {skipped} skipped (no playback, or one that changes nothing or only the Lighting); extraction {:.0} -> {:.0} ms in all", whole_ms, reused_ms);
	if failed > 0 {
		std::process::exit(1);
	}
	Ok(())
}
