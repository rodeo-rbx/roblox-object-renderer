//! A Luau script as the input of `render` and `video` (docs/scripts.md): the engine emulator (ree,
//! roblox-engine-emulator) runs it and records what it puts in the Workspace over time, and ror draws
//! that recording, its `scene.rbxm` with its `playback.json`. The script is a ree harness: a
//! ModuleScript returning `function(Context)` that parents what it builds to the Workspace.
//!
//! ree is found through `ROR_REE` (its binary, or a checkout whose release build is run as it is:
//! `cargo build --release` there), else the release build of a checkout beside this one
//! (`../roblox-engine-emulator`) when ror runs from its checkout, else `ree` on the PATH (a tool
//! manager's install, pinned per project).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

/// Whether an input names a script rather than a model, a place or an asset.
pub fn is_script(input: &str) -> bool {
	let lower = input.to_ascii_lowercase();
	lower.ends_with(".luau") || lower.ends_with(".lua")
}

/// ree's binary, if there is one (the module's doc says where it is looked for).
pub fn locate() -> Option<PathBuf> {
	let checkout = std::env::current_exe().ok().and_then(|exe| crate::output::checkout_of(&exe));
	locate_from(std::env::var_os("ROR_REE"), checkout.as_deref(), std::env::var_os("PATH"))
}

/// [`locate`] with its inputs given: `ROR_REE`, the checkout ror runs from (none for a binary outside
/// one, whose working directory says nothing of where ree is) and the PATH. A `ROR_REE` naming nothing
/// runnable finds nothing rather than another ree.
fn locate_from(given: Option<std::ffi::OsString>, checkout: Option<&Path>, path: Option<std::ffi::OsString>) -> Option<PathBuf> {
	let name = if cfg!(windows) { "ree.exe" } else { "ree" };
	let release = |checkout: &Path| crate::output::plain_path(checkout).join("target").join("release").join(name);
	if let Some(given) = given.filter(|given| !given.is_empty()).map(PathBuf::from) {
		let binary = if given.is_file() { crate::output::plain_path(&given) } else { release(&given) };
		return binary.is_file().then_some(binary);
	}
	if let Some(beside) = checkout.map(|checkout| release(&checkout.join("..").join("roblox-engine-emulator"))).filter(|binary| binary.is_file()) {
		return Some(beside);
	}
	std::env::split_paths(&path?).map(|dir| dir.join(name)).find(|binary| binary.is_file())
}

/// What a script's run is given: `--context` (its Context fields, JSON) and `--mount`
/// (`Service/Name=<dir|file>`, for its requires).
#[derive(Default)]
pub struct Options<'a> {
	pub context: Option<&'a str>,
	pub mounts: &'a [String],
	pub offline: bool,
}

/// A recorded run: the files ror draws and what the run reported.
pub struct Recording {
	pub scene: PathBuf,
	pub playback: PathBuf,
	/// ree's summary (status, frames, instances, errors) with the error lines of `events.json`
	pub report: Value,
	pub errors: Vec<String>,
}

/// Runs `script` in ree for `seconds` of play-mode time (60 frames a second) and records it into `out`.
/// Errors when the run put nothing in the Workspace; error lines of a run that did are in `errors`.
pub fn record(script: &Path, seconds: f32, options: &Options, out: &Path) -> Result<Recording> {
	if !seconds.is_finite() || seconds <= 0.0 {
		bail!("a script's recording duration must be finite and positive (got {seconds})");
	}
	if !script.is_file() {
		bail!("{}: no such script", script.display());
	}
	let ree = locate().context(
		"a .luau input runs in the engine emulator, which was not found: set ROR_REE to ree's binary or to a roblox-engine-emulator checkout built with `cargo build --release`, put such a checkout beside this one, or put ree on the PATH",
	)?;
	let mut command = Command::new(&ree);
	command.arg("record").arg("--harness").arg(crate::output::plain_path(script)).arg("--out").arg(out).arg(format!("--seconds={seconds}"));
	// An explicit service mount keeps ree from loading its unrelated default project or snapshots.
	// A class mount at the service itself is a no-op (the service already exists).
	if options.mounts.is_empty() {
		command.arg("--instance").arg("ReplicatedStorage=ReplicatedStorage");
	}
	if options.offline {
		command.arg("--offline");
	}
	if let Some(context) = options.context {
		command.arg("--context").arg(context);
	}
	for mount in options.mounts {
		command.arg("--mount").arg(mount);
	}
	let output = command.output().with_context(|| format!("running {}", ree.display()))?;
	if !output.status.success() {
		bail!("ree record {} ({}): {}", script.display(), output.status, String::from_utf8_lossy(&output.stderr).trim());
	}
	let stdout = String::from_utf8_lossy(&output.stdout);
	let summary: Value = stdout
		.find('{')
		.and_then(|start| serde_json::from_str(&stdout[start..]).ok())
		.with_context(|| format!("ree record {}: {}", script.display(), String::from_utf8_lossy(&output.stderr).trim()))?;

	let events_file = out.join("events.json");
	let events: Value = serde_json::from_slice(&std::fs::read(&events_file).with_context(|| format!("reading {}", events_file.display()))?).with_context(|| format!("reading {}", events_file.display()))?;
	let errors: Vec<String> = events["output"]
		.as_array()
		.into_iter()
		.flatten()
		.filter(|line| line["kind"] == "error")
		.filter_map(|line| line["message"].as_str().map(str::to_string))
		.collect();
	if summary["instances"].as_u64().unwrap_or(0) == 0 {
		let why = summary["failure"].as_str().map(str::to_string).or_else(|| errors.first().cloned());
		match why {
			Some(why) => bail!("{}: {why}", script.display()),
			None => bail!("{}: the script put nothing in the Workspace (parent what it builds to workspace)", script.display()),
		}
	}
	let report = json!({
		"script": crate::output::plain_path(script).display().to_string(),
		"status": summary["status"],
		"frames": summary["frames"],
		"instances": summary["instances"],
		"errors": errors,
		"record": crate::output::plain_path(out).display().to_string(),
	});
	crate::output::write_json_atomic(&out.join("script.json"), &report, true)?;
	Ok(Recording { scene: out.join("scene.rbxm"), playback: out.join("playback.json"), report, errors })
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn ree_is_ror_ree_else_a_checkout_beside_ror_s_else_on_the_path() {
		let name = if cfg!(windows) { "ree.exe" } else { "ree" };
		let root = std::env::temp_dir().join(format!("ror-locate-ree-test-{}", std::process::id()));
		let _ = std::fs::remove_dir_all(&root);
		let ror = root.join("ror");
		let beside = root.join("roblox-engine-emulator").join("target").join("release");
		let installed = root.join("installs").join("ree");
		let elsewhere = root.join("elsewhere");
		for dir in [&ror, &beside, &installed, &elsewhere] {
			std::fs::create_dir_all(dir).unwrap();
		}
		std::fs::write(installed.join(name), "").unwrap();
		let path = Some(std::env::join_paths([&elsewhere, &installed]).unwrap());
		let found = |given: Option<&Path>, checkout: Option<&Path>, path: Option<std::ffi::OsString>| locate_from(given.map(|given| given.as_os_str().to_owned()), checkout, path);

		// on the PATH: a binary outside any checkout, or a checkout with no ree built beside it
		assert_eq!(found(None, None, path.clone()), Some(installed.join(name)));
		assert_eq!(found(None, Some(&ror), path.clone()), Some(installed.join(name)));
		assert_eq!(found(None, None, None), None);
		// a ree built beside ror's checkout comes before the PATH's, but only for ror run from its checkout
		std::fs::write(beside.join(name), "").unwrap();
		assert_eq!(found(None, Some(&ror), path.clone()), Some(crate::output::plain_path(&ror.join("..").join("roblox-engine-emulator")).join("target").join("release").join(name)));
		assert_eq!(found(None, None, path.clone()), Some(installed.join(name)));
		// ROR_REE comes first: ree's binary, or a checkout's release build
		assert_eq!(found(Some(&installed.join(name)), Some(&ror), None), Some(installed.join(name)));
		assert_eq!(found(Some(&root.join("roblox-engine-emulator")), None, path.clone()), Some(beside.join(name)));
		// and names the one to run: nothing there finds nothing, not another ree
		assert_eq!(found(Some(&elsewhere), Some(&ror), path.clone()), None);
		assert_eq!(found(Some(&elsewhere.join(name)), None, path.clone()), None);
		// an empty ROR_REE is unset
		assert_eq!(found(Some(Path::new("")), None, path), Some(installed.join(name)));
		std::fs::remove_dir_all(&root).unwrap();
	}
}
