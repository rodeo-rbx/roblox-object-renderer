//! A ror binary outside a checkout takes its folders from the per-user cache, not from the current
//! directory or the folder it sits in. The same binary inside a checkout keeps the checkout's folders.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The file name of an executable called `stem` on this platform.
fn executable(stem: &str) -> String {
	if cfg!(windows) { format!("{stem}.exe") } else { stem.to_string() }
}

/// ror's per-user cache when every cache variable points at `home`.
fn cache_under(home: &Path) -> PathBuf {
	if cfg!(target_os = "macos") { home.join("Library/Caches/ror") } else { home.join("ror") }
}

/// A command with every per-user folder variable pointing at `home`, so the real ones are never
/// touched. `state` sets `ROR_STATE_DIR`.
fn defaults_command(binary: &Path, cwd: &Path, home: &Path, state: Option<&Path>) -> Command {
	let mut command = Command::new(binary);
	command.current_dir(cwd);
	command.env("LOCALAPPDATA", home).env("HOME", home).env("XDG_CACHE_HOME", home);
	command.env("ROR_STORE", "0").env("ROR_DEFAULTS_PROBE", home);
	command.env_remove("ROR_STATE_DIR").env_remove("ROR_RUR").env_remove("ROR_REE");
	if let Some(state) = state {
		command.env("ROR_STATE_DIR", state);
	}
	command
}

/// Runs `binary` with isolated defaults and returns what it printed.
fn run(binary: &Path, cwd: &Path, home: &Path, state: Option<&Path>, args: &[&str]) -> String {
	let output = defaults_command(binary, cwd, home, state).args(args).output().unwrap();
	assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
	String::from_utf8(output.stdout).unwrap()
}

/// Runs the `store_defaults_probe` in `probe` with `ROR_STORE` unset, so the probe sees the default
/// store folder.
fn run_store_probe(probe: &Path, cwd: &Path, home: &Path) {
	let mut command = defaults_command(probe, cwd, home, None);
	command.env_remove("ROR_STORE").args(["--exact", "store_defaults_probe"]);
	let output = command.output().unwrap();
	// the test harness prints a failed probe's assertion on stdout
	assert!(output.status.success(), "{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
}

/// Runs `binary` in `cwd` on a machine with no per-user cache location: none of the variables that
/// name one is set. `state` sets `ROR_STATE_DIR` again.
fn run_without_cache_location(binary: &Path, cwd: &Path, state: Option<&Path>, args: &[&str]) -> std::process::Output {
	let mut command = Command::new(binary);
	command.current_dir(cwd).args(args);
	for name in ["LOCALAPPDATA", "HOME", "XDG_CACHE_HOME", "ROR_STATE_DIR"] {
		command.env_remove(name);
	}
	if let Some(state) = state {
		command.env("ROR_STATE_DIR", state);
	}
	command.output().unwrap()
}

#[test]
fn standalone_defaults_ignore_the_working_and_binary_folders() {
	let root = std::env::temp_dir().join(format!("ror-standalone-{}", std::process::id()));
	let bin = root.join("bin");
	let cwd = root.join("working");
	let home = root.join("home");
	for dir in [&bin, &cwd, &home] {
		std::fs::create_dir_all(dir).unwrap();
	}
	// copies of ror and of this test binary, in a folder that is not a checkout
	let binary = bin.join(executable("ror"));
	std::fs::copy(env!("CARGO_BIN_EXE_ror"), &binary).unwrap();
	let probe = bin.join(executable("probe"));
	std::fs::copy(std::env::current_exe().unwrap(), &probe).unwrap();

	// Bait that a standalone binary must ignore: an unrelated Cargo project above it, a `.env` beside it
	// and in the working folder, and rur and ree where a checkout's sibling folders would be.
	std::fs::write(root.join("Cargo.lock"), "").unwrap();
	for dir in [&bin, &cwd] {
		std::fs::write(dir.join(".env"), "ROR_STATE_DIR=unexpected\n").unwrap();
	}
	for file in ["roblox-ui-renderer/cli/rur.mjs".to_string(), format!("roblox-engine-emulator/target/release/{}", executable("ree"))] {
		let path = root.join(file);
		std::fs::create_dir_all(path.parent().unwrap()).unwrap();
		std::fs::write(path, "").unwrap();
	}

	// A host with only Cargo.lock keeps its old library defaults; ror explicitly opts into its own.
	run(&probe, &cwd, &home, None, &["--exact", "host_defaults_probe"]);
	let explicit_host = root.join("host-state");
	run(&probe, &cwd, &home, Some(&explicit_host), &["--exact", "host_defaults_probe"]);
	run(&probe, &cwd, &home, None, &["--exact", "host_setters_probe"]);
	run(&probe, &cwd, &home, None, &["--exact", "runtime_defaults_probe"]);
	run_store_probe(&probe, &cwd, &home);

	// the binary: state and the default output folder come from the per-user cache
	let state = cache_under(&home).join("state");
	assert_eq!(run(&binary, &cwd, &home, None, &["--version"]).trim(), concat!("ror ", env!("CARGO_PKG_VERSION")));
	assert!(run(&binary, &cwd, &home, None, &["service", "status"]).contains(&state.display().to_string()));
	let cleaned = run(&binary, &cwd, &home, None, &["clean", "--dry-run"]);
	assert!(cleaned.contains(&state.join("out").display().to_string()));

	// ROR_STATE_DIR still wins
	let explicit = root.join("explicit");
	assert!(run(&binary, &cwd, &home, Some(&explicit), &["service", "status"]).contains(&explicit.display().to_string()));

	// With no per-user cache location, the commands that need the state dir fail with an ordinary
	// error, not a panic, and `ROR_STATE_DIR` still makes them work.
	for args in [["service", "status"], ["clean", "--dry-run"]] {
		let failed = run_without_cache_location(&binary, &cwd, None, &args);
		assert_eq!(failed.status.code(), Some(1));
		let error = String::from_utf8(failed.stderr).unwrap();
		assert!(error.contains("ror: no per-user cache directory; set ROR_STATE_DIR"), "{error}");
		assert!(!error.contains("panicked"), "{error}");
		assert!(run_without_cache_location(&binary, &cwd, Some(&explicit), &args).status.success());
	}
	// Version and help are handled by clap, and inspecting or comparing local renders needs no cache,
	// so these succeed with no cache location.
	let rendered = root.join("rendered");
	ror::output::write_png(&rendered.join("iso.png"), 1, 1, &[1, 2, 3, 255], true).unwrap();
	let printed_without_cache = |args: &[&str]| {
		let result = run_without_cache_location(&binary, &cwd, None, args);
		assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
		String::from_utf8(result.stdout).unwrap()
	};
	assert_eq!(printed_without_cache(&["--version"]).trim(), concat!("ror ", env!("CARGO_PKG_VERSION")));
	assert!(printed_without_cache(&["--help"]).contains("Usage:"));
	let inspected = printed_without_cache(&["inspect", "../rendered", "--at", "0,0", "--json"]);
	let inspected: serde_json::Value = serde_json::from_str(&inspected).unwrap();
	assert_eq!(inspected["rgba"], serde_json::json!([1, 2, 3, 255]));
	let diffed = printed_without_cache(&["diff", "../rendered", "../rendered", "--json"]);
	let diffed: serde_json::Value = serde_json::from_str(&diffed).unwrap();
	assert_eq!(diffed[0]["changed"], 0);

	// the same binary inside a checkout keeps the checkout's folders, whatever the working folder is
	let checkout = root.join("checkout");
	std::fs::create_dir_all(checkout.join("crates/ror")).unwrap();
	std::fs::write(checkout.join("Cargo.lock"), "").unwrap();
	std::fs::write(checkout.join("crates/ror/Cargo.toml"), "").unwrap();
	let checkout_binary = checkout.join(binary.file_name().unwrap());
	std::fs::copy(&binary, &checkout_binary).unwrap();
	let canonical = checkout.canonicalize().unwrap();
	assert!(run(&checkout_binary, &cwd, &home, None, &["service", "status"]).contains(&canonical.join(".cache").display().to_string()));
	assert!(run(&checkout_binary, &cwd, &home, None, &["clean", "--dry-run"]).contains(&canonical.join("out").display().to_string()));
	// A service copy inherits the starter's choice, rather than detecting its own folder.
	let checkout_probe = checkout.join(executable("probe"));
	std::fs::copy(&probe, &checkout_probe).unwrap();
	run(&checkout_probe, &cwd, &home, None, &["--exact", "inherited_defaults_probe"]);
	run_store_probe(&checkout_probe, &cwd, &home);

	// nothing was written beside the binary or into the working folder
	assert!(!cwd.join(".cache").exists());
	assert!(!bin.join(".cache").exists());
	std::fs::remove_dir_all(root).unwrap();
}

// The probes below are not tests of their own: `standalone_defaults_ignore_the_working_and_binary_folders`
// runs a copy of this test binary where each one needs it. Run normally (`ROR_DEFAULTS_PROBE` unset),
// they do nothing.

/// ror's binary outside any checkout: `init_binary` takes the per-user cache, and no checkout is found.
/// The copy sits in a folder that is not a checkout, so `current_exe()` is a standalone location.
#[test]
fn runtime_defaults_probe() {
	let Some(home) = std::env::var_os("ROR_DEFAULTS_PROBE").map(PathBuf::from) else { return };
	ror::output::init_binary(&std::env::current_exe().unwrap(), None).unwrap();
	let state = cache_under(&home).join("state");
	assert_eq!(ror::output::checkout_dir(), None);
	assert_eq!(ror::output::state_dir(), state);
	assert_eq!(ror::output::studio_capture_path(None).unwrap(), state.join("out/studio-capture.png"));
	for explicit in [PathBuf::from("out/studio-capture.png"), home.join("capture.png")] {
		assert_eq!(ror::output::studio_capture_path(Some(explicit.clone())).unwrap(), explicit);
	}
	assert_eq!(ror::output::store_dir(), None);
	assert_eq!(ror::output::asset_config(state, true).repo_dir, None);
	assert_eq!(ror::script::locate(), None);
	assert_eq!(ror::render::rur::locate(), None);
}

/// A binary in a checkout keeps the host's store; only a standalone binary uses the platform's
/// per-user cache. The two differ on macOS. Run with `ROR_STORE` unset.
#[test]
fn store_defaults_probe() {
	let Some(home) = std::env::var_os("ROR_DEFAULTS_PROBE").map(PathBuf::from) else { return };
	ror::output::init_binary(&std::env::current_exe().unwrap(), None).unwrap();
	let in_checkout = ror::output::checkout_dir().is_some();
	// every cache variable points at `home`, so the host's cache folder is `home/ror`
	let cache = if in_checkout { home.join("ror") } else { cache_under(&home) };
	assert_eq!(ror::output::store_dir(), Some(cache.join("store")));
}

/// A service copy: the probe sits inside a ror checkout, but its starter (`source`) did not, so the
/// service takes the starter's lack of a checkout and the `--state-dir` it was given.
#[test]
fn inherited_defaults_probe() {
	let Some(home) = std::env::var_os("ROR_DEFAULTS_PROBE").map(PathBuf::from) else { return };
	// the starter was in a folder with only a `Cargo.lock`, which is not a ror checkout
	let source = home.parent().unwrap().join("bin/ror");
	let chosen = home.join("chosen");
	ror::output::init_binary(&source, Some(chosen.clone())).unwrap();
	assert_eq!(ror::output::checkout_dir(), None);
	assert_eq!(ror::output::state_dir(), chosen);
	assert_eq!(ror::output::studio_capture_path(None).unwrap(), chosen.join("out/studio-capture.png"));
}

/// A host: a program that links the library from a Cargo project that has no ror crate. It keeps the
/// library's defaults, that project's `.cache` (or `ROR_STATE_DIR`), and never calls `init_binary`.
#[test]
fn host_defaults_probe() {
	let Some(home) = std::env::var_os("ROR_DEFAULTS_PROBE").map(PathBuf::from) else { return };
	let host = home.parent().unwrap().canonicalize().unwrap();
	assert!(!host.join("crates/ror/Cargo.toml").exists());
	assert_eq!(ror::output::repo_dir(), host);
	assert_eq!(ror::output::checkout_of(&std::env::current_exe().unwrap()), Some(host.clone()));
	assert_eq!(ror::output::checkout_dir(), Some(host.clone()));
	let state = ror::output::state_dir();
	let expected = std::env::var_os("ROR_STATE_DIR").map(PathBuf::from).unwrap_or_else(|| host.join(".cache"));
	assert_eq!(state, expected);
	assert_eq!(ror::output::asset_config(state, true).repo_dir, Some(host));
}

/// A host chooses its folders with `set_repo_dir` and `set_state_dir`, and the state dir follows the
/// checkout until it is set itself.
#[test]
fn host_setters_probe() {
	let Some(home) = std::env::var_os("ROR_DEFAULTS_PROBE").map(PathBuf::from) else { return };
	let chosen = home.join("host-checkout");
	ror::output::set_repo_dir(chosen.clone());
	assert_eq!(ror::output::repo_dir(), chosen);
	assert_eq!(ror::output::state_dir(), chosen.join(".cache"));
	assert_eq!(ror::output::asset_config(ror::output::state_dir(), true).repo_dir, Some(chosen));
	let state = home.join("host-state");
	ror::output::set_state_dir(state.clone());
	assert_eq!(ror::output::state_dir(), state);
}
