//! The content store: downloads, and the data made from them whose bytes depend only on their input
//! and the code that makes them, kept once for every ror build, checkout and state dir of the user on
//! this machine (`%LOCALAPPDATA%\ror\store`; `ROR_STORE` names another folder, `ROR_STORE=0` turns it
//! off). A fresh clone, worktree or an executable built outside any checkout starts with an empty
//! state dir: without the store its first render downloaded and transcoded everything again (a 4K Sun
//! Temple clip 89 s instead of 4.6).
//!
//! - `assets/`: downloads, as a state dir's ([`crate::AssetConfig::store`]): a state dir missing one
//!   takes the store's (hard-linked in), and gives the store each one it downloads or takes from
//!   another state dir.
//! - The renderer keeps its texture files under `textures/<code>/` (the code that makes their texels:
//!   another build's never stand in for this one's), its Vulkan pipeline cache under `pipelines/` and
//!   the GPU its services start on (`gpu.json`) for state dirs that have none.
//! - `state-dirs.txt`: the state dirs that used the store, looked in for what the store lacks (their
//!   downloads and texture files, kept before the store was), so even its first use finds them.
//!
//! Files are only ever placed by a rename (written whole under a temporary name first, or hard-linked),
//! never written into, so concurrent writers and readers in any process see a file whole or not at
//! all; a file taken from the store is checked (a download's size and URI against its metadata, a
//! texture's header against its length) before it is used. The store keeps [`budget`] bytes: the least
//! recently used files go past it ([`maybe_prune`], at most hourly for the machine; a file taken counts
//! as used).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

/// The registry of state dirs that used the store.
const STATE_DIRS: &str = "state-dirs.txt";
/// The state dirs kept in the registry (the most recently registered).
const MAX_STATE_DIRS: usize = 32;
/// Pruned at most this often on the machine (`.pruned`'s modification time).
const PRUNE_EVERY: Duration = Duration::from_secs(3600);
/// A prune claim (`.prune.lock`) this old was left by a process that ended mid-prune.
const STALE_CLAIM: Duration = Duration::from_secs(600);
/// A file taken from the store counts as used again once its time is older than this (one write of
/// its time an hour at most, not one each read).
const TOUCH_AFTER: Duration = Duration::from_secs(3600);
/// Temporary files left by a process that ended mid-write go after this.
const STALE_TEMP: Duration = Duration::from_secs(3600);

/// The store's folder for a program that links the library (a host): `ROR_STORE` (`0`, `off` or
/// empty: none), else `%LOCALAPPDATA%\ror\store` on Windows, `$XDG_CACHE_HOME/ror/store` or
/// `~/.cache/ror/store` elsewhere. ror's own binary uses it too when it runs from a checkout, so a
/// store that checkout builds already have stays where it is.
pub fn default_dir() -> Option<PathBuf> {
	store_in(host_cache_dir)
}

/// The store's folder for ror's binary outside a checkout: `ROR_STORE` as in [`default_dir`], else
/// `store/` in [`user_cache_dir`]. Hosts and binaries in a checkout use [`default_dir`] instead.
pub fn standalone_dir() -> Option<PathBuf> {
	store_in(user_cache_dir)
}

/// The per-user cache folder a host has always used. It is [`user_cache_dir`] except on macOS, where
/// it is `~/.cache/ror` as on Linux, not `~/Library/Caches/ror`.
fn host_cache_dir() -> Option<PathBuf> {
	host_cache_dir_for(std::env::consts::OS, |name| std::env::var_os(name).map(PathBuf::from))
}

/// [`host_cache_dir`] with an explicit operating system and environment for tests.
fn host_cache_dir_for(os: &str, env: impl Fn(&str) -> Option<PathBuf>) -> Option<PathBuf> {
	let base = if os == "windows" {
		env("LOCALAPPDATA")
	} else {
		env("XDG_CACHE_HOME").or_else(|| env("HOME").map(|home| home.join(".cache")))
	};
	base.filter(|base| base.is_absolute()).map(|base| base.join("ror"))
}

/// `ROR_STORE` when it is set (no store if it says `0`, `off`, `none` or is empty), else `store/` in
/// the cache folder that `cache_dir` finds.
fn store_in(cache_dir: impl FnOnce() -> Option<PathBuf>) -> Option<PathBuf> {
	let Some(value) = std::env::var_os("ROR_STORE") else {
		return cache_dir().map(|cache| cache.join("store"));
	};
	let text = value.to_string_lossy();
	let text = text.trim();
	if text.is_empty() || matches!(text, "0" | "off" | "none") {
		return None;
	}
	Some(PathBuf::from(value))
}

/// ror's per-user cache folder, which holds both the content store and a standalone binary's state dir.
/// `None` when the environment names no absolute home or cache folder.
pub fn user_cache_dir() -> Option<PathBuf> {
	cache_dir_for(std::env::consts::OS, |name| std::env::var_os(name).map(PathBuf::from))
}

/// [`user_cache_dir`] for operating system `os`, reading variables through `env` so a test can
/// stand in for the environment.
fn cache_dir_for(os: &str, env: impl Fn(&str) -> Option<PathBuf>) -> Option<PathBuf> {
	// a relative folder would follow the current directory, so it counts as unset
	let absolute = |name| env(name).filter(|path| path.is_absolute());
	let base = match os {
		"windows" => absolute("LOCALAPPDATA"),
		"macos" => absolute("HOME").map(|home| home.join("Library/Caches")),
		_ => absolute("XDG_CACHE_HOME").or_else(|| absolute("HOME").map(|home| home.join(".cache"))),
	};
	base.map(|base| base.join("ror"))
}

/// The byte budget (`ROR_STORE_MB`, default 4096).
pub fn budget() -> u64 {
	std::env::var("ROR_STORE_MB").ok().and_then(|value| value.trim().parse::<u64>().ok()).unwrap_or(4096) << 20
}

/// A path as the registry compares it: absolute, without Windows' verbatim prefix, one kind of
/// separator, no trailing one, case folded on Windows (no file system access: the registry's lines
/// are written canonical, [`register`]).
fn normal(path: &Path) -> String {
	let path = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
	let text = path.to_string_lossy();
	let text = text.strip_prefix(r"\\?\").unwrap_or(&text);
	if cfg!(windows) { text.replace('/', r"\").trim_end_matches('\\').to_lowercase() } else { text.trim_end_matches('/').to_string() }
}

/// `path` as the registry records it (canonical where it exists).
fn canonical(path: &Path) -> String {
	let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
	let text = path.to_string_lossy();
	text.strip_prefix(r"\\?\").unwrap_or(&text).to_string()
}

/// Records `state_dir` in the store's registry (once).
pub fn register(store: &Path, state_dir: &Path) {
	let file = store.join(STATE_DIRS);
	let path = canonical(state_dir);
	let own = normal(Path::new(&path));
	let text = std::fs::read_to_string(&file).unwrap_or_default();
	let lines: Vec<&str> = text.lines().map(str::trim).filter(|line| !line.is_empty()).collect();
	if lines.iter().any(|line| normal(Path::new(line)) == own) {
		return;
	}
	let _ = std::fs::create_dir_all(store);
	if lines.len() >= MAX_STATE_DIRS * 2 {
		// compacted: the ones that still exist, the newest kept
		let mut kept: Vec<String> = Vec::new();
		for line in lines.iter().rev() {
			if Path::new(line).is_dir() && !kept.iter().any(|seen| normal(Path::new(seen)) == normal(Path::new(line))) {
				kept.push(line.to_string());
			}
		}
		kept.truncate(MAX_STATE_DIRS - 1);
		kept.reverse();
		kept.push(path.to_string());
		let _ = crate::cache::write_atomic(&file, (kept.join("\n") + "\n").as_bytes());
	} else if let Ok(mut append) = std::fs::OpenOptions::new().create(true).append(true).open(&file) {
		let _ = append.write_all(format!("{path}\n").as_bytes());
	}
}

/// The state dirs in the store's registry that exist, the most recently registered first, `own` left
/// out.
pub fn state_dirs(store: &Path, own: &Path) -> Vec<PathBuf> {
	let own = normal(Path::new(&canonical(own)));
	let text = std::fs::read_to_string(store.join(STATE_DIRS)).unwrap_or_default();
	let mut out: Vec<PathBuf> = Vec::new();
	let mut seen = vec![own];
	for line in text.lines().rev().map(str::trim).filter(|line| !line.is_empty()) {
		let key = normal(Path::new(line));
		if seen.contains(&key) {
			continue;
		}
		seen.push(key);
		if Path::new(line).is_dir() {
			out.push(PathBuf::from(line));
		}
		if out.len() >= MAX_STATE_DIRS {
			break;
		}
	}
	out
}

/// Marks a file in the store used (its modification time now, when older than [`TOUCH_AFTER`]).
pub fn touch(path: &Path) {
	let old = std::fs::metadata(path).and_then(|metadata| metadata.modified()).ok().and_then(|time| time.elapsed().ok()).is_none_or(|age| age > TOUCH_AFTER);
	if old && let Ok(file) = std::fs::File::options().write(true).open(path) {
		let _ = file.set_modified(SystemTime::now());
	}
}

/// Makes `path` another name of `source`'s file (a hard link, a copy across volumes), placed by a
/// rename: an existing `path` stays as it is (the same content by its key).
pub fn link(source: &Path, path: &Path) -> std::io::Result<()> {
	crate::cache::link_atomic(source, path)
}

/// Writes `bytes` at `path` whole (a temporary file renamed into place).
pub fn write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
	crate::cache::write_atomic(path, bytes)
}

/// Keeps the store within [`budget`]: at most once an hour on the machine (whichever process claims
/// it first), on a thread of its own.
pub fn maybe_prune(store: &Path) {
	static STARTED: AtomicBool = AtomicBool::new(false);
	let marker = store.join(".pruned");
	let recent = std::fs::metadata(&marker).and_then(|metadata| metadata.modified()).ok().and_then(|time| time.elapsed().ok()).is_some_and(|age| age < PRUNE_EVERY);
	if recent || STARTED.swap(true, Ordering::SeqCst) {
		return;
	}
	let store = store.to_path_buf();
	let _ = std::thread::Builder::new().name("ror-store-prune".into()).spawn(move || {
		let claim = store.join(".prune.lock");
		let stale = std::fs::metadata(&claim).and_then(|metadata| metadata.modified()).ok().and_then(|time| time.elapsed().ok()).is_some_and(|age| age > STALE_CLAIM);
		if stale {
			let _ = std::fs::remove_file(&claim);
		}
		if std::fs::OpenOptions::new().write(true).create_new(true).open(&claim).is_err() {
			STARTED.store(false, Ordering::SeqCst);
			return;
		}
		prune(&store, budget());
		let _ = std::fs::write(&marker, b"");
		let _ = std::fs::remove_file(&claim);
	});
}

/// Removes the least recently used files past `budget` (down to three quarters of it, so a full store
/// is not pruned again at once) and temporary files left by processes that ended mid-write. Returns
/// (bytes before, bytes removed).
pub fn prune(store: &Path, budget: u64) -> (u64, u64) {
	let mut files: Vec<(SystemTime, u64, PathBuf)> = Vec::new();
	let mut stack = vec![store.to_path_buf()];
	while let Some(dir) = stack.pop() {
		let Ok(entries) = std::fs::read_dir(&dir) else { continue };
		for entry in entries.flatten() {
			let Ok(kind) = entry.file_type() else { continue };
			let path = entry.path();
			if kind.is_dir() {
				stack.push(path);
				continue;
			}
			let name = entry.file_name();
			let name = name.to_string_lossy();
			if dir == store && (name == STATE_DIRS || name.starts_with('.')) {
				continue;
			}
			let Ok(metadata) = entry.metadata() else { continue };
			let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
			if name.starts_with('.') && name.ends_with(".tmp") {
				if modified.elapsed().is_ok_and(|age| age > STALE_TEMP) {
					let _ = std::fs::remove_file(&path);
				}
				continue;
			}
			files.push((modified, metadata.len(), path));
		}
	}
	let total: u64 = files.iter().map(|(_, size, _)| size).sum();
	let mut left = total;
	if total > budget {
		files.sort();
		for (_, size, path) in files {
			if left <= budget / 4 * 3 {
				break;
			}
			if std::fs::remove_file(&path).is_ok() {
				left -= size;
			}
		}
	}
	(total, total - left)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn standalone_cache_defaults_use_only_absolute_per_user_folders() {
		let root = std::env::temp_dir().join("ror-cache-defaults");
		let env = |name: &str| match name {
			"LOCALAPPDATA" => Some(root.join("local")),
			"HOME" => Some(root.join("home")),
			"XDG_CACHE_HOME" => Some(root.join("xdg")),
			_ => None,
		};
		assert_eq!(cache_dir_for("windows", env), Some(root.join("local/ror")));
		assert_eq!(cache_dir_for("macos", env), Some(root.join("home/Library/Caches/ror")));
		assert_eq!(cache_dir_for("linux", env), Some(root.join("xdg/ror")));
		// Linux without XDG_CACHE_HOME falls back to ~/.cache
		let only_home = |name: &str| (name == "HOME").then(|| root.join("home"));
		assert_eq!(cache_dir_for("linux", only_home), Some(root.join("home/.cache/ror")));
		// relative folders are ignored
		let relative = |_: &str| Some(PathBuf::from("relative"));
		assert_eq!(cache_dir_for("linux", relative), None);
	}

	#[test]
	fn checkout_and_host_stores_keep_the_original_cache_on_every_platform() {
		let root = std::env::temp_dir().join("ror-store-defaults");
		let env = |name: &str| match name {
			"LOCALAPPDATA" => Some(root.join("local")),
			"HOME" => Some(root.join("home")),
			"XDG_CACHE_HOME" => Some(root.join("xdg")),
			_ => None,
		};
		let only_home = |name: &str| (name == "HOME").then(|| root.join("home"));
		let store = |cache: Option<PathBuf>| cache.map(|cache| cache.join("store"));

		assert_eq!(store(host_cache_dir_for("windows", env)), Some(root.join("local/ror/store")));
		// macOS uses the Linux folders here; only a standalone binary uses ~/Library/Caches
		for os in ["macos", "linux"] {
			assert_eq!(store(host_cache_dir_for(os, env)), Some(root.join("xdg/ror/store")));
			assert_eq!(store(host_cache_dir_for(os, only_home)), Some(root.join("home/.cache/ror/store")));
		}
		// A relative XDG_CACHE_HOME has always disabled the host's default store, even with HOME set.
		let relative_xdg = |name: &str| if name == "XDG_CACHE_HOME" { Some(PathBuf::from("relative")) } else { env(name) };
		assert_eq!(host_cache_dir_for("macos", relative_xdg), None);
	}

	#[test]
	fn the_registry_lists_other_state_dirs_newest_first() {
		let root = std::env::temp_dir().join(format!("ror-store-registry-{}", std::process::id()));
		let (store, a, b) = (root.join("store"), root.join("a"), root.join("b"));
		for dir in [&a, &b] {
			std::fs::create_dir_all(dir).unwrap();
		}
		register(&store, &a);
		register(&store, &b);
		register(&store, &a);
		assert_eq!(std::fs::read_to_string(store.join(STATE_DIRS)).unwrap().lines().count(), 2, "registered once each");
		let others = state_dirs(&store, &a);
		assert_eq!(others.len(), 1);
		// the registry holds canonical paths; the temp dir may not be one (macOS: /var is /private/var)
		assert_eq!(normal(&others[0]), normal(Path::new(&canonical(&b))));
		std::fs::remove_dir_all(&b).unwrap();
		assert!(state_dirs(&store, &a).is_empty(), "a removed state dir is not looked in");
		let _ = std::fs::remove_dir_all(&root);
	}

	#[test]
	fn pruning_removes_the_least_recently_used_past_the_budget() {
		let store = std::env::temp_dir().join(format!("ror-store-prune-{}", std::process::id()));
		let dir = store.join("assets").join("00");
		std::fs::create_dir_all(&dir).unwrap();
		let now = SystemTime::now();
		for (index, age) in [(0u32, 30u64), (1, 20), (2, 10), (3, 0)] {
			let path = dir.join(format!("{index}.bin"));
			std::fs::write(&path, vec![0u8; 1000]).unwrap();
			std::fs::File::options().write(true).open(&path).unwrap().set_modified(now - Duration::from_secs(age * 60)).unwrap();
		}
		std::fs::write(store.join(STATE_DIRS), b"x\n").unwrap();
		let (total, removed) = prune(&store, 3000);
		assert_eq!((total, removed), (4000, 2000), "down to three quarters of the budget, the oldest first");
		assert!(!dir.join("0.bin").exists() && !dir.join("1.bin").exists() && dir.join("2.bin").exists() && dir.join("3.bin").exists());
		assert!(store.join(STATE_DIRS).exists(), "the registry is not pruned");
		let _ = std::fs::remove_dir_all(&store);
	}
}
