//! The Roblox account cookie (`.ROBLOSECURITY`), read at runtime from the environment, env files or
//! Roblox Studio's sign-in.
//!
//! Sources, first match wins: env `ROBLOX_ASSET_DELIVERY_COOKIE`, env `ROBLOSECURITY`, then the env
//! files named by `ROR_COOKIE_ENV` and `RUR_COOKIE_ENV`, then `<repo>/.env`, then the account signed
//! in to Roblox Studio on this machine (read by the `rbx_cookie` crate from Windows' Credential
//! Manager or macOS's `~/Library/HTTPStorages`; `ROR_STUDIO_COOKIE=0` turns it off).
//! An env file line is `[export ]ROBLOX_ASSET_DELIVERY_COOKIE=value` or `[export ]ROBLOSECURITY=value`;
//! values may be quoted and may carry a `.ROBLOSECURITY=` prefix; `#` starts a comment.
//!
//! The value is held in [`Secret`], which has no `Display`, a redacting `Debug`, and is zeroed on
//! drop. Nothing in this module logs, prints or returns the value in an error.

use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};

const KEYS: [&str; 2] = ["ROBLOX_ASSET_DELIVERY_COOKIE", "ROBLOSECURITY"];

/// A secret string. Only [`Secret::expose`] reveals it.
pub(crate) struct Secret(String);

impl Secret {
	/// Wraps a cookie value after validation (printable ASCII, no separators or whitespace).
	pub(crate) fn new(raw: &str) -> Option<Secret> {
		let mut value = raw.trim();
		if let Some(rest) = value.strip_prefix(".ROBLOSECURITY=") {
			value = rest;
		}
		let valid = !value.is_empty() && value.bytes().all(|b| b.is_ascii_graphic() && b != b';' && b != b',' && b != b'"');
		valid.then(|| Secret(value.to_string()))
	}

	pub(crate) fn expose(&self) -> &str {
		&self.0
	}
}

impl fmt::Debug for Secret {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str("Secret(<redacted>)")
	}
}

impl Drop for Secret {
	fn drop(&mut self) {
		// Best effort: overwrite the bytes before the allocation is freed
		let mut bytes = std::mem::take(&mut self.0).into_bytes();
		for b in bytes.iter_mut() {
			// SAFETY: `b` is a valid, aligned, exclusive reference into `bytes`
			unsafe { std::ptr::write_volatile(b, 0) };
		}
		drop(bytes);
	}
}

/// Where the cookie came from (never the value).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CookieSource {
	Env(&'static str),
	File(PathBuf),
	/// The account signed in to Roblox Studio on this machine.
	Studio,
}

/// An environment variable by name: the process's ([`process_env`]), or a test's.
type Env<'a> = &'a dyn Fn(&str) -> Option<OsString>;

fn process_env(key: &str) -> Option<OsString> {
	std::env::var_os(key)
}

/// Loads the cookie from the documented sources. `repo_dir` enables the repo-relative env files.
pub(crate) fn load(repo_dir: Option<&Path>) -> Option<(Secret, CookieSource)> {
	load_from(&process_env, repo_dir, studio_cookie)
}

fn load_from(env: Env, repo_dir: Option<&Path>, studio: impl FnOnce() -> Option<Secret>) -> Option<(Secret, CookieSource)> {
	for key in KEYS {
		if let Some(value) = env(key)
			&& let Some(secret) = value.to_str().and_then(unquote).and_then(Secret::new)
		{
			return Some((secret, CookieSource::Env(key)));
		}
	}
	for path in env_files_from(env, repo_dir) {
		if let Some(secret) = read_env_file(&path).as_deref().and_then(parse_env_file) {
			return Some((secret, CookieSource::File(path)));
		}
	}
	if studio_skipped(env).is_none()
		&& let Some(secret) = studio()
	{
		return Some((secret, CookieSource::Studio));
	}
	None
}

/// Why Studio's cookie is not read, if it isn't: `ROR_STUDIO_COOKIE=0`, or `ROBLOSECURITY` set (to a
/// value refused above), which rbx_cookie would return in place of Studio's.
fn studio_skipped(env: Env) -> Option<&'static str> {
	if env("ROR_STUDIO_COOKIE").is_some_and(|value| value == "0") {
		Some("off (ROR_STUDIO_COOKIE=0)")
	} else if env("ROBLOSECURITY").is_some() {
		Some("not read while env ROBLOSECURITY is set")
	} else {
		None
	}
}

/// The cookie of the account signed in to Roblox Studio on this machine, through rbx_cookie (no
/// logger is installed, so its log lines go nowhere). Its own copies are freed without being zeroed.
fn studio_cookie() -> Option<Secret> {
	let raw = Secret(rbx_cookie::get_value()?);
	Secret::new(raw.expose())
}

/// One line per candidate source with its status (`missing`, `unreadable`, `no cookie line`,
/// `invalid value`, `ok`; Studio's `off`, `not read`, `no signed-in account found`). Never contains a
/// value.
pub(crate) fn diagnose(repo_dir: Option<&Path>) -> Vec<String> {
	let mut lines = Vec::new();
	for key in KEYS {
		let status = match std::env::var_os(key) {
			None => "not set",
			Some(v) if v.to_str().and_then(unquote).and_then(Secret::new).is_some() => "ok",
			Some(_) => "invalid value",
		};
		lines.push(format!("env {key}: {status}"));
	}
	for path in env_files(repo_dir) {
		let status = if !path.is_file() {
			"missing"
		} else {
			match read_env_file(&path) {
				None => "unreadable",
				Some(text) => {
					let has_line = text.lines().any(|l| {
						let l = l.trim();
						let l = l.strip_prefix("export ").map(str::trim_start).unwrap_or(l);
						l.split_once('=').is_some_and(|(k, _)| KEYS.contains(&k.trim()))
					});
					match (has_line, parse_env_file(&text).is_some()) {
						(_, true) => "ok",
						(true, false) => "invalid value",
						(false, false) => "no cookie line",
					}
				}
			}
		};
		lines.push(format!("file {}: {status}", path.display()));
	}
	let studio = studio_skipped(&process_env).unwrap_or_else(|| if studio_cookie().is_some() { "ok" } else { "no signed-in account found" });
	lines.push(format!("studio: {studio}"));
	lines
}

/// Env file text: UTF-8 (BOM optional) or UTF-16 with a BOM (what Windows PowerShell 5 writes).
fn read_env_file(path: &Path) -> Option<String> {
	let bytes = std::fs::read(path).ok()?;
	if let Some(rest) = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]) {
		return String::from_utf8(rest.to_vec()).ok();
	}
	let utf16 = |rest: &[u8], le: bool| {
		let units: Vec<u16> = rest.chunks_exact(2).map(|c| if le { u16::from_le_bytes([c[0], c[1]]) } else { u16::from_be_bytes([c[0], c[1]]) }).collect();
		String::from_utf16(&units).ok()
	};
	if let Some(rest) = bytes.strip_prefix(&[0xff, 0xfe]) {
		return utf16(rest, true);
	}
	if let Some(rest) = bytes.strip_prefix(&[0xfe, 0xff]) {
		return utf16(rest, false);
	}
	String::from_utf8(bytes).ok()
}

/// Candidate env files in order.
pub(crate) fn env_files(repo_dir: Option<&Path>) -> Vec<PathBuf> {
	env_files_from(&process_env, repo_dir)
}

fn env_files_from(env: Env, repo_dir: Option<&Path>) -> Vec<PathBuf> {
	let mut files = Vec::new();
	for var in ["ROR_COOKIE_ENV", "RUR_COOKIE_ENV"] {
		if let Some(path) = env(var).filter(|p| !p.is_empty()) {
			files.push(PathBuf::from(path));
		}
	}
	if let Some(repo) = repo_dir {
		files.push(repo.join(".env"));
	}
	files
}

/// The cookie from env file text: `ROBLOX_ASSET_DELIVERY_COOKIE` wins over `ROBLOSECURITY`.
pub(crate) fn parse_env_file(text: &str) -> Option<Secret> {
	let mut found: [Option<Secret>; 2] = [None, None];
	for line in text.lines() {
		let line = line.trim();
		if line.is_empty() || line.starts_with('#') {
			continue;
		}
		let line = line.strip_prefix("export ").map(str::trim_start).unwrap_or(line);
		let Some((key, value)) = line.split_once('=') else {
			continue;
		};
		let Some(slot) = KEYS.iter().position(|k| *k == key.trim()) else {
			continue;
		};
		if found[slot].is_none() {
			found[slot] = unquote(value).and_then(Secret::new);
		}
	}
	let [primary, fallback] = found;
	primary.or(fallback)
}

/// Strips matching quotes, or a trailing ` # comment` from an unquoted value.
fn unquote(value: &str) -> Option<&str> {
	let value = value.trim();
	for quote in ['"', '\''] {
		if let Some(rest) = value.strip_prefix(quote) {
			return rest.find(quote).map(|end| &rest[..end]);
		}
	}
	let value = match value.find(" #").or_else(|| value.find("\t#")) {
		Some(i) => &value[..i],
		None => value,
	};
	Some(value.trim())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn env_file_lines() {
		let text = "# comment\nOTHER=1\nexport ROBLOSECURITY=\"_|WARNING:-test|_ABC\"\n";
		assert_eq!(parse_env_file(text).unwrap().expose(), "_|WARNING:-test|_ABC");

		// ROBLOX_ASSET_DELIVERY_COOKIE wins regardless of order
		let text = "ROBLOSECURITY=second\nROBLOX_ASSET_DELIVERY_COOKIE=first # trailing comment\n";
		assert_eq!(parse_env_file(text).unwrap().expose(), "first");

		// .ROBLOSECURITY= prefix, single quotes, spaces around '='
		let text = "ROBLOSECURITY = '.ROBLOSECURITY=VALUE123'\n";
		assert_eq!(parse_env_file(text).unwrap().expose(), "VALUE123");

		// Invalid values are ignored: whitespace, separators, empty
		assert!(parse_env_file("ROBLOSECURITY=a b\n").is_none());
		assert!(parse_env_file("ROBLOSECURITY=a;b\n").is_none());
		assert!(parse_env_file("ROBLOSECURITY=\n").is_none());
		assert!(parse_env_file("ROBLOSECURITY=\"unterminated\n").is_none());
		assert!(parse_env_file("# ROBLOSECURITY=x\n").is_none());
	}

	#[test]
	fn env_file_encodings() {
		let dir = std::env::temp_dir().join(format!("ror-assets-envfile-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let text = "ROBLOSECURITY=FAKE-VALUE\r\n";
		let utf8_bom = [&[0xef, 0xbb, 0xbf][..], text.as_bytes()].concat();
		let utf16: Vec<u8> = [0xff, 0xfe].into_iter().chain(text.encode_utf16().flat_map(|u| u.to_le_bytes())).collect();
		for (name, bytes) in [("bom", utf8_bom), ("utf16", utf16), ("plain", text.as_bytes().to_vec())] {
			let path = dir.join(name);
			std::fs::write(&path, &bytes).unwrap();
			let parsed = read_env_file(&path).as_deref().and_then(parse_env_file);
			assert_eq!(parsed.unwrap().expose(), "FAKE-VALUE", "{name}");
		}
		let _ = std::fs::remove_dir_all(&dir);
	}

	#[test]
	fn debug_is_redacted() {
		let secret = Secret::new("TOPSECRET-VALUE").unwrap();
		let shown = format!("{secret:?}");
		assert!(!shown.contains("TOPSECRET"));
	}

	#[test]
	fn env_file_order() {
		let files = env_files(Some(Path::new("C:/repo/ror")));
		let last = files.last().map(|p| p.to_string_lossy().replace('\\', "/"));
		assert_eq!(last.as_deref(), Some("C:/repo/ror/.env"), "the checkout's .env is the last source");
	}

	#[test]
	fn studio_s_signed_in_account_is_the_last_source() {
		let nothing = |_: &str| -> Option<OsString> { None };
		let studio = || Secret::new("FAKE-STUDIO");
		let source = |env: Env, repo: Option<&Path>| load_from(env, repo, studio).map(|(secret, source)| (secret.expose().to_string(), source));

		// with no other source, Studio's
		assert_eq!(source(&nothing, None), Some(("FAKE-STUDIO".into(), CookieSource::Studio)));
		assert_eq!(load_from(&nothing, None, || None).map(|(_, source)| source), None, "no account signed in to Studio");

		// a variable or an env file wins
		let variable = |key: &str| (key == "ROBLOX_ASSET_DELIVERY_COOKIE").then(|| OsString::from("FAKE-ENV"));
		assert_eq!(source(&variable, None), Some(("FAKE-ENV".into(), CookieSource::Env("ROBLOX_ASSET_DELIVERY_COOKIE"))));
		// its own folder: studio.rs's lookup_order runs beside this test and removes `ror-assets-studio-<pid>`
		let dir = std::env::temp_dir().join(format!("ror-assets-cookie-file-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		std::fs::write(dir.join(".env"), "ROBLOSECURITY=FAKE-FILE\n").unwrap();
		assert_eq!(source(&nothing, Some(&dir)), Some(("FAKE-FILE".into(), CookieSource::File(dir.join(".env")))));
		let _ = std::fs::remove_dir_all(&dir);

		// ROR_STUDIO_COOKIE=0 turns it off; a ROBLOSECURITY ror refused keeps rbx_cookie from reading Studio
		let not_read = || -> Option<Secret> { panic!("Studio's cookie was read") };
		let off = |key: &str| (key == "ROR_STUDIO_COOKIE").then(|| OsString::from("0"));
		assert!(load_from(&off, None, not_read).is_none());
		let refused = |key: &str| (key == "ROBLOSECURITY").then(|| OsString::from("a b"));
		assert!(load_from(&refused, None, not_read).is_none());
		let on = |key: &str| (key == "ROR_STUDIO_COOKIE").then(|| OsString::from("1"));
		assert_eq!(source(&on, None).map(|(_, source)| source), Some(CookieSource::Studio));

		// Studio's value goes through the same check as the others
		assert!(load_from(&nothing, None, || Secret::new("a;b")).is_none());
	}
}
