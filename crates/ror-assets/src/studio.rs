//! The local Roblox install whose built-in content ror reads: detection and `rbxasset://` lookup.
//!
//! Detection order: [`crate::AssetConfig::studio_content`], env `ROR_ROBLOX_CONTENT` (the `content`
//! folder or the version folder containing it), the registry value
//! `HKCU\Software\Roblox\RobloxStudio\ContentFolder`, then the newest Studio install
//! (`%LOCALAPPDATA%\Roblox\Versions\*\RobloxStudioBeta.exe`; on macOS, `RobloxStudio.app` in
//! `/Applications` or `~/Applications`), else the newest Roblox Player install (`RobloxPlayerBeta.exe`
//! there; on macOS, `Roblox.app`). Neither has to run: ror only reads their files.
//!
//! The Player ships every file ror reads. Its `content`, `PlatformContent/pc` and `ExtraContent` held
//! the same textures (1,779), fonts (119), skies (12), avatar meshes (5), platform textures (46) and
//! terrain files (3) as Studio's on 2026-11-07; what Studio adds is its own scripts, packages, UI
//! images and docs. Studio goes first because ror's parity measurements were made against its files.
//!
//! Lookup order for `rbxasset://<path>`: `PlatformContent/pc/<path>`, `content/<path>`,
//! `ExtraContent/<path>`. In every install on this machine (12 versions) no relative path exists
//! in both `PlatformContent/pc` and another root, and the 71 paths in both `content` and
//! `ExtraContent` are byte-identical, so the order is unobservable today; the platform folder goes
//! first because it holds the per-platform overrides (`textures/studs.dds`, `textures/sky/*.tex`,
//! `textures/plastic`, `textures/water`, `terrain/*.json` exist only there).

use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub(crate) struct StudioInstall {
	pub content: PathBuf,
	/// Lookup roots in order
	pub roots: Vec<PathBuf>,
}

impl StudioInstall {
	/// An install from its `content` folder (or the version folder holding `content`).
	pub(crate) fn from_dir(dir: &Path) -> Option<StudioInstall> {
		let content = if dir.join("content").is_dir() && !dir.join("textures").is_dir() { dir.join("content") } else { dir.to_path_buf() };
		if !content.is_dir() {
			return None;
		}
		let version = content.parent().map(Path::to_path_buf).unwrap_or_else(|| content.clone());
		let roots: Vec<PathBuf> = [version.join("PlatformContent").join("pc"), content.clone(), version.join("ExtraContent")]
			.into_iter()
			.filter(|p| p.is_dir())
			.collect();
		Some(StudioInstall { content, roots })
	}

	/// The file for a normalised `rbxasset://` path.
	pub(crate) fn find(&self, path: &str) -> Option<PathBuf> {
		self.roots.iter().map(|root| root.join(path)).find(|candidate| candidate.is_file())
	}
}

pub(crate) fn detect(explicit: Option<&Path>) -> Option<StudioInstall> {
	if let Some(dir) = explicit {
		return StudioInstall::from_dir(dir);
	}
	if let Some(dir) = std::env::var_os("ROR_ROBLOX_CONTENT").filter(|v| !v.is_empty()) {
		return StudioInstall::from_dir(Path::new(&dir));
	}
	if let Some(found) = registry_content_folder().as_deref().and_then(StudioInstall::from_dir) {
		return Some(found);
	}
	installed()
}

/// The programs whose installs hold Roblox's built-in content, as (Windows executable, macOS app), in the
/// order they are looked for: Studio, then the Player (see the module's notes)
const PROGRAMS: [(&str, &str); 2] = [("RobloxStudioBeta.exe", "RobloxStudio.app"), ("RobloxPlayerBeta.exe", "Roblox.app")];

/// The first program's app with a `content` folder in its `Contents/Resources`, which holds `content`,
/// `PlatformContent` and `ExtraContent` as a Windows version folder does: in `/Applications`, else the
/// user's `~/Applications`.
#[cfg(target_os = "macos")]
fn installed() -> Option<StudioInstall> {
	PROGRAMS.iter().find_map(|(_, app)| {
		let resources = |applications: PathBuf| applications.join(app).join("Contents").join("Resources");
		let system = resources(PathBuf::from("/Applications"));
		let user = std::env::var_os("HOME").map(|home| resources(PathBuf::from(home).join("Applications")));
		[Some(system), user].into_iter().flatten().find(|dir| dir.join("content").is_dir()).as_deref().and_then(StudioInstall::from_dir)
	})
}

/// The newest install in `%LOCALAPPDATA%\Roblox\Versions` (none where that is unset).
#[cfg(not(target_os = "macos"))]
fn installed() -> Option<StudioInstall> {
	installed_in(&PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join("Roblox").join("Versions"))
}

/// The newest install of the first program in a `Versions` folder: Studio's, else the Player's.
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn installed_in(versions: &Path) -> Option<StudioInstall> {
	PROGRAMS.iter().find_map(|(exe, _)| newest_in(versions, exe).as_deref().and_then(StudioInstall::from_dir))
}

/// The newest `<versions>\<v>` holding `exe` (by the executable's time), as its `content`.
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn newest_in(versions: &Path, exe: &str) -> Option<PathBuf> {
	let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
	for entry in std::fs::read_dir(versions).ok()?.flatten() {
		let Ok(meta) = entry.path().join(exe).metadata() else {
			continue;
		};
		let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
		if best.as_ref().is_none_or(|(t, _)| modified > *t) {
			best = Some((modified, entry.path().join("content")));
		}
	}
	best.map(|(_, p)| p)
}

#[cfg(windows)]
fn registry_content_folder() -> Option<PathBuf> {
	use std::os::windows::ffi::OsStringExt;
	use windows_sys::Win32::Foundation::ERROR_SUCCESS;
	use windows_sys::Win32::System::Registry::{HKEY_CURRENT_USER, RRF_RT_REG_SZ, RegGetValueW};

	let wide = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
	let key = wide("Software\\Roblox\\RobloxStudio");
	let value = wide("ContentFolder");
	let mut size: u32 = 0;
	// SAFETY: valid NUL-terminated wide strings; a null data pointer queries the size
	let status = unsafe { RegGetValueW(HKEY_CURRENT_USER, key.as_ptr(), value.as_ptr(), RRF_RT_REG_SZ, std::ptr::null_mut(), std::ptr::null_mut(), &mut size) };
	if status != ERROR_SUCCESS || !(2..=64 * 1024).contains(&size) {
		return None;
	}
	let mut buffer = vec![0u16; (size as usize).div_ceil(2)];
	let mut size = (buffer.len() * 2) as u32;
	// SAFETY: `buffer` holds `size` bytes
	let status = unsafe { RegGetValueW(HKEY_CURRENT_USER, key.as_ptr(), value.as_ptr(), RRF_RT_REG_SZ, std::ptr::null_mut(), buffer.as_mut_ptr().cast(), &mut size) };
	if status != ERROR_SUCCESS {
		return None;
	}
	let len = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
	let path = PathBuf::from(std::ffi::OsString::from_wide(&buffer[..len]));
	(!path.as_os_str().is_empty()).then_some(path)
}

#[cfg(not(windows))]
fn registry_content_folder() -> Option<PathBuf> {
	None
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn lookup_order() {
		let dir = std::env::temp_dir().join(format!("ror-assets-studio-{}", std::process::id()));
		let _ = std::fs::remove_dir_all(&dir);
		let content = dir.join("content");
		let platform = dir.join("PlatformContent").join("pc");
		let extra = dir.join("ExtraContent");
		for (root, name, data) in [
			(&platform, "textures/studs.dds", "pc"),
			(&content, "textures/studs.dds", "content"),
			(&content, "textures/face.png", "content"),
			(&extra, "textures/face.png", "extra"),
			(&extra, "textures/ui/x.png", "extra"),
		] {
			let path = root.join(name);
			std::fs::create_dir_all(path.parent().unwrap()).unwrap();
			std::fs::write(&path, data).unwrap();
		}

		// Either the version dir or the content dir identifies the install
		for start in [&dir, &content] {
			let install = StudioInstall::from_dir(start).unwrap();
			assert_eq!(install.content, content);
			let read = |p: &str| install.find(p).map(|f| std::fs::read_to_string(f).unwrap());
			assert_eq!(read("textures/studs.dds").as_deref(), Some("pc"));
			assert_eq!(read("textures/face.png").as_deref(), Some("content"));
			assert_eq!(read("textures/ui/x.png").as_deref(), Some("extra"));
			assert_eq!(read("textures/missing.png"), None);
		}
		let _ = std::fs::remove_dir_all(&dir);
	}

	#[test]
	fn a_player_install_serves_without_studio_and_studio_goes_first() {
		let versions = std::env::temp_dir().join(format!("ror-assets-versions-{}", std::process::id()));
		let _ = std::fs::remove_dir_all(&versions);
		let install = |name: &str, exe: &str| {
			let dir = versions.join(name);
			std::fs::create_dir_all(dir.join("content").join("textures")).unwrap();
			std::fs::write(dir.join("content").join("textures").join("studs.dds"), name).unwrap();
			std::fs::write(dir.join(exe), b"").unwrap();
		};
		let found = || installed_in(&versions).map(|install| std::fs::read_to_string(install.find("textures/studs.dds").unwrap()).unwrap());

		assert_eq!(found(), None);
		// only the Player: its content serves
		install("version-player-old", "RobloxPlayerBeta.exe");
		std::thread::sleep(std::time::Duration::from_millis(20));
		install("version-player", "RobloxPlayerBeta.exe");
		assert_eq!(found().as_deref(), Some("version-player"));
		// a Studio install goes first, even beside a newer Player
		install("version-studio", "RobloxStudioBeta.exe");
		std::thread::sleep(std::time::Duration::from_millis(20));
		install("version-player-newer", "RobloxPlayerBeta.exe");
		assert_eq!(found().as_deref(), Some("version-studio"));
		let _ = std::fs::remove_dir_all(&versions);
	}

	#[test]
	fn detects_this_machine() {
		// Informational: the real install, when present
		if let Some(install) = detect(None) {
			assert!(install.content.is_dir());
			assert!(!install.roots.is_empty());
		}
	}
}
