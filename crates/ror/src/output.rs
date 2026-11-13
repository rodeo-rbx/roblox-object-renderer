//! Writing renders: PNG encoding (fast settings) and paths.

use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};

/// Encodes RGBA8 pixels as a PNG. `fast`: fdeflate's ultra-fast DEFLATE, each row filtered the way
/// that packs best (adaptive), a few ms per image; otherwise zlib's default level, 8-20 times slower
/// and smaller again. Measured 2026-10-06 (MB, fast / not): 30 calibration runs' studio.png 6.1 /
/// 2.5 and ror.png 4.4 / 2.5, 25 capture-cache files (1644 x 1116) 29.0 / 21.8. The adaptive
/// filter alone took the fast ones down from 7.9, 7.0 and 33.1 MB (`Up` before) at about 1 ms more
/// per image.
pub fn encode_png(width: u32, height: u32, rgba: &[u8], fast: bool) -> Result<Vec<u8>> {
	let mut bytes = Vec::with_capacity(rgba.len() / 2);
	{
		let mut encoder = png::Encoder::new(&mut bytes, width, height);
		encoder.set_color(png::ColorType::Rgba);
		encoder.set_depth(png::BitDepth::Eight);
		if fast {
			encoder.set_deflate_compression(png::DeflateCompression::FdeflateUltraFast);
			encoder.set_filter(png::Filter::Adaptive);
		} else {
			encoder.set_compression(png::Compression::Balanced);
			encoder.set_filter(png::Filter::Adaptive);
		}
		let mut writer = encoder.write_header().context("PNG header")?;
		writer.write_image_data(rgba).context("PNG data")?;
	}
	Ok(bytes)
}

pub fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8], fast: bool) -> Result<()> {
	let bytes = encode_png(width, height, rgba, fast)?;
	write_atomic(path, &bytes)
}

/// Writes 8-bit indices into `palette` (RGB triples) as an indexed PNG: an image of at most 256
/// colours in a byte a pixel instead of four, the same pixels decoded. fdeflate packs these smaller
/// than zlib's default level does, and faster. Measured 2026-11-02 on a 32-view `ror diff` (1280 x
/// 720): RGBA at zlib's default 16.9 MB, 72 % of the diff's 4.1 s; indexed at zlib's default
/// 10.6 MB and 2.1 s in all; indexed with fdeflate 9.8 MB and 0.8 s in all.
pub fn write_png_indexed(path: &Path, width: u32, height: u32, indices: &[u8], palette: &[u8]) -> Result<()> {
	let mut bytes = Vec::with_capacity(indices.len() / 4);
	{
		let mut encoder = png::Encoder::new(&mut bytes, width, height);
		encoder.set_color(png::ColorType::Indexed);
		encoder.set_depth(png::BitDepth::Eight);
		encoder.set_palette(palette);
		encoder.set_deflate_compression(png::DeflateCompression::FdeflateUltraFast);
		encoder.set_filter(png::Filter::Adaptive);
		let mut writer = encoder.write_header().context("PNG header")?;
		writer.write_image_data(indices).context("PNG data")?;
	}
	write_atomic(path, &bytes)
}

/// Writes through a temporary file and a rename, so readers never see a partial file. A file is
/// never written into: one that already holds `bytes` is kept (its time set as a write would), so
/// other names of it (hard links: `ror clean`'s, kept renders) go on sharing it, and a changed one is
/// replaced, which leaves those names on the old content.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
	if unchanged(path, bytes) {
		return Ok(());
	}
	if let Some(parent) = path.parent() {
		std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
	}
	let temporary = path.with_extension(format!("tmp{}", std::process::id()));
	{
		use std::io::Write;
		let file = std::fs::File::create(&temporary).with_context(|| format!("creating {}", temporary.display()))?;
		let mut writer = BufWriter::new(file);
		writer.write_all(bytes)?;
		writer.flush()?;
	}
	std::fs::rename(&temporary, path).with_context(|| format!("writing {}", path.display()))?;
	Ok(())
}

/// [`write_atomic`] of `value` as JSON (pretty or compact), written as it serializes rather than
/// built in memory first: a cost grid's cost.json is 140 MB of text, and its string growing to it
/// took 570 MB of commit. A file that already holds the same bytes is kept as `write_atomic` keeps it.
pub fn write_json_atomic(path: &Path, value: &impl serde::Serialize, pretty: bool) -> Result<()> {
	write_atomic_with(path, |writer| {
		if pretty {
			serde_json::to_writer_pretty(writer, value)?;
		} else {
			serde_json::to_writer(writer, value)?;
		}
		Ok(())
	})
}

/// [`write_json_atomic`]'s way with whatever `write` writes: through a temporary file and a rename,
/// an unchanged file kept (its time set as a write would).
pub fn write_atomic_with(path: &Path, write: impl FnOnce(&mut BufWriter<std::fs::File>) -> Result<()>) -> Result<()> {
	use std::io::Write;
	if let Some(parent) = path.parent() {
		std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
	}
	let temporary = path.with_extension(format!("tmp{}", std::process::id()));
	let written = (|| -> Result<()> {
		let file = std::fs::File::create(&temporary).with_context(|| format!("creating {}", temporary.display()))?;
		let mut writer = BufWriter::with_capacity(1 << 20, file);
		write(&mut writer)?;
		writer.flush()?;
		Ok(())
	})();
	if let Err(error) = written {
		let _ = std::fs::remove_file(&temporary);
		return Err(error);
	}
	if same_files(&temporary, path) && std::fs::File::options().write(true).open(path).and_then(|file| file.set_modified(SystemTime::now())).is_ok() {
		let _ = std::fs::remove_file(&temporary);
		return Ok(());
	}
	std::fs::rename(&temporary, path).with_context(|| format!("writing {}", path.display()))?;
	Ok(())
}

/// Whether two files hold the same bytes, read a chunk at a time.
fn same_files(a: &Path, b: &Path) -> bool {
	use std::io::Read;
	let (Ok(meta_a), Ok(meta_b)) = (std::fs::metadata(a), std::fs::metadata(b)) else { return false };
	if !meta_b.is_file() || meta_a.len() != meta_b.len() {
		return false;
	}
	let (Ok(file_a), Ok(file_b)) = (std::fs::File::open(a), std::fs::File::open(b)) else { return false };
	let (mut reader_a, mut reader_b) = (std::io::BufReader::with_capacity(1 << 20, file_a), std::io::BufReader::with_capacity(1 << 20, file_b));
	let (mut chunk_a, mut chunk_b) = (vec![0u8; 1 << 16], vec![0u8; 1 << 16]);
	loop {
		let Ok(read) = reader_a.read(&mut chunk_a) else { return false };
		if read == 0 {
			return true;
		}
		if reader_b.read_exact(&mut chunk_b[..read]).is_err() || chunk_a[..read] != chunk_b[..read] {
			return false;
		}
	}
}

/// Prints `value` as JSON (pretty or compact) and a newline to stdout, as it serializes: a cost grid's
/// is 77 MB of text, through stdout's line buffer a write every kilobyte.
pub fn print_json(value: &impl serde::Serialize, pretty: bool) -> Result<()> {
	use std::io::Write;
	let stdout = std::io::stdout();
	let mut writer = BufWriter::with_capacity(1 << 20, stdout.lock());
	if pretty {
		serde_json::to_writer_pretty(&mut writer, value)?;
	} else {
		serde_json::to_writer(&mut writer, value)?;
	}
	writer.write_all(b"\n")?;
	writer.flush()?;
	Ok(())
}

/// Whether the file at `path` holds exactly `bytes`; if so its modification time becomes now (tools
/// find fresh results by their times).
pub fn unchanged(path: &Path, bytes: &[u8]) -> bool {
	let same = std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && metadata.len() == bytes.len() as u64) && std::fs::read(path).is_ok_and(|old| old == bytes);
	same && std::fs::File::options().write(true).open(path).and_then(|file| file.set_modified(SystemTime::now())).is_ok()
}

pub fn decode_png(path: &Path) -> Result<(u32, u32, Vec<u8>)> {
	let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
	decode_png_from(std::io::BufReader::new(file))
}

/// [`decode_png`] of a PNG file's bytes.
pub fn decode_png_bytes(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
	decode_png_from(std::io::Cursor::new(bytes))
}

fn decode_png_from(reader: impl std::io::BufRead + std::io::Seek) -> Result<(u32, u32, Vec<u8>)> {
	let mut decoder = png::Decoder::new(reader);
	decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
	let mut reader = decoder.read_info().context("PNG header")?;
	let mut buffer = vec![0; reader.output_buffer_size().context("PNG size")?];
	let info = reader.next_frame(&mut buffer).context("PNG data")?;
	buffer.truncate(info.buffer_size());
	let rgba = match info.color_type {
		png::ColorType::Rgba => buffer,
		png::ColorType::Rgb => buffer.chunks_exact(3).flat_map(|px| [px[0], px[1], px[2], 255]).collect(),
		png::ColorType::GrayscaleAlpha => buffer.chunks_exact(2).flat_map(|px| [px[0], px[0], px[0], px[1]]).collect(),
		png::ColorType::Grayscale => buffer.iter().flat_map(|&v| [v, v, v, 255]).collect(),
		png::ColorType::Indexed => anyhow::bail!("indexed PNG after expansion"),
	};
	Ok((info.width, info.height, rgba))
}

/// The folder that holds the default `out/` and, for a host, the `.cache` state dir. For a host (a
/// program that links this library): what [`set_repo_dir`] set, else the nearest folder above the
/// executable that holds a `Cargo.lock` ([`checkout_of`]), else `.`. ror's own binary replaces this at
/// startup ([`init_binary`]) with its checkout, or with its state dir when it runs from none.
pub fn repo_dir() -> PathBuf {
	REPO_DIR.get_or_init(|| std::env::current_exe().ok().and_then(|exe| checkout_of(&exe)).unwrap_or_else(|| PathBuf::from("."))).clone()
}

/// Where `ror studio capture` writes: the explicit `--out`, else `out/studio-capture.png`, relative to
/// the working directory in a checkout (as it always was) or under the state dir for a standalone
/// binary, whose working directory is not ours to write into.
pub fn studio_capture_path(out: Option<PathBuf>) -> Result<PathBuf> {
	let default = Path::new("out/studio-capture.png");
	match out {
		Some(out) => Ok(out),
		None if checkout_dir().is_some() => Ok(default.to_path_buf()),
		None => Ok(repo_dir().join(default)),
	}
}

/// The checkout whose `.env`, rur and ree this process may use. A host always has one ([`repo_dir`]).
/// ror's own binary has one only when it runs from a ror checkout: `None` for a standalone binary,
/// which must not look in the working folder or beside itself.
pub fn checkout_dir() -> Option<PathBuf> {
	BINARY_CHECKOUT.get().cloned().unwrap_or_else(|| Some(repo_dir()))
}

/// For hosts: the checkout an executable belongs to, the nearest folder above it holding a
/// `Cargo.lock`. Any Cargo project counts, as a host is usually not ror's own checkout.
pub fn checkout_of(exe: &Path) -> Option<PathBuf> {
	nearest_checkout(exe, |dir| dir.join("Cargo.lock").is_file())
}

/// For ror's binary: the ror checkout an executable belongs to, the nearest folder above it holding
/// a `Cargo.lock` and `crates/ror/Cargo.toml`. Another Cargo project above a released binary is not
/// its checkout.
pub fn ror_checkout_of(exe: &Path) -> Option<PathBuf> {
	nearest_checkout(exe, |dir| dir.join("Cargo.lock").is_file() && dir.join("crates/ror/Cargo.toml").is_file())
}

/// The nearest folder above `exe` (not `exe` itself) that `is_checkout` accepts, canonical.
fn nearest_checkout(exe: &Path, is_checkout: impl Fn(&Path) -> bool) -> Option<PathBuf> {
	exe.ancestors().skip(1).find(|dir| is_checkout(dir))?.canonicalize().ok()
}

/// For hosts: sets [`repo_dir`] before anything reads it, such as a service's starter's checkout. It
/// does nothing once `repo_dir` has been read or set.
pub fn set_repo_dir(dir: PathBuf) {
	let _ = REPO_DIR.set(dir);
}

/// For ror's binary: chooses its folders before anything reads them. A host never calls this and
/// keeps the defaults of [`repo_dir`] and [`state_dir`].
///
/// `source` is the executable whose checkout counts: the binary itself, or for a service (which runs
/// a copy from its state dir) the ror that started it. `state` is `serve --state-dir`. The state dir
/// is that, else `ROR_STATE_DIR`, else `.cache` in the checkout, else `state/` in the per-user
/// cache, so a standalone binary never writes beside itself or into the working folder.
///
/// The only fallible step, finding the per-user cache, happens here so that the path getters stay
/// infallible. It must run first: it does nothing for what was already read or set.
pub fn init_binary(source: &Path, state: Option<PathBuf>) -> Result<()> {
	let checkout = ror_checkout_of(source);
	let state = binary_state_dir(checkout.as_deref(), state)?;
	// without a checkout, the state dir also holds the default `out/`
	let repo = checkout.clone().unwrap_or_else(|| state.clone());
	set_repo_dir(repo);
	set_state_dir(state);
	let _ = BINARY_CHECKOUT.set(checkout);
	Ok(())
}

/// The absolute state dir of ror's binary: `requested`, else `ROR_STATE_DIR`, else `.cache` in
/// `checkout`, else `state/` in the per-user cache.
fn binary_state_dir(checkout: Option<&Path>, requested: Option<PathBuf>) -> Result<PathBuf> {
	let state = match (requested.or_else(state_dir_from_env), checkout) {
		(Some(state), _) => state,
		(None, Some(checkout)) => checkout.join(".cache"),
		(None, None) => ror_assets::store::user_cache_dir().context("no per-user cache directory; set ROR_STATE_DIR")?.join("state"),
	};
	std::path::absolute(state).context("resolving the state dir")
}

/// What [`repo_dir`] returns, once set or first read.
static REPO_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
/// Set by [`init_binary`] and by nothing else: ror's own checkout, `None` for a standalone binary.
/// Unset in a host.
static BINARY_CHECKOUT: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();

/// A path without Windows' verbatim prefix (`\\?\C:\...`, what canonicalize returns), `..`
/// resolved; a verbatim UNC path (`\\?\UNC\...`) stays as it is.
pub fn plain_path(path: &Path) -> PathBuf {
	let text = path.to_string_lossy();
	let path = match text.strip_prefix(r"\\?\") {
		Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
		_ => path.to_path_buf(),
	};
	std::path::absolute(&path).unwrap_or(path)
}

/// Makes `directory` this process's working directory in its plain form (else as given). A
/// verbatim one (`\\?\C:\...`, as the default state dir is) is inherited by every process this
/// one starts, and turns their rooted paths without a drive into network names: ffmpeg's gnutls
/// opens `/etc/gnutls/config` before main, which became `\\etc\gnutls\config`, a lookup of a host
/// named `etc` that failed after 2.7 s whenever Windows had forgotten the last failure (a few
/// minutes), on the first clip after a pause.
pub fn work_from(directory: &Path) -> std::io::Result<()> {
	let plain = plain_path(directory);
	std::env::set_current_dir(&plain).or_else(|_| std::env::set_current_dir(directory))
}

/// Where this process keeps its state: what [`set_state_dir`] fixed (ror's binary does at startup, in
/// [`init_binary`]), else `ROR_STATE_DIR`, else `.cache` in a host's [`repo_dir`].
pub fn state_dir() -> PathBuf {
	if let Some(dir) = STATE_DIR.get() {
		return dir.clone();
	}
	state_dir_from_env().unwrap_or_else(|| repo_dir().join(".cache"))
}

/// `ROR_STATE_DIR`, unless it is unset or empty.
fn state_dir_from_env() -> Option<PathBuf> {
	std::env::var_os("ROR_STATE_DIR").filter(|dir| !dir.is_empty()).map(PathBuf::from)
}

/// Fixes the state dir for the rest of this process, so a service's pipeline, texture and GUI caches
/// follow its `--state-dir` and not the environment. It does nothing once the state dir has been set.
pub fn set_state_dir(dir: PathBuf) {
	let _ = STATE_DIR.set(dir);
}

static STATE_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// The content store every build, checkout and state dir of the user shares (ror_assets::store):
/// `ROR_STORE` (`0`: none), else `store/` in the per-user cache; none in unit tests. ror's binary
/// uses `ror_assets::store::standalone_dir` outside a checkout, else `default_dir` as a host does.
/// They differ in where macOS keeps the cache. The first call decides, so ror's binary calls
/// [`init_binary`] before it.
pub fn store_dir() -> Option<PathBuf> {
	if cfg!(test) {
		return None;
	}
	static DIR: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
	DIR.get_or_init(|| {
		let host = ror_assets::store::default_dir;
		let standalone = ror_assets::store::standalone_dir;
		default_store(BINARY_CHECKOUT.get(), host, standalone)
	})
	.clone()
}

/// Picks the store folder from what [`init_binary`] decided. `checkout` is `None` when it never ran
/// (a host), `Some(None)` for a standalone binary and `Some(Some(_))` for a binary in a checkout.
/// Only a standalone binary takes the new per-user folder; the others keep the store they have.
fn default_store(checkout: Option<&Option<PathBuf>>, host: impl FnOnce() -> Option<PathBuf>, standalone: impl FnOnce() -> Option<PathBuf>) -> Option<PathBuf> {
	match checkout {
		Some(None) => standalone(),
		Some(Some(_)) | None => host(),
	}
}

/// The other state dirs that used the content store (`ror_assets::store::state_dirs`), looked in for
/// what neither this state dir nor the store keeps: texture files, GUI images.
pub fn other_state_dirs() -> &'static [PathBuf] {
	static DIRS: std::sync::OnceLock<Vec<PathBuf>> = std::sync::OnceLock::new();
	DIRS.get_or_init(|| store_dir().map(|store| ror_assets::store::state_dirs(&store, &state_dir())).unwrap_or_default())
}

/// The asset resolver's settings for `state_dir`: its downloads shared with `ROR_SHARED_STATE_DIRS`'
/// and the content store's.
pub fn asset_config(state_dir: PathBuf, offline: bool) -> ror_assets::AssetConfig {
	ror_assets::AssetConfig { offline, repo_dir: checkout_dir(), shared: shared_state_dirs(), store: store_dir(), ..ror_assets::AssetConfig::new(state_dir) }
}

/// A state dir's file `name` (what one learnt about this machine: the GPU its services start on, its
/// pipeline cache), else the content store's of the same name, which a state dir without one starts
/// from. A state dir's own goes to a store that has none (a state dir made before the store, whose
/// file is written again only when it changes, would otherwise never give it).
pub fn read_state_file(state_dir: &Path, name: &str) -> Option<Vec<u8>> {
	let store = store_dir().map(|store| store.join(name));
	match std::fs::read(state_dir.join(name)) {
		Ok(bytes) => {
			if let Some(store) = store.filter(|store| !store.is_file()) {
				let _ = write_atomic(&store, &bytes);
			}
			Some(bytes)
		}
		Err(_) => std::fs::read(store?).ok(),
	}
}

/// Writes a state dir's file `name` whole, and the content store's.
pub fn write_state_file(state_dir: &Path, name: &str, bytes: &[u8]) -> Result<()> {
	if let Some(store) = store_dir() {
		let _ = write_atomic(&store.join(name), bytes);
	}
	write_atomic(&state_dir.join(name), bytes)
}

/// `ROR_SHARED_STATE_DIRS` (paths separated by `;`, `:` off Windows): other state dirs whose
/// downloads this one shares (`ror_assets::AssetConfig::shared`), such as another tool's `.cache`.
pub fn shared_state_dirs() -> Vec<PathBuf> {
	std::env::var_os("ROR_SHARED_STATE_DIRS").map(|dirs| std::env::split_paths(&dirs).filter(|dir| !dir.as_os_str().is_empty()).collect()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn only_a_binary_without_a_checkout_uses_the_standalone_store() {
		// Distinct cache folders, as on macOS, even when this test runs on another platform.
		let root = std::env::temp_dir().join("ror-store-selection");
		let host = || Some(root.join("home/.cache/ror/store"));
		let standalone = || Some(root.join("home/Library/Caches/ror/store"));
		let in_checkout = Some(root.join("checkout"));
		// init_binary never ran: a host
		assert_eq!(default_store(None, host, standalone), host());
		assert_eq!(default_store(Some(&in_checkout), host, standalone), host());
		assert_eq!(default_store(Some(&None), host, standalone), standalone());
	}

	#[test]
	fn studio_capture_keeps_the_checkout_default_and_explicit_paths() {
		// The unit tests run from a checkout; tests/standalone.rs covers the standalone default.
		assert!(checkout_dir().is_some());
		let default = PathBuf::from("out/studio-capture.png");
		assert_eq!(studio_capture_path(None).unwrap(), default);
		for explicit in [default, std::env::temp_dir().join("capture.png")] {
			assert_eq!(studio_capture_path(Some(explicit.clone())).unwrap(), explicit);
		}
	}

	#[cfg(windows)]
	#[test]
	fn plain_paths_drop_the_verbatim_prefix_but_not_unc() {
		assert_eq!(plain_path(Path::new(r"\\?\C:\a\b")), PathBuf::from(r"C:\a\b"));
		assert_eq!(plain_path(Path::new(r"C:\a\..\b")), PathBuf::from(r"C:\b"));
		assert_eq!(plain_path(Path::new(r"\\?\UNC\host\share\a")), PathBuf::from(r"\\?\UNC\host\share\a"));
	}

	#[test]
	fn an_executable_belongs_to_the_nearest_checkout_above_it_wherever_it_runs() {
		let root = std::env::temp_dir().join(format!("ror-checkout-test-{}", std::process::id()));
		let repo = root.join("ror");
		std::fs::create_dir_all(&repo).unwrap();
		std::fs::write(repo.join("Cargo.lock"), "").unwrap();
		let canonical = repo.canonicalize().unwrap();
		assert_eq!(checkout_of(&repo.join("target/release/host.exe")), Some(canonical.clone()));
		assert_eq!(ror_checkout_of(&repo.join("target/release/ror.exe")), None);
		std::fs::create_dir_all(repo.join("crates/ror")).unwrap();
		std::fs::write(repo.join("crates/ror/Cargo.toml"), "").unwrap();
		for exe in ["target/release/ror.exe", "target/release/deps/ror-0a1b.exe", "target-dist/release/ror.exe", ".cache/bin/ror-0.1.0-1-2.exe"] {
			assert_eq!(checkout_of(&repo.join(exe)), Some(canonical.clone()), "{exe}");
			assert_eq!(ror_checkout_of(&repo.join(exe)), Some(canonical.clone()), "{exe}");
		}
		// a service copy in a state dir outside any checkout has none: serve takes its starter's (--source)
		assert_eq!(checkout_of(&root.join("state/bin/ror-0.1.0-1-2.exe")), None);
		std::fs::remove_dir_all(&root).unwrap();
	}
}
