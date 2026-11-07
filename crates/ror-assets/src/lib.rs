//! Roblox content resolution and downloads.
//!
//! - `rbxasset://path` → the local Roblox Studio install, looked up in `PlatformContent/pc/`, then
//!   `content/`, then `ExtraContent/` (see `studio.rs` for detection and the evidence for the order)
//! - `rbxassetid://N`, `rbxassetid://N?version=V`, `http(s)://www.roblox.com/asset/?id=N` (and the
//!   other asset URL spellings), bare numeric ids → assetdelivery.roblox.com (public route first,
//!   then the account cookie; a refusal carries assetdelivery's reason, [`RefusalReason`]), then the
//!   CDN location it returns (never with the cookie)
//! - `rbxthumb://type=Asset&id=N&w=W&h=H` → thumbnails.roblox.com
//! - a local file path → the file
//!
//! [`Assets::fetch_transcode`] fetches the pack files of an image or TexturePack asset's KTX2
//! texture transcode instead (what Roblox clients draw; see [`Transcode`]), cached apart from the
//! uploaded image.
//!
//! Every network fetch is cached on disk (shared by every process using the same state dir, written
//! atomically, by other state dirs through hard links: [`AssetConfig::shared`], and by every state
//! dir of the user through the content store: [`AssetConfig::store`], [`store`]) and in memory;
//! concurrent fetches of one URI in a process download it once. Studio install and local files are
//! read in place (not copied to the disk cache) and memory-cached.
//!
//! The account cookie is read from the environment or env files (see [`AssetConfig`]), only ever
//! sent in the `Cookie` header of requests to `https://assetdelivery.roblox.com`, never forwarded
//! across redirects, and never printed, logged, cached or written anywhere.

mod cache;
mod cookie;
mod http;
mod sniff;
pub mod store;
mod studio;
mod transcode;
mod uri;
mod warm;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};

pub use sniff::sniff_kind;
pub use transcode::Transcode;
pub use uri::{ContentId, Thumbnail, ThumbnailKind};
pub use warm::{BATCH, Download, WarmReport};

use cache::{DiskCache, FileStamp, MemoryCache, Meta, Refusal};
use cookie::Secret;
use http::{Client, Endpoints, Reply};

#[derive(Clone, Debug)]
pub struct AssetConfig {
	/// Root of the state dir; downloads go to `<state_dir>/assets`.
	pub state_dir: PathBuf,
	/// The Studio install's `content` directory (or its version directory); auto-detected when
	/// `None`: env `ROR_ROBLOX_CONTENT`, then the registry value
	/// `HKCU\Software\Roblox\RobloxStudio\ContentFolder`, then the newest
	/// `%LOCALAPPDATA%\Roblox\Versions\*\RobloxStudioBeta.exe`.
	pub studio_content: Option<PathBuf>,
	/// No network access: only the Studio install, local files and the disk cache.
	pub offline: bool,
	/// Re-download network assets even when cached on disk (once per URI per [`Assets`]; repeats
	/// within the same `Assets` are served from memory).
	pub refresh: bool,
	/// Directory of the renderer checkout (its `.env` is a cookie source).
	///
	/// Cookie sources, first match wins: env `ROBLOX_ASSET_DELIVERY_COOKIE`, env `ROBLOSECURITY`,
	/// then the env files named by `ROR_COOKIE_ENV` and `RUR_COOKIE_ENV`, then `<repo_dir>/.env` (lines
	/// `ROBLOX_ASSET_DELIVERY_COOKIE=...` or `ROBLOSECURITY=...`), then the account signed in to Roblox
	/// Studio on this machine (unless `ROR_STUDIO_COOKIE=0`). Read lazily, on the first request that
	/// needs authentication.
	pub repo_dir: Option<PathBuf>,
	/// Byte budget of the in-memory cache.
	pub memory_budget: usize,
	/// Concurrent HTTP requests per [`Assets`] (and worker threads of [`Assets::fetch_many`]).
	pub max_concurrent_downloads: usize,
	/// Extra headers for assetdelivery API requests (e.g. `Roblox-Place-Id`). Never sent to the CDN
	/// or across redirects; `Cookie` and `Authorization` are ignored.
	pub assetdelivery_headers: Vec<(String, String)>,
	/// Other state dirs whose downloads this one shares: a file this state dir has not cached but
	/// one of them has is hard-linked in from there (one file on disk for both; copied where no link
	/// can be made) instead of downloaded again, offline too. Cached files are only ever replaced by
	/// a rename, never written into, so a refresh here leaves theirs as it was.
	pub shared: Vec<PathBuf>,
	/// The content store ([`store`], [`store::default_dir`]): a download this state dir lacks is taken
	/// from it (hard-linked in) or from another state dir that used it, and every download goes to it,
	/// so a fresh state dir downloads only what no state dir of the user has. `None` (the default): this
	/// state dir and `shared` alone.
	pub store: Option<PathBuf>,
}

impl AssetConfig {
	/// Defaults with the given state dir: Studio auto-detected, online, 64 MiB memory cache,
	/// 16 concurrent downloads, no shared state dirs, no content store.
	pub fn new(state_dir: impl Into<PathBuf>) -> AssetConfig {
		AssetConfig {
			state_dir: state_dir.into(),
			studio_content: None,
			offline: false,
			refresh: false,
			repo_dir: None,
			// the disk cache reads back in milliseconds what the GPU already holds anyway
			memory_budget: 64 << 20,
			max_concurrent_downloads: 16,
			assetdelivery_headers: Vec::new(),
			shared: Vec::new(),
			store: None,
		}
	}
}

impl Default for AssetConfig {
	/// State dir `.cache` (relative to the working directory).
	fn default() -> AssetConfig {
		AssetConfig::new(".cache")
	}
}

/// How long a refused download is remembered, by HTTP status: an asset the API will not give (not
/// authorised, not found) a day; a transcode still being made after the waits (406) a few minutes at
/// first, so a fresh process draws the upload at once while Roblox makes it (a game's attack bundles'
/// first renders spent seconds per process asking again), and then for as long as it has been made,
/// up to a day ([`Assets::remember_refusal`]). Other failures are not remembered.
fn refusal_ttl(status: u16) -> Option<u64> {
	match status {
		401 | 403 | 404 | 409 => Some(24 * 3600),
		406 => Some(10 * 60),
		_ => None,
	}
}

/// The error a remembered refusal answers with.
fn refused(uri: &str, refusal: Refusal) -> AssetError {
	match refusal.status {
		404 => AssetError::NotFound(uri.to_string()),
		status => http(uri, status, refusal.reason),
	}
}

/// An [`AssetError::Http`] about `uri`.
fn http(uri: &str, status: u16, reason: Option<RefusalReason>) -> AssetError {
	AssetError::Http { uri: uri.to_string(), status, reason }
}

/// Why assetdelivery refused an asset, from its error's `customErrorCode`, else the status.
///
/// [`Unreviewed`](RefusalReason::Unreviewed), [`Moderated`](RefusalReason::Moderated) and
/// [`Archived`](RefusalReason::Archived) are refused to every account and every place: no cookie,
/// `Roblox-Place-Id` or endpoint gets them, Studio cannot load them either and draws nothing for
/// them (notes/assets.md "Refusals"). The others depend on the account.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusalReason {
	/// `customErrorCode` 18, "Asset has not been reviewed"
	Unreviewed,
	/// `customErrorCode` 19, "Asset is not approved for the requester" (Studio logs it as
	/// `AssetDelivery403Moderated`)
	Moderated,
	/// `customErrorCode` 22, "Requested asset is archived"
	Archived,
	/// 401, "Authentication required to access Asset."
	AuthenticationRequired,
	/// 403 (`customErrorCode` 1 or none) or 409, "User is not authorized to access Asset."
	NotAuthorized,
	/// Another `customErrorCode`: assetdelivery's message (printable ASCII, at most 100 characters)
	Other(String),
}

impl RefusalReason {
	/// The reason of an assetdelivery error with refusal `status` (401, 403, 409); `None` for
	/// other statuses.
	fn of(error: Option<&ApiError>, status: u16) -> Option<RefusalReason> {
		if !matches!(status, 401 | 403 | 409) {
			return None;
		}
		let custom = error.and_then(|error| error.custom_code);
		Some(match custom {
			Some(18) => RefusalReason::Unreviewed,
			Some(19) => RefusalReason::Moderated,
			Some(22) => RefusalReason::Archived,
			Some(code) if code != 1 => {
				let message: String = error.and_then(|error| error.message.as_deref()).unwrap_or("").chars().filter(|c| (' '..='~').contains(c)).take(100).collect();
				RefusalReason::Other(if message.trim().is_empty() { format!("error {code}") } else { message.trim().to_string() })
			}
			_ if status == 401 => RefusalReason::AuthenticationRequired,
			_ => RefusalReason::NotAuthorized,
		})
	}

	/// Whether every account and place is refused alike (unreviewed, moderated, archived).
	pub fn universal(&self) -> bool {
		matches!(self, RefusalReason::Unreviewed | RefusalReason::Moderated | RefusalReason::Archived)
	}
}

impl std::fmt::Display for RefusalReason {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			RefusalReason::Unreviewed => f.write_str("not yet reviewed by Roblox moderation"),
			RefusalReason::Moderated => f.write_str("rejected by Roblox moderation"),
			RefusalReason::Archived => f.write_str("archived by its creator"),
			RefusalReason::AuthenticationRequired => f.write_str("authentication required"),
			RefusalReason::NotAuthorized => f.write_str("not authorized for this account"),
			RefusalReason::Other(message) => write!(f, "{message:?}"),
		}
	}
}

/// `, <reason>` after an HTTP status, when there is one.
fn reason_suffix(reason: &Option<RefusalReason>) -> String {
	reason.as_ref().map_or_else(String::new, |reason| format!(", {reason}"))
}

/// Where a fetched asset came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
	StudioInstall,
	LocalFile,
	DiskCache,
	Network,
}

#[derive(Debug)]
pub struct Fetched {
	/// The URI as requested (for repeats served from memory: the spelling of this request).
	pub uri: String,
	pub bytes: Vec<u8>,
	/// Where the bytes were loaded from (memory hits report the original source).
	pub source: Source,
}

impl Fetched {
	/// Sniffed content type, see [`sniff_kind`].
	pub fn kind(&self) -> &'static str {
		sniff_kind(&self.bytes)
	}
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum AssetError {
	#[error("invalid content URI {0:?}")]
	InvalidUri(String),
	#[error("{0} not found")]
	NotFound(String),
	#[error("{uri}: offline and not cached")]
	Offline { uri: String },
	/// An HTTP status, or an assetdelivery error code (401 authentication required, 403 / 409
	/// refused, ...), with why assetdelivery refused it (`, rejected by Roblox moderation`).
	#[error("{uri}: HTTP {status}{}", reason_suffix(.reason))]
	Http { uri: String, status: u16, reason: Option<RefusalReason> },
	#[error("{uri}: {detail}")]
	Network { uri: String, detail: String },
	#[error("{0}")]
	Io(String),
	/// The content has no texture transcode: not an asset id (Studio files, local files,
	/// thumbnails), or assetdelivery answered with something other than a KTX2 file.
	#[error("{0}: no texture transcode")]
	NoTranscode(String),
}

/// The result of one fetch.
pub type FetchResult = Result<Arc<Fetched>, AssetError>;

/// In-flight fetch shared by every caller asking for the same URI.
struct Slot {
	result: Mutex<Option<FetchResult>>,
	ready: Condvar,
}

struct Inner {
	config: AssetConfig,
	client: Client,
	disk: DiskCache,
	studio: OnceLock<Option<studio::StudioInstall>>,
	cookie: OnceLock<Option<(Secret, cookie::CookieSource)>>,
	memory: Mutex<MemoryCache>,
	inflight: Mutex<HashMap<String, Arc<Slot>>>,
	/// Network URIs already re-downloaded in refresh mode
	refreshed: Mutex<HashSet<String>>,
}

/// The shared asset resolver. Cheap to clone; safe to use from many threads at once.
#[derive(Clone)]
pub struct Assets {
	inner: Arc<Inner>,
}

impl Assets {
	pub fn new(config: AssetConfig) -> Self {
		Self::with_endpoints(config, Endpoints::production(), None)
	}

	fn with_endpoints(config: AssetConfig, endpoints: Endpoints, cookie: Option<Option<(Secret, cookie::CookieSource)>>) -> Self {
		let client = Client::new(endpoints, config.max_concurrent_downloads, &config.assetdelivery_headers);
		let cookie_cell = OnceLock::new();
		if let Some(preset) = cookie {
			let _ = cookie_cell.set(preset);
		}
		Assets {
			inner: Arc::new(Inner {
				disk: DiskCache::new(&config.state_dir, &config.shared, config.store.as_deref()),
				memory: Mutex::new(MemoryCache::new(config.memory_budget)),
				client,
				studio: OnceLock::new(),
				cookie: cookie_cell,
				inflight: Mutex::new(HashMap::new()),
				refreshed: Mutex::new(HashSet::new()),
				config,
			}),
		}
	}

	pub fn config(&self) -> &AssetConfig {
		&self.inner.config
	}

	/// The detected Studio `content` directory.
	pub fn studio_content_dir(&self) -> Option<&Path> {
		self.studio().map(|s| s.content.as_path())
	}

	/// The file an `rbxasset://` path resolves to, without reading it.
	pub fn studio_file(&self, path: &str) -> Option<PathBuf> {
		match ContentId::parse(&format!("rbxasset://{path}")).ok()? {
			ContentId::StudioAsset(p) => self.studio()?.find(&p),
			_ => None,
		}
	}

	/// Bytes of content held by the in-memory cache.
	pub fn memory_cache_bytes(&self) -> usize {
		self.inner.memory.lock().used()
	}

	/// Lets go of the in-memory cache (a later fetch reads the disk cache); returns its bytes.
	pub fn release_memory(&self) -> usize {
		let mut memory = self.inner.memory.lock();
		let used = memory.used();
		*memory = MemoryCache::new(self.inner.config.memory_budget);
		used
	}

	/// Whether a cookie is configured (reads the sources; never reveals the value).
	pub fn has_cookie(&self) -> bool {
		self.cookie().is_some()
	}

	/// A description of where the cookie comes from (`env ROBLOSECURITY`, `file <path>`, `studio`),
	/// never the value itself.
	pub fn cookie_source(&self) -> Option<String> {
		self.inner.cookie.get_or_init(|| cookie::load(self.inner.config.repo_dir.as_deref())).as_ref().map(|(_, source)| match source {
			cookie::CookieSource::Env(name) => format!("env {name}"),
			cookie::CookieSource::File(path) => format!("file {}", path.display()),
			cookie::CookieSource::Studio => "studio".to_string(),
		})
	}

	/// One line per cookie source with its status (`not set`, `missing`, `no cookie line`,
	/// `invalid value`, `ok`, ...), in lookup order. Never contains the value.
	pub fn cookie_diagnostics(&self) -> Vec<String> {
		cookie::diagnose(self.inner.config.repo_dir.as_deref())
	}

	fn studio(&self) -> Option<&studio::StudioInstall> {
		self.inner.studio.get_or_init(|| studio::detect(self.inner.config.studio_content.as_deref())).as_ref()
	}

	fn cookie(&self) -> Option<&Secret> {
		self.inner.cookie.get_or_init(|| cookie::load(self.inner.config.repo_dir.as_deref())).as_ref().map(|(s, _)| s)
	}

	/// Fetches a content URI (blocking). Concurrent calls for the same URI share one download.
	pub fn fetch(&self, uri: &str) -> Result<Arc<Fetched>, AssetError> {
		let id = ContentId::parse(uri)?;
		let key = id.canonical();
		let stamp = match &id {
			ContentId::File(path) => Some(FileStamp::of(path).ok_or_else(|| AssetError::NotFound(uri.to_string()))?),
			_ => None,
		};
		self.fetch_keyed(&key, uri, stamp, || self.load(&id, &key, uri))
	}

	/// Fetches one pack file of an asset's KTX2 texture transcode (blocking): what Roblox clients draw
	/// in place of the uploaded image. Pack 0 holds the smallest levels and lists the others in its
	/// `pack_N` keys (`ror_image::ktx2_pack_table`); `ror_image::join_packs` joins them. Each pack is
	/// cached under its own key, apart from the uploaded image. Errors with
	/// [`AssetError::NoTranscode`] for content that is not an asset id; a transcode that is still
	/// being made (assetdelivery's 406) is waited for a few seconds, then reported as HTTP 406.
	pub fn fetch_transcode(&self, uri: &str, transcode: Transcode, pack: u32) -> Result<Arc<Fetched>, AssetError> {
		let id = ContentId::parse(uri)?;
		let ContentId::Asset { id: asset, version } = id else {
			return Err(AssetError::NoTranscode(uri.to_string()));
		};
		if pack >= transcode::MAX_PACKS {
			return Err(AssetError::NoTranscode(format!("{uri} pack {pack}")));
		}
		let key = transcode.cache_key(&id.canonical(), pack);
		self.fetch_keyed(&key, uri, None, || {
			self.load_disk_or_network(&key, uri, || {
				// one still being made when last asked is asked once more, not waited for again
				let waits: &[u64] = if self.inner.disk.last_refusal(&key).is_some_and(|last| last.status == 406) { &[] } else { &TRANSCODE_WAITS_MS };
				self.download_transcode(asset, version, transcode, pack, uri, waits)
			})
		})
	}

	/// Remembers that the download of `key` was refused with `status` ([`refusal_ttl`]). A transcode
	/// still being made (406) is remembered for as long as it has been answered so, 10 minutes at
	/// least and a day at most: a map's textures whose transcodes Roblox had been "making" all
	/// day cost the first render of the map every 10 minutes 7.5 s of waits.
	fn remember_refusal(&self, key: &str, status: u16, reason: Option<RefusalReason>, now: u64) {
		let Some(ttl) = refusal_ttl(status) else { return };
		let (ttl, since) = if status == 406 {
			let since = self.inner.disk.last_refusal(key).filter(|last| last.status == 406).and_then(|last| last.since).unwrap_or(now);
			(now.saturating_sub(since).clamp(ttl, 24 * 3600), Some(since))
		} else {
			(ttl, None)
		};
		let _ = self.inner.disk.write_refusal(key, &Refusal { status, until: now + ttl, reason, since });
	}

	/// The memory cache, then one shared load per key (concurrent callers wait for the first).
	fn fetch_keyed(&self, key: &str, uri: &str, stamp: Option<FileStamp>, load: impl FnOnce() -> Result<Fetched, AssetError>) -> Result<Arc<Fetched>, AssetError> {
		let key = key.to_string();
		if let Some(hit) = self.inner.memory.lock().get(&key, stamp) {
			return Ok(respell(hit, uri));
		}

		// Join or start the in-flight fetch for this key
		let (slot, leader) = {
			let mut inflight = self.inner.inflight.lock();
			match inflight.get(&key) {
				Some(slot) => (slot.clone(), false),
				None => {
					let slot = Arc::new(Slot { result: Mutex::new(None), ready: Condvar::new() });
					inflight.insert(key.clone(), slot.clone());
					(slot, true)
				}
			}
		};

		if !leader {
			let mut result = slot.result.lock();
			while result.is_none() {
				slot.ready.wait(&mut result);
			}
			return match result.as_ref() {
				Some(Ok(fetched)) => Ok(respell(fetched.clone(), uri)),
				Some(Err(e)) => Err(e.clone()),
				None => Err(AssetError::Io("in-flight fetch vanished".into())),
			};
		}

		// Leader: publish the result even if loading panics
		struct Publish<'a> {
			assets: &'a Assets,
			key: &'a str,
			slot: &'a Slot,
			done: bool,
		}
		impl Drop for Publish<'_> {
			fn drop(&mut self) {
				if !self.done {
					*self.slot.result.lock() = Some(Err(AssetError::Io(format!("{}: fetch panicked", self.key))));
					self.slot.ready.notify_all();
					self.assets.inner.inflight.lock().remove(self.key);
				}
			}
		}
		let mut publish = Publish { assets: self, key: &key, slot: &slot, done: false };

		let result = load().map(Arc::new);
		if let Ok(fetched) = &result {
			// Memory first, so a fetch arriving after the slot is gone hits the cache
			self.inner.memory.lock().insert(&key, fetched.clone(), stamp);
		}
		*slot.result.lock() = Some(result.clone());
		slot.ready.notify_all();
		self.inner.inflight.lock().remove(&key);
		publish.done = true;
		result
	}

	/// Fetches many URIs with bounded parallelism ([`AssetConfig::max_concurrent_downloads`]
	/// workers). Results are in input order.
	pub fn fetch_many<S: AsRef<str> + Sync>(&self, uris: &[S]) -> Vec<FetchResult> {
		let workers = self.inner.config.max_concurrent_downloads.clamp(1, 64).min(uris.len());
		if workers <= 1 {
			return uris.iter().map(|u| self.fetch(u.as_ref())).collect();
		}
		let next = std::sync::atomic::AtomicUsize::new(0);
		let results: Vec<Mutex<Option<FetchResult>>> = uris.iter().map(|_| Mutex::new(None)).collect();
		std::thread::scope(|scope| {
			for _ in 0..workers {
				scope.spawn(|| {
					loop {
						let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
						if i >= uris.len() {
							break;
						}
						*results[i].lock() = Some(self.fetch(uris[i].as_ref()));
					}
				});
			}
		});
		results
			.into_iter()
			.enumerate()
			.map(|(i, r)| r.into_inner().unwrap_or_else(|| Err(AssetError::Io(format!("{}: not fetched", uris[i].as_ref())))))
			.collect()
	}

	fn load(&self, id: &ContentId, key: &str, uri: &str) -> Result<Fetched, AssetError> {
		match id {
			ContentId::StudioAsset(path) => {
				let file = self.studio().and_then(|s| s.find(path)).ok_or_else(|| AssetError::NotFound(uri.to_string()))?;
				let bytes = std::fs::read(&file).map_err(|e| AssetError::Io(format!("{uri}: {e}")))?;
				Ok(Fetched { uri: uri.to_string(), bytes, source: Source::StudioInstall })
			}
			ContentId::File(path) => {
				let bytes = std::fs::read(path).map_err(|e| match e.kind() {
					std::io::ErrorKind::NotFound => AssetError::NotFound(uri.to_string()),
					_ => AssetError::Io(format!("{uri}: {e}")),
				})?;
				Ok(Fetched { uri: uri.to_string(), bytes, source: Source::LocalFile })
			}
			ContentId::Asset { .. } | ContentId::Thumbnail(_) => self.load_network(id, key, uri),
		}
	}

	fn load_network(&self, id: &ContentId, key: &str, uri: &str) -> Result<Fetched, AssetError> {
		self.load_disk_or_network(key, uri, || match id {
			ContentId::Asset { id, version } => self.download_asset(*id, *version, uri),
			ContentId::Thumbnail(t) => self.download_thumbnail(t, uri),
			_ => Err(AssetError::InvalidUri(uri.to_string())),
		})
	}

	/// The disk cache under `key` (unless refreshing), else `download` (bytes, host), cached on disk.
	/// A refused download is remembered on disk too ([`refusal_ttl`]), so other processes sharing the
	/// state dir answer it at once instead of asking again.
	fn load_disk_or_network(&self, key: &str, uri: &str, download: impl FnOnce() -> Result<(Vec<u8>, String), AssetError>) -> Result<Fetched, AssetError> {
		let config = &self.inner.config;
		let use_disk = !config.refresh || self.inner.refreshed.lock().contains(key);
		if (use_disk || config.offline)
			&& let Some(bytes) = self.inner.disk.read(key)
		{
			return Ok(Fetched { uri: uri.to_string(), bytes, source: Source::DiskCache });
		}
		if config.offline {
			return Err(AssetError::Offline { uri: uri.to_string() });
		}
		let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
		if !config.refresh
			&& let Some(refusal) = self.inner.disk.read_refusal(key, now)
		{
			return Err(refused(uri, refusal));
		}

		let (bytes, host) = match download() {
			Ok(downloaded) => downloaded,
			Err(error) => {
				let refusal = match &error {
					AssetError::Http { status, reason, .. } => Some((*status, reason.clone())),
					AssetError::NotFound(_) => Some((404, None)),
					_ => None,
				};
				// an authorisation refusal is only final once the cookie was tried too
				let final_refusal = |status: u16| !matches!(status, 401 | 403 | 409) || self.cookie().is_some();
				if let Some((status, reason)) = refusal
					&& final_refusal(status)
				{
					self.remember_refusal(key, status, reason, now);
				}
				return Err(error);
			}
		};

		let meta = Meta {
			uri: key.to_string(),
			fetched_at: now,
			source_host: host,
			size: bytes.len(),
			kind: sniff_kind(&bytes).to_string(),
		};
		// A cache write failure does not fail the fetch
		let _ = self.inner.disk.write(key, &bytes, &meta);
		if config.refresh {
			self.inner.refreshed.lock().insert(key.to_string());
		}
		Ok(Fetched { uri: uri.to_string(), bytes, source: Source::Network })
	}

	/// assetdelivery `GET /v1/assetId/{id}[/version/{v}]` (public, then with the cookie on an
	/// authorisation error), then the CDN location without the cookie. Returns (bytes, CDN host).
	fn download_asset(&self, id: u64, version: Option<u64>, uri: &str) -> Result<(Vec<u8>, String), AssetError> {
		let client = &self.inner.client;
		let base = &client.endpoints().assetdelivery;
		let url = match version {
			Some(v) => format!("{base}/v1/assetId/{id}/version/{v}"),
			None => format!("{base}/v1/assetId/{id}"),
		};

		let location = match self.asset_location(&url, None, uri)? {
			Located::Url(location) => location,
			Located::Unauthorized(status, reason) => match self.cookie() {
				Some(secret) => match self.asset_location(&url, Some(secret), uri)? {
					Located::Url(location) => location,
					Located::Unauthorized(status, reason) => return Err(http(uri, status, reason)),
					Located::Generating => return Err(http(uri, 406, None)),
				},
				None => return Err(http(uri, status, reason)),
			},
			Located::Generating => return Err(http(uri, 406, None)),
		};

		let reply = self.get(&location, None, "*/*", uri)?;
		if reply.status != 200 {
			return Err(http(uri, reply.status, None));
		}
		Ok((gunzip_if_needed(reply.body, uri)?, reply.host))
	}

	/// assetdelivery `GET /v1/assetId/{id}[/version/{v}]` naming the transcode's pack in
	/// `contentRepresentationPriorityList`, without falling back to the uploaded image (public, then
	/// with the cookie on an authorisation error; while the transcode is being made, retried after
	/// 0.5, 1, 2 and 4 s), then the CDN location without the cookie. Returns (KTX2 bytes, CDN host).
	fn download_transcode(&self, id: u64, version: Option<u64>, transcode: Transcode, pack: u32, uri: &str, waits: &[u64]) -> Result<(Vec<u8>, String), AssetError> {
		let client = &self.inner.client;
		let base = &client.endpoints().assetdelivery;
		let path = match version {
			Some(v) => format!("{base}/v1/assetId/{id}/version/{v}"),
			None => format!("{base}/v1/assetId/{id}"),
		};
		let url = format!("{path}?contentRepresentationPriorityList={}&doNotFallbackToBaselineRepresentation=true", transcode::query_escape(&transcode.priority_list(pack)));

		let mut cookie = None;
		let mut waits = waits.iter();
		let location = loop {
			match self.transcode_location(&url, cookie, uri)? {
				Located::Url(location) => break location,
				Located::Unauthorized(status, reason) => match (cookie, self.cookie()) {
					(None, Some(secret)) => cookie = Some(secret),
					_ => return Err(http(uri, status, reason)),
				},
				Located::Generating => match waits.next() {
					Some(&ms) => std::thread::sleep(std::time::Duration::from_millis(ms)),
					None => return Err(http(uri, 406, None)),
				},
			}
		};

		let reply = self.get(&location, None, "*/*", uri)?;
		if reply.status != 200 {
			return Err(http(uri, reply.status, None));
		}
		let body = gunzip_if_needed(reply.body, uri)?;
		if sniff_kind(&body) != "ktx2" {
			return Err(AssetError::NoTranscode(uri.to_string()));
		}
		Ok((body, reply.host))
	}

	/// A transcode's location: only a reply naming the requested representation counts (without one
	/// the location is the uploaded image's).
	fn transcode_location(&self, url: &str, cookie: Option<&Secret>, uri: &str) -> Result<Located, AssetError> {
		let reply = self.get(url, cookie, "application/json", uri)?;
		let parsed: Option<TranscodeLocation> = serde_json::from_slice(&reply.body).ok();
		if let Some(parsed) = parsed.as_ref()
			&& let Some(location) = parsed.location.as_ref().filter(|l| !l.is_empty())
		{
			let named = parsed.specifier.as_ref().is_some_and(|s| s.format.as_deref() == Some(transcode::FORMAT));
			return if named { Ok(Located::Url(location.clone())) } else { Err(AssetError::NoTranscode(uri.to_string())) };
		}
		let (status, reason) = error_status(parsed.as_ref().and_then(|p| p.errors.as_deref()), reply.status);
		match status {
			401 | 403 | 409 => Ok(Located::Unauthorized(status, reason)),
			406 => Ok(Located::Generating),
			404 => Err(AssetError::NotFound(uri.to_string())),
			_ => Err(http(uri, status, None)),
		}
	}

	fn asset_location(&self, url: &str, cookie: Option<&Secret>, uri: &str) -> Result<Located, AssetError> {
		let reply = self.get(url, cookie, "application/json", uri)?;
		let parsed: Option<AssetLocation> = serde_json::from_slice(&reply.body).ok();
		if let Some(location) = parsed.as_ref().and_then(|p| p.location.clone()).filter(|l| !l.is_empty()) {
			return Ok(Located::Url(location));
		}
		// Errors arrive as HTTP 200 with {"errors":[{"code":403,"customErrorCode":19,...}]} or as plain statuses
		let (status, reason) = error_status(parsed.as_ref().and_then(|p| p.errors.as_deref()), reply.status);
		match status {
			401 | 403 | 409 => Ok(Located::Unauthorized(status, reason)),
			404 => Err(AssetError::NotFound(uri.to_string())),
			_ => Err(http(uri, status, None)),
		}
	}

	/// thumbnails.roblox.com: JSON with `imageUrl` (retrying while `Pending`), then the image.
	fn download_thumbnail(&self, thumb: &Thumbnail, uri: &str) -> Result<(Vec<u8>, String), AssetError> {
		let endpoints = self.inner.client.endpoints().clone();
		let size = format!("{}x{}", thumb.width, thumb.height);
		let circular = if thumb.circular { "true" } else { "false" };

		let mut target = thumb.id;
		if thumb.kind == ThumbnailKind::GameThumbnail {
			// rbxthumb takes a place id; the API wants the universe
			let reply = self.get(&format!("{}/universes/v1/places/{}/universe", endpoints.universes, thumb.id), None, "application/json", uri)?;
			#[derive(Deserialize)]
			struct Universe {
				#[serde(rename = "universeId")]
				universe_id: Option<u64>,
			}
			target = serde_json::from_slice::<Universe>(&reply.body).ok().and_then(|u| u.universe_id).ok_or_else(|| AssetError::NotFound(uri.to_string()))?;
		}

		let (path, param) = match thumb.kind {
			ThumbnailKind::Asset | ThumbnailKind::FontFamily => ("v1/assets", "assetIds"),
			ThumbnailKind::Avatar => ("v1/users/avatar", "userIds"),
			ThumbnailKind::AvatarBust => ("v1/users/avatar-bust", "userIds"),
			ThumbnailKind::AvatarHeadShot => ("v1/users/avatar-headshot", "userIds"),
			ThumbnailKind::BadgeIcon => ("v1/badges/icons", "badgeIds"),
			ThumbnailKind::BundleThumbnail => ("v1/bundles/thumbnails", "bundleIds"),
			ThumbnailKind::GameIcon => ("v1/games/icons", "universeIds"),
			ThumbnailKind::GamePass => ("v1/game-passes", "gamePassIds"),
			ThumbnailKind::GameThumbnail => ("v1/games/multiget/thumbnails", "universeIds"),
			ThumbnailKind::GroupIcon => ("v1/groups/icons", "groupIds"),
			ThumbnailKind::Outfit => ("v1/users/outfits", "userOutfitIds"),
		};
		let mut url = format!("{}/{path}?{param}={target}&size={size}&format=Png&isCircular={circular}", endpoints.thumbnails);
		if thumb.kind == ThumbnailKind::GameThumbnail {
			url.push_str("&countPerUniverse=1");
		}

		for attempt in 0..10u32 {
			let reply = self.get(&url, None, "application/json", uri)?;
			if reply.status != 200 {
				return Err(http(uri, reply.status, None));
			}
			let parsed: ThumbnailReply = serde_json::from_slice(&reply.body).map_err(|_| AssetError::Network { uri: uri.to_string(), detail: "unexpected thumbnail API reply".into() })?;
			if let Some(error) = parsed.errors.as_ref().and_then(|e| e.first()) {
				let detail = error.message.clone().unwrap_or_else(|| "thumbnail API error".into());
				return Err(AssetError::Network { uri: uri.to_string(), detail });
			}
			let Some(entry) = parsed.data.into_iter().next().map(|d| d.thumbnails.and_then(|t| t.into_iter().next()).unwrap_or(d.entry)) else {
				return Err(AssetError::NotFound(uri.to_string()));
			};
			match (entry.state.as_deref(), entry.image_url) {
				// Blocked thumbnails point at Roblox's "unapproved" placeholder, as in game
				(Some("Completed") | Some("Blocked"), Some(image)) if !image.is_empty() => {
					let reply = self.get(&image, None, "image/png,image/*", uri)?;
					if reply.status != 200 {
						return Err(http(uri, reply.status, None));
					}
					return Ok((gunzip_if_needed(reply.body, uri)?, reply.host));
				}
				(Some("Pending"), _) => std::thread::sleep(std::time::Duration::from_millis(500 + 250 * attempt as u64)),
				(state, _) => {
					return Err(AssetError::Network { uri: uri.to_string(), detail: format!("thumbnail state {}", state.unwrap_or("unknown")) });
				}
			}
		}
		Err(AssetError::Network { uri: uri.to_string(), detail: "thumbnail still pending".into() })
	}

	/// GET with retries; failures become errors about `uri` (never the request URL or headers).
	fn get(&self, url: &str, cookie: Option<&Secret>, accept: &str, uri: &str) -> Result<Reply, AssetError> {
		self.inner.client.get_retrying(url, cookie, accept).map_err(|f| AssetError::Network { uri: uri.to_string(), detail: f.describe() })
	}
}

/// The same entry under the caller's spelling of the URI.
fn respell(fetched: Arc<Fetched>, uri: &str) -> Arc<Fetched> {
	if fetched.uri == uri {
		fetched
	} else {
		Arc::new(Fetched { uri: uri.to_string(), bytes: fetched.bytes.clone(), source: fetched.source })
	}
}

/// CDN objects are stored gzip-compressed; ureq decodes `Content-Encoding: gzip`, this handles
/// bodies that arrive compressed without the header.
fn gunzip_if_needed(body: Vec<u8>, uri: &str) -> Result<Vec<u8>, AssetError> {
	if !body.starts_with(&[0x1f, 0x8b, 0x08]) {
		return Ok(body);
	}
	use std::io::Read;
	let mut out = Vec::with_capacity(body.len() * 3);
	flate2::read::MultiGzDecoder::new(&body[..])
		.take(1 << 30)
		.read_to_end(&mut out)
		.map_err(|e| AssetError::Network { uri: uri.to_string(), detail: format!("gzip: {e}") })?;
	Ok(out)
}

enum Located {
	Url(String),
	/// Refused, other credentials may help: the status / assetdelivery error code, and why
	Unauthorized(u16, Option<RefusalReason>),
	/// A transcode assetdelivery is still making (code 406, "Asset content representation is being
	/// generated")
	Generating,
}

/// Waits between requests for a transcode that is being made.
#[cfg(not(test))]
const TRANSCODE_WAITS_MS: [u64; 4] = [500, 1000, 2000, 4000];
#[cfg(test)]
const TRANSCODE_WAITS_MS: [u64; 4] = [10, 10, 10, 10];

#[derive(Deserialize)]
struct AssetLocation {
	location: Option<String>,
	errors: Option<Vec<ApiError>>,
}

#[derive(Deserialize)]
struct TranscodeLocation {
	location: Option<String>,
	errors: Option<Vec<ApiError>>,
	#[serde(rename = "contentRepresentationSpecifier")]
	specifier: Option<RepresentationSpecifier>,
}

#[derive(Deserialize)]
struct RepresentationSpecifier {
	format: Option<String>,
}

#[derive(Deserialize)]
struct ApiError {
	code: Option<i64>,
	message: Option<String>,
	/// assetdelivery's finer code (18 unreviewed, 19 moderated, 21 being generated, 22 archived)
	#[serde(rename = "customErrorCode", alias = "CustomErrorCode")]
	custom_code: Option<i64>,
}

/// The status of an assetdelivery reply without a location (its first error's code, else the HTTP
/// status; 502 for a 200 without either) and, for a refusal, why.
fn error_status(errors: Option<&[ApiError]>, http_status: u16) -> (u16, Option<RefusalReason>) {
	let error = errors.and_then(|errors| errors.first());
	let code = error.and_then(|e| e.code).filter(|c| (100..600).contains(c)).map(|c| c as u16);
	let status = code.unwrap_or(if http_status == 200 { 502 } else { http_status });
	(status, RefusalReason::of(error, status))
}

#[derive(Deserialize)]
struct ThumbnailReply {
	#[serde(default)]
	data: Vec<ThumbnailData>,
	errors: Option<Vec<ApiError>>,
}

#[derive(Deserialize)]
struct ThumbnailData {
	#[serde(flatten)]
	entry: ThumbnailEntry,
	/// `games/multiget/thumbnails` nests the entries
	thumbnails: Option<Vec<ThumbnailEntry>>,
}

#[derive(Deserialize)]
struct ThumbnailEntry {
	state: Option<String>,
	#[serde(rename = "imageUrl")]
	image_url: Option<String>,
}
