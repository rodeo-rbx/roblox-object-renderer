//! Filling the disk cache many files at a time ([`Assets::warm`]): assetdelivery's batch endpoint
//! (`POST /v1/assets/batch`) answers the locations of many assets per request, and the CDN files
//! come side by side. A later [`Assets::fetch`] / [`Assets::fetch_transcode`] then reads the disk.
//!
//! One file through [`Assets::fetch`] costs a round trip to assetdelivery (two for a private asset:
//! the public route first, then the cookie's) and one to the CDN, under a cap of
//! [`AssetConfig::max_concurrent_downloads`] at a time: a renderer's first look at a scene of
//! hundreds of files waits on round trips, not bandwidth. Here a batch request (with the cookie when
//! there is one, so private assets cost no extra trip) locates up to [`BATCH`] files at once.
//!
//! [`AssetConfig::max_concurrent_downloads`]: crate::AssetConfig::max_concurrent_downloads

use std::collections::{HashSet, VecDeque};

use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};

use crate::cache::Meta;
use crate::transcode::{self, MAX_PACKS, Transcode};
use crate::uri::ContentId;
use crate::{ApiError, Assets, RefusalReason, RepresentationSpecifier, gunzip_if_needed, sniff_kind};

/// Items per batch request (assetdelivery answers 100 in one).
pub const BATCH: usize = 100;
/// Rounds: the files and the transcodes' first packs; the packs those list (and the uploads of
/// images without a transcode); whatever a third round still finds.
const ROUNDS: usize = 3;

/// Something [`Assets::warm`] downloads.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Download {
	/// A content URI, as [`Assets::fetch`] fetches it.
	Asset(String),
	/// Every pack of an asset's transcode, as [`Assets::fetch_transcode`] fetches them; for an image
	/// asset without a transcode, its upload (what a renderer draws instead).
	Transcode(String, Transcode),
}

/// What [`Assets::warm`] did.
#[derive(Clone, Debug, Default)]
pub struct WarmReport {
	/// Files downloaded into the disk cache, and their bytes.
	pub downloaded: usize,
	pub bytes: u64,
	/// Files that were on disk already (or refused before).
	pub cached: usize,
	/// Files that could not be fetched now (a later fetch tries again and reports why).
	pub failed: usize,
	/// Batch requests sent.
	pub batches: usize,
}

#[derive(Clone, Debug)]
struct Item {
	key: String,
	uri: String,
	canonical: String,
	id: u64,
	version: Option<u64>,
	kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
	Plain,
	Pack(Transcode, u32),
}

struct Queue {
	next_chunk: usize,
	locating: usize,
	located: VecDeque<(usize, Place)>,
}

enum Job {
	Locate(usize),
	Download((usize, Place)),
}

/// Where a batch request placed an item.
enum Place {
	Url(String),
	/// The asset has no such transcode.
	NoTranscode,
	/// Refused (an HTTP status or assetdelivery error code), and why.
	Refused(u16, Option<RefusalReason>),
	/// A transcode assetdelivery is still making (406): a fetch would wait seconds for it, then draw
	/// the upload; remembered as it would be, the upload comes now.
	Generating,
	/// Not known: the request failed, the reply was odd.
	Unknown,
}

/// What became of an item: downloaded (with the packs a pack 0 lists), without a transcode, failed.
enum Outcome {
	Downloaded { packs: u32 },
	NoTranscode,
	Failed,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchRequest {
	request_id: String,
	asset_id: u64,
	#[serde(skip_serializing_if = "Option::is_none")]
	version: Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	content_representation_priority_list: Option<String>,
	#[serde(skip_serializing_if = "std::ops::Not::not")]
	do_not_fallback_to_baseline_representation: bool,
}

#[derive(Deserialize)]
struct BatchReply {
	#[serde(rename = "requestId", alias = "RequestId")]
	request_id: Option<String>,
	#[serde(alias = "Location")]
	location: Option<String>,
	#[serde(alias = "Errors")]
	errors: Option<Vec<ApiError>>,
	#[serde(rename = "contentRepresentationSpecifier", alias = "ContentRepresentationSpecifier")]
	specifier: Option<RepresentationSpecifier>,
}

fn now() -> u64 {
	std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl Assets {
	/// Whether nothing is fetched: [`Assets::warm`] then does nothing, so a caller can skip what it would hand it.
	pub fn offline(&self) -> bool {
		self.inner.config.offline
	}

	/// Downloads into the disk cache whatever of `downloads` is not there yet, many files at a time:
	/// their locations from assetdelivery's batch endpoint, the files from the CDN side by side (up
	/// to [`crate::AssetConfig::max_concurrent_downloads`]). A transcode's pack 0 comes first; `packs`
	/// tells from its bytes how many packs the transcode has (`ror_image::ktx2_pack_table`), and
	/// the others follow. Nothing here fails: what could not be fetched is left to a later fetch,
	/// which tries again and reports it. Offline, it does nothing.
	pub fn warm(&self, downloads: &[Download], packs: &(dyn Fn(&[u8]) -> u32 + Sync)) -> WarmReport {
		let mut report = WarmReport::default();
		if self.inner.config.offline {
			return report;
		}
		let mut seen = HashSet::new();
		let mut round = Vec::new();
		for download in downloads {
			let (uri, transcode) = match download {
				Download::Asset(uri) => (uri, None),
				Download::Transcode(uri, transcode) => (uri, Some(*transcode)),
			};
			let Ok(ContentId::Asset { id, version }) = ContentId::parse(uri) else {
				continue;
			};
			let canonical = ContentId::Asset { id, version }.canonical();
			let item = |key: String, kind: Kind| Item { key, uri: uri.clone(), canonical: canonical.clone(), id, version, kind };
			match transcode {
				None => self.consider(item(canonical.clone(), Kind::Plain), &mut seen, &mut round, &mut report),
				Some(transcode) => {
					let first = transcode.cache_key(&canonical, 0);
					// a cached pack 0 tells which packs must follow
					match self.on_disk(&first).then(|| self.inner.disk.read(&first)).flatten() {
						Some(bytes) => {
							seen.insert(first);
							report.cached += 1;
							for pack in 1..packs(&bytes).min(MAX_PACKS) {
								self.consider(item(transcode.cache_key(&canonical, pack), Kind::Pack(transcode, pack)), &mut seen, &mut round, &mut report);
							}
						}
						None => self.consider(item(first, Kind::Pack(transcode, 0)), &mut seen, &mut round, &mut report),
					}
				}
			}
		}
		for _ in 0..ROUNDS {
			if round.is_empty() {
				break;
			}
			let outcomes = self.fetch_items(&round, packs, &mut report);
			let mut next = Vec::new();
			for (item, outcome) in round.iter().zip(outcomes) {
				match (item.kind, outcome) {
					(Kind::Pack(transcode, 0), Outcome::Downloaded { packs }) => {
						for pack in 1..packs.min(MAX_PACKS) {
							let follow = Item { key: transcode.cache_key(&item.canonical, pack), kind: Kind::Pack(transcode, pack), ..item.clone() };
							self.consider(follow, &mut seen, &mut next, &mut report);
						}
					}
					// an image without a transcode is drawn from its upload
					(Kind::Pack(Transcode::Image, 0), Outcome::NoTranscode) => {
						let upload = Item { key: item.canonical.clone(), kind: Kind::Plain, ..item.clone() };
						self.consider(upload, &mut seen, &mut next, &mut report);
					}
					_ => {}
				}
			}
			round = next;
		}
		report
	}

	/// Queues `item` unless it was met already, is on disk or was refused before.
	fn consider(&self, item: Item, seen: &mut HashSet<String>, round: &mut Vec<Item>, report: &mut WarmReport) {
		if !seen.insert(item.key.clone()) {
			return;
		}
		if self.on_disk(&item.key) || (!self.inner.config.refresh && self.inner.disk.read_refusal(&item.key, now()).is_some()) {
			report.cached += 1;
			return;
		}
		round.push(item);
	}

	/// Whether a fetch would read `key` from disk (in refresh mode: only once re-downloaded).
	fn on_disk(&self, key: &str) -> bool {
		let use_disk = !self.inner.config.refresh || self.inner.refreshed.lock().contains(key);
		use_disk && self.inner.disk.contains(key)
	}

	/// Locates and downloads `items` on as many threads as downloads may run at once: each thread
	/// downloads a located file while there is one waiting, else locates the next chunk with one batch
	/// request (a chunk's locations are used within seconds), so a chunk's files download side by side.
	fn fetch_items(&self, items: &[Item], packs: &(dyn Fn(&[u8]) -> u32 + Sync), report: &mut WarmReport) -> Vec<Outcome> {
		let workers = self.inner.config.max_concurrent_downloads.clamp(1, 64);
		let chunk = items.len().div_ceil(workers).clamp(8, BATCH);
		let chunks: Vec<&[Item]> = items.chunks(chunk).collect();
		let outcomes: Vec<Mutex<Option<Outcome>>> = items.iter().map(|_| Mutex::new(None)).collect();
		let queue = Mutex::new(Queue { next_chunk: 0, locating: 0, located: VecDeque::new() });
		let wake = Condvar::new();
		let totals = Mutex::new(WarmReport::default());
		std::thread::scope(|scope| {
			for _ in 0..workers.min(items.len()) {
				scope.spawn(|| {
					loop {
						let job = {
							let mut queue = queue.lock();
							loop {
								if let Some(located) = queue.located.pop_front() {
									break Some(Job::Download(located));
								}
								if queue.next_chunk < chunks.len() {
									queue.next_chunk += 1;
									queue.locating += 1;
									break Some(Job::Locate(queue.next_chunk - 1));
								}
								if queue.locating == 0 {
									break None;
								}
								wake.wait(&mut queue);
							}
						};
						match job {
							Some(Job::Locate(index)) => {
								let places = self.locate(chunks[index]);
								totals.lock().batches += 1;
								let mut queue = queue.lock();
								queue.locating -= 1;
								queue.located.extend(places.into_iter().enumerate().map(|(offset, place)| (index * chunk + offset, place)));
								wake.notify_all();
							}
							Some(Job::Download((index, place))) => {
								let outcome = self.download_item(&items[index], place, packs, &totals);
								*outcomes[index].lock() = Some(outcome);
							}
							None => break,
						}
					}
				});
			}
		});
		let totals = totals.into_inner();
		report.downloaded += totals.downloaded;
		report.bytes += totals.bytes;
		report.failed += totals.failed;
		report.batches += totals.batches;
		outcomes.into_iter().map(|outcome| outcome.into_inner().unwrap_or(Outcome::Failed)).collect()
	}

	/// One batch request for `items`' locations (with the cookie when there is one), in their order.
	fn locate(&self, items: &[Item]) -> Vec<Place> {
		let requests: Vec<BatchRequest> = items
			.iter()
			.enumerate()
			.map(|(index, item)| {
				let representation = match item.kind {
					Kind::Plain => None,
					Kind::Pack(transcode, pack) => Some(transcode.priority_list(pack)),
				};
				BatchRequest {
					request_id: index.to_string(),
					asset_id: item.id,
					version: item.version,
					do_not_fallback_to_baseline_representation: representation.is_some(),
					content_representation_priority_list: representation,
				}
			})
			.collect();
		let unknown = || items.iter().map(|_| Place::Unknown).collect();
		let Ok(body) = serde_json::to_vec(&requests) else {
			return unknown();
		};
		let url = format!("{}/v1/assets/batch", self.inner.client.endpoints().assetdelivery);
		let Ok(reply) = self.inner.client.post_json_retrying(&url, &body, self.cookie()) else {
			return unknown();
		};
		if reply.status != 200 {
			return unknown();
		}
		let Ok(replies) = serde_json::from_slice::<Vec<BatchReply>>(&reply.body) else {
			return unknown();
		};
		let mut places: Vec<Place> = unknown();
		for (position, reply) in replies.into_iter().enumerate() {
			// by request id, else by position
			let index = reply.request_id.as_deref().and_then(|id| id.parse::<usize>().ok()).unwrap_or(position);
			let Some(item) = items.get(index) else {
				continue;
			};
			places[index] = match (reply.location.filter(|location| !location.is_empty()), item.kind) {
				(Some(location), Kind::Plain) => Place::Url(location),
				// only a reply naming the requested representation is the transcode
				(Some(location), Kind::Pack(..)) => match reply.specifier.as_ref().and_then(|s| s.format.as_deref()) {
					Some(transcode::FORMAT) => Place::Url(location),
					_ => Place::NoTranscode,
				},
				(None, _) => {
					let error = reply.errors.as_ref().and_then(|errors| errors.first());
					match error.and_then(|error| error.code).filter(|code| (100..600).contains(code)).map(|code| code as u16) {
						Some(406) => Place::Generating,
						Some(code) => Place::Refused(code, RefusalReason::of(error, code)),
						None => Place::Unknown,
					}
				}
			};
		}
		places
	}

	/// Downloads one located item into the disk cache (a refusal is remembered as a fetch would).
	fn download_item(&self, item: &Item, place: Place, packs: &(dyn Fn(&[u8]) -> u32 + Sync), totals: &Mutex<WarmReport>) -> Outcome {
		let failed = || {
			totals.lock().failed += 1;
			Outcome::Failed
		};
		let location = match place {
			Place::Url(location) => location,
			Place::NoTranscode => return Outcome::NoTranscode,
			Place::Generating => {
				self.remember_refusal(&item.key, 406, None, now());
				return Outcome::NoTranscode;
			}
			Place::Refused(status, reason) => {
				// remembered only when final: an authorisation refusal once the cookie was tried
				if !matches!(status, 401 | 403 | 409) || self.cookie().is_some() {
					self.remember_refusal(&item.key, status, reason, now());
				}
				return failed();
			}
			Place::Unknown => return failed(),
		};
		let Ok(reply) = self.get(&location, None, "*/*", &item.uri) else {
			return failed();
		};
		if reply.status != 200 {
			return failed();
		}
		let Ok(bytes) = gunzip_if_needed(reply.body, &item.uri) else {
			return failed();
		};
		let kind = sniff_kind(&bytes);
		if matches!(item.kind, Kind::Pack(..)) && kind != transcode::FORMAT {
			return Outcome::NoTranscode;
		}
		let meta = Meta { uri: item.key.clone(), fetched_at: now(), source_host: reply.host, size: bytes.len(), kind: kind.to_string() };
		if self.inner.disk.write(&item.key, &bytes, &meta).is_err() {
			return failed();
		}
		if self.inner.config.refresh {
			self.inner.refreshed.lock().insert(item.key.clone());
		}
		let mut totals = totals.lock();
		totals.downloaded += 1;
		totals.bytes += bytes.len() as u64;
		drop(totals);
		Outcome::Downloaded { packs: if matches!(item.kind, Kind::Pack(_, 0)) { packs(&bytes) } else { 0 } }
	}
}
