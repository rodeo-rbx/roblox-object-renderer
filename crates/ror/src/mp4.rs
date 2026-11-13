//! An MP4 of one H.264 track, written as its frames come (video.rs's in-process encoder, [`crate::x264`]):
//! the boxes ffmpeg's muxer writes for `-movflags +faststart` (the index before the frames, so a player
//! starts before the whole file has arrived), laid out without its second pass: the index's room is kept
//! ahead of the frames when the file starts (the clip's frame count is known then; what is left over
//! becomes a `free` box) and it is written there at the end.
//!
//! Timing as ffmpeg's: a track time scale of the frame rate's numerator doubled until it is 10,000 or
//! more (30 fps: 15,360, each frame 512 ticks), the movie's in milliseconds, one edit from 0. The
//! colour as the stream's VUI says it (an `nclx` `colr`: BT.709 primaries and matrix, the sRGB transfer,
//! limited range). Frames come in decode order: x264's preset has no B-frames, so that is the display
//! order; an encoder with B-frames ([`crate::nvenc`]) names each frame's place in the display order, kept
//! as composition offsets (`ctts`) with the edit starting at the first frame shown.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};

/// The writer of one MP4.
pub struct Mp4Writer {
	file: BufWriter<File>,
	width: u32,
	height: u32,
	/// Ticks a second of the track, and of each frame
	timescale: u32,
	delta: u32,
	sps: Vec<u8>,
	pps: Vec<u8>,
	/// Every frame's size and offset in the file, and the keyframes' numbers (from 1)
	sizes: Vec<u32>,
	offsets: Vec<u64>,
	keyframes: Vec<u32>,
	/// Each frame's place in the display order, where any is out of it (B-frames)
	order: Vec<u32>,
	reordered: bool,
	/// The encoder's name in the sample entry
	label: &'static str,
	frames: usize,
	/// Where the index's room starts and how long it is; where the frames' box starts
	room_at: u64,
	room: u64,
	mdat_at: u64,
	written: u64,
}

/// A box: its size and type, then `body`.
fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
	let mut out = Vec::with_capacity(body.len() + 8);
	out.extend_from_slice(&(body.len() as u32 + 8).to_be_bytes());
	out.extend_from_slice(kind);
	out.extend_from_slice(body);
	out
}

/// A full box: version and flags before `body`.
fn full(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
	let mut inner = Vec::with_capacity(body.len() + 4);
	inner.extend_from_slice(&((version as u32) << 24 | flags).to_be_bytes());
	inner.extend_from_slice(body);
	boxed(kind, &inner)
}

/// The unity matrix of mvhd and tkhd.
const MATRIX: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

fn be32(values: &[u32]) -> Vec<u8> {
	values.iter().flat_map(|value| value.to_be_bytes()).collect()
}

/// `fps` as ffmpeg's encoder takes `-framerate` (to three places): a fraction in lowest terms.
pub fn frame_rate(fps: f32) -> (u32, u32) {
	let thousandths = ((fps as f64 * 1000.0).round() as u64).max(1);
	let (mut a, mut b) = (thousandths, 1000u64);
	while b != 0 {
		(a, b) = (b, a % b);
	}
	((thousandths / a) as u32, (1000 / a) as u32)
}

impl Mp4Writer {
	/// Starts `path` for `frames` frames of `width` x `height` at `fps` (a fraction), the stream's SPS
	/// and PPS as given (no lengths), from the encoder `label` names; `reordered`: its frames may come
	/// out of display order ([`Mp4Writer::write_at`]).
	#[allow(clippy::too_many_arguments)]
	pub fn create(path: &Path, width: u32, height: u32, fps: (u32, u32), sps: &[u8], pps: &[u8], frames: usize, label: &'static str, reordered: bool) -> Result<Self> {
		if sps.len() < 4 || pps.is_empty() {
			bail!("mp4: no SPS or PPS");
		}
		let (numerator, denominator) = fps;
		let mut timescale = numerator.max(1);
		while timescale < 10_000 {
			timescale *= 2;
		}
		let delta = denominator * (timescale / numerator.max(1));
		let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
		let mut writer = Mp4Writer {
			file: BufWriter::with_capacity(1 << 20, file),
			width,
			height,
			timescale,
			delta,
			sps: sps.to_vec(),
			pps: pps.to_vec(),
			sizes: Vec::with_capacity(frames),
			offsets: Vec::with_capacity(frames),
			keyframes: Vec::new(),
			order: Vec::new(),
			reordered,
			label,
			frames,
			room_at: 0,
			room: 0,
			mdat_at: 0,
			written: 0,
		};
		let ftyp = boxed(b"ftyp", b"isom\0\0\x02\0isomiso2avc1mp41");
		writer.file.write_all(&ftyp)?;
		// the index of `frames` frames at the most: 64-bit offsets, a sync sample table (none when every
		// frame is a keyframe) of all but one; and the room for a free box's header after it
		writer.room_at = ftyp.len() as u64;
		let stss = 16 + 4 * frames as u64;
		let ctts = if reordered { 16 + 8 * frames as u64 } else { 0 };
		writer.room = writer.moov(&vec![0; frames], &vec![u64::MAX; frames], &(1..=frames as u32).collect::<Vec<_>>(), frames).len() as u64 + stss + ctts + 8;
		writer.file.write_all(&free(writer.room))?;
		writer.mdat_at = writer.room_at + writer.room;
		// the frames' box, 64-bit sized: its size is written at the end
		writer.file.write_all(&[0, 0, 0, 1, b'm', b'd', b'a', b't', 0, 0, 0, 0, 0, 0, 0, 0])?;
		writer.written = writer.mdat_at + 16;
		Ok(writer)
	}

	/// Adds the next frame: its NAL units with 4-byte lengths, in display order.
	pub fn write(&mut self, data: &[u8], keyframe: bool) -> Result<()> {
		self.write_at(data, keyframe, self.sizes.len())
	}

	/// Adds the next frame in decode order, shown as the clip's frame `display` (from 0).
	pub fn write_at(&mut self, data: &[u8], keyframe: bool, display: usize) -> Result<()> {
		if display != self.sizes.len() && !self.reordered {
			bail!("mp4: frame {display} came as number {}: frames out of order", self.sizes.len());
		}
		if display >= self.frames {
			bail!("mp4: a frame shown at {display} of {} frames", self.frames);
		}
		self.order.push(display as u32);
		if self.sizes.len() >= self.frames {
			bail!("mp4: more than the {} frames it was started for", self.frames);
		}
		self.file.write_all(data)?;
		self.offsets.push(self.written);
		self.sizes.push(data.len() as u32);
		if keyframe {
			self.keyframes.push(self.sizes.len() as u32);
		}
		self.written += data.len() as u64;
		Ok(())
	}

	/// Writes the index into its room and the frames' size; the file is complete.
	pub fn finish(mut self) -> Result<()> {
		if self.sizes.len() != self.frames {
			bail!("mp4: {} frames of the {} it was started for", self.sizes.len(), self.frames);
		}
		let moov = self.moov(&self.sizes, &self.offsets, &self.keyframes, self.frames);
		if moov.len() as u64 + 8 > self.room {
			bail!("mp4: an index of {} bytes in room for {}", moov.len(), self.room);
		}
		let mdat = self.written - self.mdat_at;
		self.file.flush()?;
		let file = self.file.get_mut();
		file.seek(SeekFrom::Start(self.room_at))?;
		file.write_all(&moov)?;
		file.write_all(&free(self.room - moov.len() as u64))?;
		file.seek(SeekFrom::Start(self.mdat_at + 8))?;
		file.write_all(&mdat.to_be_bytes())?;
		file.flush()?;
		Ok(())
	}

	/// The index (`moov`) of frames of these sizes at these offsets with these keyframes.
	fn moov(&self, sizes: &[u32], offsets: &[u64], keyframes: &[u32], frames: usize) -> Vec<u8> {
		let media_duration = frames as u64 * self.delta as u64;
		// the movie in milliseconds, rounded up (as ffmpeg's av_rescale_rnd rounds it up)
		let duration = (media_duration * 1000).div_ceil(self.timescale as u64) as u32;
		let mut mvhd = be32(&[0, 0, 1000, duration, 0x0001_0000]);
		mvhd.extend_from_slice(&[0x01, 0x00, 0, 0]);
		mvhd.extend_from_slice(&[0; 8]);
		mvhd.extend(be32(&MATRIX));
		mvhd.extend_from_slice(&[0; 24]);
		mvhd.extend(be32(&[2]));
		let mvhd = full(b"mvhd", 0, 0, &mvhd);

		let mut tkhd = be32(&[0, 0, 1, 0, duration, 0, 0]);
		tkhd.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
		tkhd.extend(be32(&MATRIX));
		tkhd.extend(be32(&[self.width << 16, self.height << 16]));
		let tkhd = full(b"tkhd", 0, 3, &tkhd);
		// frames out of display order: each decoded `delay` frames before the earliest it could be shown,
		// its offset from there to its place, and the edit starting where the first frame shows
		let delay = self.order.iter().enumerate().map(|(index, &display)| (index as u32).saturating_sub(display)).max().unwrap_or(0);
		let ctts = if delay == 0 {
			Vec::new()
		} else {
			let offsets: Vec<u32> = self.order.iter().enumerate().flat_map(|(index, &display)| [1, (display + delay - index as u32) * self.delta]).collect();
			full(b"ctts", 0, 0, &[&be32(&[self.order.len() as u32])[..], &be32(&offsets)].concat())
		};
		let elst = full(b"elst", 0, 0, &be32(&[1, duration, delay * self.delta, 0x0001_0000]));
		let edts = boxed(b"edts", &elst);

		let mut mdhd = be32(&[0, 0, self.timescale, media_duration as u32]);
		// "und"
		mdhd.extend_from_slice(&[0x55, 0xc4, 0, 0]);
		let mdhd = full(b"mdhd", 0, 0, &mdhd);
		let hdlr = full(b"hdlr", 0, 0, b"\0\0\0\0vide\0\0\0\0\0\0\0\0\0\0\0\0VideoHandler\0");

		let vmhd = full(b"vmhd", 0, 1, &[0; 8]);
		let dref = full(b"dref", 0, 0, &[&be32(&[1])[..], &full(b"url ", 0, 1, &[])].concat());
		let dinf = boxed(b"dinf", &dref);

		let stsd = full(b"stsd", 0, 0, &[&be32(&[1])[..], &self.avc1()].concat());
		let stts = full(b"stts", 0, 0, &be32(&[1, frames as u32, self.delta]));
		// every frame a keyframe: no stss (all are sync samples)
		let stss = if keyframes.len() == frames { Vec::new() } else { full(b"stss", 0, 0, &[&be32(&[keyframes.len() as u32])[..], &be32(keyframes)].concat()) };
		// a chunk a frame
		let stsc = full(b"stsc", 0, 0, &be32(&[1, 1, 1, 1]));
		let stsz = full(b"stsz", 0, 0, &[&be32(&[0, frames as u32])[..], &be32(sizes)].concat());
		let wide = offsets.iter().any(|&offset| offset > u32::MAX as u64);
		let chunks = if wide {
			full(b"co64", 0, 0, &[&be32(&[frames as u32])[..], &offsets.iter().flat_map(|offset| offset.to_be_bytes()).collect::<Vec<u8>>()].concat())
		} else {
			full(b"stco", 0, 0, &[&be32(&[frames as u32])[..], &be32(&offsets.iter().map(|&offset| offset as u32).collect::<Vec<u32>>())].concat())
		};
		let stbl = boxed(b"stbl", &[stsd, stts, ctts, stss, stsc, stsz, chunks].concat());
		let minf = boxed(b"minf", &[vmhd, dinf, stbl].concat());
		let mdia = boxed(b"mdia", &[mdhd, hdlr, minf].concat());
		let trak = boxed(b"trak", &[tkhd, edts, mdia].concat());
		boxed(b"moov", &[mvhd, trak].concat())
	}

	/// The sample entry: an `avc1` with its `avcC` (the parameter sets) and `colr`.
	fn avc1(&self) -> Vec<u8> {
		let mut entry = vec![0u8; 6];
		entry.extend_from_slice(&1u16.to_be_bytes());
		entry.extend_from_slice(&[0; 16]);
		entry.extend_from_slice(&(self.width as u16).to_be_bytes());
		entry.extend_from_slice(&(self.height as u16).to_be_bytes());
		entry.extend(be32(&[0x0048_0000, 0x0048_0000, 0]));
		entry.extend_from_slice(&1u16.to_be_bytes());
		let mut name = [0u8; 32];
		let label = self.label.as_bytes();
		name[0] = label.len() as u8;
		name[1..=label.len()].copy_from_slice(label);
		entry.extend_from_slice(&name);
		entry.extend_from_slice(&[0x00, 0x18, 0xff, 0xff]);
		let mut avcc = vec![1, self.sps[1], self.sps[2], self.sps[3], 0xff, 0xe1];
		avcc.extend_from_slice(&(self.sps.len() as u16).to_be_bytes());
		avcc.extend_from_slice(&self.sps);
		avcc.push(1);
		avcc.extend_from_slice(&(self.pps.len() as u16).to_be_bytes());
		avcc.extend_from_slice(&self.pps);
		// the High profiles' extension (4:2:0, 8 bits, no SPS extensions)
		if matches!(self.sps[1], 100 | 110 | 122 | 144) {
			avcc.extend_from_slice(&[0xfc | 1, 0xf8, 0xf8, 0]);
		}
		entry.extend(boxed(b"avcC", &avcc));
		// BT.709 primaries, the sRGB transfer (13), BT.709 matrix, limited range
		entry.extend(boxed(b"colr", &[b"nclx".as_slice(), &[0, 1, 0, 13, 0, 1, 0]].concat()));
		boxed(b"avc1", &entry)
	}
}

/// A `free` box `size` bytes long (header included; at least 8).
fn free(size: u64) -> Vec<u8> {
	let mut out = vec![0u8; size as usize];
	out[..4].copy_from_slice(&(size as u32).to_be_bytes());
	out[4..8].copy_from_slice(b"free");
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn rates_are_ffmpegs_fractions() {
		assert_eq!(frame_rate(30.0), (30, 1));
		assert_eq!(frame_rate(29.97), (2997, 100));
		assert_eq!(frame_rate(12.5), (25, 2));
		assert_eq!(frame_rate(60.0), (60, 1));
	}

	#[test]
	fn an_mp4_holds_its_frames_and_index() {
		let dir = std::env::temp_dir().join(format!("ror-mp4-test-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join("clip.mp4");
		let sps = [0x67, 0x42, 0xc0, 0x1e, 0xda];
		let pps = [0x68, 0xce, 0x0f, 0xc8];
		let mut writer = Mp4Writer::create(&path, 64, 48, (30, 1), &sps, &pps, 3, "ror x264", false).unwrap();
		for (index, size) in [10usize, 20, 30].into_iter().enumerate() {
			writer.write(&vec![index as u8 + 1; size], index == 0).unwrap();
		}
		writer.finish().unwrap();
		let bytes = std::fs::read(&path).unwrap();
		// ftyp, moov, free, mdat in that order; the mdat's 64-bit size covers its frames
		let mut offset = 0usize;
		let mut kinds = Vec::new();
		while offset < bytes.len() {
			let mut size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as u64;
			let kind = String::from_utf8_lossy(&bytes[offset + 4..offset + 8]).into_owned();
			if size == 1 {
				size = u64::from_be_bytes(bytes[offset + 8..offset + 16].try_into().unwrap());
				assert_eq!(size, 16 + 60);
			}
			kinds.push(kind);
			offset += size as usize;
		}
		assert_eq!(kinds, ["ftyp", "moov", "free", "mdat"]);
		assert_eq!(offset, bytes.len());
		let find = |kind: &str| bytes.windows(4).position(|window| window == kind.as_bytes());
		for kind in ["mvhd", "tkhd", "elst", "mdhd", "hdlr", "avcC", "colr", "stts", "stss", "stsz", "stco"] {
			assert!(find(kind).is_some(), "{kind}");
		}
		// the frames where the index says
		let stco = find("stco").unwrap();
		let first = u32::from_be_bytes(bytes[stco + 12..stco + 16].try_into().unwrap()) as usize;
		assert_eq!(&bytes[first..first + 10], &[1u8; 10]);
		assert!(find("ctts").is_none(), "frames in display order have no composition offsets");
		assert!(Mp4Writer::create(&path, 64, 48, (30, 1), &sps, &pps, 1, "ror x264", false).unwrap().finish().is_err(), "a frame short");
		// decoded I P B B, shown I B B P: each frame's offset from its decode time, the edit a frame in
		let mut writer = Mp4Writer::create(&path, 64, 48, (30, 1), &sps, &pps, 4, "ror nvenc", true).unwrap();
		for (index, display) in [0usize, 3, 1, 2].into_iter().enumerate() {
			writer.write_at(&[index as u8; 8], index == 0, display).unwrap();
		}
		writer.finish().unwrap();
		let bytes = std::fs::read(&path).unwrap();
		let at = |kind: &str| bytes.windows(4).position(|window| window == kind.as_bytes()).unwrap();
		let word = |offset: usize| u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap());
		let ctts = at("ctts");
		assert_eq!(word(ctts + 8), 4);
		assert_eq!([word(ctts + 16), word(ctts + 24), word(ctts + 32), word(ctts + 40)], [512, 3 * 512, 0, 0]);
		assert_eq!(word(at("elst") + 16), 512, "the edit starts at the first frame shown");
		let mut writer = Mp4Writer::create(&path, 64, 48, (30, 1), &sps, &pps, 2, "ror x264", false).unwrap();
		assert!(writer.write_at(&[0; 8], true, 1).is_err(), "out of order where none was said");
		std::fs::remove_dir_all(&dir).unwrap();
	}
}
