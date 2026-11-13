//! Whether a loop is worth rayon's threads. A loop handed to the pool from a thread outside it wakes the
//! pool's workers and parks the caller until they have run it: some hundred microseconds of CPU across
//! them whatever its size, each worker spinning for more work before it parks again. A still pays that a
//! handful of times. A clip's threads (its producer and helpers, its draws, its encoder) paid it ten times
//! a frame for loops of a few dozen items, and half of a clip's CPU time went to workers waking and
//! parking: a 286-frame 1440p attack clip took 10.9 core-seconds with every loop on the pool and 5.4 with
//! the pool held to two threads, drawing faster (6.8 against 5.5 ms a frame). So a clip's threads keep a
//! loop to themselves unless it is long enough to be worth the hand-over; frames are made side by side by
//! the producer's helpers instead ([`crate::video`]).

use std::cell::Cell;

thread_local! {
	static CLIP: Cell<bool> = const { Cell::new(false) };
}

/// The emitters a clip's thread plays and simulates itself: an attack's 500, some 3,000 particles, took
/// 0.4 ms a frame on the pool's eight threads and no longer on the frame's own, the clip 0.55 core-seconds
/// less of its 5.2.
pub(crate) const CLIP_EMITTERS: usize = 2048;

/// Marks the calling thread as one of a clip's until dropped.
pub(crate) struct ClipThread(bool);

impl ClipThread {
	pub(crate) fn enter() -> Self {
		ClipThread(CLIP.replace(true))
	}
}

impl Drop for ClipThread {
	fn drop(&mut self) {
		CLIP.set(self.0);
	}
}

/// Whether a loop of `items` goes to rayon's threads: from a clip's thread only when it has `least` or more.
pub(crate) fn spread(items: usize, least: usize) -> bool {
	items >= least || !CLIP.get()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_clips_thread_keeps_short_loops() {
		assert!(spread(3, 64));
		{
			let _clip = ClipThread::enter();
			assert!(!spread(3, 64) && spread(64, 64));
			let _inner = ClipThread::enter();
		}
		assert!(spread(3, 64));
	}
}
