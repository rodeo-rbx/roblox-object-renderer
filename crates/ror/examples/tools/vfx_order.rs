// The draw order of a sweep batch's emitters at a moment (CPU only: the batch's model, playback,
// camera and Studio's frame times, simulated, nothing drawn): each emitter's distance (its position
// moved by ZOffset), largest Size, how far its particles moved since they spawned along the view, on
// average, and where it falls in ror's order and in the order without that move:
//   cargo run -p ror --release --example tools -- vfx_order <batch dir> <moment label, e.g. t0.1> [path filter]
#[cfg(windows)]
pub fn main() -> anyhow::Result<()> {
	use anyhow::Context;
	use ror::render::camera::CameraSpec;
	use ror::studio::calibrate::timeline::{Captured, Timeline, scene};

	let mut args = std::env::args().skip(1);
	let batch = std::path::PathBuf::from(args.next().context("batch dir")?);
	let label = args.next().context("moment label")?;
	let filter = args.next().unwrap_or_default();
	let bytes = std::fs::read(batch.join("model.rbxm"))?;
	let playback = ror::studio::calibrate::timeline::stored_playback(&batch.join("playback.json"))?;
	let record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(batch.join("camera.json"))?)?;
	let spec: CameraSpec = serde_json::from_value(record["camera"].clone())?;
	let camera = spec.place(None, 1644, 1116);
	let captured: Captured = serde_json::from_str(&std::fs::read_to_string(batch.join(&label).join("studio.json"))?)?;
	let scene = scene(&bytes, &Timeline { playback, duration: 0.0 }, &captured)?;
	let forward = -glam::Vec3::from(camera.world.z_axis);
	struct Row {
		path: String,
		distance: f32,
		largest: f32,
		offset: f32,
		count: usize,
		emission: f32,
		brightness: f32,
	}
	let mut rows = Vec::new();
	for emitter in &scene.particles {
		if !emitter.path.contains(&filter) {
			continue;
		}
		let particles: Vec<_> = ror::render::particles::simulate(emitter).into_iter().filter(|particle| particle.opacity > 0.01 && particle.size > 0.0).collect();
		if particles.is_empty() {
			continue;
		}
		let position = glam::Vec3::from(emitter.cframe.translation);
		rows.push(Row {
			path: emitter.path.clone(),
			distance: position.distance(camera.eye) - emitter.z_offset,
			largest: emitter.largest_size(),
			offset: particles.iter().map(|particle| particle.moved.dot(forward)).sum::<f32>() / particles.len() as f32,
			count: particles.len(),
			emission: emitter.light_emission,
			brightness: emitter.brightness,
		});
	}
	// back to front: farther first; at one distance the one whose particles moved further away last
	// (vfx-moving-ties), then the larger first (ror's rule)
	let mut ror_order: Vec<usize> = (0..rows.len()).collect();
	ror_order.sort_by(|&a, &b| rows[b].distance.total_cmp(&rows[a].distance).then(rows[a].offset.total_cmp(&rows[b].offset)).then(rows[b].largest.total_cmp(&rows[a].largest)));
	// the same without the move
	let mut mirrored: Vec<usize> = (0..rows.len()).collect();
	mirrored.sort_by(|&a, &b| rows[b].distance.total_cmp(&rows[a].distance).then(rows[b].largest.total_cmp(&rows[a].largest)));
	println!("{:>3} {:>3}  {:60} {:>8} {:>7} {:>7} {:>5} {:>5} {:>5}", "ror", "old", "emitter", "distance", "largest", "moved", "count", "LE", "B");
	for (place, &index) in ror_order.iter().enumerate() {
		let row = &rows[index];
		let mirrored_place = mirrored.iter().position(|&other| other == index).unwrap_or_default();
		println!("{place:3} {mirrored_place:3}  {:60} {:8.3} {:7.3} {:7.3} {:5} {:5.1} {:5.1}", row.path.chars().rev().take(60).collect::<String>().chars().rev().collect::<String>(), row.distance, row.largest, row.offset, row.count, row.emission, row.brightness);
	}
	Ok(())
}

#[cfg(not(windows))]
pub fn main() {}
