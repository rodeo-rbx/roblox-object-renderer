// Lists the emitters of a sweep batch (out/calibrate/vfx-sweep-*/<batch>) whose particles at a moment
// show larger than the view or reach the camera (CPU only: the batch's model, playback, camera and
// Studio's frame times, simulated, nothing drawn):
//   cargo run -p ror --release --example tools -- vfx_big <batch dir> <moment label, e.g. t0.1> [share of the view]
#[cfg(windows)]
pub fn main() -> anyhow::Result<()> {
	use anyhow::Context;
	use ror::render::camera::CameraSpec;
	use ror::studio::calibrate::timeline::{Captured, Timeline, scene};

	let mut args = std::env::args().skip(1);
	let batch = std::path::PathBuf::from(args.next().context("batch dir")?);
	let label = args.next().context("moment label")?;
	let share: f32 = args.next().and_then(|value| value.parse().ok()).unwrap_or(0.5);
	let bytes = std::fs::read(batch.join("model.rbxm"))?;
	let playback = ror::studio::calibrate::timeline::stored_playback(&batch.join("playback.json"))?;
	let record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(batch.join("camera.json"))?)?;
	let spec: CameraSpec = serde_json::from_value(record["camera"].clone())?;
	let camera = spec.place(None, 1644, 1116);
	let captured: Captured = serde_json::from_str(&std::fs::read_to_string(batch.join(&label).join("studio.json"))?)?;
	let scene = scene(&bytes, &Timeline { playback, duration: 0.0 }, &captured)?;
	let tan = (camera.fov.to_radians() * 0.5).tan();
	let forward = -glam::Vec3::from(camera.world.z_axis);
	for emitter in &scene.particles {
		let particles = ror::render::particles::simulate(emitter);
		let mut largest = 0.0f32;
		let mut nearest = f32::INFINITY;
		let mut count = 0;
		for particle in &particles {
			if particle.opacity <= 0.01 || particle.size <= 0.0 {
				continue;
			}
			let depth = (particle.position - camera.eye).dot(forward);
			nearest = nearest.min(depth - particle.size * (1.0 + particle.squash.abs()));
			// the quad's height over the view's height at its depth
			let shown = particle.size * 2.0 * (1.0 + particle.squash.abs()) / (2.0 * tan * depth.max(1e-3));
			if shown > share || depth < particle.size {
				count += 1;
			}
			largest = largest.max(if depth > 0.0 { shown } else { f32::INFINITY });
		}
		if std::env::var_os("VFX_BIG_ALL").is_some() {
			for particle in &particles {
				println!("  {}: pos {:.3?} vel {:.4?} size {:.2} squash {:.2} rot {:.0} opacity {:.3} frames {:?} age {:.3}", emitter.path, particle.position, particle.velocity, particle.size, particle.squash, particle.rotation, particle.opacity, particle.frames, particle.age);
			}
		}
		if std::env::var_os("VFX_BIG_QUADS").is_some() {
			let mut vertices = Vec::new();
			ror::render::particles::quads(emitter, &particles, &camera, 1.0, None, &mut vertices);
			for particle in &particles {
				let velocity = particle.velocity.normalize_or_zero();
				let up = (glam::Vec3::Y - velocity * velocity.y).normalize_or(velocity.any_orthonormal_vector());
				println!("    velocity {:?} (bits {:x} {:x} {:x}) up {:?} right {:?}", velocity, velocity.x.to_bits(), velocity.y.to_bits(), velocity.z.to_bits(), up, velocity.cross(up));
			}
			let view_projection = camera.view_projection(1644.0 / 1116.0);
			for vertex in vertices.iter().take(6) {
				let clip = view_projection * glam::Vec3::from(vertex.position).extend(1.0);
				println!("    corner {:.3?} -> screen ({:.0}, {:.0}) w {:.2} uv {:.3?} colour {:.3?}", vertex.position, (clip.x / clip.w * 0.5 + 0.5) * 1644.0, (0.5 - clip.y / clip.w * 0.5) * 1116.0, clip.w, vertex.uv, vertex.color);
			}
		}
		if count > 0 {
			println!("{}: {count} of {} particles over {share} of the view or reaching the camera; largest {largest:.2} views, nearest edge {nearest:.1} studs ahead (Size {:?}, LightEmission {}, Brightness {}, ZOffset {})", emitter.path, particles.len(), emitter.size_keys.iter().map(|key| key.1).fold(0.0f32, f32::max), emitter.light_emission, emitter.brightness, emitter.z_offset);
		}
	}
	Ok(())
}

#[cfg(not(windows))]
pub fn main() {}
