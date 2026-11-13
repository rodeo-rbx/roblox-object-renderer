// Runs a release binary the way a user gets it: outside any checkout, with the per-user cache folders
// redirected to a temporary folder. Checks `--version`, a render without and with the service, and that
// nothing was written outside the redirected cache and the folders the run was given, in the workspace
// or in the folder the binary was extracted to.
//
//   node smoke.mjs --binary <extracted ror> --fixture <shapes.rbxmx> --version <version> [--workspace <folder>]
//
// `--workspace` defaults to GITHUB_WORKSPACE, else this checkout.
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { setTimeout } from "node:timers/promises";
import { ROOT, main } from "./common.mjs";

const PNG_SIGNATURE = "89504e470d0a1a0a";
const RENDER_SIZE = 128;

// Folders that a graphics driver may fill in the redirected home, named one by one: a parent such as
// `.cache` or `NVIDIA` is not allowed as a whole, or a stray write next to a driver cache would pass.
// ror's own cache is allowed too.
const DRIVER_CACHE_FOLDERS = [
	// Mesa
	"mesa_shader_cache",
	"mesa_shader_cache_db",
	"mesa_shader_cache_sf",
	".cache/mesa_shader_cache",
	".cache/mesa_shader_cache_db",
	".cache/mesa_shader_cache_sf",
	// NVIDIA
	"nvidia/GLCache",
	".cache/nvidia/GLCache",
	"NVIDIA/DXCache",
	"NVIDIA/GLCache",
	// AMD
	"AMD/DxCache",
	"AMD/GLCache",
	"AMD/VkCache",
	// Direct3D shader cache and the Windows profile locations of the same caches
	"D3DSCache",
	"AppData/Local/D3DSCache",
	"AppData/Local/NVIDIA/DXCache",
	"AppData/Local/NVIDIA/GLCache",
	"AppData/LocalLow/NVIDIA/DXCache",
];

/** Whether an environment variable is a ror, rur, ree or Roblox setting, which the run must not inherit. */
function isToolSetting(name) {
	return /^(ROR_|RUR_|REE_|ROBLOX_|ROBLOSECURITY)/.test(name) || name === ".ROBLOSECURITY";
}

/** The run's environment: the real one without ror or Roblox settings, with every user folder redirected. */
function smokeEnvironment({ root, home, tmp }) {
	const env = { ...process.env };
	for (const name of Object.keys(env)) {
		if (isToolSetting(name)) delete env[name];
	}
	return Object.assign(env, {
		LOCALAPPDATA: home,
		APPDATA: path.join(home, "AppData/Roaming"),
		HOME: home,
		USERPROFILE: home,
		XDG_CACHE_HOME: home,
		TMP: tmp,
		TEMP: tmp,
		TMPDIR: tmp,
		ROR_SERVICE_IDLE: "2", // the service stops 2 s after its last job
		ROR_ROBLOX_CONTENT: path.join(root, "no-content"),
	});
}

/** Every path under `base` with its type, size, mode and times, except `ignored`. Access times are not writes. */
export function snapshot(base, { ignored = [] } = {}) {
	const result = {};
	function visit(dir) {
		for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
			const file = path.join(dir, entry.name);
			if (ignored.includes(file)) continue;
			// a link could lead the walk out of the folders made for this run
			assert.ok(!entry.isSymbolicLink(), `unexpected link: ${file}`);
			// only metadata is read: a graphics driver's cache file can be locked or still changing
			const stat = fs.lstatSync(file);
			result[path.relative(base, file)] = {
				directory: entry.isDirectory(),
				size: stat.size,
				mode: stat.mode,
				mtime: stat.mtimeMs,
				ctime: stat.ctimeMs,
			};
			if (entry.isDirectory()) visit(file);
		}
	}
	visit(base);
	return result;
}

/** Snapshots these folders now and returns a check that throws if any of them has changed since. */
export function watchFolders(folders) {
	const snapshots = folders.map((folder) => ({ folder, before: snapshot(folder) }));
	return function checkUnchanged() {
		for (const { folder, before } of snapshots) {
			assert.deepEqual(snapshot(folder), before, `files outside the allowed folders were modified: ${folder}`);
		}
	};
}

/** Adds the PIDs of the services recorded in `<state>/services/` to `pids`. */
function rememberServices(state, pids) {
	const folder = path.join(state, "services");
	if (!fs.existsSync(folder)) return;
	for (const name of fs.readdirSync(folder).filter((name) => name.endsWith(".json"))) {
		try {
			const { pid } = JSON.parse(fs.readFileSync(path.join(folder, name), "utf8"));
			assert.ok(Number.isInteger(pid) && pid > 0, "invalid service PID");
			pids.add(pid);
		} catch (error) {
			// a service removes its record just before it exits
			if (error.code !== "ENOENT") throw error;
		}
	}
}

/** Whether a process with this PID exists. */
function isRunning(pid) {
	try {
		process.kill(pid, 0); // signal 0 only checks that the process exists
		return true;
	} catch (error) {
		if (error.code === "ESRCH") return false;
		if (error.code === "EPERM") return true; // it exists but belongs to someone else
		throw error;
	}
}

/**
 * Asks the services to stop, then waits until their processes are gone. `service stop` only sends a
 * request, and a service deletes its record before it exits, so the PIDs are noted while records exist
 * and kept in `pids`.
 */
export async function stopServices(state, run, { pids = new Set(), timeout = 30_000 } = {}) {
	rememberServices(state, pids);
	run(["service", "stop"]);
	const deadline = Date.now() + timeout;
	while (true) {
		rememberServices(state, pids);
		const running = [...pids].filter(isRunning);
		if (running.length === 0) return;
		assert.ok(Date.now() < deadline, `service processes did not exit: ${running.join(", ")}`);
		await setTimeout(50);
	}
}

/** Checks that a file is a PNG of the size the render asked for. */
function checkPng(file) {
	const png = fs.readFileSync(file);
	assert.equal(png.subarray(0, 8).toString("hex"), PNG_SIGNATURE);
	assert.equal(png.readUInt32BE(16), RENDER_SIZE, "PNG width");
	assert.equal(png.readUInt32BE(20), RENDER_SIZE, "PNG height");
}

/** Only ror's cache folder and the known graphics-driver cache folders may appear in the redirected home. */
export function checkHome(home, cache) {
	// Windows file names do not differ by case
	const comparable = (file) => (process.platform === "win32" ? file.toLowerCase() : file);
	const allowedFolders = [cache, ...DRIVER_CACHE_FOLDERS.map((folder) => path.join(home, folder))].map(comparable);

	for (const [relative, metadata] of Object.entries(snapshot(home))) {
		const file = comparable(path.join(home, relative));
		const isInside = (folder) => file.startsWith(folder + path.sep);
		// the folders on the way to an allowed folder have to exist
		const isOnTheWayTo = (folder) => metadata.directory && (file === folder || folder.startsWith(file + path.sep));
		assert.ok(allowedFolders.some((folder) => isInside(folder) || isOnTheWayTo(folder)), `unexpected user-home write: ${relative}`);
	}
}

/** The temporary folders of a run, each made for it, and the copies of the binary and the fixture in them. */
function createFolders(options) {
	const root = fs.mkdtempSync(path.join(os.tmpdir(), "ror-release-smoke-"));
	const folders = { root };
	for (const name of ["working", "home", "tmp", "output", "bin", "input"]) {
		folders[name] = path.join(root, name);
		fs.mkdirSync(folders[name]);
	}
	// The binary runs from a copy in `bin/`; both that copy and the original extraction are watched.
	folders.binary = path.join(folders.bin, path.basename(options.binary));
	folders.fixture = path.join(folders.input, path.basename(options.fixture));
	fs.copyFileSync(path.resolve(options.binary), folders.binary);
	fs.copyFileSync(path.resolve(options.fixture), folders.fixture);
	return folders;
}

/** A function that runs the binary with `argv` in the working folder and returns what it printed. */
function binaryRunner({ binary, cwd, env }) {
	return function run(argv) {
		const child = spawnSync(binary, argv, { cwd, env, encoding: "utf8", timeout: 240_000, windowsHide: true });
		if (child.error || child.status !== 0) throw new Error(`ror ${argv.join(" ")}: ${child.error ?? child.stderr}`);
		return child.stdout;
	};
}

/** Renders the fixture in this process (`direct`) or through the render service, and checks the PNG. */
function renderAndCheck({ mode, run, fixture, destination, state, pids }) {
	const argv = ["render", fixture, "--views", "iso", "--size", `${RENDER_SIZE}x${RENDER_SIZE}`, "--offline", "--out", destination];
	run(mode === "direct" ? [...argv, "--no-service"] : argv);
	// the service is running now: note its PID before its record goes
	rememberServices(state, pids);
	const pngs = fs.readdirSync(destination).filter((name) => name.endsWith(".png"));
	assert.ok(pngs.length > 0, "render produced no PNG");
	checkPng(path.join(destination, pngs[0]));
}

async function smoke(options) {
	const folders = createFolders(options);
	const { root, home, tmp, output: out } = folders;
	const run = binaryRunner({ binary: folders.binary, cwd: folders.working, env: smokeEnvironment(folders) });

	// where a standalone ror keeps its files, given the redirected home
	const cache = process.platform === "darwin" ? path.join(home, "Library/Caches/ror") : path.join(home, "ror");
	const state = path.join(cache, "state");
	const pids = new Set();

	// A `.env` in the working folder and beside the binary: a checkout would load one, a standalone binary must not.
	for (const folder of [folders.working, path.dirname(folders.binary)]) {
		fs.writeFileSync(path.join(folder, ".env"), "ROR_STATE_DIR=unexpected\n");
	}

	// This run's own folders, except the ones the binary may write to.
	const ignored = [home, tmp, out];
	const before = snapshot(root, { ignored });

	// Folders outside the run that must stay as they are: the workspace, where a build path baked into
	// the binary could put caches, and the extraction the binary was copied from.
	const workspace = path.resolve(options.workspace ?? process.env.GITHUB_WORKSPACE ?? ROOT);
	const extraction = path.dirname(path.resolve(options.binary));
	const checkOutside = watchFolders([workspace, extraction]);

	try {
		assert.equal(run(["--version"]).trim(), `ror ${options.version}`);
		assert.ok(run(["service", "status"]).includes(state), "service status does not name the standalone state dir");

		// the same render twice: in this process, then through the render service
		for (const mode of ["direct", "service"]) {
			renderAndCheck({ mode, run, fixture: folders.fixture, destination: path.join(out, mode), state, pids });
		}
		assert.ok(fs.existsSync(path.join(state, "services")), "service did not use standalone state");
		assert.ok(run(["clean", "--dry-run"]).includes(path.join(state, "out")), "clean does not name the standalone out folder");
	} finally {
		await stopServices(state, run, { pids });
	}

	assert.deepEqual(snapshot(root, { ignored }), before, "files outside the allowed folders were modified");
	checkOutside();
	checkHome(home, cache);

	console.log(`standalone version, direct render, service render and write checks passed: ${root}`);
}

main(import.meta.url, smoke);
