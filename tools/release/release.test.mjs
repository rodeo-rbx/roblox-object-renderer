// Tests of the release scripts: node --test tools/release/*.test.mjs
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { once } from "node:events";
import { pathToFileURL } from "node:url";
import { execFileSync, spawn, spawnSync } from "node:child_process";
import { test } from "node:test";
import { ROOT, REPOSITORY, sourceText } from "./common.mjs";
import { version } from "./version.mjs";
import { crc32, inspect, packageRelease, zip } from "./package.mjs";
import { inventory, normalPackages } from "./licenses.mjs";
import { checkHome, snapshot, stopServices, watchFolders } from "./smoke.mjs";
import { notes } from "./notes.mjs";

/** A new temporary folder, removed after the test (kept when it is a git repository, to look at a failure). */
function temporaryFolder(t) {
	const root = fs.mkdtempSync(path.join(os.tmpdir(), "ror-release-test-"));
	t.after(() => {
		if (!fs.existsSync(path.join(root, ".git"))) fs.rmSync(root, { recursive: true, force: true });
	});
	return root;
}

/** A workspace in `root` with one member, `crates/ror`. `memberVersion` is the member's version line. */
function writeWorkspace(root, memberVersion = "version.workspace = true") {
	fs.mkdirSync(path.join(root, "crates/ror"), { recursive: true });
	fs.writeFileSync(path.join(root, "Cargo.toml"), '[workspace]\nmembers = ["crates/ror"]\n[workspace.package]\nversion = "0.1.0"\n');
	fs.writeFileSync(path.join(root, "crates/ror/Cargo.toml"), `[package]\nname = "ror"\n${memberVersion}\n`);
}

test("version reads every manifest and rejects a mismatching tag or member", (t) => {
	const root = temporaryFolder(t);
	writeWorkspace(root);
	assert.equal(version(root, "v0.1.0"), "0.1.0");
	assert.throws(() => version(root, "v0.1.1"), /does not match/);
	assert.throws(() => version(root, "v0.1.0-extra"), /does not match/);

	writeWorkspace(root, 'version = "0.2.0"');
	assert.throws(() => version(root), /differs/);
});

test("repo manifest versions agree", () => {
	assert.match(version(), /^\d+\.\d+\.\d+(-rodeo\.\d+)?$/); // the fork's releases are x.y.z-rodeo.n
});

test("ZIP preserves order, content, fixed timestamps and Unix permissions", () => {
	const entries = [
		{ name: "ror", mode: 0o755, bytes: Buffer.from("binary") },
		{ name: "LICENSE", mode: 0o644, bytes: Buffer.from("licence") },
	];
	assert.equal(crc32(Buffer.from("123456789")), 0xcbf43926); // the standard CRC-32 check value

	const bytes = zip(entries);
	assert.deepEqual(zip(entries), bytes, "the same entries give the same bytes");
	const read = inspect(bytes);
	assert.equal(read[0].offset, 0);
	assert.deepEqual(read.map(({ name, mode, bytes }) => ({ name, mode: mode & 0o777, bytes })), entries);

	// a damaged checksum in the directory is caught
	const broken = Buffer.from(bytes);
	const directoryStart = bytes.readUInt32LE(bytes.length - 6);
	broken.writeUInt32LE(0, directoryStart + 16);
	assert.throws(() => inspect(broken), /checksum/);
});

test("packages contain exactly one executable first on every platform", (t) => {
	const root = temporaryFolder(t);
	writeWorkspace(root);
	fs.mkdirSync(path.join(root, "licenses"));
	for (const name of ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "licenses/GPL-3.0.txt", "THIRD_PARTY_LICENSES.txt", "binary"]) {
		fs.writeFileSync(path.join(root, name), name);
	}

	for (const platform of ["windows", "linux", "macos"]) {
		const file = packageRelease({
			binary: path.join(root, "binary"),
			version: "0.1.0",
			os: platform,
			arch: platform === "macos" ? "aarch64" : "x86_64",
			licenses: path.join(root, "THIRD_PARTY_LICENSES.txt"),
			out: path.join(root, "out"),
			commit: "a".repeat(40),
		}, root);

		const entries = inspect(fs.readFileSync(file));
		const exe = platform === "windows" ? "ror.exe" : "ror";
		assert.deepEqual(entries.map((entry) => entry.name), [exe, "LICENSE", "LICENSE-GPL-3.0.txt", "NOTICE", "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_LICENSES.txt", "SOURCE.md"]);
		assert.equal(entries.filter((entry) => entry.mode & 0o111).length, 1);
		assert.match(entries.at(-1).bytes.toString(), /a{40}/);
	}
});

/** Cargo metadata for a build graph: ror depends on `normal` (which needs `transitive`), `dev` and `build`. */
function writeMetadata(root) {
	/** A required dependency a manifest declares, for every target. `kind` is null for a normal one. */
	const declared = (name, kind) => ({ name, kind, target: null, optional: false });
	const declaredDependencies = {
		ror: [declared("normal", null), declared("dev", "dev"), declared("build", "build")],
		normal: [declared("transitive", null)],
	};
	function packageOf(id, license = "MIT") {
		fs.mkdirSync(path.join(root, id), { recursive: true });
		const source = id === "ror" ? null : "registry+https://github.com/rust-lang/crates.io-index";
		const dependencies = declaredDependencies[id] ?? [];
		return { id, name: id, version: "1.0.0", source, license, dependencies, manifest_path: path.join(root, id, "Cargo.toml") };
	}
	/** An edge of the resolve graph. */
	const dependency = (pkg, kind) => ({ name: pkg, pkg, dep_kinds: [{ kind, target: null }] });
	return {
		workspace_root: root,
		workspace_members: ["ror"],
		packages: ["ror", "normal", "transitive", "dev", "build"].map((id) => packageOf(id)),
		resolve: {
			nodes: [
				{ id: "ror", deps: [dependency("normal", null), dependency("dev", "dev"), dependency("build", "build")] },
				{ id: "normal", deps: [dependency("transitive", null)] },
				{ id: "transitive", deps: [] },
				{ id: "dev", deps: [] },
				{ id: "build", deps: [] },
			],
		},
	};
}

test("licence inventory walks normal edges only, keeps notices and fills absent texts", (t) => {
	const root = temporaryFolder(t);
	const metadata = writeMetadata(root);
	const write = (file, text) => fs.writeFileSync(path.join(root, file), text);
	write("ror/LICENSE", "MIT licence text");
	write("normal/LICENSE-MIT", "MIT licence text");
	fs.mkdirSync(path.join(root, "normal/LICENSES"));
	write("normal/LICENSES/component.txt", "component notice");
	fs.mkdirSync(path.join(root, "normal/vendor"));
	write("normal/vendor/COPYING", "native source licence");
	write("transitive/NOTICE", "a notice without licence terms"); // a notice is no licence text

	assert.deepEqual(normalPackages(metadata).map((pkg) => pkg.name), ["normal", "ror", "transitive"]);
	const text = inventory(metadata);
	assert.match(text, /component notice/);
	assert.match(text, /native source licence/);
	assert.match(text, /Standard text: MIT/); // transitive ships no licence file, so MIT's text is added
	assert.doesNotMatch(text, /## (dev|build) /);

	// a licence with no text anywhere in the graph fails
	metadata.packages.find((pkg) => pkg.name === "transitive").license = "Unknown-1.0";
	assert.throws(() => inventory(metadata), /no local standard licence text/);
});

// Needs `cargo metadata --locked --format-version 1 --filter-platform <triple>` saved to a file.
test("licence inventory covers this repo from locked target metadata", { skip: !process.env.ROR_RELEASE_METADATA }, () => {
	const metadata = JSON.parse(fs.readFileSync(process.env.ROR_RELEASE_METADATA, "utf8"));
	const text = inventory(metadata);
	assert.ok(text.includes(`## ror ${version()}\n`));
	assert.match(text, /## wgpu /);
	assert.doesNotMatch(text, /## (rcl-core|dhat|rusqlite|bcdec_rs|zlib-rs) /); // private or dev-only crates
});

test("notes use public subjects as data and bound them by the preceding release", (t) => {
	const root = temporaryFolder(t);
	writeWorkspace(root);
	const git = (...argv) => execFileSync("git", argv, { cwd: root, encoding: "utf8" }).trim();
	git("init", "--quiet");
	git("config", "user.name", "Release tests");
	git("config", "user.email", "release-test");
	git("add", ".");
	git("commit", "--quiet", "-m", "First change");

	// before any release tag: every subject
	const output = path.join(root, "notes.md");
	assert.match(notes({ output }, root), /First change/);

	// after v0.0.1: only what came later, and shell syntax in a subject stays text
	git("tag", "v0.0.1");
	git("commit", "--quiet", "--allow-empty", "-m", 'A subject with $(echo bad) and `code`');
	git("tag", "v0.1.0");
	const text = notes({ output, tag: "v0.1.0" }, root);
	assert.doesNotMatch(text, /First change/);
	assert.ok(text.includes('$(echo bad) and `code`'));
	assert.match(text, /gh attestation verify/);
	assert.match(text, /Source archive:/);
});

test("source notice records the exact tag, commit, pinned crates and rebuild command", () => {
	const text = sourceText({ tag: "v0.1.0", commit: "a".repeat(40), repository: REPOSITORY });
	assert.match(text, /archive\/refs\/tags\/v0\.1\.0.zip/);
	assert.match(text, /SHA-256/);
	assert.match(text, /Rust 1\.90\.0/);
	assert.match(text, /--locked --profile dist --bin ror/);
});

const workflow = fs.readFileSync(path.join(ROOT, ".github/workflows/release.yml"), "utf8");

/** The text of every `run:` step, one-line or block. */
function runScripts(text) {
	const scripts = [];
	const lines = text.split("\n");
	for (let i = 0; i < lines.length; i++) {
		const start = lines[i].match(/^(\s*)run:(.*)$/);
		if (!start) continue;
		const keyIndent = start[1].length;
		const script = [start[2]];
		// a block script goes on while the lines are blank or indented deeper than the `run:` key
		while (i + 1 < lines.length && (lines[i + 1].trim() === "" || lines[i + 1].search(/\S/) > keyIndent)) {
			script.push(lines[++i]);
		}
		scripts.push(script.join("\n"));
	}
	return scripts;
}

test("workflow pins every action to a full commit SHA and uses no build cache", () => {
	for (const match of workflow.matchAll(/uses: ([^\s]+)/g)) assert.match(match[1], /@[a-f0-9]{40}$/);
	assert.doesNotMatch(workflow, /rust-cache|save-cache|restore-cache/);
});

test("workflow keeps the publish job free of repository code", () => {
	const publish = workflow.slice(workflow.indexOf("  publish:"));
	assert.doesNotMatch(publish, /uses: actions\/checkout|run:.*(?:cargo|node)|--target/);
	assert.match(publish, /--verify-tag/);
});

test("workflow run scripts read contexts only through env", () => {
	const scripts = runScripts(workflow);
	assert.ok(scripts.length >= 8, "the run steps were not found");
	for (const script of scripts) assert.doesNotMatch(script, /\$\{\{/);
});

test("smoke snapshots inspect driver caches without opening their contents", (t) => {
	const root = temporaryFolder(t);
	const cache = path.join(root, "NVIDIA/DXCache");
	fs.mkdirSync(cache, { recursive: true });
	const file = path.join(cache, "driver.nvph");
	fs.writeFileSync(file, "driver cache");
	t.mock.method(fs, "readFileSync", () => { throw new Error("driver cache is locked"); });
	assert.ok(Object.hasOwn(snapshot(root), path.relative(root, file)));
});

test("smoke detects writes in the workspace and original extraction outside its temporary root", (t) => {
	const root = temporaryFolder(t);
	const workspace = path.join(root, "workspace");
	const extraction = path.join(root, "extraction");
	for (const folder of [workspace, extraction]) {
		fs.mkdirSync(folder);
		fs.writeFileSync(path.join(folder, "existing"), "original");
	}
	const watched = [workspace, extraction];

	/** Starts watching, makes `change`, and expects the check to report it. */
	function assertDetected(change) {
		const check = watchFolders(watched);
		change();
		assert.throws(check, /files outside the allowed folders were modified/);
	}

	// nothing changed, so the check passes
	watchFolders(watched)();

	for (const folder of watched) {
		const existing = path.join(folder, "existing");
		assertDetected(() => fs.mkdirSync(path.join(folder, ".cache")));
		assertDetected(() => fs.utimesSync(existing, 1, 1)); // same bytes and size, only the metadata changes
		assertDetected(() => fs.unlinkSync(existing));
	}
});

test("licence inventory ignores weak optional references and follows active renamed edges", (t) => {
	const metadata = writeMetadata(temporaryFolder(t));
	const root = metadata.packages.find((pkg) => pkg.id === "ror");
	const rootNode = metadata.resolve.nodes.find((node) => node.id === "ror");
	const packageNames = () => normalPackages(metadata).map((pkg) => pkg.name);

	// ror depends on `normal`, and on `transitive` only as the optional dependency `codec-alias`
	root.dependencies = [
		{ name: "normal", kind: null, target: null, optional: false },
		{ name: "transitive", rename: "codec-alias", kind: null, target: null, optional: true },
	];
	root.features = { default: ["codec-alias?/std"], codec: ["dep:codec-alias"] };
	rootNode.features = ["default"];
	rootNode.deps = [
		{ name: "codec_library", pkg: "normal", dep_kinds: [{ kind: null, target: null }] },
		{ name: "codec_alias", pkg: "transitive", dep_kinds: [{ kind: null, target: null }] },
	];
	metadata.resolve.nodes.find((node) => node.id === "normal").deps = [];

	// `codec-alias?/std` does not turn the dependency on
	assert.deepEqual(packageNames(), ["normal", "ror"]);
	// `dep:codec-alias` does
	rootNode.features.push("codec");
	assert.deepEqual(packageNames(), ["normal", "ror", "transitive"]);
	// so does `codec-alias/std`
	root.features.codec = ["codec-alias/std"];
	assert.deepEqual(packageNames(), ["normal", "ror", "transitive"]);
	// an enabled dependency with no resolved edge is not for this target
	rootNode.deps.pop();
	assert.deepEqual(packageNames(), ["normal", "ror"]);
});

test("smoke waits for service exit after its record is removed", async (t) => {
	const state = temporaryFolder(t);
	const folder = path.join(state, "services");
	fs.mkdirSync(folder);
	const stop = path.join(state, "stop");
	// a stand-in service: it says it is ready, then exits 200 ms after the stop file appears
	const script = `
		const fs = require("node:fs");
		process.stdout.write("ready");
		const poll = setInterval(() => {
			if (!fs.existsSync(process.argv[1])) return;
			clearInterval(poll);
			setTimeout(() => process.exit(0), 200);
		}, 10);
	`;
	const child = spawn(process.execPath, ["-e", script, stop], { windowsHide: true });
	t.after(() => child.kill());
	await once(child.stdout, "data");
	const record = path.join(folder, "build.json");
	fs.writeFileSync(record, JSON.stringify({ pid: child.pid }));
	// the stop request removes the record first, as a service does, so only the remembered PID is left
	await stopServices(state, (argv) => {
		assert.deepEqual(argv, ["service", "stop"]);
		fs.unlinkSync(record);
		fs.writeFileSync(stop, "");
	});
	assert.throws(() => process.kill(child.pid, 0), { code: "ESRCH" });
});

test("smoke reports a service that does not exit", async (t) => {
	const state = temporaryFolder(t);
	fs.mkdirSync(path.join(state, "services"));
	// this test process stays alive for the whole wait
	fs.writeFileSync(path.join(state, "services/build.json"), JSON.stringify({ pid: process.pid }));
	await assert.rejects(stopServices(state, () => {}, { timeout: 0 }), /service processes did not exit/);
});

test("smoke snapshots refuse links outside their redirected folders", (t) => {
	const root = temporaryFolder(t);
	const outside = temporaryFolder(t);
	fs.symlinkSync(outside, path.join(root, "escape"), process.platform === "win32" ? "junction" : "dir");
	assert.throws(() => snapshot(root), /unexpected link/);
});

test("release entry points report asynchronous errors without an unhandled rejection", (t) => {
	const root = temporaryFolder(t);
	const script = path.join(root, "failure.mjs");
	const common = pathToFileURL(path.join(ROOT, "tools/release/common.mjs")).href;
	fs.writeFileSync(script, `import { main } from ${JSON.stringify(common)};
main(import.meta.url, async () => { await Promise.resolve(); throw new Error("asynchronous failure"); });
`);
	const child = spawnSync(process.execPath, [script], { encoding: "utf8", windowsHide: true });
	assert.equal(child.status, 1);
	assert.equal(child.stderr.trim(), "asynchronous failure");
});

test("release entry points run when started through a symlinked folder", (t) => {
	const root = temporaryFolder(t);
	const folder = path.join(root, "scripts");
	const link = path.join(root, "linked");
	fs.mkdirSync(folder);
	fs.symlinkSync(folder, link, process.platform === "win32" ? "junction" : "dir");
	const common = pathToFileURL(path.join(ROOT, "tools/release/common.mjs")).href;
	fs.writeFileSync(path.join(folder, "entry.mjs"), `import { main } from ${JSON.stringify(common)};
main(import.meta.url, () => console.log("started directly"));
`);
	const child = spawnSync(process.execPath, [path.join(link, "entry.mjs")], { encoding: "utf8", windowsHide: true });
	assert.equal(child.status, 0);
	assert.equal(child.stdout.trim(), "started directly");
	assert.equal(child.stderr, "");
});

test("smoke accepts the redirected low-integrity driver cache and rejects nearby paths", (t) => {
	const home = temporaryFolder(t);
	const cache = path.join(home, "ror");
	// Windows file names do not differ by case, so the folder is spelled in lower case there
	const driver = process.platform === "win32" ? "appdata/locallow/nvidia/dxcache" : "AppData/LocalLow/NVIDIA/DXCache";
	const folder = path.join(home, driver);
	fs.mkdirSync(folder, { recursive: true });
	fs.writeFileSync(path.join(folder, "driver.nvph"), "driver cache");
	checkHome(home, cache);
	fs.mkdirSync(folder + "-other");
	assert.throws(() => checkHome(home, cache), /unexpected user-home write/);
});

test("smoke allows named driver caches but rejects other home cache writes", (t) => {
	const home = temporaryFolder(t);
	const cache = path.join(home, "ror");
	const driverCaches = ["mesa_shader_cache", ".cache/mesa_shader_cache_db", ".cache/nvidia/GLCache", "NVIDIA/DXCache", "AMD/VkCache"];
	for (const driver of driverCaches) {
		const folder = path.join(home, driver);
		fs.mkdirSync(folder, { recursive: true });
		fs.writeFileSync(path.join(folder, "shader"), "driver cache");
	}
	checkHome(home, cache);
	// the parent folders exist now, but only the named caches inside them are allowed
	for (const unexpected of [".cache/ror", "NVIDIA/ror", "AMD/ror"]) {
		const folder = path.join(home, unexpected);
		fs.mkdirSync(folder);
		assert.throws(() => checkHome(home, cache), /unexpected user-home write/);
		fs.rmdirSync(folder);
	}
});
