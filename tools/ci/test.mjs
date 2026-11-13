// Tests of the CI tools (the runner that shares ror's library tests over the CI jobs) and of what the workflow must
// keep: node --test tools/ci/test.mjs
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { RUN_MS, chunks, parseTestList, readWeights, run, shares } from "./test-shard.mjs";

const WORKFLOW = fs.readFileSync(new URL("../../.github/workflows/ci.yml", import.meta.url), "utf8").replace(/\r\n/g, "\n");
const binary = path.resolve("target/release/deps/ror-0123abcd.exe");

// A fake spawn: `--list` answers with `listing`, every other call with `code`; records the arguments
function fakeSpawn(listing, code = 0) {
	const calls = [];
	const spawn = (file, args) => {
		calls.push(args);
		return args[0] === "--list" ? { status: 0, stdout: listing } : { status: code, stdout: null };
	};
	return { calls, spawn };
}

test("the shares hold every listed test exactly once, for any number of jobs", () => {
	const names = parseTestList("a::first: test\r\na::first_extended: test\r\nz: test\r\nnew_test: test\r\nslow::one: test\r\n");
	const weights = new Map([["slow::one", 90000], ["a::first", 30000], ["gone::test", 50000]]);
	for (const count of [1, 2, 3, 6, 9]) {
		const all = shares(names, weights, count).flatMap((share) => share.names);
		assert.deepEqual(all.sort(), names);
		assert.equal(new Set(all).size, names.length);
	}
});

test("the slow tests are spread longest first, each to the job with the least time so far", () => {
	const names = parseTestList(["s1", "s2", "s3", "s4", "f1", "f2"].map((name) => `${name}: test`).join("\n"));
	const weights = new Map([["s1", 100000], ["s2", 60000], ["s3", 50000], ["s4", 40000]]);
	const [first, second] = shares(names, weights, 2);
	// 100 alone; 60 + 50 next to it; 40 goes to the first job (100 < 110), the two fast tests to the second (110 + 25 + 25 < 140)
	assert.deepEqual(first.names, ["s1", "s4"]);
	assert.deepEqual(second.names, ["f1", "f2", "s2", "s3"]);
	assert.equal(first.ms, 140000);
	assert.equal(second.ms, 110000 + 2 * RUN_MS);
});

test("the measured times are read as milliseconds, and a time that is not a number of seconds fails", (t) => {
	const weights = readWeights();
	assert.ok(weights.size >= 10, "the slow tests have times");
	for (const ms of weights.values()) assert.ok(Number.isInteger(ms) && ms >= 0);
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ror-ci-test-"));
	t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
	const bad = path.join(dir, "weights.json");
	fs.writeFileSync(bad, JSON.stringify({ seconds: { "a::b": "slow" } }));
	assert.throws(() => readWeights(bad), /a::b/);
});

test("a list that is empty, repeats a name or does not read as libtest's fails, and so does a bad share", () => {
	for (const text of ["", "a: test\na: test", "a: test\nunknown", "--skip: test"]) assert.throws(() => parseTestList(text));
	for (const count of [0, -1, 0.5, NaN]) assert.throws(() => shares(["a"], new Map(), count));
	assert.throws(() => run(["6", "6", binary], { spawn: fakeSpawn("a: test").spawn, log: () => {}, weights: new Map() }));
	assert.throws(() => run(["0", "6"], { spawn: fakeSpawn("a: test").spawn, log: () => {}, weights: new Map() }), /missing test binary/);
});

test("a job lists the binary, runs its share by exact name with the given options and exits with the tests' code", () => {
	const listing = "a: test\nb: test\nc: test\n";
	const weights = new Map([["b", 5000]]);
	const { calls, spawn } = fakeSpawn(listing, 101);
	const code = run(["1", "2", binary, "--test-threads=1"], { spawn, log: () => {}, weights });
	assert.equal(code, 101);
	assert.deepEqual(calls[0], ["--list", "--format", "terse"]);
	// b (5 s) goes to the first job; a and c to the second
	assert.deepEqual(calls[1], ["--exact", "--test-threads=1", "a", "c"]);
	assert.equal(calls.length, 2);
});

test("a long share runs in several commands, all of them even after a failure, and a failed listing fails the job", () => {
	const names = Array.from({ length: 3000 }, (_, i) => `module::a_test_with_a_fairly_long_name_${String(i).padStart(4, "0")}`);
	assert.ok(chunks(names).length > 1);
	assert.deepEqual(chunks(names).flat(), names);
	for (const chunk of chunks(names)) assert.ok(chunk.reduce((sum, name) => sum + name.length + 3, 0) <= 24000);
	const calls = [];
	let n = 0;
	const spawn = (file, args) => {
		calls.push(args);
		if (args[0] === "--list") return { status: 0, stdout: names.map((name) => `${name}: test`).join("\n") };
		n++;
		return { status: n === 1 ? 1 : 0 };
	};
	assert.equal(run(["0", "1", binary], { spawn, log: () => {}, weights: new Map() }), 1);
	assert.equal(calls.length, 1 + chunks(names).length);
	assert.equal(run(["0", "1", binary], { spawn: () => ({ status: 3, stdout: "" }), log: () => {}, weights: new Map() }), 3);
	assert.throws(() => run(["0", "1", binary], { spawn: () => ({ error: new Error("spawn failed") }), log: () => {}, weights: new Map() }), /spawn failed/);
});

test("a job whose share is empty runs nothing", () => {
	const { calls, spawn } = fakeSpawn("a: test\n");
	assert.equal(run(["1", "2", binary], { spawn, log: () => {}, weights: new Map() }), 0);
	assert.equal(calls.length, 1);
});

// The entry runs as cargo starts it, as a separate process. node stands in for the test binary: it rejects the listing's
// --list option with its exit code 9, which the entry must pass on, and a bad command line must exit 1, never 0.
test("the entry starts the test binary and exits with the runner's code, never 0 after a failure", () => {
	const entry = fileURLToPath(new URL("./test-entry.mjs", import.meta.url));
	const listed = spawnSync(process.execPath, [entry, "0", "1", process.execPath], { encoding: "utf8", windowsHide: true });
	assert.ifError(listed.error);
	assert.equal(listed.status, 9, listed.stderr);
	const bad = spawnSync(process.execPath, [entry, "1", "1", process.execPath], { encoding: "utf8", windowsHide: true });
	assert.equal(bad.status, 1);
	assert.match(bad.stderr, /shard index/);
});

test("the workflow runs six library jobs through the entry, each building only the library's test binary", () => {
	assert.match(WORKFLOW, /part: \[1, 2, 3, 4, 5, 6\]/);
	assert.match(WORKFLOW, /cargo test --release --locked --no-fail-fast -p ror --lib --config "target\.x86_64-pc-windows-msvc\.runner=\['node','\$entry','\$\{\{ strategy\.job-index \}\}','\$\{\{ strategy\.job-total \}\}'\]" -- --test-threads=1/);
	assert.match(WORKFLOW, /\$entry = \(Resolve-Path tools\/ci\/test-entry\.mjs\)\.Path\.Replace\('\\', '\/'\)/);
	// everything else of ror's runs in the targets job: the binary's and the integration tests, the doc tests, the examples
	assert.match(WORKFLOW, /cargo test --release --locked --no-fail-fast -p ror --bins --test '\*' -- --test-threads=1/);
	assert.match(WORKFLOW, /cargo test --release --locked -p ror --doc -- --test-threads=1/);
	assert.match(WORKFLOW, /cargo build --release --locked -p ror --examples/);
	// and the tools' own tests run
	assert.match(WORKFLOW, /node --test tools\/ci\/test\.mjs/);
});
