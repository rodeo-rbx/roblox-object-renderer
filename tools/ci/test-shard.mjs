// The runner behind the CI jobs that share ror's library tests (.github/workflows/ci.yml). Cargo starts the test binary
// through test-entry.mjs, which calls run() here:
//
//   node tools/ci/test-entry.mjs <index> <count> <binary> [test arguments]
//
// It lists the binary's tests, takes this job's share of them and runs it, one test at a time (--test-threads=1 comes
// from the workflow, after `--`). Fourteen tests that draw on the software adapter take nearly all of the time, so the
// shares are cut by time, not by count: test-weights.json holds the seconds a CI run measured for each slow test, and
// every other test counts as RUN_MS. The tests are dealt out longest first, each to the job with the least time so far
// (ties to the lowest job), so every job computes the same shares from the same list, and the shares together hold
// every listed test exactly once. A new test needs no change here; a new slow one runs in some job either way, and its
// time in test-weights.json only evens the jobs out again (the file says how to refresh it).
//
// The exit code is the test binary's (the last failing one when the share runs in several commands), so a failing
// test fails the job. A list that is empty, has a duplicate or does not read as libtest's fails the job too.
import { spawnSync } from "node:child_process";
import fs from "node:fs";

// What a test without a measured time counts as, in milliseconds: the 364 fast tests take about 9 s together
export const RUN_MS = 25;
// The names go on the command line, and Windows allows 32767 UTF-16 characters there. A command holds at most this many
// characters of names (each name and 3 for its quotes and the space), which leaves room for the binary and its options.
export const NAME_BUDGET = 24000;

// The names in libtest's `--list --format terse` output (one `name: test` per line), sorted by character code
export function parseTestList(text) {
	const names = [];
	for (const line of text.split(/\r?\n/)) {
		if (!line.trim()) continue;
		const match = /^(.+): (?:test|benchmark)$/.exec(line);
		// a name that starts with "-" would read as an option when it is passed back to the binary
		if (!match || match[1].startsWith("-")) throw new Error(`unrecognized libtest --list row: ${line}`);
		names.push(match[1]);
	}
	if (names.length === 0 || new Set(names).size !== names.length) throw new Error("the test list is empty or has duplicates");
	return names.sort(byCode);
}

function byCode(a, b) {
	return a < b ? -1 : a > b ? 1 : 0;
}

// Fails unless `index` is a whole number from 0 up to `count`, which is a whole number of at least 1
export function checkShard(index, count) {
	if (!Number.isSafeInteger(count) || count < 1 || !Number.isSafeInteger(index) || index < 0 || index >= count) {
		throw new Error("expected a zero-based shard index and a positive shard count");
	}
}

// The measured seconds per test name, as whole milliseconds
export function readWeights(file = new URL("./test-weights.json", import.meta.url)) {
	const table = JSON.parse(fs.readFileSync(file, "utf8"));
	const weights = new Map();
	for (const [name, seconds] of Object.entries(table.seconds ?? {})) {
		if (typeof seconds !== "number" || !(seconds >= 0)) throw new Error(`test-weights.json: ${name} has no time in seconds`);
		weights.set(name, Math.round(seconds * 1000));
	}
	return weights;
}

// The `count` shares of `names`: longest first, each to the job with the least time so far
export function shares(names, weights, count) {
	checkShard(0, count);
	const items = names.map((name) => ({ name, ms: weights.get(name) ?? RUN_MS }));
	items.sort((a, b) => b.ms - a.ms || byCode(a.name, b.name));
	const loads = new Array(count).fill(0);
	const result = Array.from({ length: count }, () => []);
	for (const item of items) {
		let least = 0;
		for (let i = 1; i < count; i++) if (loads[i] < loads[least]) least = i;
		result[least].push(item.name);
		loads[least] += item.ms;
	}
	return result.map((share, i) => ({ names: share.sort(byCode), ms: loads[i] }));
}

// Splits `names` into runs of names that fit the command line
export function chunks(names, budget = NAME_BUDGET) {
	const result = [];
	let chunk = [];
	let length = 0;
	for (const name of names) {
		const cost = name.length + 3;
		if (cost > budget) throw new Error(`test name longer than the command line allows: ${name}`);
		if (length + cost > budget) {
			result.push(chunk);
			chunk = [];
			length = 0;
		}
		chunk.push(name);
		length += cost;
	}
	if (chunk.length) result.push(chunk);
	return result;
}

// Runs this job's share and returns the exit code. `cwd`, `env`, `spawn`, `log` and `weights` are replaced by the
// tests. The binary starts in `cwd` with `env`, the folder and variables cargo gave the runner.
export function run(argv, { cwd = process.cwd(), env = process.env, spawn = spawnSync, log = console.error, weights = readWeights() } = {}) {
	const [indexText, countText, binary, ...testArguments] = argv;
	const index = Number(indexText);
	const count = Number(countText);
	checkShard(index, count);
	if (!binary) throw new Error("missing test binary");

	function start(argumentsForBinary, stdio = "inherit") {
		const result = spawn(binary, argumentsForBinary, { cwd, env, windowsHide: true, encoding: "utf8", stdio, maxBuffer: 16 * 1024 * 1024 });
		if (result.error) throw result.error;
		return { code: result.status ?? 1, output: result.stdout };
	}

	const listing = start(["--list", "--format", "terse"], ["ignore", "pipe", "inherit"]);
	if (listing.code !== 0) return listing.code;
	const names = parseTestList(listing.output);
	const all = shares(names, weights, count);
	const mine = all[index];
	log(`ror tests, part ${index + 1} of ${count}: ${mine.names.length} of ${names.length} tests, about ${Math.round(mine.ms / 1000)} s (the parts: ${all.map((share) => Math.round(share.ms / 1000)).join(" / ")} s)`);
	// A share with no tests runs nothing: libtest with no names would run every test
	let exitCode = 0;
	for (const chunk of chunks(mine.names)) {
		// --exact makes each name match that test alone, never every test whose name contains it. Every chunk runs, so
		// one log shows every failing test.
		const code = start(["--exact", ...testArguments, ...chunk]).code;
		if (code !== 0) exitCode = code;
	}
	return exitCode;
}
