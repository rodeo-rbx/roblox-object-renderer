// Tests of dry-run.mjs, the choice of the dry run whose zips a tag publishes
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { WORKFLOW, pickDryRun, run } from "./dry-run.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
// The artifacts a release publishes: one zip per platform
const ARTIFACTS = ["ror-windows-x86_64", "ror-linux-x86_64", "ror-macos-aarch64"];
const SHA = "a".repeat(40);

function dryRun(id, fields = {}) {
	return { id, run_number: id, run_attempt: 1, head_sha: SHA, head_branch: "main", path: WORKFLOW, event: "push", status: "completed", conclusion: "success", ...fields };
}
const passedJobs = [
	{ name: "version", conclusion: "success" },
	{ name: "build (windows)", conclusion: "success" },
	{ name: "smoke (windows)", conclusion: "success" },
	{ name: "notes", conclusion: "skipped" },
	{ name: "publish", conclusion: "skipped" },
];
const allArtifacts = ARTIFACTS.map((name) => ({ name, expired: false }));

function pick(runs, { jobs = {}, artifacts = {} } = {}) {
	return pickDryRun({
		runs,
		jobsOf: async (r) => jobs[r.id] ?? passedJobs,
		artifactsOf: async (r) => artifacts[r.id] ?? allArtifacts,
		sha: SHA,
		branch: "main",
		artifacts: ARTIFACTS,
	});
}

test("a tag publishes the newest dry run of its commit that passed every job and kept every zip", async () => {
	const { run: found } = await pick([dryRun(10), dryRun(12, { event: "workflow_dispatch" }), dryRun(11)]);
	assert.equal(found.id, 12);
});

test("runs of another commit, branch, workflow file or event are never taken", async () => {
	const { run: found, reasons } = await pick([
		dryRun(1, { head_sha: "b".repeat(40) }),
		dryRun(2, { head_branch: "other" }),
		dryRun(3, { path: ".github/workflows/ci.yml" }),
		dryRun(4, { event: "pull_request" }),
		dryRun(5, { event: "schedule" }),
	]);
	assert.equal(found, null);
	assert.equal(reasons.length, 2, "the two runs of the right file, branch and commit are passed over by event");
});

test("a dry run that failed, is still going, or lost a zip is passed over for an older one that passed", async () => {
	const runs = [dryRun(1), dryRun(2, { conclusion: "failure" }), dryRun(3, { status: "in_progress", conclusion: null }), dryRun(4), dryRun(5), dryRun(6)];
	const jobs = {
		4: [...passedJobs, { name: "smoke (macos)", conclusion: "cancelled" }],
		5: [...passedJobs.slice(0, 3), { name: "notes", conclusion: "skipped" }, { name: "build (linux)", conclusion: "skipped" }],
	};
	const artifacts = { 6: allArtifacts.map((a, i) => (i === 2 ? { ...a, expired: true } : a)) };
	const { run: found, reasons } = await pick(runs, { jobs, artifacts });
	assert.equal(found.id, 1);
	assert.equal(reasons.length, 5);
	assert.match(reasons.join("\n"), /run 6 .*no artifact ror-macos-aarch64/);
	assert.match(reasons.join("\n"), /run 5 .*build \(linux\) skipped/);
});

test("no run found leaves the run id empty, so the release builds", async (t) => {
	const calls = [];
	t.mock.method(globalThis, "fetch", async (url) => {
		calls.push(String(url));
		return { ok: true, json: async () => ({ workflow_runs: [] }) };
	});
	const line = await run(["--repository", "owner/name", "--sha", SHA, "--artifacts", ARTIFACTS.join(",")], { GH_TOKEN: "x" }, () => {});
	assert.equal(line, "run-id=");
	assert.match(calls[0], new RegExp(`/repos/owner/name/actions/workflows/release.yml/runs\\?head_sha=${SHA}&branch=main`));
});

test("the workflow asks for every platform's zip and publishes from the dry run only on a tag", () => {
	// Windows checkouts may have CRLF line ends
	const workflow = fs.readFileSync(path.join(ROOT, WORKFLOW), "utf8").replace(/\r\n/g, "\n");
	assert.deepEqual(/--artifacts (\S+)/.exec(workflow)[1].split(","), ARTIFACTS);
	assert.match(workflow, /name: Find the passing dry run of the tagged commit\n\s+id: dry-run\n\s+if: \$\{\{ github.event_name == 'push' && github.ref_type == 'tag' \}\}/);
	// building and smoke-testing are skipped only when a dry run's zips are published
	for (const job of ["build", "smoke"]) {
		assert.match(workflow, new RegExp(`\\n  ${job}:\\n    if: \\$\\{\\{ !github.event.repository.private && needs.version.outputs.dry-run == '' \\}\\}`));
	}
	// and the downloads from another run are the publish job's alone
	assert.equal(workflow.match(/run-id: \$\{\{ needs.version.outputs.dry-run \}\}/g).length, workflow.split(/^  publish:/m)[1].match(/run-id: /g).length);
});
