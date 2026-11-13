// Finds the release dry run a tag's release can publish from: a run of the release workflow on the default branch, at
// the tagged commit, that passed on every platform and still has every platform's zip.
//
//   node tools/release/dry-run.mjs --repository <owner/name> --sha <commit> --artifacts <name,name,...>
//
// Prints `run-id=<id>` for $GITHUB_OUTPUT, or `run-id=` when there is no such run (the release then builds and tests
// every platform itself, as a dry run does). Says on stderr which run it found or why it took none. Reads the GitHub
// API with GH_TOKEN (or GITHUB_TOKEN), which needs `actions: read`. Node.js only, no npm packages; the same file in
// each tool repository.
//
// A dry run qualifies when it:
//   - ran the release workflow file itself (not another workflow at the same commit), on the default branch, started by
//     a push or by hand (the only events that run it there), at exactly the tagged commit;
//   - completed with success, and every job of its latest attempt succeeded, except the tag-only jobs (notes, publish),
//     which a dry run skips;
//   - still has an unexpired artifact of every name given.
// The newest such run is taken. Its zips are a build of the same source by the same workflow file on clean runners.
import fs from "node:fs";
import { fileURLToPath } from "node:url";

export const WORKFLOW = ".github/workflows/release.yml";
const TAG_ONLY_JOB = /^(?:notes|publish)\b/;

// The qualifying run among `runs` (the API's workflow_runs), or null with the reasons the others were passed over.
// `jobsOf(run)` and `artifactsOf(run)` give a run's jobs (latest attempt) and artifacts as the API lists them.
export async function pickDryRun({ runs, jobsOf, artifactsOf, sha, branch, artifacts }) {
	const reasons = [];
	const candidates = runs
		.filter((run) => run.head_sha === sha && run.head_branch === branch && run.path === WORKFLOW)
		.sort((a, b) => b.run_number - a.run_number || b.id - a.id);
	for (const run of candidates) {
		const label = `run ${run.id} (attempt ${run.run_attempt ?? 1})`;
		if (run.event !== "push" && run.event !== "workflow_dispatch") { reasons.push(`${label}: started by ${run.event}`); continue; }
		if (run.status !== "completed" || run.conclusion !== "success") { reasons.push(`${label}: ${run.status} ${run.conclusion ?? ""}`.trim()); continue; }
		const jobs = await jobsOf(run);
		const failed = jobs.filter((job) => !(job.conclusion === "success" || (job.conclusion === "skipped" && TAG_ONLY_JOB.test(job.name))));
		if (jobs.length === 0 || failed.length) { reasons.push(`${label}: jobs not passed: ${failed.map((job) => `${job.name} ${job.conclusion}`).join(", ") || "none listed"}`); continue; }
		const present = new Set((await artifactsOf(run)).filter((artifact) => !artifact.expired).map((artifact) => artifact.name));
		const missing = artifacts.filter((name) => !present.has(name));
		if (missing.length) { reasons.push(`${label}: no artifact ${missing.join(", ")}`); continue; }
		return { run, reasons };
	}
	if (candidates.length === 0) reasons.push(`no run of ${WORKFLOW} on ${branch} at ${sha}`);
	return { run: null, reasons };
}

async function api(token, url) {
	const response = await fetch(url, { headers: { accept: "application/vnd.github+json", authorization: `Bearer ${token}`, "x-github-api-version": "2022-11-28" } });
	if (!response.ok) throw new Error(`GET ${url}: ${response.status} ${await response.text()}`);
	return response.json();
}

function options(argv) {
	const result = {};
	for (let i = 0; i < argv.length; i += 2) {
		const name = argv[i]?.replace(/^--/, "");
		if (!["repository", "sha", "artifacts", "branch"].includes(name) || argv[i + 1] === undefined || name in result) throw new Error("usage: dry-run.mjs --repository <owner/name> --sha <commit> --artifacts <name,...> [--branch <default branch>]");
		result[name] = argv[i + 1];
	}
	if (!result.repository || !/^[0-9a-f]{40}$/.test(result.sha ?? "") || !result.artifacts) throw new Error("--repository, a full --sha and --artifacts are required");
	return result;
}

export async function run(argv, env = process.env, log = console.error) {
	const { repository, sha, artifacts, branch = "main" } = options(argv);
	const token = env.GH_TOKEN || env.GITHUB_TOKEN;
	if (!token) throw new Error("GH_TOKEN or GITHUB_TOKEN is needed to read the workflow runs");
	const base = `${env.GITHUB_API_URL || "https://api.github.com"}/repos/${repository}`;
	const listed = await api(token, `${base}/actions/workflows/release.yml/runs?head_sha=${sha}&branch=${encodeURIComponent(branch)}&per_page=50`);
	const { run: found, reasons } = await pickDryRun({
		runs: listed.workflow_runs ?? [],
		jobsOf: async (r) => (await api(token, `${base}/actions/runs/${r.id}/attempts/${r.run_attempt ?? 1}/jobs?per_page=100`)).jobs ?? [],
		artifactsOf: async (r) => (await api(token, `${base}/actions/runs/${r.id}/artifacts?per_page=100`)).artifacts ?? [],
		sha,
		branch,
		artifacts: artifacts.split(",").map((name) => name.trim()).filter(Boolean),
	});
	for (const reason of reasons) log(`passed over ${reason}`);
	if (found) log(`publishing what dry run ${found.html_url ?? found.id} built and tested at ${sha}`);
	else log("no passing dry run of this commit: this run builds and tests every platform");
	return `run-id=${found ? found.id : ""}`;
}

// Started as the script, compared by real path so that a start through a linked folder counts too
function startedAsScript() {
	try {
		return Boolean(process.argv[1]) && fs.realpathSync.native(process.argv[1]) === fs.realpathSync.native(fileURLToPath(import.meta.url));
	} catch {
		return false;
	}
}

// A bad command line fails the step. A failing API call does not: the release then builds every platform itself, which
// is what it did before it could publish a dry run's zips, and the warning shows on the run's page.
if (startedAsScript()) {
	try {
		options(process.argv.slice(2));
		run(process.argv.slice(2)).then((line) => console.log(line), (error) => {
			console.error(`::warning::could not look for a dry run to publish from, so this run builds every platform: ${error.message}`);
			console.log("run-id=");
		});
	} catch (error) {
		console.error(error.message);
		process.exitCode = 1;
	}
}
