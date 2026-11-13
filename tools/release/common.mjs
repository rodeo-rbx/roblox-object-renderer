// Helpers shared by the release scripts.
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
// The public repository as owner/name. It is derived from the URL because the publishing gate allows the
// repository's name only in a URL. A workflow names the repository it runs in, so a fork's release names its own
// source.
export const REPOSITORY = process.env.GITHUB_REPOSITORY || new URL("https://github.com/Anime-Vanguards/roblox-object-renderer").pathname.slice(1);

/** Reads `--name value` pairs into an object: `--tag v0.1.0` becomes `{ tag: "v0.1.0" }`. */
export function args(argv = process.argv.slice(2)) {
	const options = {};
	for (let i = 0; i < argv.length; i += 2) {
		const flag = argv[i];
		const value = argv[i + 1];
		if (!flag.startsWith("--") || value === undefined) {
			throw new Error("expected arguments as --name value pairs");
		}
		options[flag.slice(2)] = value;
	}
	return options;
}

/**
 * Whether two paths are the same file once links are followed. The real paths are compared because one script can
 * be reached by two names: on macOS the temp folder /var is a link to /private/var. A path that cannot be resolved
 * is compared as written.
 */
function sameFile(first, second) {
	try {
		return fs.realpathSync(first) === fs.realpathSync(second);
	} catch {
		return first === second;
	}
}

/** Runs `action` with the command line options when the script was started directly, not imported (by the tests). */
export async function main(url, action) {
	const entryPoint = process.argv[1];
	if (!entryPoint || !sameFile(path.resolve(entryPoint), fileURLToPath(url))) return;
	try {
		await action(args());
	} catch (error) {
		console.error(error.message);
		process.exitCode = 1;
	}
}

/** The output of a git command run in `cwd`, without its trailing newline. */
export function git(argv, cwd = ROOT) {
	return execFileSync("git", argv, { cwd, encoding: "utf8", maxBuffer: 32 * 1024 * 1024 }).trimEnd();
}

/** Writes a file, making its folder first. */
export function write(file, text) {
	fs.mkdirSync(path.dirname(file), { recursive: true });
	fs.writeFileSync(file, text);
}

/** Checks which release a build belongs to and returns it as `{ version, tag, commit, repository }`. */
export function identity(version, tag, commit, repository = REPOSITORY) {
	if (!/^\d+\.\d+\.\d+(-rodeo\.\d+)?$/.test(version) || tag !== `v${version}`) {
		throw new Error(`tag ${tag} is not v<manifest version> (manifests say ${version})`);
	}
	if (!/^[a-f0-9]{40}$/.test(commit)) {
		throw new Error(`commit ${commit} is not a full SHA`);
	}
	if (!/^[\w.-]+\/[\w.-]+$/.test(repository)) {
		throw new Error(`repository ${repository} is not owner/name`);
	}
	return { version, tag, commit, repository };
}

/** SOURCE.md: where the Corresponding Source of a binary built from this tag is. */
export function sourceText({ tag, commit, repository }) {
	const repo = `https://github.com/${repository}`;
	const paragraphs = [
		"# Corresponding Source",
		"This binary is distributed under GPL-3.0-or-later.",
		`Source: ${repo}/tree/${tag}\nCommit: ${commit}\nSource archive: ${repo}/archive/refs/tags/${tag}.zip`,
		"The source includes Cargo.lock and the release scripts. Obtain the Rust crates pinned in Cargo.lock from crates.io; the lockfile records their SHA-256 checksums. Any git dependencies are available from their public repository URLs at the commits pinned in Cargo.lock.",
		"Build with Rust 1.90.0: cargo build --locked --profile dist --bin ror\nUse the public default features and the native target triple (--target <triple>). On Windows set RUSTFLAGS=\"-C target-feature=+crt-static\" to include the C runtime. No x264 library or other downloaded build archive is required.",
		"See .github/workflows/release.yml at this tag for the target triples, runner setup, licence inventory and packaging commands.",
	];
	return paragraphs.join("\n\n") + "\n";
}
