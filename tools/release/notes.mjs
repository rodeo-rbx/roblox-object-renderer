// Writes the release notes: the public commit subjects since the previous release, how to install and
// verify the binaries, and where the source is. Needs the full git history and the release tags.
import { ROOT, git, identity, main, sourceText, write } from "./common.mjs";
import { version } from "./version.mjs";

// upstream's releases, and the rodeo-rbx fork's v<version>-rodeo.N ones
const RELEASE_TAG = /^v\d+\.\d+\.\d+(-rodeo\.\d+)?$/;
const FENCE = "```";

/** The newest release tag before `commit`, or undefined for the first release. */
function previousReleaseTag(tag, commit, root) {
	// only release tags that are ancestors of this commit bound the changelog
	const tags = git(["tag", "--merged", "HEAD", "--sort=-version:refname"], root)
		.split("\n")
		.filter((name) => RELEASE_TAG.test(name) && name !== tag);
	return tags.find((name) => git(["rev-parse", `${name}^{commit}`], root) !== commit);
}

/** One bullet per commit subject, oldest first. */
function changelog(previous, root) {
	const range = previous ? `${previous}..HEAD` : "HEAD";
	const subjects = git(["log", "--reverse", "--format=%s", range], root).split("\n").filter(Boolean);
	// A subject is only ever text in a file, never a shell command or a workflow expression.
	const escape = (subject) => subject.replace(/[<>]/g, (char) => (char === "<" ? "&lt;" : "&gt;"));
	return subjects.map((subject) => `- ${escape(subject)}`).join("\n");
}

export function notes(options, root = ROOT) {
	const manifestVersion = version(root, options.tag);
	const tag = options.tag ?? `v${manifestVersion}`;
	const commit = git(["rev-parse", "HEAD"], root);
	const info = identity(manifestVersion, tag, commit, options.repository);

	const bullets = changelog(previousReleaseTag(tag, commit, root), root);
	const sections = [
		`# ror ${tag}`,
		bullets || "No commit subjects.",
		"## Installing",
		"Download the zip for your platform from this release and extract ror, or add to rokit.toml:",
		`${FENCE}toml\nror = "${info.repository}@${manifestVersion}"\n${FENCE}`,
		"Use rokit 1.0 or later on Linux and macOS; 0.2.x cannot identify this executable by the repository name. Built-in Roblox content is read from your own Roblox install or ROR_ROBLOX_CONTENT. See docs/building.md at the tag for standalone commands and cache paths.",
		"## Verifying",
		"Check the zip against SHA256SUMS, then run:",
		`${FENCE}sh\ngh attestation verify <zip> -R ${info.repository}\n${FENCE}`,
		sourceText(info),
	];
	const text = sections.join("\n\n");
	write(options.output, text);
	return text;
}

main(import.meta.url, (options) => notes(options));
