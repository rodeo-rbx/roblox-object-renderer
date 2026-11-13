// Writes THIRD_PARTY_LICENSES.txt for one build target: every crate compiled into the binary, with its
// source, declared licence and the licence and notice files of its package. Reads `cargo metadata`;
// downloads nothing.
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { ROOT, main, write } from "./common.mjs";

// the file names that count as a licence or notice
const LICENSE_FILE = /^(LICENSE|LICENCE|COPYING|NOTICE|UNLICENSE|COPYRIGHT)/i;
// the same without the notice kinds: a file that holds licence terms
const LICENSE_TERMS_FILE = /^(LICENSE|LICENCE|COPYING|UNLICENSE)/i;
const SKIPPED_FOLDERS = [".git", "target", "node_modules"];

// How each common licence names its file when a crate has several. Used to pick the standard text.
const STANDARD_TEXT_FILE = {
	"MIT": /^(LICENSE|LICENCE)(?:[._-]MIT(?:[._-].*)?)?$/i,
	"Apache-2.0": /^(LICENSE|LICENCE)(?:[._-](?:APACHE(?:[._-]2\.0)?|Apache-2\.0)(?:\.txt)?)?$/i,
	"BSD-3-Clause": /^(LICENSE|COPYING)(?:[._-]BSD(?:[._-].*)?)?$/i,
	"ISC": /^(LICENSE|COPYING)(?:[._-]ISC(?:[._-].*)?)?$/i,
	"Zlib": /^(LICENSE|COPYING)(?:[._-]ZLIB(?:[._-].*)?)?$/i,
	"Unlicense": /^UNLICENSE(?:\.txt)?$/i,
	"CC0-1.0": /^(LICENSE[._-])?CC0(?:[._-].*)?$/i,
	"Unicode-3.0": /^(LICENSE[._-])?UNICODE(?:[._-].*)?$/i,
};

function slashed(file) {
	return file.replaceAll("\\", "/");
}

function byName(a, b) {
	return a.name.localeCompare(b.name, "en");
}

/** Every licence or notice file under a package root, plus all files of a LICENSES folder, in name order. */
function licenseFiles(root) {
	const files = [];
	function visit(dir, inLicensesFolder) {
		for (const entry of fs.readdirSync(dir, { withFileTypes: true }).sort(byName)) {
			const file = path.join(dir, entry.name);
			if (entry.isDirectory()) {
				if (SKIPPED_FOLDERS.includes(entry.name)) continue;
				visit(file, inLicensesFolder || /^LICENSES$/i.test(entry.name));
			} else if (entry.isFile() && (inLicensesFolder || LICENSE_FILE.test(entry.name))) {
				files.push({ name: slashed(path.relative(root, file)), text: fs.readFileSync(file, "utf8") });
			}
		}
	}
	visit(root, false);
	return files;
}

/**
 * The optional dependencies of `pkg` that an enabled feature turns on. A feature turns one on as
 * `dep:name` or `name/feature`; `name?/feature` only configures it when something else has.
 */
function activeDependencies(pkg, node) {
	const active = new Set();
	const seen = new Set();
	function visit(feature) {
		if (seen.has(feature)) return;
		seen.add(feature);
		if (feature.startsWith("dep:")) {
			active.add(feature.slice("dep:".length));
		} else if (feature.includes("/")) {
			const [dependency] = feature.split("/");
			if (!dependency.endsWith("?")) active.add(dependency);
		} else {
			// a feature that enables other features of this package
			for (const member of pkg.features?.[feature] ?? []) visit(member);
		}
	}
	for (const feature of node.features ?? []) visit(feature);
	return active;
}

/**
 * The normal dependencies `pkg` declares that one resolved edge (`edge`, of the kind `kind`, for one
 * target) comes from. The resolve names a renamed dependency by its alias with `-` as `_`.
 */
function declarationsOfEdge(pkg, edge, kind, packages) {
	const edgeName = packages.get(edge.pkg).name;
	return (pkg.dependencies ?? []).filter((declared) => {
		if (declared.kind !== null || declared.target !== kind.target) return false;
		if (declared.rename) return declared.rename.replaceAll("-", "_") === edge.name;
		return declared.name === edgeName;
	});
}

/** The packages compiled into the binary: `name`'s normal dependencies, not dev or build ones. */
export function normalPackages(metadata, name = "ror") {
	const packages = new Map(metadata.packages.map((pkg) => [pkg.id, pkg]));
	const nodes = new Map(metadata.resolve.nodes.map((node) => [node.id, node]));
	const root = metadata.packages.find((pkg) => pkg.name === name && metadata.workspace_members.includes(pkg.id));
	if (!root) throw new Error(`no workspace package ${name}`);

	const seen = new Set();
	function visit(id) {
		if (seen.has(id)) return;
		seen.add(id);
		const node = nodes.get(id);
		if (!node) throw new Error(`missing resolve node ${id}`);
		const pkg = packages.get(id);
		const active = activeDependencies(pkg, node);
		// Follow the resolve's edges: they are already filtered for the target. The manifest lists every
		// dependency it could have, so it only tells which edges an optional dependency's features turn on.
		for (const edge of node.deps) {
			const isCompiledIn = edge.dep_kinds.some((kind) => {
				if (kind.kind !== null) return false; // dev and build dependencies are not in the binary
				const declarations = declarationsOfEdge(pkg, edge, kind, packages);
				return declarations.some((declared) => !declared.optional || active.has(declared.rename ?? declared.name));
			});
			if (isCompiledIn) visit(edge.pkg);
		}
	}
	visit(root.id);

	const selected = [...seen].map((id) => packages.get(id));
	return selected.sort((a, b) => a.name.localeCompare(b.name, "en") || a.version.localeCompare(b.version, "en"));
}

/** The licence names in an SPDX expression: "MIT OR Apache-2.0" gives MIT and Apache-2.0. */
function identifiers(expression) {
	const words = (expression ?? "").match(/[A-Za-z0-9][A-Za-z0-9.+-]*/g) ?? [];
	return words.filter((word) => !["AND", "OR", "WITH"].includes(word));
}

/** Where a package comes from, as the inventory words it. */
function describeSource(pkg, workspaceRoot) {
	if (pkg.source?.startsWith("registry+")) return "crates.io";
	if (pkg.source) return pkg.source; // a git URL and commit
	const relative = slashed(path.relative(workspaceRoot, path.dirname(pkg.manifest_path)));
	if (relative.startsWith("..") || path.isAbsolute(relative)) {
		throw new Error(`${pkg.name}: path dependency outside source tree`);
	}
	return `path in this repo: ${relative || "."}`;
}

/** A finder of each package's licence files, remembering the answers. */
function licenseFilesByPackage(workspaceRoot) {
	const known = new Map();
	return function filesOf(pkg) {
		if (known.has(pkg.id)) return known.get(pkg.id);
		const packageRoot = path.dirname(pkg.manifest_path);
		const files = licenseFiles(packageRoot);

		// a licence file the manifest names, if it is not one of the files found above
		if (pkg.license_file) {
			const named = path.resolve(packageRoot, pkg.license_file);
			const name = slashed(path.relative(packageRoot, named));
			if (!files.some((file) => file.name === name)) files.push({ name, text: fs.readFileSync(named, "utf8") });
		}

		// crates of this repo (no source) are covered by the repo's own licence and notices too
		if (!pkg.source) {
			for (const name of ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"]) {
				const file = path.join(workspaceRoot, name);
				if (fs.existsSync(file)) files.push({ name: `workspace/${name}`, text: fs.readFileSync(file, "utf8") });
			}
		}
		known.set(pkg.id, files);
		return files;
	};
}

/** The inventory section for one package. */
function packageSection(pkg, source, files) {
	const heading = `## ${pkg.name} ${pkg.version}\nSource: ${source}\nDeclared licence: ${pkg.license ?? "see licence file"}\n\n`;
	if (!files.length) {
		return heading + "No licence file shipped in this package; see the standard text appendix.\n";
	}
	return heading + files.map((file) => `--- ${file.name} ---\n${file.text}\n`).join("\n");
}

/** Does the licence file `name` of a crate with licences `ids` hold the standard text of `id`? */
function isStandardText(name, id, ids) {
	if (name.includes("/")) return false; // only files in the package root
	if (STANDARD_TEXT_FILE[id]?.test(name)) return true;
	// A crate with one licence names it in its LICENSE or COPYING file. A notice file holds no terms.
	return ids.length === 1 && LICENSE_FILE.test(name) && !/^NOTICE|^COPYRIGHT/i.test(name);
}

/** The standard text of licence `id`, taken from any package in the build graph that ships it. */
function findStandardText(id, packages, filesOf) {
	for (const pkg of packages) {
		const ids = identifiers(pkg.license);
		if (!ids.includes(id)) continue;
		const found = filesOf(pkg).find((file) => isStandardText(file.name, id, ids));
		if (found) return found;
	}
	return undefined;
}

export function inventory(metadata) {
	const filesOf = licenseFilesByPackage(metadata.workspace_root);
	const withoutText = new Set(); // licences of crates that ship no licence file
	const sections = [];

	for (const pkg of normalPackages(metadata)) {
		const files = filesOf(pkg);
		const source = describeSource(pkg, metadata.workspace_root);
		if (!pkg.license && !files.length) {
			throw new Error(`${pkg.name}: no declared licence or licence file`);
		}
		const shipsLicenseText = pkg.license_file || files.some((file) => LICENSE_TERMS_FILE.test(path.basename(file.name)));
		if (!shipsLicenseText) {
			for (const id of identifiers(pkg.license)) withoutText.add(id);
		}
		sections.push(packageSection(pkg, source, files));
	}

	// the standard text of each such licence, once, from another package that has it
	const appendix = [];
	for (const id of [...withoutText].sort()) {
		const found = findStandardText(id, metadata.packages, filesOf);
		if (!found) throw new Error(`no local standard licence text for ${id}; refusing an incomplete inventory`);
		appendix.push(`## Standard text: ${id}\n\n${found.text}\n`);
	}

	const intro = "# Third-party licences\n\nNormal dependency graph of ror for this release target and feature selection. Package-root notices, LICENSES folders and vendored source notices are included.\n\n";
	const appendixText = appendix.length ? "\n# Standard licence text appendix\n\n" + appendix.join("\n") : "";
	return intro + sections.join("\n") + appendixText;
}

function cargoMetadata(target) {
	const argv = ["metadata", "--locked", "--format-version", "1", "--filter-platform", target];
	return execFileSync("cargo", argv, { cwd: ROOT, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
}

main(import.meta.url, (options) => {
	if (!options.target || !options.output) {
		throw new Error("usage: licenses.mjs --target <triple> --output <file> [--metadata <json>]");
	}
	const json = options.metadata ? fs.readFileSync(options.metadata, "utf8") : cargoMetadata(options.target);
	write(options.output, inventory(JSON.parse(json)));
});
