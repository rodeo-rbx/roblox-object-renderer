// Prints the version in the Cargo manifests, failing unless every workspace member agrees.
// With --tag <tag> it also fails unless the tag is exactly v<version>.
import fs from "node:fs";
import path from "node:path";
import { ROOT, main } from "./common.mjs";

/** The text of the `[name]` table in a Cargo.toml: everything up to the next line starting with `[`. */
function tableBody(toml, name) {
	const header = name.replaceAll(".", "\\.");
	const table = new RegExp(`^\\[${header}\\]([^]*?)(?=^\\[|$(?![^]))`, "m");
	return toml.match(table)?.[1] ?? "";
}

function readManifest(root, ...folder) {
	return fs.readFileSync(path.join(root, ...folder, "Cargo.toml"), "utf8");
}

export function version(root = ROOT, tag) {
	const workspace = readManifest(root);
	const value = tableBody(workspace, "workspace.package").match(/^version\s*=\s*"(\d+\.\d+\.\d+(?:-rodeo\.\d+)?)"/m)?.[1];
	if (!value) throw new Error("Cargo.toml has no [workspace.package] version");
	const memberList = tableBody(workspace, "workspace").match(/^members\s*=\s*\[([^]*?)\]/m)?.[1];
	if (!memberList) throw new Error("Cargo.toml has no workspace members");

	for (const [, member] of memberList.matchAll(/"([^"]+)"/g)) {
		const manifest = tableBody(readManifest(root, member), "package");
		const inheritsVersion = /^version\.workspace\s*=\s*true\s*$/m.test(manifest);
		const ownVersion = manifest.match(/^version\s*=\s*"([^"]+)"/m)?.[1];
		if (!inheritsVersion && ownVersion !== value) {
			throw new Error(`${member}: version differs from ${value}`);
		}
	}

	if (tag !== undefined && tag !== `v${value}`) {
		throw new Error(`tag ${tag} does not match v${value}`);
	}
	return value;
}

main(import.meta.url, (options) => console.log(version(ROOT, options.tag)));
