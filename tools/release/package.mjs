// Writes the release zip, ror-<version>-<os>-<arch>.zip, and SOURCE.md beside it.
//
// The zip is written here, with Node's zlib, so that its layout is exact and the same on every
// platform. Rokit, aftman and foreman pick the binary by its position, name and mode, so:
//   - the binary is the first entry, stored with mode 0755;
//   - no other entry is executable or ends in .exe.
import fs from "node:fs";
import path from "node:path";
import { deflateRawSync, inflateRawSync } from "node:zlib";
import { ROOT, git, identity, main, sourceText, write } from "./common.mjs";
import { version } from "./version.mjs";

// ZIP record signatures
const LOCAL_FILE = 0x04034b50;
const CENTRAL_FILE = 0x02014b50;
const END_OF_ARCHIVE = 0x06054b50;
const LOCAL_HEADER_SIZE = 30;
const CENTRAL_HEADER_SIZE = 46;
const END_RECORD_SIZE = 22;

const VERSION_NEEDED = 20; // 2.0, for deflate
const VERSION_MADE_BY_UNIX = 0x0314; // "made on Unix", so readers trust the mode below
const DEFLATE = 8;
const FIXED_DATE = 33; // 1980-01-01, the earliest; with a time of 0 the zip is the same on every run
const REGULAR_FILE = 0o100000;

// the files after the binary: the name in the zip, and the file in this repo it comes from
const SHIPPED_FILES = [
	["LICENSE", "LICENSE"],
	["LICENSE-GPL-3.0.txt", "licenses/GPL-3.0.txt"],
	["NOTICE", "NOTICE"],
	["THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"],
];

export function crc32(bytes) {
	let crc = 0xffffffff;
	for (const byte of bytes) {
		crc ^= byte;
		for (let bit = 0; bit < 8; bit++) {
			crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
		}
	}
	return (crc ^ 0xffffffff) >>> 0;
}

function localHeader(filename, crc, compressedSize, size) {
	const header = Buffer.alloc(LOCAL_HEADER_SIZE);
	header.writeUInt32LE(LOCAL_FILE, 0);
	header.writeUInt16LE(VERSION_NEEDED, 4);
	header.writeUInt16LE(DEFLATE, 8);
	header.writeUInt16LE(FIXED_DATE, 12);
	header.writeUInt32LE(crc, 14);
	header.writeUInt32LE(compressedSize, 18);
	header.writeUInt32LE(size, 22);
	header.writeUInt16LE(filename.length, 26);
	return header;
}

/** The central directory entry for the file whose local header is `local`, starting at `offset`. */
function centralHeader(local, mode, offset) {
	const header = Buffer.alloc(CENTRAL_HEADER_SIZE);
	header.writeUInt32LE(CENTRAL_FILE, 0);
	header.writeUInt16LE(VERSION_MADE_BY_UNIX, 4);
	local.copy(header, 6, 4, LOCAL_HEADER_SIZE); // from "version needed" to the name length, the same fields
	header.writeUInt32LE(((REGULAR_FILE | mode) << 16) >>> 0, 38); // the Unix mode is in the top 16 bits
	header.writeUInt32LE(offset, 42);
	return header;
}

/** A zip of `entries` (`{ name, bytes, mode }`) in the given order. Same input, same bytes. */
export function zip(entries) {
	const localParts = [];
	const centralParts = [];
	let offset = 0;
	for (const { name, bytes, mode } of entries) {
		if (path.basename(name) !== name || !/^[\w.-]+$/.test(name)) {
			throw new Error(`archive entry ${name} must be a file name, not a path`);
		}
		const filename = Buffer.from(name);
		const data = deflateRawSync(bytes, { level: 9 });
		if (bytes.length > 0xffffffff || data.length > 0xffffffff) {
			throw new Error(`${name} is too large for a zip without ZIP64`);
		}
		const local = localHeader(filename, crc32(bytes), data.length, bytes.length);
		localParts.push(local, filename, data);
		centralParts.push(centralHeader(local, mode, offset), filename);
		offset += local.length + filename.length + data.length;
	}

	const directory = Buffer.concat(centralParts);
	const end = Buffer.alloc(END_RECORD_SIZE);
	end.writeUInt32LE(END_OF_ARCHIVE, 0);
	end.writeUInt16LE(entries.length, 8); // entries on this disk
	end.writeUInt16LE(entries.length, 10); // entries in all
	end.writeUInt32LE(directory.length, 12);
	end.writeUInt32LE(offset, 16); // where the directory starts: after the last local entry
	return Buffer.concat([...localParts, directory, end]);
}

/** Reads a zip made by [`zip`] back as `{ name, mode, offset, bytes }` entries, checking each checksum. */
export function inspect(bytes) {
	const end = bytes.length - END_RECORD_SIZE; // no archive comment
	if (bytes.readUInt32LE(end) !== END_OF_ARCHIVE) throw new Error("zip has no end record");
	const count = bytes.readUInt16LE(end + 10);
	let position = bytes.readUInt32LE(end + 16); // the start of the central directory

	const entries = [];
	for (let i = 0; i < count; i++) {
		if (bytes.readUInt32LE(position) !== CENTRAL_FILE) throw new Error("zip directory entry is invalid");
		const nameLength = bytes.readUInt16LE(position + 28);
		const extraLength = bytes.readUInt16LE(position + 30);
		const commentLength = bytes.readUInt16LE(position + 32);
		const name = bytes.subarray(position + CENTRAL_HEADER_SIZE, position + CENTRAL_HEADER_SIZE + nameLength).toString();
		const start = bytes.readUInt32LE(position + 42);
		if (bytes.readUInt32LE(start) !== LOCAL_FILE) throw new Error("zip entry is invalid");

		const dataStart = start + LOCAL_HEADER_SIZE + bytes.readUInt16LE(start + 26) + bytes.readUInt16LE(start + 28);
		const compressedSize = bytes.readUInt32LE(position + 20);
		const data = inflateRawSync(bytes.subarray(dataStart, dataStart + compressedSize));
		const sizeMatches = data.length === bytes.readUInt32LE(position + 24);
		if (!sizeMatches || crc32(data) !== bytes.readUInt32LE(position + 16)) throw new Error(`zip entry ${name} fails its checksum`);

		entries.push({ name, mode: bytes.readUInt32LE(position + 38) >>> 16, offset: start, bytes: data });
		position += CENTRAL_HEADER_SIZE + nameLength + extraLength + commentLength;
	}
	return entries;
}

/** Fails unless the binary is first and the only executable, as rokit, aftman and foreman need. */
function checkExecutableLayout(entries, exe) {
	const [first, ...others] = entries;
	if (first.name !== exe || first.offset !== 0) throw new Error(`the first zip entry must be ${exe}`);
	if ((first.mode & 0o777) !== 0o755) throw new Error(`${exe} must be stored with mode 0755`);
	for (const entry of others) {
		if (entry.mode & 0o111) throw new Error(`${entry.name} must not be executable`);
		if (entry.name.endsWith(".exe")) throw new Error(`${entry.name} must not end in .exe`);
	}
}

export function packageRelease(options, root = ROOT) {
	const manifestVersion = version(root, options.tag);
	if (options.version !== manifestVersion) throw new Error("package version differs from manifests");
	if (!/^(windows|linux|macos)$/.test(options.os) || !/^(x86_64|aarch64)$/.test(options.arch)) {
		throw new Error(`unsupported archive platform ${options.os} ${options.arch}`);
	}
	const info = identity(
		manifestVersion,
		options.tag ?? `v${manifestVersion}`,
		options.commit ?? git(["rev-parse", "HEAD"], root),
		options.repository,
	);

	const exe = options.os === "windows" ? "ror.exe" : "ror";
	const source = sourceText(info);
	const entries = [{ name: exe, bytes: fs.readFileSync(options.binary), mode: 0o755 }];
	for (const [name, file] of SHIPPED_FILES) {
		entries.push({ name, bytes: fs.readFileSync(path.join(root, file)), mode: 0o644 });
	}
	entries.push({ name: "THIRD_PARTY_LICENSES.txt", bytes: fs.readFileSync(options.licenses), mode: 0o644 });
	entries.push({ name: "SOURCE.md", bytes: Buffer.from(source), mode: 0o644 });

	// read the zip back before anything is written
	const bytes = zip(entries);
	checkExecutableLayout(inspect(bytes), exe);

	write(path.join(options.out, "SOURCE.md"), source);
	const output = path.join(options.out, `ror-${manifestVersion}-${options.os}-${options.arch}.zip`);
	write(output, bytes);
	return output;
}

main(import.meta.url, (options) => console.log(packageRelease(options)));
