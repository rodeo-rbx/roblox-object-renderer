// Cargo starts this in place of ror's library test binary (the runner setting in .github/workflows/ci.yml). It hands
// the command line to test-shard.mjs and exits with its code. It is a file of its own so that the runner never has to
// check whether it was started as a script: a runner that skipped its work would exit 0 without running the test
// binary, and the job would pass with no test run.
import { run } from "./test-shard.mjs";

try {
	process.exitCode = run(process.argv.slice(2));
} catch (error) {
	console.error(error.message);
	process.exitCode = 1;
}
