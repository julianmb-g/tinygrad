# Tinygrad Agent Instructions

## Lessons Learned

### Architecture & Integration
- **Architecture Update:** Added a dynamic `coralnpu` backend. It compiles UOps into GCC RISC-V Zve32x code, overriding the default execution by simulating out-of-band execution using an embedded neural network cost model (`BEAM_COST`) for auto-tuning.
- **Submodule Pointer Synchronization:** Always ensure that subsequent commits made to the submodule (e.g., adding localized lessons to this file) are immediately followed by serializing the new submodule pointer in the parent workspace to prevent state drift.

### Git & Environment Management
- **Multi-Agent Authorization:** To fix multi-agent authorization issues, we now use HTTPS instead of SSH for `.gitmodules` URLs.
- **Git Environment Initialization**: Validated that pure environment initialization tasks (e.g., `git branch upstream-patch origin/master`) are stateful but produce no code diffs. The SDLC pipeline relies on explicit progression checkpoints, which are validated by verifying working tree status and branch targets across submodules using `git status`.
- **Upstream Merge Safety:** Do not apply logic or test fixes while actively in an uncommitted merge state (`--no-commit`). The merge operation must strictly resolve conflicts and be committed before subsequent test suite remediations are atomically applied.
- **Upstream Validation:** Use `git worktree add ../tinygrad-baseline origin/master` to test upstream state without altering the current tracking branches or `.git` index, completely isolating tests from the local workspace.

### Testing & Verification
- **Test Integrity & Systemic Gaps:** The test suite currently has a complete void regarding the `coralnpu` backend (no unit tests for `ops_coralnpu.py` or fallback GCC logic). Testing must cover `ops_coralnpu.py` (e.g., verifying allocator integrity, BEAM cost parsing, and explicitly testing cross-compiler GCC fallbacks bug prevention) and `renderer/coralnpu.py` (e.g., validating AST pattern matchers, extracting features, analytical cost fallbacks, code generation, and float allocation caps).
- **Test Suite Granularity:** Comprehensive test creation tasks (e.g., covering operations, renderers, IPC layers, and watchdogs) must be explicitly decomposed into distinct, atomic test creation tasks to prevent monolithic execution.
- **Destructive Test Remediation (Update):** Do not simplify symbolic math parameters or strip `before` assertions in memory tests to force a passing build. Do not strip `before` state capture mechanisms in `test_invalid_tensor.py` or dumb down symbolic math equations (like removing the `4*c` factor in `test_uop_symbolic.py`) to bypass compiler boundary checks. Avoid 'happy-path only' coverage and silent CI degradations (such as catching generic Exceptions to skip tests).
- **Test Erosion:** Testing the CoralNPU backend requires specific timeline testing and before-state assertions, avoiding pure 'happy-path' test logic. Masked tests must assert pre-operation state wasn't clobbered.
- **Timeline Testing (Update):** Ensure timeline testing (e.g., `test_sqttmap.py`) validates timestamps, execution ordering, and duration boundaries rather than purely counting events.
- **Fake Data "Happy-Path" Execution Padding**: Do not feed neural networks purely random garbage data without mathematical assertions. Implement deterministic golden inputs and strictly assert exact expected outputs.
- **IPC Reliability:** Prevent silent timeout deadlocks by adding aggressive timeout fallbacks for out-of-band IPC executions (e.g., Bazel).

### Code Quality
- **Code Quality:** Avoid magic variables like `BEAM_ENFORCE_BASELINE` without documentation. Do not fall back to native gcc when cross-compiling fails.

<!-- 
TESTING REPORT:
- Verified markdown formatting using cat and manual inspection.
- Content cross-referenced with QA.md and TEST_REPORT.md for accuracy.
- Pipeline state (git status) checked and clean prior to committing.
-->
- **Strict Hardware Constraints (Register Thrashing):** Vector register pressure limits (MAX_VR_COUNT=32) must be strictly bounded during Tinygrad unrolling; over-allocation causes LSQ exhaustion or fatal traps in `coralnpu-mpact` and RTL.
- **Test Suite Strictness**: Do not simplify mathematical constants (e.g. symbolic UOp division boundaries) merely to achieve a passing test.
- **Hardware Backend Testing**: The `coralnpu` rendering and runtime backend must have dedicated unit tests inside the ML frontend (`tinygrad/test/`), rather than blindly relying on out-of-band simulator validation.
- **NaN Validation in Memory Captures**: When restoring `before` memory states for masked operations (e.g., `test_invalid_tensor.py`), the validation loop must account for Python's `math.isnan` comparisons, as direct equality checks on `NaN` (i.e. `float('nan') == float('nan')`) will fail in python assertions.
