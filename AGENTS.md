# Tinygrad Agent Instructions

## Lessons Learned

### Architecture & Integration
- **Architecture Update:** Added a dynamic `coralnpu` backend. It compiles UOps into GCC RISC-V Zve32x code, overriding the default execution by simulating out-of-band execution using an embedded neural network cost model (`BEAM_COST`) for auto-tuning.
- **Submodule Pointer Synchronization:** Always ensure that subsequent commits made to the submodule (e.g., adding localized lessons to this file) are immediately followed by serializing the new submodule pointer in the parent workspace to prevent state drift.

### Git & Environment Management
- **Multi-Agent Authorization:** To fix multi-agent authorization issues, we now use HTTPS instead of SSH for `.gitmodules` URLs.
- **Merge Constraints:** We must merge *local* feature branches instead of remote tracking branches (e.g., `git merge coralnpu-dev` instead of `git merge origin/coralnpu-dev`).
- **Diff Sanity Checks:** A strict requirement is to perform a 'Final Diff Sanity Check' after submodule merges to ensure no logic was lost during conflict resolution.

### Testing & Verification
- **Test Integrity & Systemic Gaps:** The test suite currently has a complete void regarding the `coralnpu` backend (no unit tests for `ops_coralnpu.py` or fallback GCC logic). Furthermore, do not simplify symbolic math parameters or strip `before` assertions in memory tests to force a passing build. Avoid 'happy-path only' coverage and silent CI degradations (such as catching generic Exceptions to skip tests).
- **IPC Reliability:** Prevent silent timeout deadlocks by adding aggressive timeout fallbacks for out-of-band IPC executions (e.g., Bazel).

### Code Quality
- **Code Quality:** Avoid magic variables like `BEAM_ENFORCE_BASELINE` without documentation. Do not fall back to native gcc when cross-compiling fails.

## Lessons Learned
- **Backend Testing & Verification Void:** The `coralnpu` backend lacks unit testing in `tinygrad/test/`. Testing must cover `ops_coralnpu.py` (e.g., verifying allocator integrity, BEAM cost parsing, and explicitly testing cross-compiler GCC fallbacks bug prevention) and `renderer/coralnpu.py` (e.g., validating AST pattern matchers, extracting features, analytical cost fallbacks, code generation, and float allocation caps).
- **Test Integrity:** Prevent "happy-path only" tests (e.g., restoring `before` state capture in `test_invalid_tensor.py`), stripping math checks to force green builds (e.g., in `test_uop_symbolic.py`), and masking degradation with broad exception swallowing (e.g., in `test_handwritten.py`).
