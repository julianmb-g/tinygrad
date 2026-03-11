# Tinygrad Agent Instructions

## Lessons Learned
- **Architecture Update:** Added a dynamic `coralnpu` backend. It compiles UOps into GCC RISC-V Zve32x code, overriding the default execution by simulating out-of-band execution using an embedded neural network cost model (`BEAM_COST`) for auto-tuning.
- **Multi-Agent Authorization:** To fix multi-agent authorization issues, we now use HTTPS instead of SSH for `.gitmodules` URLs.
- **Merge Constraints:** We must merge *local* feature branches instead of remote tracking branches (e.g., `git merge coralnpu-dev` instead of `git merge origin/coralnpu-dev`).
- **Diff Sanity Checks:** A strict requirement is to perform a 'Final Diff Sanity Check' after submodule merges to ensure no logic was lost during conflict resolution.
- **Upstream Integration:** Feature branch integration successfully validated. No logic was lost during conflict resolution.
- **Submodule Pointer Synchronization:** Always ensure that subsequent commits made to the submodule (e.g., adding localized lessons to this file) are immediately followed by serializing the new submodule pointer in the parent workspace to prevent state drift.
- **Test Integrity & Systemic Gaps:** The test suite currently has a complete void regarding the `coralnpu` backend (no unit tests for `ops_coralnpu.py` or fallback GCC logic). Furthermore, do not simplify symbolic math parameters or strip `before` assertions in memory tests to force a passing build. Avoid 'happy-path only' coverage and silent CI degradations (such as catching generic Exceptions to skip tests).
