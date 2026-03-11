# Tinygrad Agent Instructions

## Lessons Learned
- **Architecture Update:** Added a dynamic `coralnpu` backend. It compiles UOps into GCC RISC-V Zve32x code, overriding the default execution by simulating out-of-band execution using an embedded neural network cost model (`BEAM_COST`) for auto-tuning.
- **Multi-Agent Authorization:** To fix multi-agent authorization issues, we now use HTTPS instead of SSH for `.gitmodules` URLs.
- **Merge Constraints:** We must merge *local* feature branches instead of remote tracking branches (e.g., `git merge coralnpu-dev` instead of `git merge origin/coralnpu-dev`).
- **Diff Sanity Checks:** A strict requirement is to perform a 'Final Diff Sanity Check' after submodule merges to ensure no logic was lost during conflict resolution.
- **Upstream Integration:** Feature branch integration successfully validated. No logic was lost during conflict resolution.
