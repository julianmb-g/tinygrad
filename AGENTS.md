# Tinygrad Agent Instructions

## Lessons Learned
- **Architecture Update:** Added a dynamic `coralnpu` backend. It compiles UOps into GCC RISC-V Zve32x code, overriding the default execution by simulating out-of-band execution using an embedded neural network cost model (`BEAM_COST`) for auto-tuning.
