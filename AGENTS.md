# tinygrad Agent Instructions

## Lessons Learned

### Architecture Quirks
- **.bss Section Obliteration Flaw:** When using out-of-band IPC to stream host tensors before boot, explicitly declare main I/O arrays as file-scoped global arrays tagged with `__attribute__((section(".noinit")))` to prevent the C runtime `_start` sequence from zeroing them out.
- **DTCM Tiling Limits & Double-Buffering Overflows:** Always enforce the 12KB DTCM limit in the renderer by explicitly verifying the sum of `Ops.DEFINE_LOCAL` buffer sizes. When tiling loops, chunks must be strictly scaled to accommodate stack overhead (e.g., <= 12KB chunks, not 16KB). The limit applies to the sum of all live `LOCAL` tensors to prevent runtime memory overflow, and lower DMA via `Ops.COPY` to asynchronous hardware macros like `CORAL_DMA_ASYNC`.
- **Analytical Model Trivialization:** When testing cost estimators (e.g., in `test_coralnpu_renderer.py`), tests must use a deterministic, valid `_cost_model` dictionary that tests the actual numerical bounds (like `0.1` softplus activation bounds) rather than trivializing the test by stripping parameters and asserting `cost >= 0.0` with a static 0.0.
