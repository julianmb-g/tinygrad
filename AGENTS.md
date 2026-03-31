# Tinygrad Submodule Orchestration Guidelines (AGENTS.md)

## Lessons Learned
* **IPC Garbage Collection Traps**: Enforce targeted OS exception handling (`FileNotFoundError`, `ProcessLookupError`) in `__del__` GC lifecycles instead of `OSError` to prevent shared memory deadlocks during multiprocessing teardowns.
* [FLAG: invalid] **Pytest Teardown Sanitization**: Ensure explicit `pkill` boundaries exist in `pytest_sessionfinish` to forcefully clean up orphaned execution processes.
* **.bss Obliteration & DTCM Allocation**: Main I/O buffers must be declared as file-scoped global arrays explicitly annotated with `__attribute__((section(".noinit")))` to prevent the `_start` phase from wiping host-streamed DMA weights.
* **Full INT8 Quantization & Late Dequantization**: `Int8Linear` must perform native integer vector ops (`x_int8.dot(w_int8)`) with scaling applied strictly *after* the dot product to preserve memory footprints.
* **Register Bounds Limits vs DTCM**: Explicitly cap unroll sizes at `max_upcast = 28` for the `CORALNPU` target to preserve at least 4KB of DTCM for C-stack overhead, preventing stack overflows.
* **Python Line Length & Timeout Obfuscation**: Enforce strict 80-character line lengths to prevent the obfuscation of arbitrary hardware execution bounds (e.g., `timeout=15.0`) within massive one-liners.
* **IPC Authentic Testing**: Ensure that IPC worker communication tests allocate genuine `shared_memory.SharedMemory` objects and simulate authentic worker payloads, rather than asserting on mock primitive values.
