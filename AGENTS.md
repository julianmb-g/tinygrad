# Tinygrad Orchestration Guidelines

## System Rules & Architectural Guidelines

### Build & Orchestration Execution
* **Test Invocation Routine**: Ensure all Python test scripts natively invoke `if __name__ == '__main__': unittest.main()`.

*(Global instructions regarding IPC Muzzling, OOM deadlocks, and Linker limits have been unified to the root AGENTS.md ledger).*

### DMA Coherency & Chunk Constraints
* **Hardware BSS Alignment**: C-Runtime `.bss` allocation must explicitly inject `__attribute__((section(".noinit")))` for hardware reserved zones.
* **Physical DTCM Limit**: The Tinygrad compiler must strictly enforce `AddrSpace.LOCAL` to 32KB, generating <= 12KB ping/pong chunks to accommodate `.bss` and stack overheads.
* **SLVERR Handling**: Ensure `WAIT_DMA_READY` evaluates the underlying AXI response code. If the transaction traps via `SLVERR`, it must assert a Machine Load Access Fault natively.
* **Split-K Dimension Boundaries Context**: Reductions along contiguous tensor axes exceeding the physical `CORALNPU_L1_LIMIT` (e.g., 12KB) must raise a fatal `OutOfMemoryError` early in heuristic hand-coded optimizations when split-k is not supported.
* **String-Matched Exception Muzzling**: Masking OS errors with substring matching (e.g., `"already closed"`, `"cannot send"`) instead of architecting an authentic GC lifecycle is testing fraud.
* **conftest.py Race Conditions**: Global exception muzzling specifically in `conftest.py` for `Connection.send` masks shared memory teardown race conditions. Use explicit teardowns.
* **pytest-xdist Teardown Freezes**: Require targeted exception handling around `os.kill`/`os.unlink` to prevent main thread freezing during `pytest-xdist` teardown.
\n* **Test Watchdogs**: Artificial timeouts in subprocess commands (e.g. `timeout=15.0`) must not be used on execution/cross-compiled tests like `test_dev_var.py` and `compiler_amd.py`, as they illegally mask organic structural execution hangs.
