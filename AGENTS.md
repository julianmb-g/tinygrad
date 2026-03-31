# Tinygrad Orchestration Guidelines

## System Rules & Architectural Guidelines

### Build & Orchestration Execution
* **Test Invocation Routine**: Ensure all Python test scripts natively invoke `if __name__ == '__main__': unittest.main()`.

*(Global instructions regarding IPC Muzzling, OOM deadlocks, and Linker limits have been unified to the root AGENTS.md ledger).*

### DMA Coherency & Chunk Constraints
* **Physical DTCM Limit**: The Tinygrad compiler must strictly enforce `AddrSpace.LOCAL` to 32KB, generating <= 12KB ping/pong chunks to accommodate `.bss` and stack overheads.
* **SLVERR Handling**: Ensure `WAIT_DMA_READY` evaluates the underlying AXI response code. If the transaction traps via `SLVERR`, it must assert a Machine Load Access Fault natively.
* **Host Compiler Compatibility**: Always wrap RISC-V specific inline assembly (like `_start` stubs) with `#ifdef __riscv` to ensure `g++` on x86 test harnesses do not crash.
* **IPC Teardown**: `tinygrad` OSError requires targeted exception handling around `os.kill`/`os.unlink` to prevent main thread freezing.
* **Split-K Dimension Boundaries**: Reductions along contiguous tensor axes exceeding the physical `CORALNPU_L1_LIMIT` (e.g., 12KB) must raise a fatal `OutOfMemoryError` early in heuristic hand-coded optimizations when split-k is not supported. This avoids undefined runtime chunking overflow during codegen.
# tinygrad Orchestration Guidelines & Lessons Learned

* **IPC Teardown Deadlocks**: Do not use global exception muzzling (`except OSError: pass`) in `conftest.py` for `Connection.send`. This masks shared memory teardown race conditions. Use explicit teardowns.
* **Tensor Dimension Splitting**: Trap unsplittable tensor dimension bounds (`OutOfMemoryError`) to prevent chunking overflow on restricted NPU limits (e.g., 12KB `CORALNPU_L1_LIMIT`).
* **Hardware BSS Alignment**: C-Runtime `.bss` allocation must explicitly inject `__attribute__((section(".noinit")))` for hardware reserved zones.
