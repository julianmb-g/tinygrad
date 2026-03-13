# tinygrad Agent Instructions

## Lessons Learned

### Architecture Quirks
- **Architecture Update:** Added a dynamic `coralnpu` backend compiling UOps into GCC RISC-V Zve32x code. Simulates out-of-band execution using an embedded neural network cost model (`BEAM_COST`) for auto-tuning.
- **AST Limits:** Enforce `max_upcast = 32` in the custom renderer (e.g. `CoralNPURenderer`) to prevent massive AST explosion during auto-vectorization passes.
- **Dequantization FPU Dependency:** Dequantizing via FP16 scaling factors after a VDOT reduction loop still requires floating-point operations. If the hardware lacks an FPU, this will trigger an illegal instruction trap.
- **DTCM Double-Buffering Mathematical Overflow:** When tiling loops, the limit (e.g., 12KB) must apply to the *sum* of all live `LOCAL` tensors to accommodate the stack overhead and prevent DTCM capacity overflow.
- **The Phantom DMA Engine Paradox:** An in-order scalar core cannot interleave an AXI `memcpy` loop with Vector ALU execution without a dedicated autonomous DMA controller. Memory fetch UOp passes must be strictly synchronous unless hardware DMA exists.
- **VMM Heap Base Address Collision:** The Virtual Memory Manager cannot blindly assume a base address of `0x80000000`. It must dynamically parse the `.elf` symbol table to find the `_end` of static allocations or risk overwriting `.data` and `.bss` segments.
- **W8A8 Activation Quantization Void:** Before issuing an INT8 vector MAC operation (`VDOT`), the incoming `fp16` activation tensors MUST be dynamically quantized; issuing `x_int8.dot(w_int8)` directly onto unquantized streams produces mathematical garbage.

### Build Dependencies
- **Hardcoded Executable Path Coupling:** Extract cross-compiler paths into configurable attributes rather than hardcoding to maintain hermetic Bazel execution.
- **Test Dependencies:** Ensure `huggingface_hub`, `pytest-xdist`, and `pytest-timeout` are installed.

### Git & Environment Management
- **Bazel Test Runfiles vs. Submodule Sync:** When running `bazel test` with an external repository override for an out-of-band submodule, ensure its `BUILD` file correctly exports the updated module structures or dependencies. Missing overrides cause `ModuleNotFoundError`.
- **Integration Dependencies & Overrides:** Beware of Bazel caching `(cached) PASSED` falsely reporting success. Use `--nocache_test_results`. Test targets depending on dynamically updated submodules require explicit overrides.
- **Makefile vs Bazel:** Do not assume `make test` is valid for Bazel-centric repositories. Ensure execution scripts explicitly use Bazel targets.
- **Test-Ignorant Pipeline Deadlock (Hypothesis DeadlineExceeded):** When executing baseline test suites, inherently flaky tests that use property-testing frameworks can mathematically fail an upstream baseline sync due to sandbox latency. The test execution must inject a pre-baseline remediation step to exclude them.
- **Upstream Validation:** Use `git worktree add ../tinygrad-baseline upstream/master` to test upstream state without altering the current tracking branches. Ensure `git worktree prune` and `rm -rf` are executed on the target `/tmp/` directory prior to initialization.
- **Zero-Trust Baseline Synchronization Safeguards:** Always mandate `git status` and `git stash` inside the submodule prior to invoking baseline extraction or rebasing logic.

### Testing Gotchas
- **Analytical Cost Modeling Data:** Analytical validation tests MUST NOT excessively mock stochastic variance (`random.gauss`) or replace zeros with fake "happy-path" `np.ones` padding. Use deterministic golden tensors.
- **Architecture Update (Integration of ML to HW):** The ecosystem supports an end-to-end integration path. Tinygrad compiles to GCC RISC-V Zve32x C++, verified against the `mpact` instruction simulator (with Map-Elites) and `coralnpu` hardware co-simulation.
- **AST Simplification Fidelity:** Tests MUST mathematically assert the fully simplified optimal AST representation (e.g., `(b+a*15)`) instead of hardcoding bloated asymmetric fold defects. Do not hardcode mathematically bloated compiler outputs just to achieve a passing test.
- **Autogen Dependencies (`libclang`):** When integrating or testing `tinygrad` or any autogen modules relying on `libclang`, ensure `LIBCLANG_PATH` is correctly exported (e.g., `LIBCLANG_PATH=/usr/lib/llvm-19/lib/libclang.so`) in the container environment or the required LLVM package is installed to avoid test masking.
- **CoralNPURenderer Unit Testing:** When testing AST limit boundaries (e.g. `RuntimeError` due to >32 float allocations), nodes must have identical depth constraints. Chaining dependents results in scalarized depths preventing the max-width check from exceeding limits. Use `unittest.mock.patch` to sidestep strict file constraints for missing ML models during environment-agnostic tests.
- **CPU-Only Backend Enforcement:** To prevent ROCm/GPU compiler crashes in the orchestrator container, tests must forcefully use the CPU backend and ignore hardware-specific folders using `CPU=1 pytest --timeout=120 --timeout-method=thread test/backend/ test/unit/ test/null/ test/test_tiny.py`.
- **Deterministic Mathematical Verification & Zero-Divisors:** Do not rely on unhandled zero-tensors and `Inf`/`NaN` comparisons (e.g. `RuntimeWarning: divide by zero`) during `assert_allclose`. This serves as a vector for fraudulent 'happy-path' assertions. Tests verifying complex neural networks or JIT compilation MUST explicitly forbid passing validations based on these.
- **Fake Data Execution Padding:** Replacing `np.zeros` padding with deterministic Gaussian noise (`rng.randn`) does not resolve "happy-path" testing. True deterministic golden weights must be used and exact mathematical outputs strictly asserted. Inference scripts must not use random data padding.
- **Global State Mutation in Tests:** Use `unittest.mock.patch.dict` for `os.environ` modifications to avoid race conditions during parallel testing (`pytest -n auto`). Do not mutate os.environ["PATH"] globally.
- **Hardware Backend Testing:** The `coralnpu` rendering and runtime backend must have dedicated unit tests inside the ML frontend (`tinygrad/test/`), rather than blindly relying on out-of-band simulator validation.
- **Hardware Co-Simulation Flow:** The hardware integration spans from `tinygrad` C++ kernels loaded as RISC-V ELF binaries into the `mpact` ISS to `cocotb` testbenches orchestrating Verilog/SystemVerilog RTL.
- **IPC Reliability & Timeouts:** Prevent silent timeout deadlocks by adding aggressive timeout fallbacks for out-of-band IPC executions. Use `PYTHONUNBUFFERED=1 pytest -n auto --timeout=120 --timeout-method=thread` to force thread dumps on timeout. Parallel execution (`-n auto`) is strictly required.
- **IPC Watchdog Testing:** When testing execution watchdogs, do not bypass boundaries using `subprocess.run` mocks. Organically trigger `TimeoutError` paths by creating dummy out-of-band executables (e.g., infinite sleep script) and temporarily injecting them into the `PATH`.
- **JIT Timeouts:** Property testing MUST implement explicit timeouts. Disabling deadlines completely via `@settings(deadline=None)` creates silent infinite loop masking vulnerabilities and masks flaky CI failures.
- **Missing ROCm Device Library**: In environments lacking the ROCm device library, AMD handwritten tests (like `test_mfma_fp8`) must be explicitly skipped using `@unittest.skip` to prevent CI failures.
- **ML Cost Validation:** Tests relying on ML cost model assertions must NOT filter by `cost > 0`. Because stochastic noise was replaced by deterministic outputs, the predicted cost for models can legitimately evaluate to exactly `0.0`.
- **Mock Test Pruning:** When updating methods to explicitly raise `RuntimeError` (e.g., prohibiting host compilation), ensure any legacy tests that mock these methods (like `subprocess.check_call`) are deleted.
- **Test Fidelity on Missing Dependencies**: Tests explicitly targeting specific LLVM CPU targets or missing device libraries should use `@unittest.skip` instead of generic exception swallowing.
- **Test Suite Granularity:** Comprehensive test creation tasks must be explicitly decomposed into distinct, atomic test creation tasks.
- **Test Suite Masking (tinygrad):** A 'passing' test suite does not guarantee full validation if a massive amount of tests are skipped. Ensure skipped tests are analyzed to avoid masking true architectural failure surface areas.
- **Test Suite Verification & Integrity:** Tests should not pass by blindly relying on string printing (e.g., `uart_puts("PASS")`). Structural architectural state must be verified.
- **The $TEST_TMPDIR Inode Exhaustion Leak:** `mmap.close()` does not delete the file. Mandate explicit `os.unlink(filepath)` for shared files immediately after subprocess termination to prevent inode exhaustion.
- **Zero-Divisor Resolution via Offset:** When replacing standard normal distributions with uniform/positive-only distributions to prevent zero-divisors, apply a constant offset (like `Tensor.rand() + 1.0`) to guarantee mathematical stability during `assert_allclose`.

### Miscellaneous
- **Inline Imports Anti-Pattern:** Relocate all standard library imports to the top of the file to prevent redundant scope evaluation.
- **Process Group Suicide (SIGKILL):** When a python orchestrator needs to SIGKILL subprocesses, the `multiprocessing` worker must be explicitly spawned with a distinct process group (`preexec_fn=os.setpgrp`).
- **The .noinit Symbol Resolution Black Hole:** If variables are `static` or mangled, the Host's Python ELF loader has zero visibility. The compiler must emit an explicit, predictable symbol map or define a fixed physical anchor point via a linker script.
- **Dynamic VMM Allocation Constraint:** The virtual memory allocator must strictly forbid hardcoded `0x80000000` mapping baselines. It MUST dynamically parse `.elf` bounds and accurately constrain VMM assignments against the true 32KB hardware limits without regressions.
# Lessons Learned

## JIT Latency and Environment Masking
- **Property Test Timeouts:** JIT compilation latency regressions that trigger property test timeouts (e.g., Hypothesis `DeadlineExceeded`) must never be bypassed by removing the explicit time bound (`deadline=None`). The latency bound is critical to preventing infinite loop masks and must either be explicitly scaled to accommodate the sandbox or the underlying regression fixed.
- **Environment Masking:** IPC Watchdog tests must not be bypassed using trivial test skipping (e.g., `@unittest.skipIf`) on nodes lacking cross-compilers. They must be organically evaluated using injected dummy out-of-band executables to guarantee full-fidelity execution.
- **Asymmetric AST Simplification:** Compilation defects producing mathematically unsimplified AST representations must not be "fixed" by hardcoding the bloated AST into test expectations. Tests must strictly assert the optimal, simplified representation to force compiler correction.

- **Asymmetric AST Simplification Logic**: When recombining `IDIV` and `MOD` operations within factored `Ops.ADD` UOps, the symbolic simplifier must evaluate `(base // div).simplify()` and search the expanded additive terms rather than naively checking `q.src[0] is base`. Constant term factoring causes equivalent operations (e.g., `(a*5+b)//10`) to simplify to unrelated AST branches (e.g., `(a+b//5)//2`).
