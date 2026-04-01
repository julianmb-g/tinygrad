# tinygrad Multi-Agent Constraints

### Execution & Architecture
* **.bss Obliteration & DTCM Allocation**: Main I/O buffers must be declared as file-scoped global arrays explicitly annotated with `__attribute__((section(".noinit")))` to prevent the `_start` phase from wiping host-streamed DMA weights.
* **Fake ELF Generation**: Do not utilize `create_dummy_elf` to manually write raw byte structures, bypassing authentic binary evaluation in the execution engine.
* **Full INT8 Quantization & Late Dequantization**: `Int8Linear` must perform native integer vector ops (`x_int8.dot(w_int8)`) with scaling applied strictly *after* the dot product to preserve memory footprints.
* **Ops.EXPAND Shape Requirement**: Ensure `Ops.EXPAND` operations correctly handle shape expansion mismatches to avoid unhandled `RuntimeError` and `ValueError` crashes.
* **Ops.EXPAND Symbolic Evaluation**: Use the `resolve()` helper to safely evaluate symbolic ranges (UOp) within `Ops.EXPAND` boolean matching, preventing `ValueError` crashes on unresolved vars.
* **Organic Hardware Traps**: Do NOT use Python regex assertions (`assertRaisesRegex`) to bypass memory bounds traps. The simulator MUST dynamically compile and bounds-trap the generated `.elf` payload.
* **Python Line Length & Timeout Obfuscation**: Enforce strict 80-character line lengths to prevent the obfuscation of arbitrary hardware execution bounds (e.g., `timeout=15.0`) within massive one-liners.
* **Register Bounds Limits vs DTCM**: Explicitly cap unroll sizes at `max_upcast = 28` for the `CORALNPU` target to preserve at least 4KB of DTCM for C-stack overhead, preventing stack overflows.

### IPC & Test Runner Deadlocks
* **IPC Isolation and Worker Lifecycle**: Tests such as `TestIpcWorkerPool` MUST evaluate true multiprocess lockstep execution using `multiprocessing.shared_memory`. Overarching OS signal injections are banned; worker execution timeout limits MUST trigger native graceful `TimeoutError` exceptions.
* **IPC Teardown Reversion**: Monkeypatching `pytest_sessionfinish` and `multiprocessing.shared_memory` to forcefully terminate worker groups causes premature OS-level deadlocks (e.g., `OSError: cannot send (already closed?)`) during `pytest-xdist` execution. Authentic Python IPC lifecycles must be respected without unauthorized `atexit` OS-level `SIGKILL` interventions. Do not blindly execute worker teardown monkeypatches unless explicitly authorized by SPECS.md.
* **Multiprocessing Spawn Method**: Enforcing `multiprocessing.set_start_method("spawn")` globally at module scope prevents `pytest-xdist` fork deadlocks and gracefully supports native IPC teardown.
* **Multiprocessing Teardown**: `pytest-xdist` workers crashing mid-suite indicate improper state or IPC teardown within the ML framework tests.
* **Suite Unmasking Audit**: Resolving the OSError worker crash mandates an immediate QA audit to evaluate the newly revealed test assertions.

### QA & Refactoring Constraints
* **Collection Unmasking Audit**: Resolving pytest collection failures (e.g., missing imports or API drift) will unmask massive underlying functional failures. A full QA triage is mandatory immediately after unblocking `pytest`.
* **Cross-Compiler Testing Bound**: `tinygrad` tests (like `TestCoralNPURenderer`) MUST explicitly use the host `riscv64-unknown-elf-gcc` toolchain and Clang python bindings to parse the generated C++ AST, ensuring asynchronous DMA macros (`CORAL_DMA_ASYNC`) precede barriers.
* **Decoupled API Boundaries & Missing Standard Imports**: Cross-component refactoring must be strictly atomic. Modifying bindings or decoupling core modules (like `CompilerSet` from `tinygrad.device`) without updating all downstream callers causes catastrophic test collection failures. Always ensure standard library imports (e.g., `unittest`) are explicitly declared in test files. Prevent API drift causing massive test failures; maintain strict compiler boundaries. Ensure code correctly interacts with `dtypes.weakint` and `dtypes.index` changes without throwing unhandled exceptions.
* **Exception Swallowing Prohibition**: Catching integration/compilation failures via generic `except (FileNotFoundError, subprocess.CalledProcessError)` and routing to `unittest.SkipTest` mathematically erases test failures and is explicitly forbidden. Float8 compilation errors (`float8_e4m3`, `float8_e5m2fnuz`) must be allowed to fail organically or skipped properly based on hardware capability, not exception swallowing. Cosmetic refactors that mask organic test failures with `SkipTest` must be forcefully reverted. Modifying an upstream test to skip execution subverts the pipeline. Do NOT catch `subprocess.CalledProcessError` or `FileNotFoundError` using `@unittest.skip`. Mocking compilation borders mathematically erases integration boundary failure.
* **NV Interface Global Swallowing**: Instantiating objects like `Tensor(dat, device="NV")` wrapped in a blanket `except Exception: raise SkipTest` silences real architectural crashes (AttributeError, TypeError) and is strictly forbidden.
* **Recursion Limit Masking**: Wrapping deep graph evaluations in `try...except RecursionError: raise SkipTest` instead of fixing AST bounds is mathematically erasing test failures and is prohibited.
* **Refactoring Void Trapping**: Aggressively removing imports like `run_schedule` and `unittest` without verifying downstream test file dependencies leads to catastrophic collection failures (100% test void). All refactoring must execute a full `pytest --collect-only` before committing.
## Lessons Learned
* **Upstream Synchronization Priority:** Upstream changes (e.g. CUDARenderer refactoring) can break local dependencies. JIT Upstream sync tasks must be scheduled atomically per-submodule when divergent.
* **Mocked Fetching Prevention**: Do not isolate logic by instantiating `MockResponse(img_bytes)` via `@patch('urllib.request.urlopen')` to fake image fetching. Authentic E2E validations proving cross-component fetch viability are required.
* **IPC Worker Mocking Ban**: Parallel execution state MUST NOT be mocked with purely Python-based dummy workers. Tests must schedule authentic cross-compiled compute kernels to accurately measure shared memory teardown boundaries natively.
* **Configuration Obsolescence**: Tests failing due to `CPU=1` deprecation mask authentic architecture faults. Ensure `DEV=CPU` and current API targets are strictly preserved in orchestration configs to avoid 100% test failure loops.

## IPC Worker E2E Mocking Illusion
Exercising Python SharedMemory primitives with mocked tasks (e.g., numpy array doubling) fails to prove NPU execution. Tests must schedule authentic cross-compiled compute kernels to accurately measure shared memory teardown boundaries natively.

## Global Evasion via Toolchain Bypassing
Aggressively catching missing cross-compiler errors (like `FileNotFoundError`) to skip tests mathematically erases architectural test failures and evades E2E constraints.

## Compiler & Execution Bounds
* **Unaligned DMA Evasion**: The ML compiler MUST strictly assert that all `GLOBAL` memory pointers passed to DMA copy intrinsics are aligned to the exact AXI bus width prior to compilation to prevent bus faults.
* **Beam Search False-Positive Evasion**: The Python simulation wrapper MUST evaluate semantic correctness BEFORE accepting the `mcycle` cost to prevent poisoning the optimizer with zero-cost early exits.
* **Pytest Fatal Watchdog Termination & Debugging**: When the CI pipeline is brutally terminated by a global timeout watchdog mid-execution (e.g., 120s limit), `pytest` will not emit its standard failure summary or stack traces. To diagnose explicit test failures (`F`) that occur before the timeout, developers MUST run the suite with fail-fast enabled (`pytest -x`) or disable parallel workers (`-n 0`) so the test runner organically halts and prints the exact stack trace of the first failure.
