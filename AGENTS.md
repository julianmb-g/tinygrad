# tinygrad Multi-Agent Constraints

### Execution & Architecture
* **.bss Obliteration & DTCM Allocation**: Main I/O buffers must be declared as file-scoped global arrays explicitly annotated with `__attribute__((section(".noinit")))` to prevent the `_start` phase from wiping host-streamed DMA weights.
* **Full INT8 Quantization & Late Dequantization**: `Int8Linear` must perform native integer vector ops (`x_int8.dot(w_int8)`) with scaling applied strictly *after* the dot product to preserve memory footprints.
* **Ops.EXPAND Shape Requirement**: Ensure `Ops.EXPAND` operations correctly handle shape expansion mismatches to avoid unhandled `RuntimeError` and `ValueError` crashes.
* **Python Line Length & Timeout Obfuscation**: Enforce strict 80-character line lengths to prevent the obfuscation of arbitrary hardware execution bounds (e.g., `timeout=15.0`) within massive one-liners.
* **Register Bounds Limits vs DTCM**: Explicitly cap unroll sizes at `max_upcast = 28` for the `CORALNPU` target to preserve at least 4KB of DTCM for C-stack overhead, preventing stack overflows.

### IPC & Test Runner Deadlocks
* **IPC Teardown Reversion**: Monkeypatching `pytest_sessionfinish` and `multiprocessing.shared_memory` to forcefully terminate worker groups causes premature OS-level deadlocks (e.g., `OSError: cannot send (already closed?)`) during `pytest-xdist` execution. Authentic Python IPC lifecycles must be respected without unauthorized `atexit` OS-level `SIGKILL` interventions. Do not blindly execute worker teardown monkeypatches unless explicitly authorized by SPECS.md.
* **Multiprocessing Spawn Method**: Enforcing `multiprocessing.set_start_method("spawn")` globally at module scope prevents `pytest-xdist` fork deadlocks and gracefully supports native IPC teardown.
* **Multiprocessing Teardown**: `pytest-xdist` workers crashing mid-suite indicate improper state or IPC teardown within the ML framework tests.
* **Suite Unmasking Audit**: Resolving the OSError worker crash mandates an immediate QA audit to evaluate the newly revealed test assertions.

### QA & Refactoring Constraints
* **Decoupled API Boundaries & Missing Standard Imports**: Cross-component refactoring must be strictly atomic. Modifying bindings or decoupling core modules (like `CompilerSet` from `tinygrad.device`) without updating all downstream callers causes catastrophic test collection failures. Always ensure standard library imports (e.g., `unittest`) are explicitly declared in test files.
* **Float8 & Compilation Crash Evasion**: Catching `FileNotFoundError` or `CalledProcessError` during Clang compilation and bypassing it via `unittest.SkipTest` (e.g., in `test_fp8_linear.py` or `test_gemma_decomposition.py`) masks unsupported compiler representations (`float8_e4m3`, `float8_e5m2fnuz`). These must be allowed to fail organically or explicitly skipped based on hardware capability, not exception swallowing.
* **NV Interface Global Swallowing**: Instantiating objects like `Tensor(dat, device="NV")` wrapped in a blanket `except Exception: raise SkipTest` silences real architectural crashes (AttributeError, TypeError) and is strictly forbidden.
* **Recursion Limit Masking**: Wrapping deep graph evaluations in `try...except RecursionError: raise SkipTest` instead of fixing AST bounds is mathematically erasing test failures and is prohibited.
* **Refactoring Void Trapping**: Aggressively removing imports like `run_schedule` and `unittest` without verifying downstream test file dependencies leads to catastrophic collection failures (100% test void). All refactoring must execute a full `pytest --collect-only` before committing.
* **Ops.EXPAND Symbolic Evaluation**: Use the `resolve()` helper to safely evaluate symbolic ranges (UOp) within `Ops.EXPAND` boolean matching, preventing `ValueError` crashes on unresolved vars.
* **Collection Unmasking Audit**: Resolving pytest collection failures (e.g., missing imports or API drift) will unmask massive underlying functional failures. A full QA triage is mandatory immediately after unblocking `pytest`.

### Knowledge Accumulation (Design Phase)
* **Cross-Compiler Testing Bound**: `tinygrad` tests (like `TestCoralNPURenderer`) MUST explicitly use the host `riscv64-unknown-elf-gcc` toolchain and Clang python bindings to parse the generated C++ AST, ensuring asynchronous DMA macros (`CORAL_DMA_ASYNC`) precede barriers.
* **IPC Isolation and Worker Lifecycle**: Tests such as `TestIpcWorkerPool` MUST evaluate true multiprocess lockstep execution using `multiprocessing.shared_memory`. Overarching OS signal injections are banned; worker execution timeout limits MUST trigger native graceful `TimeoutError` exceptions.
* **Exception Swallowing Prohibition**: Catching integration/compilation failures via generic `except (FileNotFoundError, subprocess.CalledProcessError)` and routing to `unittest.SkipTest` mathematically erases test failures and is explicitly forbidden.
- **API Drift**: Prevent API drift causing massive test failures. Maintain strict compiler boundaries.
- **E2E IPC and Compilation Boundaries**: Added on Wed Apr 1 2026. Do NOT catch `subprocess.CalledProcessError` or `FileNotFoundError` using `@unittest.skip`. Mocking compilation borders mathematically erases integration boundary failure.
- **Organic Hardware Traps**: Added on Wed Apr 1 2026. Do NOT use Python regex assertions (`assertRaisesRegex`) to bypass memory bounds traps. The simulator MUST dynamically compile and bounds-trap the generated `.elf` payload.
- **Testing Fraud Resolution**: Added on Wed Apr 1 2026. Cosmetic refactors that mask organic test failures with `SkipTest` must be forcefully reverted. Modifying an upstream test to skip execution subverts the pipeline.
