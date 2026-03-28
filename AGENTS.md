# Tinygrad Execution Context & Lessons

## Lessons Learned

### Tier 1: Build & Orchestration

* **Quote:** "Do not catch `subprocess.CalledProcessError` in test suites to bypass or skip failing tests."
* **Impact:** Catching these errors mathematically erases invalid cross-component execution bugs from the CI runner, presenting a false "green" build.
* **Action:** Allow underlying C/C++ compiler syntax or compilation faults to organically fail the test. Only expected missing environment dependencies (like `FileNotFoundError`) should trigger a `SkipTest`.

* **Quote:** "All execution tasks in global validation stages must explicitly append `Teardown:` validation steps."
* **Impact:** Stray processes cause resource deadlocks and worker CPU exhaustion.
* **Action:** Validate clean states via `ps aux` and explicitly kill stray PIDs. Strictly bound CPU usage (`-n 4` or `-n 8`); never use `pytest -n auto`.

### Tier 2: Build & Orchestration

* **Quote:** "Replacing `http_archive` with `local_repository` or `native.local_repository` in Bazel repository definitions is strictly forbidden."
* **Impact:** Doing so breaks hermeticity and cross-system reproducibility across submodules.
* **Action:** Always use `http_archive` or remote repository definitions in Bazel.

* **Quote:** "Tests must explicitly catch `ExceptionGroup` to natively raise `unittest.SkipTest`."
* **Impact:** Broad `except Exception:` blocks mask failures or completely crash the pipeline when multiple interface probes fail.
* **Action:** Gracefully handle missing hardware interfaces by explicitly catching `ExceptionGroup` and raising `unittest.SkipTest("hardware unsupported")`.

* **Quote:** "Always wrap hardware simulator executions with `try... except FileNotFoundError`."
* **Impact:** Ensures the CI degrades gracefully on unsupported nodes, though graceful fallbacks do NOT constitute authentic E2E evaluations.
* **Action:** Wrap missing simulator executions appropriately. Mandate parallel E2E execution tests running genuine payloads on hardware simulators to prove organic boundaries.

### Tier 1: C++ & System Programming

* **Quote:** "The `devectorizer.py` AST generator MUST replace `ImageDType` with `PtrDType(float)`."
* **Impact:** Failing to convert the index back to a linear offset results in fatal native C compiler faults during pointer arithmetic.
* **Action:** For `IMAGE=2`, use `image_dtype.base.ptr()` rather than converting the 1D channel index `x` into an `int2` vector.

* **Quote:** "Subprocess worker pools, fork bombs, and IPC boundaries are strictly forbidden for the Map-Elites fuzzer pipeline."
* **Impact:** IPC usage in the fuzzer causes catastrophic system instability and process leaks.
* **Action:** Mutator ASTs must natively invoke `EncodeSafe()` and return `absl::NotFoundError`. Do not apply globally; Tinygrad Auto-Tuner is permitted multiprocessing.

### Tier 2: C++ & System Programming

* **Quote:** "Strictly enforce `multiprocessing.set_start_method('spawn')`."
* **Impact:** Using `fork()` duplicates lock state and deadlocks runtime. Initializing context dynamically inside classes causes fatal `RuntimeError` crashes upon re-instantiation.
* **Action:** Initialize `set_start_method` globally at module level when interfacing Python multiprocessing with C++ PyBind11 bindings.

### Tier 1: Python Code Style & Testing

* **Quote:** "The `pytest-xdist` IPC worker teardown crashes (`OSError: cannot send`) and test watchdog timeouts are fatal synchronization failures."
* **Impact:** Unhandled exceptions in `__del__`, zombie processes, or infinite loops kill workers abruptly, leaking file descriptors and hanging pytest-xdist indefinitely.
* **Action:** Explicitly trap `AttributeError`, `KeyError`, `OSError`, `FileNotFoundError` in destructors natively. Use `memoryview(shm.buf).release()`. Ensure daemon processes implement `__del__` sending exit signals and calling `terminate()`/`join()`. Implement explicit bounds-trapping for all simulation loops to avoid global pytest watchdogs.

* **Quote:** "Never extract `.schedule()` from the actual tensor intended to be evaluated."
* **Impact:** `Tensor.schedule()` permanently consumes the lazy computation graph, causing subsequent `.realize()` calls to execute an empty schedule and silently return uninitialized zeroes.
* **Action:** Use structurally independent dummy tensors to extract ASTs before evaluating the true target hardware tensor.

* **Quote:** "The test must naturally fail and trap the architectural bound natively."
* **Impact:** Inverted assertions, tautological tests (`assert True`), or mocking memory (e.g., Python dictionaries for AxiSlave) bypass cross-component routing and create testing illusions.
* **Action:** Permit tests to organically throw limits (`AssertionError`, native `RuntimeError`). Execute authentic E2E network requests and route real compiled ELF payloads. Tests mimicking external memory MUST instantiate REAL synthesized DDR controllers and SRAM RTL block responders.

### Tier 2: Python Code Style & Testing

* **Quote:** "Tests evaluating allocation and OOM bounds must naturally fail and trap organically using native Python boundaries."
* **Impact:** Masking successful limits via `assertRaises` or ignoring allocation bounds breaks critical hardware validation.
* **Action:** Explicitly validate DTCM limits (`Active FP Allocations > 32`, `OOM: 32KB`). Wrap `GeGLU` and `RoPE` in `assertRaises(RuntimeError)` for allocations, and operations within threshold (`RMSNorm`) in `try.. except FileNotFoundError`.

* **Quote:** "Explicitly calling `.realize()` within the loop prevents unbound AST explosion and fixes depth limit bounds."
* **Impact:** Failure to evaluate cyclic boundaries eagerly causes recursive failure. Masking these with `assertRaises` hides infinite loop graph rewrite defects.
* **Action:** Strictly forbid `SkipTest` for cyclic limits. Evaluate eager boundaries natively. Organically trap `RecursionError` via `assertRaises` only for negative cases.

* **Quote:** "Use `int(math.fmod())` and `int(x / y)` to correctly simulate C-style truncation towards zero."
* **Impact:** Python's native truncation towards negative infinity causes incorrect integer division evaluation during UOp rewrite tests.
* **Action:** Ensure simulated truncation accurately matches C-style math. Explicitly supply `dtype=dtypes.float` when initializing variables for transcendental functions to prevent `KeyError`.

* **Quote:** "The `Ops.STORE` node's value must be upcasted explicitly to `vec(4)` before emitting to `write_imagef`."
* **Impact:** OpenCL cannot perform a read-modify-write channel mask natively, resulting in an `AssertionError` if not upcasted.
* **Action:** For normal image environments (`IMAGE!=2`), explicitly upcast scalar pixel stores to `vec(4)` when the compiler fails to group them.

* **Quote:** "Move all standard library and third-party imports to the top-level scope (except PyBind11 bindings)."
* **Impact:** Scattered imports violate style guides and complicate dependencies.
* **Action:** Follow Google Python Style Guide for imports.

* **Quote:** "Abstracting root cause locations hides the source of crashes."
* **Impact:** Abstracting crash locations or relying on broad mocking obscures exact component failures.
* **Action:** Explicitly fix files like `tinygrad/runtime/ops_coralnpu.py` and `conftest.py`. Assert specific network parameters like `FETCHED_AVATAR_SIZE = (460, 460)` rather than using `@patch('urllib.request.urlopen')`.

### Tier 3: Python Code Style & Testing

* **Quote:** "Do not bypass line-length limits in tests or core codebase files using `# noqa: E501`."
* **Impact:** Extremely long lines mask complex assertions or data structure creations, decreasing code readability and hiding structural logic.
* **Action:** Actively refactor long lines into properly indented multi-line blocks. Do not blindly delete variables to fix unused variable errors if they are tied to structural assertions—restore the validation check instead.

* **Quote:** "Legacy keyword arguments must be strictly preserved via `**kwargs` or safe fallbacks."
* **Impact:** Removing legacy keyword arguments causes cascading API contract breakages for downstream users and tests.
* **Action:** Ensure safe fallbacks and backward compatibility when refactoring core APIs like `UOp.cast` or `UOp.bitcast`.

### Tier 1: Runtime & Execution Logic

* **Quote:** "Hardware trap executions (mcause != 0) organically terminate the subprocess with a non-zero exit code."
* **Impact:** Throwing `RuntimeError` on evaluation failure crashes the main evaluation loop and prevents valid beam search candidate exploration.
* **Action:** Python wrappers MUST explicitly catch `p.returncode != 0` and return `math.inf` to cleanly discard the candidate.

* **Quote:** "Do not rely on the default toolchain linker script."
* **Impact:** Using the default linker script for bare-metal C payloads destroys strict DTCM and EXTMEM memory layouts on CoralNPU hardware.
* **Action:** Dynamically generate a `.ld` script and pass `-T script.ld` enforcing boundaries (`EXTMEM` at `0x20000000`, `PING` at `0x00010000`, etc.) and preserving `.noinit` zones.

### Tier 2: Runtime & Execution Logic

* **Quote:** "Ensure the implementation directly modifies the `children` and `in_degree` maps before the linearize pass queue execution."
* **Impact:** Dynamically injecting sequential edges reduces peak memory consumption but does not reduce the per-kernel AST size if applied incorrectly.
* **Action:** Explicitly map sequential edges between independent kernels directly in the dependency graph structure rather than relying on global memory buffers.

* **Quote:** "Slow tests masked as deadlocks during pytest -n 4 runs..."
* **Impact:** Misdiagnosing slow pipelines as deadlocks leads to false fixes.
* **Action:** If runs take `>300s`, bump timeout in `pyproject.toml` (`timeout = 1200`).

### Tier 2: Documentation & Ledger Maintenance

* **Quote:** "Leaving 'Restored Knowledge' blocks at the bottom of the submodule AGENTS.md fragments execution constraints."
* **Impact:** Fragments submodule-specific execution constraints and causes ledger bloat.
* **Action:** Immediately integrate audit restorations into the primary strict execution mandates and remove the restoration headers.

### New Lessons Learned (Cycle 166)
* **Deadlock Masking via Timeouts:** Trapping `SimTimeoutError` with artificial bounds (`with_timeout()`) masks RTL stalling loops without fixing them. Diagnose AXI Memory Interface drops directly and migrate from `AxiSlave` Python dictionaries to synthesized DDR controllers.
* **CLI Bounds and Argument Parsing:** Execution binaries (e.g., `coralnpu_v2_sim.cc`) must be organically tested with `argc == 0, 1, and 3+` to ensure they natively catch unhandled limits, preventing "Happy-Path" verification bias.
* **E2E Integration Testing Rigor:** Mocking components like `TargetEncoder` or injecting raw hexadecimal words into memory (e.g., bypassing `NativeTextualAssembler`) is testing fraud. Authentic tests must route raw assembly through the full compilation-to-execution loop.
* **Python Linter Integrity:** Scattered Python import violations must be resolved by moving standard/third-party imports to the top of the file (exceptions apply to PyBind11 simulator bindings). Remove unused variables or use them in verifiable assertions.
* **OS Boot Artifact Graceful Degradation:** Pre-compiled OS artifacts must be probed; if missing, raise `unittest.SkipTest` or `pytest.skip` organically to avoid pipeline-crashing null pointer defects.

### New Lessons Learned (Cycle 167)
* **Tier 1: Test Suite Subprocess Deadlocks:**
  * **Quote:** "Parallel execution under pytest -n 4 deadlocked or timed out violently after running 5000+ tests."
  * **Impact:** Subprocess calls in tests hanging infinitely mask execution boundaries and crash the CI orchestrator.
  * **Action:** Individual tests executing `subprocess.run` or complex loops MUST enforce strict, native timeouts (e.g., `timeout=15.0`) to fail fast and prevent orchestrator deadlocks.
