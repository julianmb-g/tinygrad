# Tinygrad Execution Context & Lessons

## Lessons Learned

### Build & Orchestration

- **[Tier 1] Compilation Failure Masking**
  - **Quote:** "Do not catch `subprocess.CalledProcessError` in test suites to bypass or skip failing tests."
  - **Impact:** Catching these errors mathematically erases invalid cross-component execution bugs from the CI runner, presenting a false "green" build.
  - **Action:** Allow underlying C/C++ compiler syntax or compilation faults to organically fail the test. Only expected missing environment dependencies (like `FileNotFoundError`) should trigger a `SkipTest`.

- **[Tier 1] Pipeline Safety & Teardowns**
  - **Quote:** "All execution tasks in global validation stages must explicitly append `Teardown:` validation steps."
  - **Impact:** Stray processes cause resource deadlocks and worker CPU exhaustion.
  - **Action:** Validate clean states via `ps aux` and explicitly kill stray `pytest`/`bazel`/`python` PIDs. Never use `pytest -n auto`; strictly bound CPU usage (`-n 4` or `-n 8`).

- **[Tier 2] Local Repository Prohibition**
  - **Quote:** "Replacing `http_archive` with `local_repository` or `native.local_repository` in Bazel repository definitions is strictly forbidden."
  - **Impact:** Doing so breaks hermeticity and cross-system reproducibility across submodules.
  - **Action:** Always use `http_archive` or remote repository definitions in Bazel.

- **[Tier 2] ExceptionGroup Trapping for Hardware Boundaries**
  - **Quote:** "Tests must explicitly catch `ExceptionGroup` to natively raise `unittest.SkipTest`."
  - **Impact:** Broad `except Exception:` blocks mask failures or completely crash the pipeline when multiple interface probes (like PCI and NVIDIACTL) fail.
  - **Action:** Gracefully handle missing hardware interfaces by explicitly catching `ExceptionGroup` and raising `unittest.SkipTest("hardware unsupported")`.

- **[Tier 2] Missing Hardware Simulators & E2E Verification**
  - **Quote:** "Always wrap hardware simulator executions with `try... except FileNotFoundError`."
  - **Impact:** Ensures the CI pipeline degrades gracefully rather than crashing outright on unsupported nodes. However, graceful fallbacks do NOT constitute authentic E2E hardware evaluations.
  - **Action:** Wrap missing simulator executions appropriately, but ensure parallel E2E execution tests running genuine payloads on the hardware simulator are mandated to prove organic boundaries.

### C++ & System Programming

- **[Tier 1] IMAGE=2 Pointer Math Syntax Faults**
  - **Quote:** "The `devectorizer.py` AST generator MUST replace `ImageDType` with `PtrDType(float)`."
  - **Impact:** Failing to convert the index back to a linear offset results in fatal native C compiler faults (`cannot convert between vector and non-scalar values ('float *' and 'int2')`) during pointer arithmetic.
  - **Action:** When `IMAGE=2` is used, use `image_dtype.base.ptr()` rather than converting the 1D channel index `x` into an `int2` vector.

- **[Tier 1] Fuzzer Generation IPC Ban & Scope**
  - **Quote:** "Subprocess worker pools, fork bombs, and IPC boundaries are strictly forbidden for the Map-Elites fuzzer pipeline."
  - **Impact:** IPC usage in the fuzzer causes catastrophic system instability and process leaks.
  - **Action:** Mutator ASTs must natively invoke `EncodeSafe()` and return `absl::NotFoundError`. Do NOT apply this ban globally; the Tinygrad Auto-Tuner is explicitly permitted to use `multiprocessing`.

- **[Tier 2] Multiprocessing Start Method & Context Integrity**
  - **Quote:** "Strictly enforce `multiprocessing.set_start_method('spawn')`."
  - **Impact:** Using `fork()` duplicates lock state and deadlocks the runtime. Initializing context dynamically inside classes causes fatal `RuntimeError` crashes upon re-instantiation.
  - **Action:** Initialize the `set_start_method` context globally at the module level when interfacing Python multiprocessing with C++ PyBind11 bindings.

### Python Code Style & Testing

- **[Tier 1] Pytest-Xdist Deadlocks & SharedMemory Teardown**
  - **Quote:** "The `pytest-xdist` IPC worker teardown crashes (`OSError: cannot send`) are fatal parallel worker synchronization failures."
  - **Impact:** Unhandled exceptions in `__del__` methods or teardown hooks utilizing blanket `except Exception: pass` kill the worker abruptly and crash the main test suite.
  - **Action:** Replace blanket exception swallowing with explicit traps for `(AttributeError, KeyError, OSError, FileNotFoundError)` to allow the Python GC to cleanly tear down `SharedMemory` segments. Bump test timeouts to distinguish slow runs from actual deadlocks.

- **[Tier 1] Schedule Consumption Mismatch Masking**
  - **Quote:** "Never extract `.schedule()` from the actual tensor intended to be evaluated."
  - **Impact:** `Tensor.schedule()` permanently consumes the lazy computation graph. Subsequent `.realize()` calls execute an empty schedule, silently returning uninitialized zeroes and registering as a false "green" test.
  - **Action:** Use structurally independent dummy tensors to extract ASTs before evaluating the true target hardware tensor.

- **[Tier 1] Assertion Evasion & Tautological Mocking Ban**
  - **Quote:** "The test must naturally fail and trap the architectural bound natively."
  - **Impact:** Wrapping failing evaluations in inverted assertions or replacing pipeline tests with tautological tests (`assert True`) completely bypasses verification and creates testing illusions.
  - **Action:** Permit tests to organically throw limits (e.g., `AssertionError: 0 != 1` or native `RuntimeError`). Execute authentic E2E network requests and structurally assert against organic response payloads instead of using mocked APIs.

- **[Tier 2] Gemma Hardware Bounds Evaluation**
  - **Quote:** "Tests evaluating allocation and OOM bounds must naturally fail and trap organically using native Python boundaries."
  - **Impact:** Wrapping successful limits in `assertRaises(RuntimeError)` causes the test suite to fail with `AssertionError: RuntimeError not raised`.
  - **Action:** Explicitly wrap `GeGLU` and `RoPE` in `assertRaises(RuntimeError)` to hit allocation limits. Operations within the threshold (`RMSNorm`) must be wrapped in `try.. except FileNotFoundError:` to degrade gracefully.

- **[Tier 2] Recursion & Cyclic Graph Limits**
  - **Quote:** "Explicitly calling `.realize()` within the loop prevents unbound AST explosion and fixes depth limit bounds."
  - **Impact:** Failure to evaluate cyclic boundaries eagerly leads to recursive execution failure limits.
  - **Action:** Strictly forbid `SkipTest` for cyclic limits. Evaluate eager boundaries natively, and if testing the negative case, organically trap `RecursionError` via `assertRaises`.

- **[Tier 2] Python Semantics for UOp Rewrites**
  - **Quote:** "Use `int(math.fmod())` and `int(x / y)` to correctly simulate C-style truncation towards zero."
  - **Impact:** Python's native truncation towards negative infinity causes incorrect integer division evaluation during UOp rewrite tests.
  - **Action:** Ensure simulated truncation accurately matches C-style math. Explicitly supply `dtype=dtypes.float` when initializing variables for transcendental functions to prevent `KeyError`.

- **[Tier 2] Upcasting Vector Image Stores**
  - **Quote:** "The `Ops.STORE` node's value must be upcasted explicitly to `vec(4)` before emitting to `write_imagef`."
  - **Impact:** OpenCL cannot perform a read-modify-write channel mask natively, resulting in an `AssertionError` if not upcasted.
  - **Action:** For normal image environments (`IMAGE!=2`), explicitly upcast scalar pixel stores to `vec(4)` when the compiler fails to group them.

- **[Tier 3] Linter Evasion & Code Style Bypasses**
  - **Quote:** "Do not bypass line-length limits in tests or core codebase files using `# noqa: E501`."
  - **Impact:** Extremely long lines mask complex assertions or data structure creations, decreasing code readability and hiding structural logic.
  - **Action:** Actively refactor long lines into properly indented multi-line blocks. Do not blindly delete variables to fix unused variable errors if they are tied to structural assertions—restore the validation check instead.

- **[Tier 3] Legacy Keyword Arguments API Contracts**
  - **Quote:** "Legacy keyword arguments must be strictly preserved via `**kwargs` or safe fallbacks."
  - **Impact:** Removing legacy keyword arguments (like `allow_buffer_view` or `bitcast`) causes cascading API contract breakages for downstream users and tests.
  - **Action:** Ensure safe fallbacks and backward compatibility when refactoring core APIs like `UOp.cast` or `UOp.bitcast`.

### Runtime & Execution Logic

- **[Tier 1] Auto-Tuner Concurrency IPC Poisoning**
  - **Quote:** "Hardware trap executions organically terminate the subprocess with a non-zero exit code."
  - **Impact:** Throwing `RuntimeError` on evaluation failure crashes the main evaluation loop and prevents valid beam search candidate exploration.
  - **Action:** Explicitly catch `p.returncode != 0` in the Python wrapper and return `math.inf` to cleanly discard the candidate.

- **[Tier 1] Linker Contract Enforcement**
  - **Quote:** "Do not rely on the default toolchain linker script."
  - **Impact:** Using the default linker script for bare-metal C payloads destroys strict DTCM and EXTMEM memory layouts on CoralNPU hardware.
  - **Action:** Dynamically generate a `.ld` script and pass `-T script.ld` enforcing boundaries (`EXTMEM` at `0x20000000`, `PING` at `0x00010000`, etc.) and preserving `.noinit` zones.

- **[Tier 2] Dependency Chaining in Schedule**
  - **Quote:** "Ensure the implementation directly modifies the `children` and `in_degree` maps before the linearize pass queue execution."
  - **Impact:** Dynamically injecting sequential edges reduces peak memory consumption but does not reduce the per-kernel AST size if applied incorrectly.
  - **Action:** Explicitly map sequential edges between independent kernels directly in the dependency graph structure rather than relying on global memory buffers.
### Restored Architectural Constraints & Concrete Fault Locations
- **Tier 1: Architectural Constraints (Gemma/RoPE)**
  - **Quote:** "The GemmaDecomposition tests explicitly hit hardware allocation limits..."
  - **Impact:** Erasing these limits removes critical boundaries.
  - **Action:** Tests must explicitly validate `Active FP Allocations > 32` and `OOM: 32KB` for DTCM limits.

- **Tier 1: RISC-V Hardware Trap Codes**
  - **Quote:** "Hardware trap executions (mcause != 0) organically terminate the subprocess..."
  - **Impact:** Prevents beam search poisoning.
  - **Action:** Python wrappers MUST explicitly catch `p.returncode != 0` and return `math.inf`.

- **Tier 2: Operational Timeout Thresholds**
  - **Quote:** "Slow tests masked as deadlocks during pytest -n 4 runs..."
  - **Impact:** Misdiagnosing slow pipelines as deadlocks leads to false fixes.
  - **Action:** If runs take `>300s`, bump timeout in `pyproject.toml` (`timeout = 1200`).

- **Tier 2: Concrete Fault Locations**
  - **Quote:** "Root cause locations for SharedMemory GC crashes..."
  - **Impact:** Abstracting these locations hides the source of the crash.
  - **Action:** Explicitly fix `tinygrad/runtime/ops_coralnpu.py` and `conftest.py`. Also, assert specific network parameters like `FETCHED_AVATAR_SIZE = (460, 460)` rather than using `@patch('urllib.request.urlopen')`.
### Restored Knowledge (Data-Loss Audit Remediation - Latest Cycle)
- Trap `OSError: cannot send (already closed?)` in IPC worker teardown (`pytest-xdist`) to avoid session finish crashes masking execution validations.
- Move all standard library and third-party imports to the top-level scope (except PyBind11 bindings) per Google Python Style Guide.
- Do not use `assertRaises(RuntimeError)` to mask infinite loop architectural defects in graph rewrites or schedule assertions.

### Pytest IPC Teardown Deadlock Rules
When using `pytest-xdist` parallel workers, crashes with `OSError: cannot send` occur due to improperly handled `__del__` garbage collection on `multiprocessing.shared_memory.SharedMemory`. You MUST release memory using `memoryview(shm.buf).release()` and explicitly trap `OSError`, `KeyError`, and `AttributeError` in the destructor natively.


### Restored Knowledge (Data-Loss Audit Remediation - Testing Illusion Mandates)
- **Tier 1: Hardware Simulation & Mock Isolation Rules**
  - **Quote:** "Mocking AxiSlave with Python dictionaries and swallowing test exceptions."
  - **Impact:** False positive 100% unit tests that fail to simulate RTL component boundaries.
  - **Action:** tests mimicking external memory MUST instantiate REAL synthesized DDR controllers and SRAM RTL block responders. Eviscerating memory boundaries to trap timeouts is strictly forbidden.
- **Tier 1: E2E Integration Boundary Rules**
  - **Quote:** "Validating bytes written to memory does not prove cross-component hardware integration."
  - **Impact:** Systemic testing illusions where isolated string matching or individual instruction evaluations bypass cross-component routing.
  - **Action:** MUST introduce rigorous Integration/E2E execution tests. A mutator or wrapper component is invalid until an authentic AST payload (compiled ELF) routes through the entire execution loop natively and verifies execution traces. Mocking TargetEncoder classes is insufficient.

- **[Tier 1] Multiprocessing Zombie Deadlocks**
  - **Quote:** "The test suite did not complete. Parallel execution under pytest -n 4 deadlocked or timed out violently after running 5000+ tests."
  - **Impact:** Subprocesses without a proper `__del__` GC lifecycle that calls `terminate()` and `join()` leak file descriptors and stdout pipes, causing pytest-xdist to hang indefinitely waiting for EOF.
  - **Action:** Ensure any daemon processes like `TinygradAutoTunerIPC` explicitly implement `__del__` that sends exit signals (`None`) and explicitly calls `worker.terminate()` and `worker.join()`.
### Tinygrad Specific Lessons
- **Tier 1: Pytest Parallel Worker Deadlocks**
  - **Quote:** "Parallel execution under pytest -n 4 deadlocked or timed out violently after running 5000+ tests."
  - **Impact:** 1200s orchestrator timeouts mask execution validations.
  - **Action:** Ensure individual test timeouts organically trap infinite loops before the global pytest watchdog is triggered.
