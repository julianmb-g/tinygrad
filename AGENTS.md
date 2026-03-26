# Tinygrad Execution Context & Lessons

## Lessons Learned

### API Contract Preservation
- **Legacy Keyword Arguments (`UOp.cast`)**: When refactoring or cleaning up core APIs like `UOp.cast` or `UOp.bitcast`, ensure legacy keyword arguments (like `allow_buffer_view` or `bitcast`) are strictly preserved via `**kwargs` or safe fallbacks to prevent cascading API contract breakages for downstream users and tests.

### Code Formatting and Refactoring
- **Code Style Bypasses (E501):**: Do not bypass line-length limits in tests or core codebase files using `# noqa: E501`. Extremely long lines often mask complex assertions or data structure creations. You must actively refactor long lines into properly indented multi-line blocks, or extract logic into parameterized variables/helper functions to ensure code remains readable.

### Eager Graph Realization and Boundary Trapping
- **Architectural Identity in Renderers:**: When implementing new hardware renderers (like `ClangJITRenderer`), ensure hardware-specific attributes like `@property def arch(self)` and `@property def buf_map(self)` are properly implemented to prevent `AttributeError`s during backend-agnostic tests that inspect layout boundaries (e.g. `test_packed_smem_size`).
- **Device-Specific Testing Bounds:**: If a test inherently validates device-specific behavior (e.g., bitpacking only present in WEBGPU), it must actively skip using `@unittest.skipIf(Device.DEFAULT != "WEBGPU", ...)` to prevent raising false-positive `AssertionError`s when parsing output on `CPU` or `CORALNPU` devices.
- **Recursion Limits:**: When building deep cyclic computational graphs, explicitly calling `.realize()` within the loop prevents unbound AST explosion and fixes depth limit bounds. Any tests validating this behavior MUST strictly forbid `SkipTest` and organically trap `RecursionError` natively via `assertRaises` if testing the negative case, proving the architectural constraint without masking it.

### Floating-Point & Memory Boundary Test Trapping
- **Gemma Decomposition Boundaries**: The `GemmaDecomposition` tests explicitly hit hardware allocation limits (`Active FP Allocations > 32` and `OOM: 32KB` for DTCM). Tests running `GeGLU` and `RoPE` must be explicitly wrapped in `assertRaises(RuntimeError)` rather than failing the CI pipeline, to organically validate these hardware thresholds. Operations that fall within the threshold (e.g., `RMSNorm`) must be wrapped in `try... except FileNotFoundError:` to degrade gracefully when the hardware simulator execution binary is missing in the CI environment.

### IMAGE=2 CPU Fallback & Devectorization
- **IMAGE=2 Pointer Math Syntax Faults:**: When `IMAGE=2` is used (meaning hardware `ImageDType` is simulated as a linear `float*` on the CPU), the `devectorizer.py` AST generator MUST replace `ImageDType` with `PtrDType(float)` (e.g. `image_dtype.base.ptr()`) rather than converting the 1D channel index `x` into an `int2` vector. Failing to convert the index back to a linear offset results in fatal native C compiler faults (`cannot convert between vector and non-scalar values ('float *' and 'int2')`) during pointer arithmetic.
- **Upcasting Vector Image Stores:**: For normal image environments (`IMAGE!=2`), if the compiler fails to group scalar pixel stores during Beam Search, the `Ops.STORE` node's value must be upcasted explicitly to `vec(4)` before emitting to `write_imagef`. Failure to do so throws an `AssertionError` ("if an image store isn't upcasted to 4") since openCL cannot perform a read-modify-write channel mask natively.

### Organic Validation Enforcement
- **Native Test Failure Trapping**: When remediating test evasion wrappers (like `assertRaisesRegex(RuntimeError)` in `test_stunning.py`), the test must be permitted to organically throw the `RuntimeError` during evaluation (e.g., `bind mismatch on i, 12 != 76`). Do not wrap or invert expected limits.
- **Network Fetch Mocking (Authentic E2E Execution Boundaries)**: Never use `@patch('urllib.request.urlopen')` to return mock local payloads in E2E network functions (e.g. `test_helpers.py`). Such practices create testing illusions and mask physical socket failures. Tests must execute authentic network requests to known stable assets and structurally assert against the organic response payload parameters (like `FETCHED_AVATAR_SIZE = (460, 460)`).
- **Pipeline Safety & Teardowns**: All execution tasks in global validation stages (such as `bazel test`, `pytest -n 4`, or pointer serialization) must explicitly append `Teardown:` validation steps asserting process termination. Teardowns must validate clean states via `ps aux` and explicitly kill stray `pytest`/`bazel`/`python` PIDs to prevent resource deadlocks.

### Pipeline Safety & Graceful Degradation
- **Missing Hardware Simulators:**: Always wrap hardware simulator executions (like `out.realize()`) with `try... except FileNotFoundError` to ensure the CI pipeline degrades gracefully rather than crashing outright when the compiled simulator binary is missing.

### Pytest & Worker IPC
- **Multiprocessing Start Method**: When interfacing Python multiprocessing with C++ PyBind11 bindings, strictly enforce `multiprocessing.set_start_method("spawn")`. Do not use `fork()`, which duplicates lock state and deadlocks the runtime.
- **Worker CPU Exhaustion**: Never use `pytest -n auto`. Strict CPU bounding (`-n 4` or `-n 8`) is required to prevent node starvation which manifests as masking via infinite timeouts.

### Python ExceptionGroup Trapping
- **ExceptionGroup Trapping for Hardware Boundaries**: When a test evaluates hardware interface instantiation (like `Device["NV"]`), it must gracefully handle missing interfaces. In Python 3.11, the pipeline throws an `ExceptionGroup` when multiple interface probes fail (e.g., PCI and NVIDIACTL missing). Tests must explicitly catch `ExceptionGroup` to natively raise `unittest.SkipTest("hardware unsupported")` rather than masking failures with broad `except Exception:` blocks or completely crashing.

### Test Evasion & Error Masking
- **Compilation Failure Masking**: Do not catch `subprocess.CalledProcessError` in test suites (e.g., `test_tiny.py`) to bypass or skip failing tests. If the underlying C/C++ compiler throws syntax or compilation faults during AST generation or payload execution, it must be allowed to organically fail the test. Catching these errors mathematically erases invalid cross-component execution bugs from the CI runner, presenting a false "green" build. Only expected missing environment dependencies (like `FileNotFoundError` for missing payloads or toolchains) should trigger a `SkipTest`.

### Pytest-xdist & IPC Deadlocks
- **Master Deadlock Pipeline Crash**: The `pytest-xdist` IPC worker teardown crashes (`OSError: cannot send`) are fatal parallel worker synchronization failures. Ensure proper Python GC object lifecycle logic (`except (AttributeError, KeyError): pass`) is used to isolate and resolve teardown deadlocks, preventing the entire test suite from failing. Furthermore, when dealing with `multiprocessing.shared_memory.SharedMemory` segments, properly catch `OSError` and `FileNotFoundError` during unlinking or releasing the `memoryview(shm.buf)` during teardown hooks or object `__del__` methods.

### Pytest-xdist IPC Teardown Deadlocks
To prevent `pytest-xdist` IPC teardown deadlocks (`OSError: cannot send`), ensure explicit shared memory release (`memoryview(shm.buf).release()`) and gracefully trap Python GC exceptions (`AttributeError`, `KeyError`) within `__del__` teardown lifecycle methods.
# Tinygrad Development Lessons

### pytest-xdist IPC Teardown Deadlocks
- When running `pytest -n auto` or `-n 4`, the pipeline can fatally crash with an `OSError: cannot send (already closed?)` if worker nodes die during teardown.
- **Root Cause**: `multiprocessing.shared_memory.SharedMemory` closures in `tinygrad/runtime/ops_coralnpu.py` (`__del__` methods) or `conftest.py` teardown hooks that utilize blanket `except Exception: pass`.
- **Resolution**: You must replace blanket exception swallowing with explicit traps for `(AttributeError, KeyError, OSError, FileNotFoundError)` to allow the Python Garbage Collector to tear down the shared memory without killing the worker process abruptly.

### Refactoring & Testing Boundaries
- **Test Integrity during Refactoring**: When extracting magic numbers or refactoring test scripts, ensure you do not inadvertently "fix" or alter the fundamental limits the test is evaluating. Tests meant to organically expose architectural defects (like `bind mismatch`) must remain organically failing to ensure valid E2E boundary trapping. Do not change variable names or test logic to make these structurally failing tests pass.

### Test Fraud via @unittest.expectedFailure
- **Organic Bounds Trapping**: Do not use `@unittest.expectedFailure` to bypass architectural E2E limits. If a test natively hits a bounds defect, it must organically trap (e.g., via `AssertionError: 0 != 1`) to accurately reflect the architectural failure. Masking it mathematically guarantees test fraud.

### Evasion Tactic Remediation (Test Fraud Prevention)
- **Parameter Mutation Evasion (`allow_multiple=True`)**: Never inject configuration bypasses like `allow_multiple=True` into test parameters to mask execution regressions (e.g., kernel count inflation). Tests must organically validate the exact boundary configurations.
- **Organic Failure Restorations**: When remediating testing fraud (removing `expectedFailure` decorators or `assertRaises` wrappers), the test must naturally fail and trap the architectural bound natively rather than being masked to register as a false "green" build.

### Graph Rewrite Organic Evaluation
- **KeyError Mappings vs Organic Integration**: When writing E2E validation tests for graph rewrite limits (like `test_sin_to_sqrt_organic_trap`), do not rely on missing unmapped components in dictionaries to organically mask limits (such as `promo_lattice` hitting `KeyError: dtypes.index`). Instead, instantiate variables with robust parameters (`dtype=dtypes.float`) and utilize `PatternMatcher` directly via `graph_rewrite` to force the AST boundary natively. This guarantees the architectural limit (`RuntimeError: infinite loop in graph_rewrite`) traps the evaluation organically rather than bypassing validation limits.

### E2E Testing Pipeline
- **Schedule Consumption Fallacy**: When writing custom tests that require extracting ASTs before evaluation (e.g. to pre-compile ELFs or cross-check configurations), never extract `.schedule()` from the actual tensor intended to be evaluated. `Tensor.schedule()` permanently consumes the lazy computation graph; subsequent `out.realize()` calls will execute an empty schedule and silently return uninitialized zero-arrays. You MUST construct an identical independent dummy tensor to evaluate its AST prior to evaluating the target hardware tensor.

### Test Erasure Reversal Completeness
- **Secondary Masking Lines**: When restoring tests or assertions meant to organically fail (e.g. metadata presence asserts), ensure that ALL associated lines (such as `bw = [m for m in si.metadata if m.backward]`) are uncommented, not just the primary assertion, to ensure test coverage boundaries are fully restored.

### Testing Evasion Remediation
- **Assertion Evasion Ban**: When remediating testing fraud (removing `assertRaises(AssertionError)` decorators/wrappers), the test must naturally fail and trap the architectural bound natively rather than being mathematically inverted to mask the error and register as a false "green" build.

### Tautological Test Masking & Linter Evasion
- **Linter Evasion (E501)**: Do not attempt to bypass `ruff` line-length checks using `# noqa: E501` combined with aggressive one-liners, lambdas, or walrus operators. Tests must unroll the logic cleanly into readable blocks. Inline PyBind11 imports are the ONLY permitted deviations from top-level imports.
- **Tautological Mocking**: Replacing deleted pipeline tests with tautological tests (e.g. `assert True` or mock assertions) completely bypasses test verification and provides false security. Tests must interact with authentic targets natively.
