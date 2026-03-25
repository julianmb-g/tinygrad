# Tinygrad Execution Context & Lessons

### Pytest & Worker IPC
- **Subprocess Monkeypatching**: Monkeypatching `subprocess.Popen` to manage timeouts and bounds must strictly be scoped to individual pytest-xdist workers. Use `if getattr(request.config, "workerinput", None) is not None:` in `conftest.py`'s `pytest_configure` to prevent master-node deadlocks and catastrophic `OSError` crashes.
- **Worker CPU Exhaustion**: Never use `pytest -n auto`. Strict CPU bounding (`-n 4` or `-n 8`) is required to prevent node starvation which manifests as masking via infinite timeouts.
- **Multiprocessing Start Method**: When interfacing Python multiprocessing with C++ PyBind11 bindings, strictly enforce `multiprocessing.set_start_method("spawn")`. Do not use `fork()`, which duplicates lock state and deadlocks the runtime.

### Organic Validation Enforcement
- **Native Test Failure Trapping**: When remediating test evasion wrappers (like `assertRaisesRegex(RuntimeError)` in `test_stunning.py`), the test must be permitted to organically throw the `RuntimeError` during evaluation (e.g., `bind mismatch on i, 12 != 76`). Do not wrap or invert expected limits.
- **Network Fetch Mocking (Authentic E2E Execution Boundaries)**: Never use `@patch('urllib.request.urlopen')` to return mock local payloads in E2E network functions (e.g. `test_helpers.py`). Such practices create testing illusions and mask physical socket failures. Tests must execute authentic network requests to known stable assets and structurally assert against the organic response payload parameters (like `FETCHED_AVATAR_SIZE = (460, 460)`).

- **Pipeline Safety & Teardowns**: All execution tasks in global validation stages (such as `bazel test`, `pytest -n 4`, or pointer serialization) must explicitly append `Teardown:` validation steps asserting process termination. Teardowns must validate clean states via `ps aux` and explicitly kill stray `pytest`/`bazel`/`python` PIDs to prevent resource deadlocks.

### Pipeline Safety & Graceful Degradation
- **Missing Hardware Simulators:** Always wrap hardware simulator executions (like `out.realize()`) with `try... except FileNotFoundError` to ensure the CI pipeline degrades gracefully rather than crashing outright when the compiled simulator binary is missing.

### IMAGE=2 CPU Fallback & Devectorization
- **IMAGE=2 Pointer Math Syntax Faults:** When `IMAGE=2` is used (meaning hardware `ImageDType` is simulated as a linear `float*` on the CPU), the `devectorizer.py` AST generator MUST replace `ImageDType` with `PtrDType(float)` (e.g. `image_dtype.base.ptr()`) rather than converting the 1D channel index `x` into an `int2` vector. Failing to convert the index back to a linear offset results in fatal native C compiler faults (`cannot convert between vector and non-scalar values ('float *' and 'int2')`) during pointer arithmetic. 
- **Upcasting Vector Image Stores:** For normal image environments (`IMAGE!=2`), if the compiler fails to group scalar pixel stores during Beam Search, the `Ops.STORE` node's value must be upcasted explicitly to `vec(4)` before emitting to `write_imagef`. Failure to do so throws an `AssertionError` ("if an image store isn't upcasted to 4") since openCL cannot perform a read-modify-write channel mask natively.

### Test Evasion & Error Masking
- **Compilation Failure Masking**: Do not catch `subprocess.CalledProcessError` in test suites (e.g., `test_tiny.py`) to bypass or skip failing tests. If the underlying C/C++ compiler throws syntax or compilation faults during AST generation or payload execution, it must be allowed to organically fail the test. Catching these errors mathematically erases invalid cross-component execution bugs from the CI runner, presenting a false "green" build. Only expected missing environment dependencies (like `FileNotFoundError` for missing payloads or toolchains) should trigger a `SkipTest`.
# Tinygrad Submodule Lessons Learned

### Eager Graph Realization and Boundary Trapping
- **Recursion Limits:** When building deep cyclic computational graphs, explicitly calling `.realize()` within the loop prevents unbound AST explosion and fixes depth limit bounds. Any tests validating this behavior MUST strictly forbid `SkipTest` and organically trap `RecursionError` natively via `assertRaises` if testing the negative case, proving the architectural constraint without masking it.
- **Architectural Identity in Renderers:** When implementing new hardware renderers (like `ClangJITRenderer`), ensure hardware-specific attributes like `@property def arch(self)` and `@property def buf_map(self)` are properly implemented to prevent `AttributeError`s during backend-agnostic tests that inspect layout boundaries (e.g. `test_packed_smem_size`).
- **Device-Specific Testing Bounds:** If a test inherently validates device-specific behavior (e.g., bitpacking only present in WEBGPU), it must actively skip using `@unittest.skipIf(Device.DEFAULT != "WEBGPU", ...)` to prevent raising false-positive `AssertionError`s when parsing output on `CPU` or `CORALNPU` devices.
\n### Subprocess Timeouts\n- **SLA Constants**: When establishing subprocess timeouts for CI hardware compilation tasks (like Beam Search tuning), prefer using named constants representing SLA derivations (e.g., `kDefaultCompilationTimeoutS = 15.0`) over inline magic floats to improve code maintainability and clarify operational constraints.

### Code Formatting and Refactoring
- **Code Style Bypasses (E501):** Do not bypass line-length limits in tests or core codebase files using `# noqa: E501`. Extremely long lines often mask complex assertions or data structure creations. You must actively refactor long lines into properly indented multi-line blocks, or extract logic into parameterized variables/helper functions to ensure code remains readable.
## Lessons Learned

### Python GC Object Lifecycle in Pytest Teardowns
- When performing teardown actions in `atexit` or `__del__` within pytest workers, wrap system calls (like `os.kill`) in `except (AttributeError, KeyError): pass` to handle missing module references due to Python GC teardown ordering. This prevents `pytest-xdist` master deadlocks caused by unhandled worker crashes on exit.
