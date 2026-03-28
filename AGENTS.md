# Tinygrad Execution Context & Lessons

## Lessons Learned

### Tier 1: Build, Orchestration & Execution Logic

* **Compiler Syntax Integrity & Error Catching**
  * **Quote:** "Do not catch `subprocess.CalledProcessError` in test suites to bypass or skip failing tests."
  * **Impact:** Catching these errors mathematically erases invalid cross-component execution bugs from the CI runner, presenting a false "green" build.
  * **Action:** Allow underlying C/C++ compiler syntax or compilation faults to organically fail the test. Only expected missing environment dependencies (like `FileNotFoundError`) should trigger a `SkipTest`.

* **Resource Leakage & Teardown Verification**
  * **Quote:** "All execution tasks in global validation stages must explicitly append `Teardown:` validation steps."
  * **Impact:** Stray processes cause resource deadlocks and worker CPU exhaustion. Unhandled exceptions in `__del__` kill workers abruptly, leaking file descriptors.
  * **Action:** Validate clean states via `ps aux` and explicitly kill stray PIDs. Explicitly trap expected exceptions (`AttributeError`, `KeyError`, `OSError`) in Python destructors. Use `memoryview(shm.buf).release()`. Ensure daemon processes implement `__del__` sending exit signals. Strictly bound CPU usage (`-n 4` or `-n 8`); never use `pytest -n auto`.

* **E2E Validation & Memory Boundary Authenticities**
  * **Quote:** "The test must naturally fail and trap the architectural bound natively."
  * **Impact:** Mocking memory (e.g., Python dictionaries for AxiSlave), mocking `TargetEncoder`, or bypassing `NativeTextualAssembler` bypasses cross-component routing and creates testing illusions.
  * **Action:** Execute authentic E2E network requests and route real compiled ELF payloads. Tests mimicking external memory MUST instantiate REAL synthesized DDR controllers and SRAM RTL block responders.

* **Fuzzer IPC Boundaries**
  * **Quote:** "Subprocess worker pools, fork bombs, and IPC boundaries are strictly forbidden for the Map-Elites fuzzer pipeline."
  * **Impact:** IPC usage in the fuzzer causes catastrophic system instability and process leaks.
  * **Action:** Mutator ASTs must natively invoke `EncodeSafe()` and return `absl::NotFoundError`. Do not apply globally; Tinygrad Auto-Tuner is permitted multiprocessing.

* **Pointer Arithmetic & ImageDType Conversion**
  * **Quote:** "The `devectorizer.py` AST generator MUST replace `ImageDType` with `PtrDType(float)`."
  * **Impact:** Failing to convert the index back to a linear offset results in fatal native C compiler faults during pointer arithmetic.
  * **Action:** For `IMAGE=2`, use `image_dtype.base.ptr()` rather than converting the 1D channel index `x` into an `int2` vector.

* **OS Boot & Linker Scripts Constraints**
  * **Quote:** "Do not rely on the default toolchain linker script."
  * **Impact:** Using the default linker script for bare-metal C payloads destroys strict DTCM and EXTMEM memory layouts on CoralNPU hardware.
  * **Action:** Dynamically generate a `.ld` script and pass `-T script.ld` enforcing boundaries (`EXTMEM` at `0x20000000`, `PING` at `0x00010000`, etc.) and preserving `.noinit` zones. Pre-compiled OS artifacts must be probed; if missing, raise `unittest.SkipTest` organically to avoid pipeline-crashing null pointer defects.

* **CLI Argument Verification**
  * **Quote:** "Execution binaries (e.g., `coralnpu_v2_sim.cc`) must be organically tested with `argc == 0, 1, and 3+`."
  * **Impact:** Fails to trap unhandled argument limits, creating "Happy-Path" verification bias.
  * **Action:** Ensure execution binaries natively catch unhandled argument limits.

* **Evaluation Loop & Trap Handling**
  * **Quote:** "Hardware trap executions (mcause != 0) organically terminate the subprocess with a non-zero exit code."
  * **Impact:** Throwing `RuntimeError` on evaluation failure crashes the main evaluation loop and prevents valid beam search candidate exploration.
  * **Action:** Python wrappers MUST explicitly catch `p.returncode != 0` and return `math.inf` to cleanly discard the candidate.

### Tier 2: System Architecture & Code Style

* **Hermetic Bazel Dependencies**
  * **Quote:** "Replacing `http_archive` with `local_repository` or `native.local_repository` in Bazel repository definitions is strictly forbidden."
  * **Impact:** Breaks hermeticity and cross-system reproducibility across submodules.
  * **Action:** Always use `http_archive` or remote repository definitions in Bazel.

* **Hardware Feature Graceful Degradation**
  * **Quote:** "Tests must explicitly catch `ExceptionGroup` to natively raise `unittest.SkipTest`."
  * **Impact:** Broad `except Exception:` blocks mask failures or completely crash the pipeline when multiple interface probes fail.
  * **Action:** Gracefully handle missing hardware interfaces by explicitly catching `ExceptionGroup` and raising `unittest.SkipTest("hardware unsupported")`. Wrap missing simulator executions appropriately.

* **Multiprocessing Start Methods**
  * **Quote:** "Strictly enforce `multiprocessing.set_start_method('spawn')`."
  * **Impact:** Using `fork()` duplicates lock state and deadlocks runtime.
  * **Action:** Initialize `set_start_method` globally at module level when interfacing Python multiprocessing with C++ PyBind11 bindings.

* **Graph Evaluation & Ast Unbound Constraints**
  * **Quote:** "Never extract `.schedule()` from the actual tensor intended to be evaluated. Explicitly calling `.realize()` within the loop prevents unbound AST explosion."
  * **Impact:** Consumes lazy computation graphs permanently or causes recursive failure.
  * **Action:** Use structurally independent dummy tensors to extract ASTs before evaluating the true target hardware tensor. Strictly forbid `SkipTest` for cyclic limits. Evaluate eager boundaries natively. Organically trap `RecursionError` via `assertRaises` only for negative cases.

* **Allocation & Math Bounds Evaluation**
  * **Quote:** "Tests evaluating allocation and OOM bounds must naturally fail and trap organically using native Python boundaries."
  * **Impact:** Masking successful limits via `assertRaises` breaks critical hardware validation.
  * **Action:** Explicitly validate DTCM limits. Use `int(math.fmod())` and `int(x / y)` to correctly simulate C-style truncation towards zero. Provide `dtype=dtypes.float` when initializing variables for transcendental functions. For normal image environments (`IMAGE!=2`), explicitly upcast scalar pixel stores to `vec(4)` before emitting to `write_imagef`.

* **Sequential Edge Memory Mappings**
  * **Quote:** "Ensure the implementation directly modifies the `children` and `in_degree` maps before the linearize pass queue execution."
  * **Impact:** Dynamically injecting sequential edges reduces peak memory consumption but does not reduce the per-kernel AST size if applied incorrectly.
  * **Action:** Explicitly map sequential edges between independent kernels directly in the dependency graph structure rather than relying on global memory buffers.

* **Code Style, Linters & Imports**
  * **Quote:** "Do not bypass line-length limits in tests or core codebase files using `# noqa: E501`."
  * **Impact:** Scattered imports and long lines hide structural logic and violate code standards.
  * **Action:** Move all standard/third-party imports to the top-level scope (except PyBind11 bindings). Actively refactor long lines into properly indented multi-line blocks. Remove unused variables or use them in verifiable assertions.

* **Backward Compatibility & Crash Tracing**
  * **Quote:** "Abstracting root cause locations hides the source of crashes. Legacy keyword arguments must be strictly preserved via `**kwargs`."
  * **Impact:** Causes cascading API contract breakages and obscures exact component failures.
  * **Action:** Explicitly fix files instead of relying on broad mocking. Ensure safe fallbacks via `**kwargs` when refactoring core APIs like `UOp.cast`.

* **Ledger Fragmentation Constraints**
  * **Quote:** "Leaving 'Restored Knowledge' blocks at the bottom of the submodule AGENTS.md fragments execution constraints."
  * **Impact:** Fragments submodule-specific execution constraints and causes ledger bloat.
  * **Action:** Immediately integrate audit restorations into the primary strict execution mandates and remove the restoration headers.

* **Memory Allocation Exact Block Evasion**
  * **Quote:** "Mutating `if bsize >= size_aligned:` to `if bsize > size_aligned:` in `ops_coralnpu.py`."
  * **Impact:** Survives execution natively because tests never request an exact memory block size equal to a free chunk.
  * **Action:** E2E allocation tests MUST explicitly request allocation boundaries equal to remaining chunk sizes to ensure boundaries are inclusive natively.

* **Beam Search Configuration Evasion**
  * **Quote:** "Mutating `BEAM > 0` to `BEAM > 1` in `ops_coralnpu.py`."
  * **Impact:** Test suite is blind to exact configuration limits, meaning beam search is never structurally verified on `BEAM=1`.
  * **Action:** Write rigorous bounds tests explicitly invoking `BEAM=1` to enforce native threshold boundary execution.

* **Complete Test Erasure via Skipping**
  * **Quote:** "In `test_uop_graph.py`, failing tests... were blanketed with `@unittest.skip('invalid uops')`."
  * **Impact:** Erasing execution failures from the CI pipeline entirely is catastrophic structural masking.
  * **Action:** Never use `@unittest.skip('invalid uops')` or similar blanket decorators to hide failing graph validations or execution tests. Fix the underlying logic natively.

* **Subprocess Masking (Deadlocks)**
  * **Quote:** "Another artificial timeout masking the fact that the actual simulation payload execution is permanently deadlocking."
  * **Impact:** Masking deadlocks with `p.wait(timeout=15.0)` or `with_timeout` hides severe cross-component subsystem failures.
  * **Action:** Remove artificial framework timeouts. The simulated RTL bounds must trap and cleanly abort invalid flows natively.

* **AxiSlave Mocking in Allocators**
  * **Quote:** "`TestCoralNPUAllocator.setUp` explicitly mocks the hardware device (`self.device = MagicMock()`) alongside mock ELF generation."
  * **Impact:** Allocator logic is tested in total isolation, never proving it can route data through real synthesized AXI allocators.
  * **Action:** E2E tests MUST instantiate REAL synthesized memory controllers. Revert the testing fraud.

# tinygrad Orchestration Guidelines

*   **IPC Shared Memory Teardown Deadlocks**: Explicitly trap expected cleanup exceptions (`AttributeError`, `KeyError`, `OSError`) in Python `__del__` GC lifecycles to prevent `pytest-xdist` parallel workers from crashing with `OSError: cannot send (already closed?)`.
*   **Test Erasure via Skipping**: Never use `@unittest.skip('invalid uops')` or similar blanket decorators to hide failing graph validations or execution tests.
