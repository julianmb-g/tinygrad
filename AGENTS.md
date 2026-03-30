# tinygrad Agent Instructions

## Lessons Learned & Orchestration Rules

### Tier 1: Critical Blocker

* **Allocation & Math Bounds Evaluation**
  * **Quote:** "Tests evaluating allocation and OOM bounds must naturally fail and trap organically using native Python boundaries."
  * **Impact:** Masking successful limits via `assertRaises` breaks critical hardware validation.
  * **Action:** Explicitly validate DTCM limits. Use `int(math.fmod())` and `int(x / y)` to correctly simulate C-style truncation towards zero. Provide `dtype=dtypes.float` when initializing variables for transcendental functions. For normal image environments (`IMAGE!=2`), explicitly upcast scalar pixel stores to `vec(4)` before emitting to `write_imagef`.

* **Backward Compatibility & Crash Tracing**
  * **Quote:** "Abstracting root cause locations hides the source of crashes. Legacy keyword arguments must be strictly preserved via `**kwargs`."
  * **Impact:** Causes cascading API contract breakages and obscures exact component failures.
  * **Action:** Explicitly fix files instead of relying on broad mocking. Ensure safe fallbacks via `**kwargs` when refactoring core APIs like `UOp.cast`.

* **Complete Test Erasure via Skipping**
  * **Quote:** "In `test_uop_graph.py`, failing tests... were blanketed with `@unittest.skip('invalid uops')`."
  * **Impact:** Erasing execution failures from the CI pipeline entirely is catastrophic structural masking. Claiming a pipeline is fixed by literally skipping the graph validations is testing fraud.
  * **Action:** Never use `@unittest.skip('invalid uops')` or similar blanket decorators to hide failing graph validations or execution tests. Fix the underlying logic natively. Agents must rigorously cross-verify `PLAN.md` against actual codebase state before attempting tasks.

* **Fuzzer IPC Boundaries**
  * **Quote:** "Subprocess worker pools, fork bombs, and IPC boundaries are strictly forbidden for the Map-Elites fuzzer pipeline."
  * **Impact:** IPC usage in the fuzzer causes catastrophic system instability and process leaks.
  * **Action:** Mutator ASTs must natively invoke `EncodeSafe()` and return `absl::NotFoundError`. Do not apply globally; Tinygrad Auto-Tuner is permitted multiprocessing.

* **Graph Evaluation & Ast Unbound Constraints**
  * **Quote:** "Never extract `.schedule()` from the actual tensor intended to be evaluated. Explicitly calling `.realize()` within the loop prevents unbound AST explosion."
  * **Impact:** Consumes lazy computation graphs permanently or causes recursive failure.
  * **Action:** Use structurally independent dummy tensors to extract ASTs before evaluating the true target hardware tensor. Strictly forbid `SkipTest` for cyclic limits. Evaluate eager boundaries natively. Organically trap `RecursionError` via `assertRaises` only for negative cases.

* **Graph Validation Failures (SCHEDULE ISSUE)**
  * **Quote:** "tinygrad Graph Validation Failures: `TypeError: unsupported operand type(s) for -: 'tuple' and 'int'` and `SCHEDULE ISSUE`."
  * **Impact:** Core regressions exist in graph scheduling and multi-dimensional tensor boundary calculations.
  * **Action:** Fix the UOp vectorize/folding logic natively and restore strict multi-dimensional tensor boundary handling.

* **Pointer Arithmetic & ImageDType Conversion**
  * **Quote:** "The `devectorizer.py` AST generator MUST replace `ImageDType` with `PtrDType(float)`."
  * **Impact:** Failing to convert the index back to a linear offset results in fatal native C compiler faults during pointer arithmetic.
  * **Action:** For `IMAGE=2`, use `image_dtype.base.ptr()` rather than converting the 1D channel index `x` into an `int2` vector.

* **Toolchain Isolation & Hermetic Boundaries**
  * **Quote:** "The RISC-V cross-compilers... must be installed on the host machine. Hermetic Build Exemption..."
  * **Impact:** Autonomous agents will pollute the hermetic Bazel workspace with host toolchains if not explicitly bounded.
  * **Action:** Agents must strictly enforce Cross-Compiler Testing Matrix & Toolchain Isolation. Exclusively `tinygrad` is allowed to use host cross-compilers; all other Bazel submodules must use strict hermetic toolchains.

* **Upstream Scope Management**
  * **Quote:** "Upstream components such as tinygrad 0-dimensional tensors are failing QA."
  * **Impact:** QA complaints on out-of-scope upstream frameworks pollute orchestration logs and block downstream progress.
  * **Action:** Agents MUST explicitly ignore and prune any QA complaints related to upstream components (e.g., tinygrad 0-dimensional tensors) to prevent orchestration pollution.

### Tier 2: System Architecture

* **Explicit Mocking of WMMA Graph Boundaries**
  * **Quote:** "wmma_arg = ('mock', ...)"
  * **Impact:** 100% unit test coverage hiding 0% systemic integration. Bypasses layout boundaries.
  * **Action:** Eradicate 'MOCK' strings and dynamically evaluate authentic tensor core bounds.
### Architectural Design & API Contracts
* **AXI Burst Segmentation Compiler Limits**
  * **Quote:** "AXI4 bursts must not exceed 256 beats."
  * **Impact:** Issuing generic DMA chunks of 4KB violates the AXI4 specification and hangs the system.
  * **Action:** Enforce strict AXI burst limits. For a 32-bit bus, the maximum burst is 1024 bytes. Segment `_dma_chunk` organically inside `emit_dma_async`.
*   **Mandate:** Ensure all unbuilt requirements and architectural designs reflect clear HW/SW boundaries, exact file paths, and strict API/ABI contracts. Use Mermaid for topology when defining tinygrad.
* [FLAG: invalid] **Targeted OSError Isolation in Pytest IPC (tinygrad)**
  * **Quote:** "Removing `OSError` from global exception masking causes `os.kill` to crash with `ProcessLookupError` during thread garbage collection."
  * **Impact:** Uncontrolled crashing during `atexit` prevents complete pytest session cleanup and spawns zombie workers.
  * **Action:** Global `OSError` exception muzzling (e.g., `except (AttributeError, KeyError, OSError): pass`) must be strictly eradicated to unmask actual deadlocks. However, targeted `except OSError: pass` wrappers MUST be retained explicitly around specific OS execution calls like `os.kill()` or `os.unlink()` where missing processes/files are an organically safe execution path.


* [FLAG: stale] **Testing Fraud vs Explicit Execution Bounds Verification**:
  * **Quote:** "Removing `try...except RuntimeError` in an explicit infinite loop recursion test causes an unhandled exception that breaks the entire test suite."
  * **Impact:** Blanket eradication of `try...except RuntimeError` without understanding context causes critical CI/CD pipeline crashes and fails to recognize explicit system validations.
  * **Action:** Agents MUST differentiate between "Exception Masking" (hiding real architectural boundaries with `assertRaises`) and "Explicit Limits Verification" (a test explicitly designed to verify a cyclic graph limit natively trapped and raised the expected error). Never remove `try...except RuntimeError` blocks from tests explicitly verifying that the system successfully rejected an infinite loop.


### Globally Relevant Execution Rules (Appended)

* **Dangerous Commands & Missing Teardown (Bazel Deadlocks)**
  * **Quote:** "Launching massive monolithic Bazel test suites concurrently without pkill -f bazel or resource constraints guarantees CI orchestrator timeouts."
  * **Impact:** CI timeout due to Bazel server memory exhaustion.
  * **Action:** Prepend and append `pkill -f "bazel" || true` to all Bazel batch executions. Apply strict `--local_resources=cpu=8 --local_resources=memory=HOST_RAM*0.5` flags.

* **Safe C++ API Boundaries**
  * **Quote:** "Natively generated APIs must utilize `absl::flat_hash_map::find()` rather than `.at(index)`."
  * **Impact:** Using `.at(index)` triggers C++ exceptions and `std::abort()` crashes on cache misses.
  * **Action:** Use `.find()` and gracefully return an `absl::NotFoundError` if an element misses.

### Verification Authenticity
* **Mocked API Bounds**: Masking exceptions like `assertRaises(RuntimeError)` on large tensors artificially evades hardware limits natively trapping.
* **Action**: Instead of catching execution traps, scale the test boundary (e.g. `seq_len`, `hidden_dim`) to fit naturally within the physical execution limits (e.g. 28KB DTCM Ping/Pong bounds).

### Orchestration Execution Insights (Cycle 166 - IPC Bridging)
* **Missing File Descriptor Cleanup in IPC `_respawn`**:
  * **Quote:** "Failing to close() old Pipe connections before respawning multiprocessing workers leaks file descriptors and starves shared memory."
  * **Impact:** Causes `OSError: cannot send (already closed?)` and crashes the entire `pytest-xdist` session during teardown.
  * **Action:** Always explicitly call `close()` on `parent_conn` and `child_conn` before recreating the `multiprocessing.Pipe(duplex=True)` in any worker `_respawn` method.

* [FLAG: invalid] **Targeted OSError Isolation for `Connection.send`**:
  * **Quote:** "The master process attempting to send data to a cleanly terminated worker throws OSError if the connection is already closed."
  * **Impact:** Crashing `pytest_sessionfinish` and spawning zombie processes.
  * **Action:** `Connection.send` wrappers MUST catch `OSError` alongside `BrokenPipeError` and `ConnectionResetError` to safely ignore severed IPC disconnections during teardown.

### Orchestration Execution Insights (Cycle 166)
* **Missing Linker Placement for .noinit**: Emitting `.noinit` in C++ is structurally insufficient. Inject a corresponding `*(.noinit)` memory segment mapping into the `coralnpu_tcm.ld.tpl` linker script to guarantee physical DTCM placement.
* **Exception Masking**: Wrapping evaluation execution blocks in `with self.assertRaises(RuntimeError):` is testing fraud and is strictly prohibited.

### Orchestration Execution Insights (Cycle 166)
* **IPC Teardown Deadlock**: Explicitly call `close()` on `parent_conn` and `child_conn` before recreating `multiprocessing.Pipe(duplex=True)` in worker `_respawn` methods to prevent file descriptor leaks and `OSError: cannot send (already closed?)` crashes during `pytest-xdist` teardown.
### Orchestration Execution Insights (Cycle 166)
* **IPC Garbage Collection Deadlocks**: Always call `close()` on `parent_conn` and `child_conn` before recreating `multiprocessing.Pipe` in `_respawn` methods to prevent `OSError: cannot send` crashes during pytest worker teardown.
* **Targeted OSError Exceptions**: Retain targeted `except OSError: pass` wrappers strictly around severed IPC disconnections or teardown routines, while unmasking it globally.
