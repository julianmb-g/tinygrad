# tinygrad Agent Instructions

## Lessons Learned & Orchestration Rules

### Tier 1: Critical Blocker

* **Complete Test Erasure via Skipping**
  * **Quote:** "In `test_uop_graph.py`, failing tests... were blanketed with `@unittest.skip('invalid uops')`."
  * **Impact:** Erasing execution failures from the CI pipeline entirely is catastrophic structural masking. Claiming a pipeline is fixed by literally skipping the graph validations is testing fraud.
  * **Action:** Never use `@unittest.skip('invalid uops')` or similar blanket decorators to hide failing graph validations or execution tests. Fix the underlying logic natively. Agents must rigorously cross-verify `PLAN.md` against actual codebase state before attempting tasks.

* **Toolchain Isolation & Hermetic Boundaries**
  * **Quote:** "The RISC-V cross-compilers... must be installed on the host machine. Hermetic Build Exemption..."
  * **Impact:** Autonomous agents will pollute the hermetic Bazel workspace with host toolchains if not explicitly bounded.
  * **Action:** Agents must strictly enforce Cross-Compiler Testing Matrix & Toolchain Isolation. Exclusively `tinygrad` is allowed to use host cross-compilers; all other Bazel submodules must use strict hermetic toolchains.

* **Upstream Scope Management**
  * **Quote:** "Upstream components such as tinygrad 0-dimensional tensors are failing QA."
  * **Impact:** QA complaints on out-of-scope upstream frameworks pollute orchestration logs and block downstream progress.
  * **Action:** Agents MUST explicitly ignore and prune any QA complaints related to upstream components (e.g., tinygrad 0-dimensional tensors) to prevent orchestration pollution.

* **Fuzzer IPC Boundaries**
  * **Quote:** "Subprocess worker pools, fork bombs, and IPC boundaries are strictly forbidden for the Map-Elites fuzzer pipeline."
  * **Impact:** IPC usage in the fuzzer causes catastrophic system instability and process leaks.
  * **Action:** Mutator ASTs must natively invoke `EncodeSafe()` and return `absl::NotFoundError`. Do not apply globally; Tinygrad Auto-Tuner is permitted multiprocessing.

* **Pointer Arithmetic & ImageDType Conversion**
  * **Quote:** "The `devectorizer.py` AST generator MUST replace `ImageDType` with `PtrDType(float)`."
  * **Impact:** Failing to convert the index back to a linear offset results in fatal native C compiler faults during pointer arithmetic.
  * **Action:** For `IMAGE=2`, use `image_dtype.base.ptr()` rather than converting the 1D channel index `x` into an `int2` vector.

* **Graph Evaluation & Ast Unbound Constraints**
  * **Quote:** "Never extract `.schedule()` from the actual tensor intended to be evaluated. Explicitly calling `.realize()` within the loop prevents unbound AST explosion."
  * **Impact:** Consumes lazy computation graphs permanently or causes recursive failure.
  * **Action:** Use structurally independent dummy tensors to extract ASTs before evaluating the true target hardware tensor. Strictly forbid `SkipTest` for cyclic limits. Evaluate eager boundaries natively. Organically trap `RecursionError` via `assertRaises` only for negative cases.

* **Allocation & Math Bounds Evaluation**
  * **Quote:** "Tests evaluating allocation and OOM bounds must naturally fail and trap organically using native Python boundaries."
  * **Impact:** Masking successful limits via `assertRaises` breaks critical hardware validation.
  * **Action:** Explicitly validate DTCM limits. Use `int(math.fmod())` and `int(x / y)` to correctly simulate C-style truncation towards zero. Provide `dtype=dtypes.float` when initializing variables for transcendental functions. For normal image environments (`IMAGE!=2`), explicitly upcast scalar pixel stores to `vec(4)` before emitting to `write_imagef`.

* **Backward Compatibility & Crash Tracing**
  * **Quote:** "Abstracting root cause locations hides the source of crashes. Legacy keyword arguments must be strictly preserved via `**kwargs`."
  * **Impact:** Causes cascading API contract breakages and obscures exact component failures.
  * **Action:** Explicitly fix files instead of relying on broad mocking. Ensure safe fallbacks via `**kwargs` when refactoring core APIs like `UOp.cast`.

* **Graph Validation Failures (SCHEDULE ISSUE)**
  * **Quote:** "tinygrad Graph Validation Failures: `TypeError: unsupported operand type(s) for -: 'tuple' and 'int'` and `SCHEDULE ISSUE`."
  * **Impact:** Core regressions exist in graph scheduling and multi-dimensional tensor boundary calculations.
  * **Action:** Fix the UOp vectorize/folding logic natively and restore strict multi-dimensional tensor boundary handling.

### Tier 2: System Architecture

* **Explicit Mocking of WMMA Graph Boundaries**
  * **Quote:** "wmma_arg = ('mock', ...)"
  * **Impact:** 100% unit test coverage hiding 0% systemic integration. Bypasses layout boundaries.
  * **Action:** Eradicate 'MOCK' strings and dynamically evaluate authentic tensor core bounds.
### Architectural Design & API Contracts
*   **Mandate:** Ensure all unbuilt requirements and architectural designs reflect clear HW/SW boundaries, exact file paths, and strict API/ABI contracts. Use Mermaid for topology when defining tinygrad.
* **IPC Thread Synchronization ()**\n  * **Quote:** "Explicitly join detached background threads before the main thread exits."\n  * **Impact:** Prevents asynchronous worker race conditions and shared memory GC deadlocks during teardown.\n  * **Action:** Synchronize IPC thread termination by explicitly joining all non-daemon threading.Thread instances and cleanly wait on PIDs during teardown.
* **IPC Thread Synchronization (`pytest-xdist`)**\n  * **Quote:** "Explicitly join detached background threads before the main thread exits."\n  * **Impact:** Prevents asynchronous worker race conditions and shared memory GC deadlocks during teardown.\n  * **Action:** Synchronize IPC thread termination by explicitly joining all non-daemon threading.Thread instances and cleanly wait on PIDs during teardown.
