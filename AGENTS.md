
## Lessons Learned

### Architecture Quirks
- **Memory Abstraction**: Implement a Virtual Memory Management (VMM) abstraction or pseudo-driver layer in `ops_coralnpu.py` to handle ITCM/DTCM mapping internally, preventing hardware abstraction leaks.

### Build Dependencies
- **Native Dependency Masking**: Ensure failure pipeline verification blocks are placed independently of native GCC compilation steps that might skip the test, avoiding bypassing of mock failure verification.

### Cycle 6 Discovered Constraints & Lessons
- **Bazel Test Scope Limitation (Python Submodules)**: When running cross-submodule zero-trust verification (`bazel test //...` inside `/workspace/louhi_ws/coralnpu`), Bazel executes hardware simulator and C++ targets natively. It inherently bypasses internal `pytest` suites inside external repositories like `tinygrad` unless explicitly bridged via `py_test`. Therefore, severe Python-level test framework deadlocks (such as SQLite locking issues) might not fail the global Bazel invocation. Submodule-specific Python testing pipelines must be explicitly evaluated outside the Bazel run before pointer serialization to ensure true zero-trust validation.

### Git & Environment Management
- **Zero-Trust Baseline Path Coupling**: When executing `bazel test --override_repository` inside an out-of-tree isolated baseline worktree (e.g. `/tmp/coralnpu-baseline`), you MUST strictly enforce absolute paths (e.g. `--override_repository=tinygrad=/workspace/louhi_ws/tinygrad`). Using relative paths (like `../tinygrad`) will fatally escape the temporary root and crash the dependency resolver because `/tmp/tinygrad` does not exist.

### Miscellaneous
- **AttributeError masking in setUpClass**: When implementing test preconditions in `setUpClass` (e.g. `Device.DEFAULT.sqtt_enabled`), safely fetch attributes using `getattr()` instead of dot-notation. Missing attributes (e.g., on the `CPUDevice`) will crash the `pytest` setup runner before the test can be organically skipped.
- **Evasion via expectedFailure**: Do not use `@unittest.expectedFailure` to mask the renderer's failure to enforce upper bounds on massive BSS allocations. Organically trap limitations via `assertRaises` or implement native bounds checking.
- **Fraudulent Deterministic Padding Shapes**: When regenerating or refactoring deterministic arrays in tests with `np.arange`, explicitly pass the correct target shape tuple directly to `math.prod()` to ensure array size matches the layout constraints.
- **High-Concurrency SQLite Caching (node down)**: When executing the test suite utilizing massive parallel workers (e.g., `pytest -n auto` with 128 workers), concurrent disk caching writes inside `tinygrad/helpers.py` to `compile_clang_jit` can cause aggressive `sqlite3.OperationalError: database is locked` exceptions, eventually cascading into `pytest-xdist` node pool collapses (`node down: Not properly terminated`). Configure resilient database journaling (like WAL) or restrict parallelization limits in `harness.yaml`.
- **IPC Deadlock Identification**: In tinygrad, if the pytest worker pool collapses with `node down: Not properly terminated` without tracebacks, this represents a Tier 1 Blocker indicating an out-of-band IPC simulator deadlock or zombie process crash. Ensure explicit `atexit` hooks and process group bounds are enforced for out-of-band execution.
- **Rigid Substring Scraping in AST Validation**: Avoid strictly scraping generated C++ source code for structural validation. Evaluate exact structural values within the parsed AST or UOp arrays.

### Testing Gotchas
- **Massive Test Bypassing & IPC Deadlocks**: Extensive skipping of tests early in a run followed by hard timeouts indicates structural execution evasion and fatal IPC deadlocks masquerading as passing subsets.
- **Orphaned Task Ledger Updating**: When resuming an aborted execution cycle, explicitly check the git commit history of submodules to verify if the previous tasks were already completed before repeating redundant codebase modifications. - **Phase 5 Ledger Serialization Atomicity**:: When checking off Phase 5 Submodule Pointer Serialization in `PLAN.md`, the root workspace orchestrator commit MUST atomcially stage the updated `PLAN.md` concurrently with the submodules (e.g. `coralnpu`, `tinygrad`) to tightly bind the pointer state to the checked-off plan phase.
- **Private Method Mocking (Framework Creep)**: Bypassing public ML APIs (like `.pool()`) via private interceptors (like `._pool()`) and injecting unauthorized integer exponentiation hacks into third-party ML frameworks (like tinygrad) purely to pass local tests is strictly forbidden. Known framework limitations must be organically trapped via `assertRaises`.
- **Test Framework Leaked into Production Runtime**: Production execution code MUST NOT raise test framework exceptions (like `unittest.SkipTest`) to silently bypass execution failures.
- **pytest-xdist IPC Worker Deadlock Resolution**: To natively resolve `pytest-xdist` locking deadlocks, strictly import `os`, `signal`, and `atexit` in the orchestrator entrypoint (`conftest.py`) and inject `os.setpgrp()` on workers with an `atexit.register(lambda: os.killpg(0, signal.SIGKILL))` hook to cleanly eradicate zombie processes.
