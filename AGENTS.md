# tinygrad Agent Instructions


## Lessons Learned

### Architecture Quirks
- **Memory Abstraction**: Implement a Virtual Memory Management (VMM) abstraction or pseudo-driver layer in `ops_coralnpu.py` to handle ITCM/DTCM mapping internally, preventing hardware abstraction leaks.

### Build Dependencies
- **Native Dependency Masking**: Ensure failure pipeline verification blocks are placed independently of native GCC compilation steps that might skip the test, avoiding bypassing of mock failure verification.

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
- **Asynchronous Subprocess Teardown**: When managing teardowns for `pytest` worker pools, explicitly monkeypatch `subprocess.Popen` inside `conftest.py` to add process IDs to a global `active_pids` set. This guarantees that all asynchronously spawned lifecycles are explicitly eradicated during `atexit` teardown using `os.kill`, preventing orphaned zombie processes from causing deadlock timeouts.
- **Massive Test Bypassing & IPC Deadlocks**: Extensive skipping of tests early in a run followed by hard timeouts indicates structural execution evasion and fatal IPC deadlocks masquerading as passing subsets.
- **Non-Deterministic Garbage Padding**: Tests utilizing `Tensor.empty` (like `test_vdot_mapping`) can generate non-deterministic memory garbage and cause erratic CI behaviors. Replace with deterministic constants or zeros.
- **Orphaned Task Ledger Updating**: When resuming an aborted execution cycle, explicitly check the git commit history of submodules to verify if the previous tasks were already completed before repeating redundant codebase modifications.
- **Phase 5 Ledger Serialization Atomicity**: When checking off Phase 5 Submodule Pointer Serialization in `PLAN.md`, the root workspace orchestrator commit MUST atomcially stage the updated `PLAN.md` concurrently with the submodules (e.g. `coralnpu`, `tinygrad`) to tightly bind the pointer state to the checked-off plan phase.
- **Test Framework Leaked into Production Runtime**: Production execution code MUST NOT raise test framework exceptions (like `unittest.SkipTest`) to silently bypass execution failures.
- **Test Math Dependencies**: When generating tensor shapes mathematically in tests like `test_fp8_linear.py`, ensure `import math` is explicitly present at the top level to prevent `NameError` execution crashes. Also ensure `math.prod` is fed valid tuples instead of undeclared variables.

