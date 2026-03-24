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
