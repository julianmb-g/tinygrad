# Tinygrad Execution Context & Lessons

### Pytest & Worker IPC
- **Subprocess Monkeypatching**: Monkeypatching `subprocess.Popen` to manage timeouts and bounds must strictly be scoped to individual pytest-xdist workers. Use `if getattr(request.config, "workerinput", None) is not None:` in `conftest.py`'s `pytest_configure` to prevent master-node deadlocks and catastrophic `OSError` crashes.
- **Worker CPU Exhaustion**: Never use `pytest -n auto`. Strict CPU bounding (`-n 4` or `-n 8`) is required to prevent node starvation which manifests as masking via infinite timeouts.
- **Multiprocessing Start Method**: When interfacing Python multiprocessing with C++ PyBind11 bindings, strictly enforce `multiprocessing.set_start_method("spawn")`. Do not use `fork()`, which duplicates lock state and deadlocks the runtime.
