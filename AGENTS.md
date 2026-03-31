# Tinygrad Submodule Orchestration Guidelines (AGENTS.md)

## Lessons Learned
* **IPC Garbage Collection Traps**: Enforce targeted OS exception handling (`FileNotFoundError`, `ProcessLookupError`) in `__del__` GC lifecycles instead of `OSError` to prevent shared memory deadlocks during multiprocessing teardowns.
* **Pytest Teardown Sanitization**: Ensure explicit `pkill` boundaries exist in `pytest_sessionfinish` to forcefully clean up orphaned execution processes.
