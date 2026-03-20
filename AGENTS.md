# tinygrad Agent Instructions

## Lessons Learned

- **Exception Evasion Constraints**: Production execution code MUST NOT raise test framework exceptions (like `unittest.SkipTest`) to silently bypass execution failures. Use `assertRaises` instead for organically trapping limitations.
- **SQLite Cache Concurrency**: Enforce strictly serialized SQLite cache-layer tests (`-n 0` or isolated) to prevent concurrency deadlocks in pytest-xdist.
