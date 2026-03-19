# Tinygrad Backend Lessons Learned
- **Test Framework Leaked into Production Runtime**: Production execution code MUST NOT raise test framework exceptions (like `unittest.SkipTest`) to silently bypass execution failures.
- **Evasion via expectedFailure**: Do not use `@unittest.expectedFailure` to mask the renderer's failure to enforce upper bounds on massive BSS allocations. Organically trap limitations via `assertRaises` or implement native bounds checking.
- **Fraudulent Deterministic Padding Shapes**: When regenerating or refactoring deterministic arrays in tests with `np.arange`, explicitly pass the correct target shape tuple directly to `math.prod()` to ensure array size matches the layout constraints.
- **Native Dependency Masking**: Ensure failure pipeline verification blocks are placed independently of native GCC compilation steps that might skip the test, avoiding bypassing of mock failure verification.
- **Rigid Substring Scraping in AST Validation**: Avoid strictly scraping generated C++ source code for structural validation. Evaluate exact structural values within the parsed AST or UOp arrays.
