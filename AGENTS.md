Repo-specific Lessons Learned:
- Tinygrad DTCM must partition arrays [Weights:Activations:Accumulators] mapping strictly inside 28KB.
- Enforce explicit OutOfMemoryError if a contiguous reduction axis cannot be mathematically split within these limits.
- When applying heuristics based on renderer attributes (like `device` or `max_upcast`), always wrap the attribute evaluation in a try/except block to handle API boundary drift (e.g. `ClangJITRenderer` attribute crashes).
