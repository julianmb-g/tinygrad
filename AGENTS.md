Repo-specific Lessons Learned:
- Tinygrad DTCM must partition arrays [Weights:Activations:Accumulators] mapping strictly inside 28KB.
- Enforce explicit OutOfMemoryError if a contiguous reduction axis cannot be mathematically split within these limits.
