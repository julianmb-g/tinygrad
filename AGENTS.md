- In tests, avoid using `allow_scale=True` to shrink matrix dimensions down for non-CDNA4 execution, as it structurally bypasses assertion failures and fakes passing boundaries.
- When resolving tinygrad exceptions, ensure exception classes like ProcessLookupError and BufferError are explicitly asserted rather than blindly passed to avoid masking deadlocks.
\n- Always drop `pytest -x` early aborts in harness.yaml if an overarching test like test_asm_gemm.py masks all downstream detonating stubs and blinds coverage.
