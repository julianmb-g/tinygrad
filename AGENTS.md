# Tinygrad Orchestration Guidelines

- Ensure tests natively invoke `if __name__ == '__main__': unittest.main()`.

- Prevent OOM deadlocks by avoiding `size='enormous'` targets.

### Orchestration Execution Insights (Cycle 167)
* **Missing Linker Placement for .noinit**: Emitting `.noinit` in C++ is structurally insufficient if the upstream linker script lacks a `*(.noinit)` rule mapped to DTCM. Inject a corresponding `*(.noinit)` memory segment mapping into the `coralnpu_tcm.ld.tpl` linker script to guarantee physical DTCM placement.
