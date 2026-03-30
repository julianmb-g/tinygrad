# Tinygrad Orchestration Guidelines

## System Rules & Architectural Guidelines

### Build & Execution Environment
* **Test Invocation**
  * **Action:** Ensure tests natively invoke `if __name__ == '__main__': unittest.main()`.
* **OOM Deadlock Prevention**
  * **Impact:** `size='enormous'` targets cause orchestrator memory deadlocks.
  * **Action:** Prevent OOM deadlocks by strictly avoiding `size='enormous'` in Bazel targets. Downgrade to `size='large'`.

### Memory Placement & Linker Topology
* **Missing Linker Placement for .noinit**
  * **Impact:** Emitting `.noinit` in C++ is structurally insufficient if the upstream linker script lacks a `*(.noinit)` rule mapped to DTCM. This breaks hardware tensor buffer mappings.
  * **Action:** Inject a corresponding `*(.noinit)` memory segment mapping explicitly into the `coralnpu_tcm.ld.tpl` linker script to guarantee physical DTCM placement.