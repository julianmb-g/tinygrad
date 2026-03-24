# tinygrad Agent Instructions

## Lessons Learned

### Architecture Quirks
- **AXI4 Protocol Bounds**: AXI bursts for tensor fetches (>4KB) must be segmented to strictly align to and not cross 4KB physical address boundaries.
- **Metal Objective-C bindings**: macOS Objective-C message forwarding on Linux will crash with AttributeError. When importing macOS-specific backend bindings (`ops_metal.py`) on Linux, use strict `if sys.platform == 'darwin':` guards to cleanly disable the bindings and raise `NotImplementedError` rather than allowing violent segmentation faults. Explicitly wrap macOS tests with `@unittest.skipIf(sys.platform != 'darwin', "macOS only")`.
- **VMM Bump Allocator Memory Leak**: Linear accumulation without reuse guarantees OOM on sequential executions. Update the VMM spec to implement true memory lifecycle (e.g., arena resets per-inference or tensor address ref-counting).

### Build Dependencies
- **PyBind11 Out-of-Band Isolation**: Native PyBind11 bindings inside the Python tuning loop cause crashes. Extract and wrap calls via `multiprocessing` workers using zero-copy shared memory.

### Testing & E2E Validation
- **150-Character Line Limit Refactoring (Ruff E501)**: When formatting long dictionary comprehensions or lambda operator mappings to conform strictly to bounds, extract them into multi-line structures rather than relying on horizontal squashing.
- **API Contract Breakages**: Do not silently remove `device` kwargs from `reshape` without providing a compatible ABI fallback or updating all downstream tests, as this triggers sweeping integration failures. Inject hardware keywords natively into generator constructors (`Tensor.arange(..., device=X)`).
- **Bazel Test Scope Limitation (Python Submodules)**: Cross-submodule zero-trust verification (`bazel test //...`) bypasses internal `pytest` suites inside external repositories like `tinygrad`. Submodule-specific Python pipelines must be explicitly evaluated outside the Bazel run before pointer serialization to ensure true zero-trust validation.
- **CPU Backend `bfloat16` Compilation Bug**: In the CPU backend, `__bf16` casting is broken because it truncates floats to integers before bitcasting. Do not assert mathematical parity for `bfloat16` on CPU until the compiler bug is fixed.
- **CStyleLanguage Context Initialization**: When extracting dynamically generated attributes from `CoralNPURenderer` context inside `CStyleLanguage` rewrites, use `getattr(ctx, 'dtcm_bump', 4096)`. Manual `render_kernel` invocations bypass standard `render()` initialization.
- **EXTMEM Boundary & Allocator Proportionality**: When migrating large buffers to `.extdata` to test external bus limits, proportionally reduce static allocators (like `kTensorArenaSize`) mapped to the same region. Failing to scale down forces `ld.bfd` to violently overflow the rigid `EXTMEM` hardware definition.
- **Kernel Optimization Upcast Bounds**: Ensure DTCM tensor allocation limits are strictly enforced (e.g., `max_upcast = 28`) to prevent upcast bounds violations.
- **Lazy Tensor Buffer Sharing Assertions**: When testing lazy tensor assignments, underlying buffers are shared. Both variables organically evaluate to the mutated value. Use `assertEqual` instead of `assertNotEqual` to accurately validate the shared execution behavior without relying on `@unittest.expectedFailure` masking.
- **Memory GC Deadlocks**: Parallel worker crashes and timeouts can be caused by catastrophic `KeyError` loops within `UOp.__del__` garbage collection. When parallel tests teardown, dictionary entries in `UOpMetaClass.ucache` may be popped before all references expire. Fix this organically by updating the destructor logic to natively trap the missing key (`except (AttributeError, KeyError): pass`) rather than swallowing all exceptions (`except Exception: pass`).
- **NaN Validation in Memory Captures**: When restoring `before` memory assertions, preserve `math.isnan` logic alongside restored assertions to prevent tests from spuriously breaking during Python memory dump equality checks.
- **OpenCL Backend Cross-Compilation Trap**: Do not feed OpenCL-specific types (`write_only`, `sampler_t`) into C target compilers. Map backend type `'l'` correctly to `int64` or `long` in C-style renderers.
- **Operation Count Bounds Enforcement**: Enforcing operation bounds (e.g. `< 1`) that assume `WMMA` execution on `AMD` fails on `CPU`. Ensure `ops_scale` degrades gracefully so performance bounds degrade deterministically.
- **Renderer Architectural Identity**: When extending C-style renderers (`ClangJITRenderer`), ensure `arch` property is explicitly defined (e.g., `arch = "CORALNPU"`). Do not hardcode renderer strings like "CORALNPU" into general optimizers; use `getattr(self.ren, "max_upcast", <default>)`.
- **SQLite Caching & Pytest Xdist Locking**: Concurrent access to shared SQLite layers (`compile_clang_jit`) causes database locking. Mandate isolated database connections scaled per-worker (`PYTEST_XDIST_WORKER`), use WAL journaling, and strictly ensure in-memory table registries are flushed correctly during cache clears.
- **Test Fidelity via Bounds Checking Preservation**: Deleting `np.iinfo` float-to-int bounds checks natively forces testing framework assertions onto C/C++ Undefined Behavior. Do not delete organic constraints.
- **Test Integrity (Non-Deterministic Padding)**: Avoid uninitialized `Tensor.empty` which generates non-deterministic memory garbage (`NaN`s) causing CI validation failures. Explicitly initialize datasets deterministically (`Tensor.full`, `math.prod(shape)`). Bound scaling input tensors (e.g., `% 10 * 0.1`) to prevent FP16 mathematical overflows. Array initializations must mathematically evaluate to the correct flattened target size without modulo padding that crashes reshaping.
- **Test Math Dependencies**: Ensure `import math` is present at the top level to prevent `NameError` crashes when mathematically generating tensor shapes. `math.prod` must be fed valid tuples (e.g., `(1, 1, 32, 32, 32)`), not generic shapes that evaluate to size 1.
- **Unsupported Backend Feature Testing**: Tests asserting linearizer functionality that natively requires specific backends (e.g. `MetalRenderer()`) MUST use `unittest.SkipTest("Reason")` to organically trap the boundary rather than abruptly crashing.

### AST Metadata Persistence & Cache Leaks
- **UOpMetaClass Caching & Metadata Bleed**: Because `all_metadata` is tracked out-of-band and the core `UOpMetaClass.__call__` caches nodes based purely on `(op, dtype, src, arg, tag)`, structurally identical execution graphs (e.g. `Ops.MUL` from unrelated matrix computations in different unit tests) will hit the cache. If the unique `.tag` property is stripped or not utilized, metadata attached via `TRACEMETA` wrapper in prior tests will organically bleed into subsequent isolated tests, causing false assertions like `len(si.metadata) == 2` instead of `1`. To isolate unit tests tracking metadata, explicitly clear `UOpMetaClass.ucache` and `all_metadata` in `setUp()`.


# Lessons Learned (Tinygrad)
- **Auto-Tuner Concurrency:** The Tinygrad auto-tuner is explicitly allowed to use Python `multiprocessing.Process` pools. The strict IPC/Worker Pool ban applies only to the C++ Map-Elites Fuzzer, not to Tinygrad.
- **DTCM Double-Buffering Mathematical Closure:** The 28KB `AddrSpace.LOCAL` allocation must explicitly account for all segments, including the 4KB output/accumulator boundary to prevent silent `.bss` overlap or capacity evasion.
- **Pytest Worker Exhaustion:** Never use `-n auto` in pytest configuration files (e.g. `run_all_tests.py`, github workflows) as it can cause fatal timeouts, CPU starvation, and undetected infinite loops by over-subscribing workers (e.g., spawning 128 workers). Strictly enforce a maximum of 4 parallel workers globally (e.g. `-n 4`).
- **Google Python Style Guide Enforcement:** All Python imports must strictly reside at the top of the file. Inline or deferred imports (e.g., inside function bodies or class methods) create tight coupling and are forbidden. Also, ensure trailing whitespaces are stripped via linters like `ruff`.
