# tinygrad Agent Instructions

## Lessons Learned

### Architecture Quirks
- **Architecture Update:** Added a dynamic `coralnpu` backend compiling UOps into GCC RISC-V Zve32x code. Simulates out-of-band execution using an embedded neural network cost model (`BEAM_COST`) for auto-tuning.
- **AST Limits:** Enforce `max_upcast = 32` in the custom renderer (e.g. `CoralNPURenderer`) to prevent massive AST explosion during auto-vectorization passes.
- **Dequantization FPU Dependency:** Dequantizing via FP16 scaling factors after a VDOT reduction loop still requires floating-point operations. If the hardware lacks an FPU, this will trigger an illegal instruction trap.
- **The Phantom DMA Engine Paradox:** An in-order scalar core cannot interleave an AXI `memcpy` loop with Vector ALU execution without a dedicated autonomous DMA controller. Memory fetch UOp passes must be strictly synchronous unless hardware DMA exists.
- **W8A8 Activation Quantization Void:** Before issuing an INT8 vector MAC operation (`VDOT`), the incoming `fp16` activation tensors MUST be dynamically quantized; issuing `x_int8.dot(w_int8)` directly onto unquantized streams produces mathematical garbage.

### Build Dependencies
- **Makefile vs Bazel:** Do not assume `make test` is valid for Bazel-centric repositories. Ensure execution scripts explicitly use Bazel targets.
- **Test Dependencies:** Ensure `huggingface_hub`, `pytest-xdist`, and `pytest-timeout` are installed.

### Git & Environment Management
- **Bazel Test Runfiles vs. Submodule Sync:** When running `bazel test` with an external repository override for an out-of-band submodule, ensure its `BUILD` file correctly exports the updated module structures or dependencies. Missing overrides cause `ModuleNotFoundError`.

### Testing Gotchas
- **Architecture Update (Integration of ML to HW):** The ecosystem supports an end-to-end integration path. Tinygrad compiles to GCC RISC-V Zve32x C++, verified against the `mpact` instruction simulator (with Map-Elites) and `coralnpu` hardware co-simulation.
- **CoralNPURenderer Unit Testing:** When testing AST limit boundaries (e.g. `RuntimeError` due to >32 float allocations), nodes must have identical depth constraints. Chaining dependents results in scalarized depths preventing the max-width check from exceeding limits. Use `unittest.mock.patch` to sidestep strict file constraints for missing ML models during environment-agnostic tests.
- **CPU-Only Backend Enforcement:** To prevent ROCm/GPU compiler crashes in the orchestrator container, tests must forcefully use the CPU backend and ignore hardware-specific folders using `CPU=1 pytest --timeout=120 --timeout-method=thread test/backend/ test/unit/ test/null/ test/test_tiny.py`.
- **Hardware Backend Testing:** The `coralnpu` rendering and runtime backend must have dedicated unit tests inside the ML frontend (`tinygrad/test/`), rather than blindly relying on out-of-band simulator validation.
- **Hardware Co-Simulation Flow:** The hardware integration spans from `tinygrad` C++ kernels loaded as RISC-V ELF binaries into the `mpact` ISS to `cocotb` testbenches orchestrating Verilog/SystemVerilog RTL.
- **Missing ROCm Device Library:** In environments lacking the ROCm device library, AMD handwritten tests (like `test_mfma_fp8`) must be explicitly skipped using `@unittest.skip` to prevent CI failures.
- **ML Cost Validation:** Tests relying on ML cost model assertions must NOT filter by `cost > 0`. Because stochastic noise was replaced by deterministic outputs, the predicted cost for models can legitimately evaluate to exactly `0.0`.
- **Test Fidelity on Missing Dependencies:** Tests explicitly targeting specific LLVM CPU targets or missing device libraries should use `@unittest.skip` instead of generic exception swallowing.
- **Test Suite Granularity:** Comprehensive test creation tasks must be explicitly decomposed into distinct, atomic test creation tasks.
- **Zero-Divisor Resolution via Offset:** When replacing standard normal distributions with uniform/positive-only distributions to prevent zero-divisors, apply a constant offset (like `Tensor.rand() + 1.0`) to guarantee mathematical stability during `assert_allclose`.
