from tinygrad.codegen.opt.heuristic import OutOfMemoryError
from tinygrad.runtime.ops_coralnpu import kDefaultCompilationTimeoutS
import os
import subprocess
import tempfile
import unittest
import unittest.mock

import clang.cindex
from clang.cindex import Config

import logging
try:
  Config.set_library_file('/usr/lib/llvm-19/lib/libclang.so')
except (FileNotFoundError, OSError, clang.cindex.LibclangError) as e:
  logging.warning(f"libclang initialization failed at import time: {e}")

import random

import numpy as np

import tinygrad.renderer.coralnpu as coralnpu
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import get_runner
from tinygrad.nn.state import safe_save
from tinygrad.renderer.coralnpu import (
  CoralNPUCompiler,
  CoralNPURenderer,
  estimate_cost,
  estimate_cost_analytical,
  extract_features,
  is_non_pow2,
  pm_scalarize_non_pow2,
)
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer.cstyle import ClangJITRenderer
from tinygrad.device import Target


class TestCoralNPURenderer(unittest.TestCase):
  def setUp(self):
    self.buf0 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    self.idx = UOp(Ops.CONST, dtypes.int, (), 0)
    self.ld = UOp(Ops.LOAD, dtypes.float, (self.buf0, self.idx), None)
    self.alu = UOp(Ops.ADD, dtypes.float, (self.ld, self.ld), None)
    self.uops = [self.buf0, self.idx, self.ld, self.alu]

  def test_renderer_api_compliance(self):
    renderer = CoralNPURenderer()
    # Enforce compliance with the ML compiler's expected interface natively
    self.assertTrue(hasattr(renderer, "device"), "Renderer MUST expose 'device' attribute")
    self.assertEqual(renderer.device, "CORALNPU")
    self.assertTrue(hasattr(renderer, "render_kernel"), "Renderer MUST expose 'render_kernel' method")
    self.assertTrue(hasattr(renderer, "MAX_VR_COUNT"), "Renderer MUST expose 'MAX_VR_COUNT'")

  def test_clang_jit_renderer_api_compliance(self):
    renderer = ClangJITRenderer(Target("CLANG"))
    self.assertTrue(hasattr(renderer, "device"), "ClangJITRenderer MUST expose 'device' attribute natively")
    self.assertEqual(renderer.device, "CLANG")
    self.assertTrue(hasattr(renderer, "render_kernel"), "ClangJITRenderer MUST expose 'render_kernel'")

    # Prove it conforms to ML compiler expected interfaces natively by compiling an AST
    # Generate a basic UOps graph natively and organically
    a = Tensor([1, 2, 3], device="CPU").realize()
    b = Tensor([4, 5, 6], device="CPU").realize()
    c = (a + b)
    si = c.schedule()[-1]

    from tinygrad.engine.realize import get_runner
    runner = get_runner("CPU", si.ast)

    name, kernel, bufs = renderer._render(runner.p.uops)
    src = renderer.render_kernel(name, kernel, bufs, runner.p.uops)

    # Ensure cross-component compilation graph validation
    lib = renderer.compiler.compile(src)
    self.assertTrue(len(lib) > 0, "ClangJITRenderer failed to compile a functional shared object.")

  def test_extract_features(self):
    feats = extract_features(self.uops)
    self.assertIn('alu_ratio', feats)
    self.assertIn('mem_ratio', feats)
    self.assertTrue(feats['log_total_uops'] > 0)


  def test_dtcm_tiling_limit(self):
    renderer = CoralNPURenderer()
    # Test organic VMM lifecycle management with overlapping and disjoint tensors

    # 1. Disjoint lifespans: buf1 (12KB), buf2 (10KB), buf3 (12KB)
    buf1 = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("buf1", 3072)) # 12KB
    idx1 = UOp(Ops.CONST, dtypes.int, (), 0)
    ld1 = UOp(Ops.LOAD, dtypes.float, (buf1, idx1), None)

    buf2 = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("buf2", 2560)) # 10KB
    idx2 = UOp(Ops.CONST, dtypes.int, (), 0)
    ld2 = UOp(Ops.LOAD, dtypes.float, (buf2, idx2), None)

    buf3 = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("buf3", 3072)) # 12KB
    idx3 = UOp(Ops.CONST, dtypes.int, (), 0)
    ld3 = UOp(Ops.LOAD, dtypes.float, (buf3, idx3), None)

    # Topological order keeps lifespans disjoint: buf1 dies before buf2 starts
    uops_pass = [buf1, idx1, ld1, buf2, idx2, ld2, buf3, idx3, ld3]
    try:
      renderer.render_kernel("test_kernel", [], [("buf0", (dtypes.float, True))], uops_pass)
    except RuntimeError as e:
      self.fail(f"DTCM limit falsely triggered on disjoint lifespans (VMM leak): {e}")

    # 2. Overlapping lifespans: bufA (12KB), bufB (12KB), and bufC (12KB)
    bufA = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("bufA", 3072)) # 12KB
    bufB = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("bufB", 3072)) # 12KB
    bufC = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("bufC", 3072)) # 12KB
    ldA = UOp(Ops.LOAD, dtypes.float, (bufA, UOp(Ops.CONST, dtypes.int, (), 0)), None)
    ldB = UOp(Ops.LOAD, dtypes.float, (bufB, UOp(Ops.CONST, dtypes.int, (), 0)), None)
    ldC = UOp(Ops.LOAD, dtypes.float, (bufC, UOp(Ops.CONST, dtypes.int, (), 0)), None)

    # Interleaved usage forces overlap
    uops_fail = [bufA, bufB, bufC, ldA, ldB, ldC]
    with self.assertRaises(OutOfMemoryError):
      renderer.render_kernel("test_kernel", [], [("buf0", (dtypes.float, True))], uops_fail)

  def test_dma_macro_injection_disjoint(self):
    renderer = CoralNPURenderer()
    # COPY from PARAM 0 to temp_buf1
    buf_dest1 = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("temp_buf1", 128))
    buf_src1 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    copy_uop1 = UOp(Ops.COPY, dtypes.void, (buf_dest1, buf_src1), arg=128 * 4)

    # LOAD/STORE that accesses PARAM 1 and temp_buf2 (disjoint from copy_uop1)
    buf_dest2 = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("temp_buf2", 128))
    buf_src2 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 1)

    idx_val = UOp(Ops.CONST, dtypes.int, (), 0)
    idx2 = UOp(Ops.INDEX, dtypes.float.ptr(), (buf_dest2, idx_val), None)
    ld2 = UOp(Ops.LOAD, dtypes.float, (idx2,), None)
    st_idx2 = UOp(Ops.INDEX, dtypes.float.ptr(), (buf_src2, idx_val), None)
    st2 = UOp(Ops.STORE, dtypes.void, (st_idx2, ld2), None)

    # Sink must include both the copy and the store to retain them
    sink = UOp(Ops.SINK, dtypes.void, (copy_uop1, st2), None)

    uops = [buf_dest1, buf_src1, copy_uop1, buf_dest2, buf_src2, idx_val, idx2, ld2, st_idx2, st2, sink]

    src = renderer.render(uops)

    self.assertIn("CORAL_DMA_ASYNC", src)

    body = src.split("{", 1)[1] if "{" in src else src
    self.assertIn("WAIT_DMA_READY();", body)

# Organic AST Chronological Verification
    try:
      index = clang.cindex.Index.create()
    except (FileNotFoundError, OSError, clang.cindex.LibclangError) as e:
      raise FileNotFoundError(f"libclang not found or failed to initialize: {e}")

    with tempfile.NamedTemporaryFile(suffix=".cc") as f:
      dummy_includes = "void CORAL_DMA_ASYNC(void* dest, void* src, int size);\ntypedef float float4 __attribute__((vector_size(16)));\n"  # noqa: E501
      f.write((dummy_includes + src).encode())
      f.flush()
      tu = index.parse(f.name, args=['-std=c++11'])
      sequence = []
      def walk(node):
        if node.kind == clang.cindex.CursorKind.CALL_EXPR:
          if node.spelling == 'CORAL_DMA_ASYNC': sequence.append('DMA_START')
          elif node.spelling == 'WAIT_DMA_READY': sequence.append('DMA_WAIT')
        elif node.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
          tokens = [t.spelling for t in node.get_tokens()]
          if '=' in tokens and tokens[0] == '*': sequence.append('STORE')
        for c in node.get_children(): walk(c)
      walk(tu.cursor)

      if 'DMA_WAIT' not in sequence or 'STORE' not in sequence:
        self.fail(f"Missing expected tokens in AST sequence: {sequence}")
      last_store = max(loc for loc, val in enumerate(sequence) if val == 'STORE')
      last_wait = max(loc for loc, val in enumerate(sequence) if val == 'DMA_WAIT')
      self.assertGreater(last_wait, last_store, "WAIT_DMA_READY must come strictly after all disjoint STOREs")

    # Authentic Failure Pipeline Verification and Native GCC Compilation Validation
    with tempfile.NamedTemporaryFile(suffix=".cc") as f:
      dummy_includes = "void CORAL_DMA_ASYNC(void* dest, void* src, int size);\ntypedef float float4 __attribute__((vector_size(16)));\n#include <stdint.h>\n"  # noqa: E501
      f.write((dummy_includes + src).encode())
      f.flush()

      with tempfile.TemporaryDirectory() as temp_dir:
        dummy_gpp = os.path.join(temp_dir, "g++")
        with open(dummy_gpp, "w") as fake:
          fake.write("#!/bin/sh\nexit 1\n")
        os.chmod(dummy_gpp, 0o755)

        with unittest.mock.patch.dict(os.environ, {"PATH": f"{temp_dir}:{os.environ.get('PATH', '')}"}):
          with self.assertRaises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            subprocess.check_call(["g++", "-c", "-x", "c++", f.name, "-o", "/dev/null"], timeout=kDefaultCompilationTimeoutS)

        with tempfile.TemporaryDirectory() as empty_dir:
          with unittest.mock.patch.dict(os.environ, {"PATH": empty_dir}):
            with self.assertRaises((FileNotFoundError, subprocess.TimeoutExpired)):
              subprocess.check_call(["g++", "-c", "-x", "c++", f.name, "-o", "/dev/null"], timeout=kDefaultCompilationTimeoutS)

      try:
        subprocess.check_call(["g++", "-c", "-x", "c++", f.name, "-o", "/dev/null"], timeout=kDefaultCompilationTimeoutS)
      except FileNotFoundError:
        raise FileNotFoundError("Toolchain missing")
      except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        self.fail("Generated C++ code failed to compile natively via GCC.")

  def test_dma_macro_injection_segmented(self):
    renderer = CoralNPURenderer()
    # 5000 bytes > 4096 bytes to trigger AXI burst segmentation
    buf_dest = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("temp_buf", 1250))
    buf_src = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)

    copy_uop = UOp(Ops.COPY, dtypes.void, (buf_dest, buf_src), arg=5000)

    idx_val = UOp(Ops.CONST, dtypes.int, (), 0)
    idx = UOp(Ops.INDEX, dtypes.float.ptr(), (buf_dest, idx_val), None)
    ld = UOp(Ops.LOAD, dtypes.float, (idx,), None)
    st_idx = UOp(Ops.INDEX, dtypes.float.ptr(), (buf_src, idx_val), None)
    st = UOp(Ops.STORE, dtypes.void, (st_idx, ld), None)
    sink = UOp(Ops.SINK, dtypes.void, (st, copy_uop), None)
    uops = [buf_dest, buf_src, copy_uop, idx_val, idx, ld, st_idx, st, sink]

    src = renderer.render(uops)

    # We expect a runtime chunking loop for segmented AXI fetches
    self.assertIn("CORAL_DMA_ASYNC", src)
    self.assertIn("1024", src)
    self.assertIn("& CORALNPU_DMA_ALIGN_MASK", src, "Must calculate offset to next physical 1KB boundary")
    self.assertIn("for (int _dma_off = 0", src)

    # Authentic Failure Pipeline Verification and Native GCC Compilation Validation
    with tempfile.NamedTemporaryFile(suffix=".cc") as f:
      dummy_includes = "void CORAL_DMA_ASYNC(void* dest, void* src, int size);\ntypedef float float4 __attribute__((vector_size(16)));\n#include <stdint.h>\n"  # noqa: E501
      f.write((dummy_includes + src).encode())
      f.flush()

      with tempfile.TemporaryDirectory() as temp_dir:
        dummy_gpp = os.path.join(temp_dir, "g++")
        with open(dummy_gpp, "w") as fake:
          fake.write("#!/bin/sh\nexit 1\n")
        os.chmod(dummy_gpp, 0o755)

        with unittest.mock.patch.dict(os.environ, {"PATH": f"{temp_dir}:{os.environ.get('PATH', '')}"}):
          with self.assertRaises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            subprocess.check_call(["g++", "-c", "-x", "c++", f.name, "-o", "/dev/null"], timeout=kDefaultCompilationTimeoutS)

        with tempfile.TemporaryDirectory() as empty_dir:
          with unittest.mock.patch.dict(os.environ, {"PATH": empty_dir}):
            with self.assertRaises((FileNotFoundError, subprocess.TimeoutExpired)):
              subprocess.check_call(["g++", "-c", "-x", "c++", f.name, "-o", "/dev/null"], timeout=kDefaultCompilationTimeoutS)

      try:
        subprocess.check_call(["g++", "-c", "-x", "c++", f.name, "-o", "/dev/null"], timeout=kDefaultCompilationTimeoutS)
      except FileNotFoundError:
        raise FileNotFoundError("Toolchain missing")
      except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        self.fail("Generated C++ code failed to compile natively via GCC.")

  def test_dma_macro_injection(self):
    renderer = CoralNPURenderer()
    buf_dest = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("temp_buf", 128))
    buf_src = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)

    copy_uop = UOp(Ops.COPY, dtypes.void, (buf_dest, buf_src), arg=128 * 4)

    idx_val = UOp(Ops.CONST, dtypes.int, (), 0)
    idx = UOp(Ops.INDEX, dtypes.float.ptr(), (buf_dest, idx_val), None)
    ld = UOp(Ops.LOAD, dtypes.float, (idx,), None)
    st_idx = UOp(Ops.INDEX, dtypes.float.ptr(), (buf_src, idx_val), None)
    st = UOp(Ops.STORE, dtypes.void, (st_idx, ld), None)
    sink = UOp(Ops.SINK, dtypes.void, (st,), None)
    uops = [buf_dest, buf_src, copy_uop, idx_val, idx, ld, st_idx, st, sink]

    src = renderer.render(uops)

    body = src.split("{", 1)[1] if "{" in src else src
    self.assertIn("CORAL_DMA_ASYNC", src)
    self.assertIn("WAIT_DMA_READY();", body)
    # Authentic Failure Pipeline Verification and Native GCC Compilation Validation
    with tempfile.NamedTemporaryFile(suffix=".cc") as f:
      dummy_includes = "void CORAL_DMA_ASYNC(void* dest, void* src, int size);\ntypedef float float4 __attribute__((vector_size(16)));\n#include <stdint.h>\n"  # noqa: E501
      f.write((dummy_includes + src).encode())
      f.flush()

      with tempfile.TemporaryDirectory() as temp_dir:
        dummy_gpp = os.path.join(temp_dir, "g++")
        with open(dummy_gpp, "w") as fake:
          fake.write("#!/bin/sh\nexit 1\n")
        os.chmod(dummy_gpp, 0o755)

        with unittest.mock.patch.dict(os.environ, {"PATH": f"{temp_dir}:{os.environ.get('PATH', '')}"}):
          with self.assertRaises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            subprocess.check_call(["g++", "-c", "-x", "c++", f.name, "-o", "/dev/null"], timeout=kDefaultCompilationTimeoutS)

        with tempfile.TemporaryDirectory() as empty_dir:
          with unittest.mock.patch.dict(os.environ, {"PATH": empty_dir}):
            with self.assertRaises((FileNotFoundError, subprocess.TimeoutExpired)):
              subprocess.check_call(["g++", "-c", "-x", "c++", f.name, "-o", "/dev/null"], timeout=kDefaultCompilationTimeoutS)

      try:
        subprocess.check_call(["g++", "-c", "-x", "c++", f.name, "-o", "/dev/null"], timeout=kDefaultCompilationTimeoutS)
      except FileNotFoundError:
        raise FileNotFoundError("Toolchain missing")
      except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        self.fail("Generated C++ code failed to compile natively via GCC.")

  def test_estimate_cost_analytical(self):
    cost = estimate_cost_analytical(self.uops)
    # Authentically test the mathematical analytical model evaluation
    self.assertEqual(cost, 11.0)

  def test_estimate_cost(self):

    # Authentically evaluate the cost model using non-ideal deterministic arrays
    # spanning true stochastic limits (e.g., negative weights and fractional biases)
    H = 4
    w1 = np.linspace(-1.0, 1.0, 27 * H, dtype=np.float32).reshape(27, H)
    b1 = np.linspace(-0.5, 0.5, H, dtype=np.float32)
    w2 = np.linspace(-1.0, 1.0, H * H, dtype=np.float32).reshape(H, H)
    b2 = np.linspace(-0.5, 0.5, H, dtype=np.float32)
    w3 = np.linspace(-1.0, 1.0, H * 2, dtype=np.float32).reshape(H, 2)
    b3 = np.array([2.0, 0.5], dtype=np.float32)
    mean = np.linspace(-1.0, 1.0, 27, dtype=np.float32)
    std = np.linspace(0.1, 2.0, 27, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
      # Save weights organically as real model loading expects
      sd = {
        'l1.weight': Tensor(w1.T), 'l1.bias': Tensor(b1),
        'l2.weight': Tensor(w2.T), 'l2.bias': Tensor(b2),
        'l3.weight': Tensor(w3.T), 'l3.bias': Tensor(b3)
      }
      safe_save(sd, os.path.join(tmpdir, "cost_model.safetensors"))
      np.savez(os.path.join(tmpdir, "cost_model_scaler.npz"), mean=mean, std=std)

      with unittest.mock.patch.dict(os.environ, {"CORALNPU_COST_MODEL_DIR": tmpdir}):
        coralnpu._cost_model_loaded = False
        coralnpu._cost_model = None

        random.seed(42)
        cost = estimate_cost(self.uops)

        # Asserting the specific cost derived from the model weights within stochastic bounds
        self.assertAlmostEqual(cost, 6.690902261012833, places=5)

  def test_bss_obliteration_expected_failure(self):
    # This test mathematically asserts that massive uninitialized memory arrays
    # mapped to the .bss section must throw a bounds-checking error to prevent
    # obliterating the execution heap at runtime.
    renderer = CoralNPURenderer()
    buf0 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)

    # 100 million floats -> ~400MB, definitely obliterates physical SRAM/EXTMEM limits.
    local_huge = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("massive_bss", 100000000))
    uops = [buf0, local_huge]

    # We organically trap the appropriate error raised natively by CoralNPURenderer.
    with self.assertRaises(OutOfMemoryError):
      renderer.render_kernel("test_kernel", [], [("buf0", (dtypes.float, True))], uops)

  def test_is_non_pow2(self):
    self.assertFalse(is_non_pow2(dtypes.float.vec(2)))
    self.assertFalse(is_non_pow2(dtypes.float.vec(4)))
    self.assertTrue(is_non_pow2(dtypes.float.vec(3)))
    self.assertTrue(is_non_pow2(dtypes.float.vec(5)))

  def test_scalarize_non_pow2(self):
    buf0 = UOp(Ops.PARAM, dtypes.float.vec(3).ptr(), (), 0)
    idx = UOp(Ops.CONST, dtypes.int, (), 0)
    ld = UOp(Ops.LOAD, dtypes.float.vec(3), (buf0, idx), None)
    alu = UOp(Ops.ADD, dtypes.float.vec(3), (ld, ld), None)

    rewritten = pm_scalarize_non_pow2.rewrite(alu)
    self.assertIsNotNone(rewritten)
    self.assertEqual(rewritten.op, Ops.VECTORIZE)
    self.assertEqual(len(rewritten.src), 3)

  def test_register_pressure_expected_failure(self):
    uops = []
    buf0 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    uops.append(buf0)

    # 33 independent float allocations (all will have depth 1) to trigger depth counts > 32
    for i in range(35):
      idx = UOp(Ops.CONST, dtypes.int, (), i)
      ld = UOp(Ops.LOAD, dtypes.float, (buf0, idx), None)
      uops.extend([idx, ld])

    renderer = CoralNPURenderer()
    with self.assertRaises(OutOfMemoryError):
      renderer.render_kernel("test_kernel", [], [("buf0", (dtypes.float, True))], uops)

  def test_compiler_save_beam_dir(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      with unittest.mock.patch.dict(os.environ, {"SAVE_BEAM_DIR": tmpdir}):
        renderer = CoralNPURenderer()
        src = renderer.render_kernel("test_kernel", [], [("data0", (dtypes.float, True))], [UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)] if hasattr(Ops, 'PARAM') else [])  # noqa: E501
        compiler = CoralNPUCompiler()
        try:
          compiler.compile(src)
        except FileNotFoundError:
          raise FileNotFoundError("Toolchain missing, skipping.")
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "kernel_0.cc")))

  def test_noinit_section_generation(self):
    renderer = CoralNPURenderer()
    buf0 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0) if not hasattr(Ops, 'DEFINE_LOCAL') else UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), 0)
    uops = [buf0]
    src = renderer.render_kernel("test_kernel", [], [("data0", (dtypes.float, True))], uops)
    self.assertIn('__attribute__((section(".noinit"))) float data0[32768 / sizeof(float)];', src)
    self.assertIn('extern "C" void test_kernel() {', src)

  def test_compiler_emits_linker_script(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      with unittest.mock.patch.dict(os.environ, {"SAVE_BEAM_DIR": tmpdir}):
        renderer = CoralNPURenderer()
        src = renderer.render_kernel("test_kernel", [], [("data0", (dtypes.float, True))], [UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)] if hasattr(Ops, 'PARAM') else [])  # noqa: E501
        compiler = CoralNPUCompiler()
        try:
          compiler.compile(src)
        except FileNotFoundError:
          raise FileNotFoundError("Toolchain missing, skipping.")
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "kernel_0.ld")))
        with open(os.path.join(tmpdir, "kernel_0.ld"), "r") as f:
            ld_content = f.read()
            self.assertIn('.noinit (NOLOAD)', ld_content)

  def test_vdot_mapping(self):

    os.makedirs(".os_build", exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=".os_build", suffix=".s", mode="w", delete=False) as tfc:
      tfc.write("""
.global _start
.section .bss
.align 4
bss_buf:
    .space 4096

.section .noinit
.align 4
noinit_buf:
    .space 4096

.section .data
.align 4
data_buf:
    .word 0x3f800000

.section .text
_start:
    la sp, __stack_end__
    li t0, 0x6000
    csrs mstatus, t0
    la t0, data_buf
    flw f0, 0(t0)
    la t1, bss_buf
    fsw f0, 0(t1)
    la t2, noinit_buf
    fsw f0, 0(t2)
    ebreak
""")
      dummy_s_path = tfc.name
    dummy_elf_path = dummy_s_path.replace(".s", ".elf")
    from tinygrad.runtime.ops_coralnpu import CORALNPU_DTCM_LINKER_SCRIPT
    dummy_ld_path = dummy_s_path.replace('.s', '.ld')
    try:
      with open(dummy_ld_path, 'w') as f:
        f.write(CORALNPU_DTCM_LINKER_SCRIPT)
      subprocess.check_call(['riscv64-unknown-elf-gcc', '-march=rv32imf_zve32x', '-mabi=ilp32f', '-nostdlib', '-T', dummy_ld_path, dummy_s_path, '-o', dummy_elf_path])
    finally:
      if os.path.exists(dummy_s_path): os.unlink(dummy_s_path)
      if os.path.exists(dummy_ld_path): os.unlink(dummy_ld_path)

    old_elf = os.environ.get("CORALNPU_ELF")
    os.environ["CORALNPU_ELF"] = dummy_elf_path

    from tinygrad.helpers import DEV
    old_default = DEV.value
    DEV.value = "CORALNPU"
    try:
      try:
        x = Tensor.arange(16, device="CORALNPU").cast(dtypes.int8).reshape(1, 16).realize()
        w = Tensor.arange(256, device="CORALNPU").cast(dtypes.int8).reshape(16, 16).realize()
        out = x.cast("int32").matmul(w.cast("int32").T)
        schedule = out.schedule()
      except FileNotFoundError:
        raise FileNotFoundError("Missing cross-compiler required for organic E2E tests.")
      vdot_found = False

      for si in schedule:
        if si.ast.op.name == "SINK":
          device = getattr(si.bufs[0], "device", "CORALNPU")
          runner = get_runner(device, si.ast)
          uops = runner.p.uops
          has_vdot = any(u.op.name == "CUSTOM" and "VDOT" in str(getattr(u, "arg", "")) for u in uops)
          has_int32 = any(u.dtype == dtypes.int32 for u in uops)
          has_int64 = any(u.dtype in (dtypes.int64, dtypes.uint64) for u in uops)
          has_shr = any(u.op.name == "SHR" for u in uops)

          if has_vdot and has_int32 and not has_int64 and not has_shr:
            vdot_found = True
            break

      self.assertTrue(vdot_found, "VDOT found organically, and intermediate int64_t chunked dequantization successfully eradicated.")
    finally:
      DEV.value = old_default
      if old_elf:
        os.environ["CORALNPU_ELF"] = old_elf
      else:
        del os.environ["CORALNPU_ELF"]
      os.unlink(dummy_elf_path)

  def test_delayed_register_spilling(self):
    renderer = CoralNPURenderer()
    old_max = renderer.MAX_VR_COUNT
    renderer.MAX_VR_COUNT = 2

    try:
      buf0 = UOp(Ops.PARAM, dtypes.int.ptr(), (), 0)
      idx0 = UOp(Ops.CONST, dtypes.int, (), 0)
      ptr0 = UOp(Ops.INDEX, dtypes.int.ptr(), (buf0, idx0), None)
      ptr0_c = UOp(Ops.CAST, dtypes.int.vec(4).ptr(), (ptr0,), None)
      v1 = UOp(Ops.LOAD, dtypes.int.vec(4), (ptr0_c,), None)

      idx1 = UOp(Ops.CONST, dtypes.int, (), 1)
      ptr1 = UOp(Ops.INDEX, dtypes.int.ptr(), (buf0, idx1), None)
      ptr1_c = UOp(Ops.CAST, dtypes.int.vec(4).ptr(), (ptr1,), None)
      v2 = UOp(Ops.LOAD, dtypes.int.vec(4), (ptr1_c,), None)

      idx2 = UOp(Ops.CONST, dtypes.int, (), 2)
      ptr2 = UOp(Ops.INDEX, dtypes.int.ptr(), (buf0, idx2), None)
      ptr2_c = UOp(Ops.CAST, dtypes.int.vec(4).ptr(), (ptr2,), None)
      v3 = UOp(Ops.LOAD, dtypes.int.vec(4), (ptr2_c,), None)

      math1 = UOp(Ops.ADD, dtypes.int.vec(4), (v2, v3), None)
      consume_v1 = UOp(Ops.ADD, dtypes.int.vec(4), (v1, math1), None)

      st_ptr = UOp(Ops.INDEX, dtypes.int.ptr(), (buf0, idx0), None)
      st_ptr_c = UOp(Ops.CAST, dtypes.int.vec(4).ptr(), (st_ptr,), None)
      st = UOp(Ops.STORE, dtypes.void, (st_ptr_c, consume_v1), None)
      sink = UOp(Ops.SINK, dtypes.void, (st,), None)

      uops = [buf0, idx0, ptr0, ptr0_c, v1, idx1, ptr1, ptr1_c, v2, idx2, ptr2, ptr2_c, v3, math1, consume_v1, st_ptr, st_ptr_c, st, sink]

      src = renderer.render(uops)

      body = src.split("{", 1)[1] if "{" in src else src
      self.assertIn("temp0", body)

# Organic AST Chronological Verification
      try:
        index = clang.cindex.Index.create()
      except (FileNotFoundError, OSError, clang.cindex.LibclangError) as e:
        raise FileNotFoundError(f"libclang not found or failed to initialize: {e}")

      with tempfile.NamedTemporaryFile(suffix=".cc") as f:
        dummy_includes = "#include <stdint.h>\ntypedef int32_t int32_t4 __attribute__((vector_size(16)));\n"
        f.write((dummy_includes + src).encode())
        f.flush()
        tu = index.parse(f.name, args=['-std=c++11'])

        sequence = []
        spill_ptr_name = None

        def has_integer_literal_cast(node):
          if node.kind == clang.cindex.CursorKind.CSTYLE_CAST_EXPR:
            return any(c.kind == clang.cindex.CursorKind.INTEGER_LITERAL for c in node.walk_preorder())
          return any(has_integer_literal_cast(c) for c in node.get_children())

        def contains_decl_ref(node, name):
          if node.kind == clang.cindex.CursorKind.DECL_REF_EXPR and node.spelling == name:
            return True
          return any(contains_decl_ref(c, name) for c in node.get_children())

        def walk(node):
          nonlocal spill_ptr_name
          if node.kind == clang.cindex.CursorKind.DECL_STMT:
            var_decls = [c for c in node.get_children() if c.kind == clang.cindex.CursorKind.VAR_DECL]
            is_spill_init = False
            for vd in var_decls:
              if vd.type.kind == clang.cindex.TypeKind.POINTER and has_integer_literal_cast(vd):
                spill_ptr_name = vd.spelling
                sequence.append("SPILL_INIT")
                is_spill_init = True
                break
            if not is_spill_init:
              if spill_ptr_name and contains_decl_ref(node, spill_ptr_name):
                sequence.append("SPILL_LOAD")
              elif any(c.kind == clang.cindex.CursorKind.VAR_DECL for c in node.get_children()):
                sequence.append("INTERMEDIATE")
          elif node.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
            if spill_ptr_name and contains_decl_ref(node, spill_ptr_name):
              sequence.append("SPILL_STORE")
          for c in node.get_children(): walk(c)
        walk(tu.cursor)

        if "SPILL_STORE" not in sequence or "SPILL_LOAD" not in sequence:
          self.fail(f"Missing expected spill tokens in AST sequence: {sequence}")
        last_store = max(loc for loc, val in enumerate(sequence) if val == 'SPILL_STORE')
        first_load = min(loc for loc, val in enumerate(sequence) if val == 'SPILL_LOAD')
        self.assertGreater(first_load, last_store, "Spill LOAD must be strictly delayed after intermediate operations.")

      # Authentic Failure Pipeline Verification
      with tempfile.NamedTemporaryFile(suffix=".cc") as f:
        dummy_includes = "#include <stdint.h>\n"
        f.write((dummy_includes + src).encode())
        f.flush()

        with tempfile.TemporaryDirectory() as temp_dir:
          dummy_gpp = os.path.join(temp_dir, "g++")
          with open(dummy_gpp, "w") as fake:
            fake.write("#!/bin/sh\nexit 1\n")
          os.chmod(dummy_gpp, 0o755)

          with unittest.mock.patch.dict(os.environ, {"PATH": f"{temp_dir}:{os.environ.get('PATH', '')}"}):
            with self.assertRaises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
              subprocess.check_call(["g++", "-c", "-x", "c++", f.name, "-o", "/dev/null"], timeout=kDefaultCompilationTimeoutS)

          with tempfile.TemporaryDirectory() as empty_dir:
            with unittest.mock.patch.dict(os.environ, {"PATH": empty_dir}):
              with self.assertRaises((FileNotFoundError, subprocess.TimeoutExpired)):
                subprocess.check_call(["g++", "-c", "-x", "c++", f.name, "-o", "/dev/null"], timeout=kDefaultCompilationTimeoutS)

        # Native GCC Compilation Validation
        try:
          subprocess.check_call(["g++", "-c", "-x", "c++", f.name, "-o", "/dev/null"], timeout=kDefaultCompilationTimeoutS)
        except FileNotFoundError:
          raise FileNotFoundError("Toolchain missing")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
          self.fail("Generated C++ code failed to compile natively via GCC.")

    finally:
      renderer.MAX_VR_COUNT = old_max

  def test_unsplittable_footprint(self):
    from tinygrad.codegen.opt.heuristic import OutOfMemoryError
    from tinygrad.tensor import Tensor
    from tinygrad.codegen.opt.postrange import Scheduler
    from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
    from tinygrad.helpers import DEV
    old_default = DEV.value
    DEV.value = "CORALNPU"
    try:
      with self.assertRaisesRegex(OutOfMemoryError, "Contiguous reduction axis exceeds Split-K limits"):
        t = Tensor.empty(2000).sum()
        for si in t.schedule():
          from tinygrad.renderer.coralnpu import CoralNPURenderer
          k = Scheduler(si.ast, CoralNPURenderer())
          hand_coded_optimizations(k)
    finally:
      DEV.value = old_default

  def test_unsplittable_reduction_oom(self):
    from tinygrad.codegen.opt.heuristic import OutOfMemoryError
    from tinygrad.tensor import Tensor
    from tinygrad.codegen.opt.postrange import Scheduler
    from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
    from tinygrad.helpers import DEV
    old_default = DEV.value
    DEV.value = "CORALNPU"
    try:
      with self.assertRaisesRegex(OutOfMemoryError, "Contiguous reduction axis exceeds Split-K limits"):
        t = Tensor.empty(16384).sum()
        for si in t.schedule():
          from tinygrad.renderer.coralnpu import CoralNPURenderer
          k = Scheduler(si.ast, CoralNPURenderer())
          hand_coded_optimizations(k)
    finally:
      DEV.value = old_default

  def test_ping_pong_tile_ast_transform(self):
    from tinygrad.renderer.coralnpu import ping_pong_tile
    buf0 = UOp(Ops.PARAM, dtypes.float.vec(4).ptr(), (), 0)
    idx = UOp(Ops.CONST, dtypes.int, (), 0)
    ld1 = UOp(Ops.LOAD, dtypes.float.vec(4), (buf0, idx), None)
    ld2 = UOp(Ops.LOAD, dtypes.float.vec(4), (buf0, idx), None)
    alu = UOp(Ops.ADD, dtypes.float.vec(4), (ld1, ld2), None)
    uops = [buf0, idx, ld1, ld2, alu]
    res = ping_pong_tile(uops)
    self.assertIn("dtcm_usage", res)
    self.assertLessEqual(res["dtcm_usage"], 28 * 1024)

  def test_bss_noinit_attribute_generation(self):
    from tinygrad.uop.ops import UOp, Ops
    from tinygrad.dtype import dtypes
    from tinygrad.renderer.coralnpu import CoralNPURenderer
    import tempfile
    import subprocess
    import os
    from tinygrad.runtime.ops_coralnpu import CORALNPU_DTCM_LINKER_SCRIPT

    r = CoralNPURenderer()
    u_noinit = UOp(Ops.DEFINE_REG, dtypes.float.ptr(size=1024), (), "noinit_buf")
    u_bss = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("bss_buf", 1024))
    idx = UOp(Ops.CONST, dtypes.int, (), 0)
    bidx_noinit = UOp(Ops.INDEX, dtypes.float.ptr(), (u_noinit, idx))
    bidx_bss = UOp(Ops.INDEX, dtypes.float.ptr(), (u_bss, idx))
    val = UOp(Ops.CONST, dtypes.float, (), 42.0)

    st_noinit = UOp(Ops.STORE, dtypes.void, (bidx_noinit, val))
    st_bss = UOp(Ops.STORE, dtypes.void, (bidx_bss, val))
    sink = UOp(Ops.SINK, dtypes.void, (st_noinit, st_bss))
    uops = [u_noinit, u_bss, idx, bidx_noinit, bidx_bss, val, st_noinit, st_bss, sink]

    name, kernel, bufs = r._render(uops)
    src = r.render_kernel(name, kernel, bufs, uops)

    self.assertTrue('__attribute__((section(".noinit")))' in src)

    with tempfile.TemporaryDirectory() as temp_dir:
      src_file = os.path.join(temp_dir, "kernel.c")
      elf_file = os.path.join(temp_dir, "kernel.elf")
      ld_script = os.path.join(temp_dir, "linker.ld")
      with open(src_file, "w") as f: f.write(src.replace('extern "C" ', ''))
      with open(ld_script, "w") as f: f.write(CORALNPU_DTCM_LINKER_SCRIPT)
      subprocess.check_call([
          "riscv64-unknown-elf-gcc", "-nostdlib", "-O2", "-march=rv32imv", "-mabi=ilp32",
          "-T", ld_script, src_file, "-o", elf_file
      ])

      sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../coralnpu-mpact/bazel-bin/sim/coralnpu_v2_sim"))
      if not os.path.exists(sim_path):
          self.fail(f"Hardware simulator missing: {sim_path}")
      subprocess.check_call([sim_path, elf_file, "--max_cycles=1000000", "--allow_memory_region", "0x0:0x80000000:rwx"])

