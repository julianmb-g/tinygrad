import unittest
import unittest.mock
import os
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from tinygrad.renderer.coralnpu import (
  extract_features, estimate_cost, estimate_cost_analytical,
  is_non_pow2, scalarize_alu, pm_scalarize_non_pow2, CoralNPURenderer, CoralNPUCompiler
)

class TestCoralNPURenderer(unittest.TestCase):
  def setUp(self):
    self.buf0 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    self.idx = UOp(Ops.CONST, dtypes.int, (), 0)
    self.ld = UOp(Ops.LOAD, dtypes.float, (self.buf0, self.idx), None)
    self.alu = UOp(Ops.ADD, dtypes.float, (self.ld, self.ld), None)
    self.uops = [self.buf0, self.idx, self.ld, self.alu]

  def test_extract_features(self):
    feats = extract_features(self.uops)
    self.assertIn('alu_ratio', feats)
    self.assertIn('mem_ratio', feats)
    self.assertTrue(feats['log_total_uops'] > 0)
    
  def test_max_upcast(self):
    renderer = CoralNPURenderer()
    
    uops = []
    buf0 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    idx = UOp(Ops.CONST, dtypes.int, (), 0)
    
    vec_srcs = []
    for i in range(33):
      ld = UOp(Ops.LOAD, dtypes.float, (buf0, UOp(Ops.CONST, dtypes.int, (), i)), None)
      vec_srcs.append(ld)
    
    vec = UOp(Ops.VECTORIZE, dtypes.float.vec(33), tuple(vec_srcs), None)
    uops = [buf0, idx] + vec_srcs + [vec]
    
    with self.assertRaisesRegex(RuntimeError, "AST upcast limit exceeded"):
      renderer.render_kernel("test_kernel", [], [("buf0", (dtypes.float, True))], uops)

  def test_dtcm_tiling_limit(self):
    renderer = CoralNPURenderer()
    # Create DEFINE_LOCAL UOp exceeding 12KB (12288 bytes)
    # 3073 floats * 4 bytes/float = 12292 bytes
    local_buf = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("temp_buf", 3073))
    with self.assertRaisesRegex(RuntimeError, "DTCM Tiling exceeded 12KB limit"):
      renderer.render_kernel("test_kernel", [], [("buf0", (dtypes.float, True))], [local_buf])

  def test_dma_macro_injection(self):
    renderer = CoralNPURenderer()
    buf_dest = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("temp_buf", 128))
    buf_src = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    # A COPY uop copies from buf_src to buf_dest. In tinygrad AST, COPY returns void and takes (dest, src) in older versions, 
    # but here we use the signature expected by our new code_for_op mapping which is (dest, src, dtype)
    # Actually, UOp(Ops.COPY, dtype, (buf_src, buf_dest))
    copy_uop = UOp(Ops.COPY, dtypes.float, (buf_dest, buf_src))
    uops = [buf_dest, buf_src, copy_uop]
    src = renderer.render_kernel("test_kernel", [], [("buf0", (dtypes.float, True))], uops)
    self.assertIn("CORAL_DMA_ASYNC", src)
    self.assertIn("CORAL_DMA_WAIT()", src)
    
  def test_estimate_cost_analytical(self):
    cost = estimate_cost_analytical(self.uops)
    self.assertTrue(cost > 0)

  def test_estimate_cost(self):
    import tinygrad.renderer.coralnpu as coralnpu
    # Test the real analytical fallback organically without mocking load_cost_model
    # We must not use artificial deterministic dummy arrays to pad coverage.
    coralnpu._cost_model_loaded = True
    coralnpu._cost_model = None
    cost = estimate_cost(self.uops)
    self.assertTrue(cost >= 0.0)

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

  def test_render_kernel_runtime_error(self):
    uops = []
    buf0 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
    uops.append(buf0)
    
    # 33 independent float allocations (all will have depth 1) to trigger depth counts > 32
    for i in range(35):
      idx = UOp(Ops.CONST, dtypes.int, (), i)
      ld = UOp(Ops.LOAD, dtypes.float, (buf0, idx), None)
      uops.extend([idx, ld])
      
    renderer = CoralNPURenderer()
    with self.assertRaisesRegex(RuntimeError, "Active floating-point variable allocations exceeded cap"):
      renderer.render_kernel("test_kernel", [], [("buf0", (dtypes.float, True))], uops)

  def test_compiler_save_beam_dir(self):
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
      with unittest.mock.patch.dict(os.environ, {"SAVE_BEAM_DIR": tmpdir}):
        compiler = CoralNPUCompiler()
        compiler.compile("int main() { return 0; }")
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "kernel_0.cc")))


  def test_noinit_section_generation(self):
    from tinygrad.renderer.coralnpu import CoralNPURenderer
    renderer = CoralNPURenderer()
    buf0 = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0) if not hasattr(Ops, 'DEFINE_LOCAL') else UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), 0)
    uops = [buf0]
    src = renderer.render_kernel("test_kernel", [], [("data0", (dtypes.float, True))], uops)
    self.assertIn('__attribute__((section(".noinit"))) float data0[32768 / sizeof(float)];', src)
    self.assertIn('extern "C" void test_kernel() {', src)
    
  def test_compiler_emits_linker_script(self):
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
      with unittest.mock.patch.dict(os.environ, {"SAVE_BEAM_DIR": tmpdir}):
        compiler = CoralNPUCompiler()
        compiler.compile("int main() { return 0; }")
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "kernel_0.ld")))
        with open(os.path.join(tmpdir, "kernel_0.ld"), "r") as f:
            ld_content = f.read()
            self.assertIn('.noinit (NOLOAD)', ld_content)

if __name__ == '__main__':
  unittest.main()
