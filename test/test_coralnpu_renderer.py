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
    
  def test_estimate_cost_analytical(self):
    cost = estimate_cost_analytical(self.uops)
    self.assertTrue(cost > 0)

  @unittest.mock.patch('tinygrad.renderer.coralnpu.load_cost_model')
  def test_estimate_cost(self, mock_load):
    # Depending on test environment, this might fallback to analytical or use ML model
    import tinygrad.renderer.coralnpu as coralnpu
    
    # Test without loaded model
    coralnpu._cost_model = None
    cost = estimate_cost(self.uops)
    self.assertEqual(cost, 0.0)
    
    # Test with mock dummy model
    import numpy as np
    import random
    w1 = np.arange(108, dtype=np.float32).reshape(27, 4) * 0.01 - 0.5
    b1 = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    w2 = np.arange(16, dtype=np.float32).reshape(4, 4) * 0.05 - 0.4
    b2 = np.array([0.5, -0.1, 0.2, 0.0], dtype=np.float32)
    w3 = np.array([[1.0, 0.5], [-1.0, 0.2], [0.5, -0.5], [0.1, 0.1]], dtype=np.float32)
    b3 = np.array([2.0, 1.0], dtype=np.float32)
    mean = np.arange(27, dtype=np.float32) * 0.1
    std = np.arange(27, dtype=np.float32) * 0.05 + 0.1
    
    coralnpu._cost_model = {
      'w1': w1, 'b1': b1,
      'w2': w2, 'b2': b2,
      'w3': w3, 'b3': b3,
      'mean': mean, 'std': std
    }
    random.seed(42)
    cost = estimate_cost(self.uops)
    self.assertAlmostEqual(cost, 8.493139132982789, places=5)

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

if __name__ == '__main__':
  unittest.main()
