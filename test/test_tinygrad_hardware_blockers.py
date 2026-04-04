import unittest
from tinygrad.codegen.opt.heuristic import OutOfMemoryError
from tinygrad.renderer.coralnpu import CoralNPURenderer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import dtypes

class TestTinygradHardwareBlockers(unittest.TestCase):
    def test_chunk_exceeds_dtcm_limit(self):
        renderer = CoralNPURenderer()
        buf_dest = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("temp_buf", 4097))
        uops = [buf_dest]
        with self.assertRaises(OutOfMemoryError):
            renderer.render(uops)

    def test_register_pressure_upcast_limit(self):
        renderer = CoralNPURenderer()

        # Use authentic UOp instead of a Python class mock
        uop = UOp(Ops.DEFINE_LOCAL, dtypes.float.vec(29), (), ("temp_vec", 1))

        with self.assertRaises(MemoryError):
            renderer.render([uop])

    def test_register_pressure_fp_allocation_cap(self):
        renderer = CoralNPURenderer()
        cst = UOp(Ops.CONST, dtypes.float, (), 1.0)
        uops = [cst]
        for i in range(33):
            alu = UOp(Ops.ADD, dtypes.float, (cst, cst), None)
            uops.append(alu)

        with self.assertRaises(MemoryError):
            renderer.render(uops)

if __name__ == '__main__':
    unittest.main()
