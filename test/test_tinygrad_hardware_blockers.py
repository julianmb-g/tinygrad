import unittest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.codegen.opt.heuristic import OutOfMemoryError

class TestTinygradHardwareBlockers(unittest.TestCase):
    def test_chunk_exceeds_dtcm_limit(self):
        # A contiguous reduction over 16384 elements forces the ML compiler 
        # to organically evaluate the 12KB DTCM / Split-K limits.
        with Context(DEV="CORALNPU"):
            a = Tensor.empty(16384)
            with self.assertRaises(OutOfMemoryError):
                a.sum().realize()

    def test_register_pressure_upcast_limit(self):
        # Explicitly force a vector cast of size 29 to test the hardware vector limit organically
        with Context(DEV="CORALNPU"):
            a = Tensor.empty(29)
            with self.assertRaises(OutOfMemoryError):
                a.cast(dtypes.float.vec(29)).realize()

    def test_register_pressure_fp_allocation_cap(self):
        # Dynamically transform ML ops into an AST graph containing 33 variables
        # at the same execution depth to authentically push register pressure past limits.
        with Context(DEV="CORALNPU"):
            sums = []
            for i in range(33):
                sums.append(Tensor.empty(1) + 1)
            res = Tensor.stack(*sums).sum()
            with self.assertRaises(OutOfMemoryError):
                res.realize()

if __name__ == '__main__':
    unittest.main()
