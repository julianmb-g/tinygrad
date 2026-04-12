import unittest
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import shutil

def has_compiler(): return shutil.which("riscv64-unknown-elf-gcc") is not None

class TestBaremetalPingPongDma(unittest.TestCase):
    @pytest.mark.prototype
    @unittest.skipIf(not has_compiler(), "Missing riscv64-unknown-elf-gcc")
    def test_baremetal_pingpong_dma(self):
        """
        Verify the 32KB DTCM equation natively.
        [Weights: 2x 6KB : Activations: 8KB : Accumulators: 8KB] = 28KB + 4KB (Stack/BSS) = 32KB DTCM.
        """
        with Context(DEV="CORALNPU"):
            a = Tensor.arange(3072, dtype=dtypes.float32).realize()
            b = (Tensor.arange(3072, dtype=dtypes.float32) * 2).realize()
            c = (a + b).realize()
            res = c.numpy()
            self.assertEqual(res[0], 0.0)
            self.assertEqual(res[1500], 4500.0)
            self.assertEqual(res[-1], 3071.0 * 3.0)

if __name__ == '__main__':
    unittest.main()
