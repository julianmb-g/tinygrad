import unittest
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import shutil

def has_compiler(): return shutil.which("riscv64-unknown-elf-gcc") is not None

class TestOpsCoralnpuParallel(unittest.TestCase):
    @pytest.mark.prototype
    @unittest.skipIf(not has_compiler(), "Missing riscv64-unknown-elf-gcc")
    def test_ops_coralnpu_parallel(self):
        with Context(DEV="CORALNPU"):
            a = Tensor.arange(1024, dtype=dtypes.float32).realize()
            b = a.sum().realize()
            # Sum of 0..1023 = 1023 * 1024 / 2 = 523776.0
            self.assertEqual(b.numpy().item(), 523776.0)

if __name__ == '__main__':
    unittest.main()
