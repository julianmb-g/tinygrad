import unittest
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context, DEV
import shutil

def has_compiler(): return shutil.which("riscv64-unknown-elf-gcc") is not None

@pytest.mark.prototype
class TestCoralMacros(unittest.TestCase):
    @pytest.mark.prototype
    @unittest.skipIf(not has_compiler(), "Missing riscv64-unknown-elf-gcc")
    def test_coral_cooperative_yield_bounds(self):
        with Context(DEV="CORALNPU"):
            a = Tensor([1], dtype=dtypes.float32).realize()
            self.assertEqual(a.numpy()[0], 1)

if __name__ == '__main__':
    unittest.main()
