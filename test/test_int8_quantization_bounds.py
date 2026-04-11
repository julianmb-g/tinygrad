import unittest
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context, DEV, getenv
import shutil

def has_compiler(): return shutil.which("riscv64-unknown-elf-gcc") is not None

class TestInt8QuantizationBounds(unittest.TestCase):
    @pytest.mark.prototype
    @unittest.skipIf(not has_compiler(), "Missing riscv64-unknown-elf-gcc")
    def test_int8_quantization_bounds(self):
        with Context(DEV="CORALNPU"):
            a = Tensor([1, 2, 127, -128], dtype=dtypes.int8).realize()
            b = Tensor([1, 1, 1, -1], dtype=dtypes.int8).realize()
            c = (a + b).realize()
            res = c.numpy()
            self.assertEqual(res[0], 2)
            self.assertEqual(res[1], 3)
            self.assertEqual(res[2], -128)
            self.assertEqual(res[3], 127)
