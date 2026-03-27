import unittest

import numpy as np

from extra.f16_decompress import u32_to_f16
from tinygrad import dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.tensor import Tensor


class TestF16Decompression(unittest.TestCase):
  def test_u32_to_f16(self):
    a = ((Tensor.arange(50) % 10) * 0.1).reshape(50).cast(dtypes.float16)
    f16_as_u32 = a.bitcast(dtypes.uint32)
    f16 = u32_to_f16(f16_as_u32)
    ref = a.numpy()
    out = f16.numpy().astype(np.float16)
    np.testing.assert_allclose(out, ref)
