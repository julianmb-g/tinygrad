#!/usr/bin/env python3
import numpy as np

from tinygrad.tensor import Tensor

while True:
  arr = np.ones(1000000, dtype=np.uint8)
  print(f"numpy: {(arr + 1)[:10]}")

  ptr = arr.ctypes.data
  tensor = Tensor.from_blob(ptr, arr.shape, dtype='uint8', device='QCOM').realize() + 1
  print(f"from_blob: {tensor.numpy()[:10]}")
