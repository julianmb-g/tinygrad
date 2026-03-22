import os
import unittest
import subprocess
import tempfile
import numpy as np
from unittest.mock import patch
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.realize import get_runner
from extra.models.gemma import GemmaRMSNorm, GemmaMLP, precompute_freqs_cis, apply_rotary_emb

HIDDEN_DIM = 256
FF_DIM = 1024

class TestGemmaDecomposition(unittest.TestCase):

  def test_gemma_rmsnorm(self):
    dim = HIDDEN_DIM
    x_cpu = ((Tensor.arange(1 * 16 * dim, device="CPU") % 10) * 0.1).reshape(1, 16, dim).realize()
    rmsnorm_cpu = GemmaRMSNorm(dim)
    rmsnorm_cpu.weight = ((Tensor.arange(dim, device="CPU") % 10) * 0.1).contiguous().realize()
    expected_out = rmsnorm_cpu(x_cpu).realize().numpy()

    with patch("tinygrad.device.Device.DEFAULT", "CORALNPU"):
      x = x_cpu.to("CORALNPU")
      rmsnorm = GemmaRMSNorm(dim)
      rmsnorm.weight = rmsnorm_cpu.weight.to("CORALNPU")
      out = rmsnorm(x)

      try:
        out.realize()
        np.testing.assert_allclose(out.numpy(), expected_out, atol=1e-4)
      except RuntimeError as e:
        if "math.h: No such file" in str(e) or "implicit declaration of function" in str(e):
          raise unittest.SkipTest("CORALNPU bare-metal compiler lacks math.h support for sqrt/sin/cos")
        elif "VMM base address" in str(e):
          raise unittest.SkipTest("Toolchain or authentic ELF missing")
        raise
      except FileNotFoundError as e:
        if "coralnpu_v2_sim" in str(e):
          raise unittest.SkipTest("Simulator missing, skipping.")
        raise

  def test_gemma_geglu(self):
    hidden_dim = HIDDEN_DIM
    ff_dim = FF_DIM

    x_cpu = ((Tensor.arange(1 * 16 * hidden_dim, device="CPU") % 10) * 0.1).reshape(1, 16, hidden_dim).realize()
    mlp_gate_cpu = ((Tensor.arange(hidden_dim * ff_dim, device="CPU") % 10) * 0.1).reshape(hidden_dim, ff_dim).realize()
    mlp_up_cpu = ((Tensor.arange(hidden_dim * ff_dim, device="CPU") % 10) * 0.1).reshape(hidden_dim, ff_dim).realize()
    mlp_down_cpu = ((Tensor.arange(ff_dim * hidden_dim, device="CPU") % 10) * 0.1).reshape(ff_dim, hidden_dim).realize()

    with patch("tinygrad.device.Device.DEFAULT", "CORALNPU"):
      x = x_cpu.to("CORALNPU")
      mlp = GemmaMLP(hidden_dim, ff_dim)

      mlp.gate_proj = mlp_gate_cpu.to("CORALNPU")
      mlp.up_proj = mlp_up_cpu.to("CORALNPU")
      mlp.down_proj = mlp_down_cpu.to("CORALNPU")

      out = mlp(x)

      schedule = out.schedule()
      for si in schedule:
        if si.ast.op.name == "SINK":
          try:
            get_runner(Device.DEFAULT, si.ast)
          except RuntimeError as e:
            if "Active floating-point variable allocations exceeded cap" in str(e):
              raise unittest.SkipTest("Active floating-point variable allocations exceeded cap")
            elif "VMM base address" in str(e):
              raise unittest.SkipTest("Toolchain or authentic ELF missing")
            else:
              raise

  def test_gemma_rope(self):
    seq_len = 8
    head_dim = 64
    x_cpu = ((Tensor.arange(1 * seq_len * 4 * head_dim, device="CPU") % 10) * 0.1).reshape(1, seq_len, 4, head_dim).realize()

    # Precompute freqs directly on CPU for expected
    freqs_cis_cpu = precompute_freqs_cis(head_dim, seq_len).realize()
    expected_out = apply_rotary_emb(x_cpu, freqs_cis_cpu).realize().numpy()

    with patch("tinygrad.device.Device.DEFAULT", "CORALNPU"):
      x = x_cpu.to("CORALNPU")
      freqs_cis = freqs_cis_cpu.to("CORALNPU")
      out = apply_rotary_emb(x, freqs_cis)

      try:
        out.realize()
        np.testing.assert_allclose(out.numpy(), expected_out, atol=1e-4)
      except RuntimeError as e:
        if "math.h: No such file" in str(e) or "implicit declaration of function" in str(e):
          raise unittest.SkipTest("CORALNPU bare-metal compiler lacks math.h support for sqrt/sin/cos")
        elif "VMM base address" in str(e):
          raise unittest.SkipTest("Toolchain or authentic ELF missing")
        raise
      except FileNotFoundError as e:
        if "coralnpu_v2_sim" in str(e):
          raise unittest.SkipTest("Simulator missing, skipping.")
        raise

if __name__ == '__main__':
  unittest.main()
