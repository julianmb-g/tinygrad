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

class TestGemmaDecomposition(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    try:
      # Natively compile a baseline ELF to provide authentic architectural boundaries for VMM initialization
      cls.tf_c = tempfile.NamedTemporaryFile(suffix=".c", delete=False)
      cls.tf_c.write(b"int main() { return 0; }\n")
      cls.tf_c.flush()

      cls.tf_elf = tempfile.NamedTemporaryFile(suffix=".elf", delete=False)
      subprocess.check_call([
        "riscv64-unknown-elf-gcc", "-march=rv32imf_zve32x", "-mabi=ilp32f",
        "-O3", "-nostdlib", cls.tf_c.name, "-o", cls.tf_elf.name,
        "-Wl,--defsym=_end=0x80000000"
      ])
      cls.native_elf_path = cls.tf_elf.name
    except (FileNotFoundError, subprocess.CalledProcessError):
      raise unittest.SkipTest("Toolchain missing, cannot natively compile ELF")

  @classmethod
  def tearDownClass(cls):
    if hasattr(cls, 'tf_c'):
      cls.tf_c.close()
      os.unlink(cls.tf_c.name)
    if hasattr(cls, 'tf_elf'):
      cls.tf_elf.close()
      os.unlink(cls.tf_elf.name)

  def test_gemma_rmsnorm(self):
    dim = 256
    x_cpu = ((Tensor.arange(1 * 16 * dim, device="CPU") % 10) * 0.1).reshape(1, 16, dim).realize()
    rmsnorm_cpu = GemmaRMSNorm(dim)
    rmsnorm_cpu.weight = ((Tensor.arange(dim, device="CPU") % 10) * 0.1).contiguous().realize()
    expected_out = rmsnorm_cpu(x_cpu).realize().numpy()

    with patch("tinygrad.device.Device.DEFAULT", "CORALNPU"), patch.dict(os.environ, {"CORALNPU_ELF": self.native_elf_path}):
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
        raise
      except FileNotFoundError as e:
        if "coralnpu_v2_sim" in str(e):
          raise unittest.SkipTest("Simulator missing, skipping.")
        raise

  def test_gemma_geglu(self):
    hidden_dim = 256
    ff_dim = 1024

    x_cpu = ((Tensor.arange(1 * 16 * hidden_dim, device="CPU") % 10) * 0.1).reshape(1, 16, hidden_dim).realize()
    mlp_gate_cpu = ((Tensor.arange(hidden_dim * ff_dim, device="CPU") % 10) * 0.1).reshape(hidden_dim, ff_dim).realize()
    mlp_up_cpu = ((Tensor.arange(hidden_dim * ff_dim, device="CPU") % 10) * 0.1).reshape(hidden_dim, ff_dim).realize()
    mlp_down_cpu = ((Tensor.arange(ff_dim * hidden_dim, device="CPU") % 10) * 0.1).reshape(ff_dim, hidden_dim).realize()

    with patch("tinygrad.device.Device.DEFAULT", "CORALNPU"), patch.dict(os.environ, {"CORALNPU_ELF": self.native_elf_path}):
      x = x_cpu.to("CORALNPU")
      mlp = GemmaMLP(hidden_dim, ff_dim)
      
      mlp.gate_proj = mlp_gate_cpu.to("CORALNPU")
      mlp.up_proj = mlp_up_cpu.to("CORALNPU")
      mlp.down_proj = mlp_down_cpu.to("CORALNPU")

      out = mlp(x)

      schedule = out.schedule()
      raised = False
      for si in schedule:
        if si.ast.op.name == "SINK":
          try:
            get_runner(Device.DEFAULT, si.ast)
          except RuntimeError as e:
            if "Active floating-point variable allocations exceeded cap" in str(e):
              raised = True
              break
            else:
              raise
      self.assertTrue(raised, "Expected Gemma GeGLU to hit active variable allocations cap")

  def test_gemma_rope(self):
    seq_len = 8
    head_dim = 64
    x_cpu = ((Tensor.arange(1 * seq_len * 4 * head_dim, device="CPU") % 10) * 0.1).reshape(1, seq_len, 4, head_dim).realize()
    
    # Precompute freqs directly on CPU for expected
    freqs_cis_cpu = precompute_freqs_cis(head_dim, seq_len).realize()
    expected_out = apply_rotary_emb(x_cpu, freqs_cis_cpu).realize().numpy()

    with patch("tinygrad.device.Device.DEFAULT", "CORALNPU"), patch.dict(os.environ, {"CORALNPU_ELF": self.native_elf_path}):
      x = x_cpu.to("CORALNPU")
      freqs_cis = freqs_cis_cpu.to("CORALNPU")
      out = apply_rotary_emb(x, freqs_cis)

      try:
        out.realize()
        np.testing.assert_allclose(out.numpy(), expected_out, atol=1e-4)
      except RuntimeError as e:
        if "math.h: No such file" in str(e) or "implicit declaration of function" in str(e):
          raise unittest.SkipTest("CORALNPU bare-metal compiler lacks math.h support for sqrt/sin/cos")
        raise
      except FileNotFoundError as e:
        if "coralnpu_v2_sim" in str(e):
          raise unittest.SkipTest("Simulator missing, skipping.")
        raise

if __name__ == '__main__':
  unittest.main()
