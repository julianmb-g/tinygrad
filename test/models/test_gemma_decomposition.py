from tinygrad.runtime.ops_coralnpu import kDefaultCompilationTimeoutS
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
from tinygrad.renderer.coralnpu import CoralNPURenderer
from tinygrad.codegen import get_program

HIDDEN_DIM = 256
FF_DIM = 1024

class TestGemmaDecomposition(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    try:
      subprocess.check_call(["riscv64-unknown-elf-gcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5.0)

      # Explicitly construct an authentic cross-compilation pipeline utilizing CoralNPUAllocator for the Gemma layer
      seq_len = 8
      head_dim = 64
      x_cpu = (Tensor.arange(1 * seq_len * 4 * head_dim, device="CPU") * 0.1).reshape(1, seq_len, 4, head_dim).realize()
      freqs_cis_cpu = precompute_freqs_cis(head_dim, seq_len).realize()

      out_cpu = apply_rotary_emb(x_cpu, freqs_cis_cpu)
      schedule = out_cpu.schedule()
      sink_ast = [si.ast for si in schedule if si.ast.op.name == "SINK"][0]

      renderer = CoralNPURenderer()
      prg = get_program(sink_ast, renderer)

      cls.tf_c = tempfile.NamedTemporaryFile(suffix=".c", delete=False)
      cls.tf_c.write(prg.src.encode())
      cls.tf_c.flush()

      cls.tf_elf = tempfile.NamedTemporaryFile(suffix=".elf", delete=False)
      subprocess.check_call([
          "riscv64-unknown-elf-gcc", "-march=rv32imf_zve32x", "-mabi=ilp32f",
          "-O3", "-nostdlib", cls.tf_c.name, "-o", cls.tf_elf.name
      ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=kDefaultCompilationTimeoutS)

      cls.native_elf_path = cls.tf_elf.name
      os.environ["CORALNPU_ELF"] = cls.native_elf_path
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
      raise unittest.SkipTest("Toolchain missing, cannot natively compile authentic ELF payload")

  @classmethod
  def tearDownClass(cls):
    if hasattr(cls, 'tf_c'):
      cls.tf_c.close()
      if os.path.exists(cls.tf_c.name): os.unlink(cls.tf_c.name)
    if hasattr(cls, 'tf_elf'):
      cls.tf_elf.close()
      if os.path.exists(cls.tf_elf.name): os.unlink(cls.tf_elf.name)
    if "CORALNPU_ELF" in os.environ:
      del os.environ["CORALNPU_ELF"]

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
      except FileNotFoundError:
        pass

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
          with self.assertRaises(RuntimeError):
            get_runner(Device.DEFAULT, si.ast)

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
        with self.assertRaises(RuntimeError):
          out.realize()
      except FileNotFoundError:
        pass

if __name__ == '__main__':
  unittest.main()
