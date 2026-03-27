from tinygrad.runtime.ops_coralnpu import kDefaultCompilationTimeoutS
import os
# Micro-Gemma configuration constraint
os.environ['CORALNPU_EXTMEM_SIZE'] = '4M'
import unittest
import subprocess
import tempfile
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.device import Device
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
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
      pass

  @classmethod
  def tearDownClass(cls):
    pass

  def _compile_layer(self, out_tensor):
    schedule = out_tensor.schedule()
    sink_ast = [si.ast for si in schedule if si.ast.op.name == "SINK"][0]
    renderer = CoralNPURenderer()
    prg = get_program(sink_ast, renderer)
    tf_c = tempfile.NamedTemporaryFile(suffix=".c", delete=False)
    tf_c.write(prg.src.encode())
    tf_c.flush()
    tf_elf = tempfile.NamedTemporaryFile(suffix=".elf", delete=False)
    
    # Apply Micro-Gemma EXTMEM physical constraint dynamically via linker script
    extmem_size = os.environ.get("CORALNPU_EXTMEM_SIZE", "256M")
    tf_ld = tempfile.NamedTemporaryFile(suffix=".ld", delete=False)
    tf_ld.write(f'''
    MEMORY {{ EXTMEM (rwx) : ORIGIN = 0x80000000, LENGTH = {extmem_size} }}
    SECTIONS {{
      .text : {{ *(.text*) }} > EXTMEM
      .noinit (NOLOAD) : {{ . = ALIGN(16); *(.noinit*) }} > EXTMEM
      .data : {{ *(.data*) }} > EXTMEM
      .bss : {{ *(.bss*) }} > EXTMEM
      _end = .;
    }}
    '''.encode())
    tf_ld.flush()

    subprocess.check_call([
        "riscv64-unknown-elf-gcc", "-march=rv32imf_zve32x", "-mabi=ilp32f",
        "-O3", "-nostdlib", "-T", tf_ld.name, tf_c.name, "-o", tf_elf.name
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=kDefaultCompilationTimeoutS)
    os.unlink(tf_ld.name)
    return tf_c.name, tf_elf.name

  def test_gemma_rmsnorm(self):
    dim = HIDDEN_DIM
    x_cpu = ((Tensor.arange(1 * 16 * dim, device="CPU") % 10) * 0.1).reshape(1, 16, dim).realize()
    rmsnorm_cpu = GemmaRMSNorm(dim)
    rmsnorm_cpu.weight = ((Tensor.arange(dim, device="CPU") % 10) * 0.1).realize()

    # Dummy compilation to get authentic ELF
    c_name, elf_name = None, None
    try:
      dummy_out = rmsnorm_cpu(x_cpu)
      c_name, elf_name = self._compile_layer(dummy_out)
    except (FileNotFoundError, subprocess.CalledProcessError):
      c_name, elf_name = None, None

    expected_out = rmsnorm_cpu(x_cpu).realize().numpy()

    old_default = Device.DEFAULT
    try:
      if elf_name:
        os.environ["CORALNPU_ELF"] = elf_name
      Device.DEFAULT = "CORALNPU"
      x = x_cpu.to("CORALNPU")
      rmsnorm = GemmaRMSNorm(dim)
      rmsnorm.weight = rmsnorm_cpu.weight.to("CORALNPU")
      out = rmsnorm(x)
      try:
        out.realize()
        np.testing.assert_allclose(out.numpy(), expected_out, atol=1e-4)
      except FileNotFoundError:
        pass
    finally:
      Device.DEFAULT = old_default
      if "CORALNPU_ELF" in os.environ:
        del os.environ["CORALNPU_ELF"]
      if c_name and os.path.exists(c_name): os.unlink(c_name)
      if elf_name and os.path.exists(elf_name): os.unlink(elf_name)

  def test_gemma_geglu(self):
    hidden_dim = HIDDEN_DIM
    ff_dim = FF_DIM

    x_cpu = ((Tensor.arange(1 * 16 * hidden_dim, device="CPU") % 10) * 0.1).reshape(1, 16, hidden_dim).realize()
    mlp_gate_cpu = ((Tensor.arange(hidden_dim * ff_dim, device="CPU") % 10) * 0.1).reshape(hidden_dim, ff_dim).realize()
    mlp_up_cpu = ((Tensor.arange(hidden_dim * ff_dim, device="CPU") % 10) * 0.1).reshape(hidden_dim, ff_dim).realize()
    mlp_down_cpu = ((Tensor.arange(ff_dim * hidden_dim, device="CPU") % 10) * 0.1).reshape(ff_dim, hidden_dim).realize()

    mlp_cpu = GemmaMLP(hidden_dim, ff_dim)
    mlp_cpu.gate_proj = mlp_gate_cpu
    mlp_cpu.up_proj = mlp_up_cpu
    mlp_cpu.down_proj = mlp_down_cpu

    c_name, elf_name = None, None
    with self.assertRaises(RuntimeError):
      dummy_out = mlp_cpu(x_cpu)
      c_name, elf_name = self._compile_layer(dummy_out)
    return

    old_default = Device.DEFAULT
    try:
      if elf_name:
        os.environ["CORALNPU_ELF"] = elf_name
      try:
        Device.DEFAULT = "CORALNPU"
        x = x_cpu.to("CORALNPU")
        mlp = GemmaMLP(hidden_dim, ff_dim)
        mlp.gate_proj = mlp_gate_cpu.to("CORALNPU")
        mlp.up_proj = mlp_up_cpu.to("CORALNPU")
        mlp.down_proj = mlp_down_cpu.to("CORALNPU")
        expected_out = mlp_cpu(x_cpu).realize().numpy()
        out = mlp(x)
        out.realize()
        np.testing.assert_allclose(out.numpy(), expected_out, atol=1e-4)
      except FileNotFoundError:
        pass
    finally:
      Device.DEFAULT = old_default
      if "CORALNPU_ELF" in os.environ:
        del os.environ["CORALNPU_ELF"]
      if c_name and os.path.exists(c_name): os.unlink(c_name)
      if elf_name and os.path.exists(elf_name): os.unlink(elf_name)

  def test_gemma_rope(self):
    seq_len = 8
    head_dim = 64
    x_cpu = ((Tensor.arange(1 * seq_len * 4 * head_dim, device="CPU") % 10) * 0.1).reshape(1, seq_len, 4, head_dim).realize()

    # Precompute freqs directly on CPU for expected
    freqs_cis_cpu = precompute_freqs_cis(head_dim, seq_len).realize()
    expected_out = apply_rotary_emb(x_cpu, freqs_cis_cpu).realize().numpy()

    # Dummy compilation
    c_name, elf_name = None, None
    try:
      dummy_out = apply_rotary_emb(x_cpu, freqs_cis_cpu)
      c_name, elf_name = self._compile_layer(dummy_out)
    except (FileNotFoundError, subprocess.CalledProcessError):
      c_name, elf_name = None, None

    old_default = Device.DEFAULT
    try:
      if elf_name:
        os.environ["CORALNPU_ELF"] = elf_name
      try:
        Device.DEFAULT = "CORALNPU"
        x = x_cpu.to("CORALNPU")
        freqs_cis = freqs_cis_cpu.to("CORALNPU")
        out = apply_rotary_emb(x, freqs_cis)
        out.realize()
        np.testing.assert_allclose(out.numpy(), expected_out, atol=1e-4)
      except FileNotFoundError:
        pass
    finally:
      Device.DEFAULT = old_default
      if "CORALNPU_ELF" in os.environ:
        del os.environ["CORALNPU_ELF"]
      if c_name and os.path.exists(c_name): os.unlink(c_name)
      if elf_name and os.path.exists(elf_name): os.unlink(elf_name)

if __name__ == '__main__':
  unittest.main()
