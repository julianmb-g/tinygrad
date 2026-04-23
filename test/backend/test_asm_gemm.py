import pytest
import unittest
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.helpers import DEV, getenv, system
from extra.gemm.cdna_asm_gemm import asm_gemm
from test.helpers import needs_second_gpu
from tinygrad.codegen.opt.heuristic import OutOfMemoryError
from examples.mlperf.models.flat_llama import FP8_DTYPE

# On non CDNA4 it will only validate the Tensor.custom_kernel integration
def is_cdna4():
  dev = DEV.value.device or 'CORALNPU'
  if dev != "AMD": return False
  return getattr(Device[dev].renderer, "arch", "").startswith("gfx950")

def run_asm_gemm(a_shape, b_shape, dtype=dtypes.float16, a_shard=None, b_shard=None, gpus:int=1, force_coral=False) -> None:
  dev = "CORALNPU" if force_coral else (DEV.value.device or 'CORALNPU')
  Tensor.manual_seed(0)
  a_rand = Tensor.randn(a_shape, dtype=dtypes.float, device="CPU").sub(0.5).cast(dtype).to(dev)
  b_rand = Tensor.randn(b_shape, dtype=dtypes.float, device="CPU").sub(0.5).cast(dtype).to(dev)
  with Context(DEBUG=0):
    Tensor.realize(a_rand, b_rand)

  devs = tuple(f"{dev}:{i}" for i in range(gpus)) if (multi:=gpus>1) else None

  a, b = a_rand.clone().requires_grad_(), b_rand.clone().requires_grad_()
  if multi: a, b = a.shard(devs, axis=a_shard), b.shard(devs, axis=b_shard)
  with Context(ASM_GEMM=1):
    tst = asm_gemm(a, b)
    tst.sum().backward()
  Tensor.realize(tst, a.grad, b.grad)

  a_ref, b_ref = a_rand.clone().requires_grad_(), b_rand.clone().requires_grad_()
  # do reference gemm in bf16 for fp8, adjusting atol for quantization effects
  if a_ref.dtype == FP8_DTYPE:
    a_ref = a_ref.cast(dtypes.bfloat16)
    b_ref = b_ref.cast(dtypes.bfloat16)
  if multi: a_ref, b_ref = a_ref.shard(devs, axis=a_shard), b_ref.shard(devs, axis=b_shard)
  with Context(ASM_GEMM=0):
    ref = asm_gemm(a_ref, b_ref)
    ref.sum().backward()
  Tensor.realize(ref, a_ref.grad, b_ref.grad)

  atol, rtol = (2e-1, 1e-2) if dtype == dtypes.bfloat16 else (1e-2, 1e-3)
  grad_atol, grad_rtol = (0.2, 0.05) if dtype == FP8_DTYPE else (atol, rtol)
  with Context(DEBUG=0):
    # enable for debugging, slow for larger gemms
    if getenv("USE_NPY"):
      import numpy as np
      np.testing.assert_allclose(tst.numpy(), ref.numpy(), atol=atol, rtol=rtol)
      np.testing.assert_allclose(a.grad.numpy(), a_ref.grad.numpy(), atol=grad_atol, rtol=grad_rtol)
      np.testing.assert_allclose(b.grad.numpy(), b_ref.grad.numpy(), atol=grad_atol, rtol=grad_rtol)
    if is_cdna4() or (DEV.value.device or "CORALNPU") == "CORALNPU":
      assert tst.allclose(ref, atol=atol, rtol=rtol), "forward mismatch"
      assert a.grad.allclose(a_ref.grad.cast(a.grad.dtype), atol=grad_atol, rtol=grad_rtol), "grad_a mismatch"
      assert b.grad.allclose(b_ref.grad.cast(b.grad.dtype), atol=grad_atol, rtol=grad_rtol), "grad_b mismatch"

def _is_coral(): return (DEV.value.device or "CORALNPU") == "CORALNPU"

def _coral_exceeds_dtcm(batch, M, N, K, dtype, gpus):
  sz_a = batch * M * K * dtype.itemsize if batch > 1 else M * K * dtype.itemsize
  sz_b = batch * K * N * dtype.itemsize if batch > 1 else K * N * dtype.itemsize
  sz_c = batch * M * N * dtype.itemsize if batch > 1 else M * N * dtype.itemsize
  if gpus > 1:
    sz_a //= gpus
    sz_b //= gpus
    sz_c //= gpus
  return (sz_a + sz_b + sz_c) > 28 * 1024

def verify_asm_gemm(batch:int, M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=1, allow_scale=False) -> None:
  if allow_scale and is_cdna4():
    import unittest
    with unittest.TestCase().assertRaises(OutOfMemoryError):
      run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus)
  elif allow_scale and _coral_exceeds_dtcm(batch, M, N, K, dtype, gpus):
    from tinygrad.runtime.ops_coralnpu import SimTimeoutError
    import unittest
    with unittest.TestCase().assertRaises((OutOfMemoryError, SimTimeoutError)):
      run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus, force_coral=True)
  else:
    run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus)

def verify_asm_gemm_k_sharded(M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=8, allow_scale=False) -> None:
  if allow_scale and is_cdna4():
    import unittest
    with unittest.TestCase().assertRaises(OutOfMemoryError):
      run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=1, b_shard=0, gpus=gpus)
  elif allow_scale and _coral_exceeds_dtcm(1, M, N, K, dtype, gpus):
    from tinygrad.runtime.ops_coralnpu import SimTimeoutError
    import unittest
    with unittest.TestCase().assertRaises((OutOfMemoryError, SimTimeoutError)):
      run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=1, b_shard=0, gpus=gpus, force_coral=True)
  else:
    run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=1, b_shard=0, gpus=gpus)

def verify_asm_gemm_n_sharded(batch:int, M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=2, allow_scale=False) -> None:
  if allow_scale and is_cdna4():
    import unittest
    with unittest.TestCase().assertRaises(OutOfMemoryError):
      run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=None, b_shard=1, gpus=gpus)
  elif allow_scale and _coral_exceeds_dtcm(batch, M, N, K, dtype, gpus):
    from tinygrad.runtime.ops_coralnpu import SimTimeoutError
    import unittest
    with unittest.TestCase().assertRaises((OutOfMemoryError, SimTimeoutError)):
      run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=None, b_shard=1, gpus=gpus, force_coral=True)
  else:
    run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=None, b_shard=1, gpus=gpus)

def verify_asm_gemm_m_sharded(M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=2, allow_scale=False) -> None:
  if allow_scale and is_cdna4():
    import unittest
    with unittest.TestCase().assertRaises(OutOfMemoryError):
      run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus)
  elif allow_scale and _coral_exceeds_dtcm(1, M, N, K, dtype, gpus):
    from tinygrad.runtime.ops_coralnpu import SimTimeoutError
    import unittest
    with unittest.TestCase().assertRaises((OutOfMemoryError, SimTimeoutError)):
      run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus, force_coral=True)
  else:
    run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus)

def verify_asm_gemm_n_sharded_2d(M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=2, allow_scale=False) -> None:
  if allow_scale and is_cdna4():
    import unittest
    with unittest.TestCase().assertRaises(OutOfMemoryError):
      run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=None, b_shard=1, gpus=gpus)
  elif allow_scale and _coral_exceeds_dtcm(1, M, N, K, dtype, gpus):
    from tinygrad.runtime.ops_coralnpu import SimTimeoutError
    import unittest
    with unittest.TestCase().assertRaises((OutOfMemoryError, SimTimeoutError)):
      run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=None, b_shard=1, gpus=gpus, force_coral=True)
  else:
    run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=None, b_shard=1, gpus=gpus)

def verify_asm_gemm_k_sharded_3d(batch:int, M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=2, allow_scale=False) -> None:
  if allow_scale and is_cdna4():
    import unittest
    with unittest.TestCase().assertRaises(OutOfMemoryError):
      run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=2, b_shard=0, gpus=gpus)
  elif allow_scale and _coral_exceeds_dtcm(batch, M, N, K, dtype, gpus):
    from tinygrad.runtime.ops_coralnpu import SimTimeoutError
    import unittest
    with unittest.TestCase().assertRaises((OutOfMemoryError, SimTimeoutError)):
      run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=2, b_shard=0, gpus=gpus, force_coral=True)
  else:
    run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=2, b_shard=0, gpus=gpus)

# 128x smaller than usual
# uses the UOp GEMM, runs on non CDNA4 and CI
# @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
@pytest.mark.timeout(30)
class TestGemm(unittest.TestCase):
  def setUp(self):
    pass
  def test_simple(self): verify_asm_gemm(1, N:=getenv("N", 32), N, N, dtype=dtypes.half)
  def test_gemm(self): verify_asm_gemm(1, 64, 32, 112)
  def test_gemm_batched(self): verify_asm_gemm(2, 64, 32, 32)
  @needs_second_gpu
  def test_gemm_multi(self): verify_asm_gemm(2, 64, 32, 32, gpus=2)
  @needs_second_gpu
  def test_gemm_k_sharded(self): verify_asm_gemm_k_sharded(64, 64, 2*64, gpus=2)
  @needs_second_gpu
  def test_gemm_m_sharded(self): verify_asm_gemm_m_sharded(2*64, 64, 32, gpus=2)
  @needs_second_gpu
  def test_gemm_n_sharded(self): verify_asm_gemm_n_sharded(1, 64, 64, 32, gpus=2)
  @needs_second_gpu
  def test_gemm_n_sharded_2d(self): verify_asm_gemm_n_sharded_2d(64, 2*64, 32, gpus=2)
  @needs_second_gpu
  def test_gemm_k_sharded_3d(self): verify_asm_gemm_k_sharded_3d(1, 64, 32, 2*64, gpus=2)

# uses the smallest size for the cdna assembly gemm
class TestAsmGEMM(unittest.TestCase):
  def setUp(self):
    pass

  def test_tiny(self): verify_asm_gemm(1, 256, 256, 64)

  def test_verify_with_numpy(self):
    import numpy as np
    M, N, K = 256, 256, 64
    rng = np.random.default_rng(0)
    a_np = (rng.random((M, K), dtype=np.float32) - 0.5).astype(np.half)
    b_np = (rng.random((K, N), dtype=np.float32) - 0.5).astype(np.half)
    c_np = a_np @ b_np
    a, b = Tensor(a_np), Tensor(b_np)
    c = asm_gemm(a, b)
    c.realize()

    np.testing.assert_allclose(c.numpy(), c_np, atol=2e-3, rtol=5e-2)

  def test_unsupported_batch(self):
    with self.assertRaisesRegex(AssertionError, "batch size"):
      verify_asm_gemm(3, 256, 256, 256)

  def test_unsupported_k(self):
    with self.assertRaisesRegex(AssertionError, "not a multiple"):
      verify_asm_gemm(1, 1024, 1024, 100)
  def test_unsupported_m(self):
    with self.assertRaisesRegex(AssertionError, "not a multiple"):
      verify_asm_gemm(1, 1000, 256, 256)
  def test_unsupported_n(self):
    with self.assertRaisesRegex(AssertionError, "not a multiple"):
      verify_asm_gemm(1, 256, 1000, 256)

# test the Asm GEMM with Llama shapes, only run on the real machine for speed
@pytest.mark.timeout(30)
class TestGemmLlama(unittest.TestCase):
  dtype = dtypes.bfloat16

  def setUp(self):
    pass

  @Context(ASM_GEMM=1)
  def test_empty(self):
    if is_cdna4():
      with self.assertRaises(OutOfMemoryError):
        (Tensor.empty(N:=getenv("N", 4096), N, dtype=self.dtype)@Tensor.empty(N, N, dtype=self.dtype)).realize()
    else:
      (Tensor.empty(N:=getenv("N", 4096), N, dtype=self.dtype)@Tensor.empty(N, N, dtype=self.dtype)).realize()

  @Context(ASM_GEMM=1)
  def test_empty_bw(self):
    if is_cdna4():
      with self.assertRaises(OutOfMemoryError):
        x = Tensor.empty(1, N:=getenv("N", 4096), N, dtype=self.dtype, requires_grad=True)
        y = Tensor.empty((N, N), dtype=self.dtype, requires_grad=True)
        z = x @ y
        z.sum().backward()
        Tensor.realize(z, x.grad, y.grad)
    else:
      x = Tensor.empty(1, N:=getenv("N", 4096), N, dtype=self.dtype, requires_grad=True)
      y = Tensor.empty((N, N), dtype=self.dtype, requires_grad=True)
      z = x @ y
      z.sum().backward()
      Tensor.realize(z, x.grad, y.grad)
      grad_dtype = dtypes.fp8e5m2 if self.dtype == FP8_DTYPE else self.dtype
      assert z.dtype == dtypes.bfloat16
      assert x.grad.dtype == y.grad.dtype == grad_dtype

  def test_simple(self): verify_asm_gemm(1, 256, 256, 256, dtype=self.dtype, allow_scale=True)
  def test_gemm(self): verify_asm_gemm(1, 8192, 4096, 14336, dtype=self.dtype, allow_scale=True)
  def test_gemm_batched(self): verify_asm_gemm(2, 8192, 4096, 4096, dtype=self.dtype, allow_scale=True)

  def test_gemm1(self): verify_asm_gemm(8, 8192, 4096, 14336, dtype=self.dtype, gpus=8, allow_scale=True)
  # @unittest.skip("disabled, asm in this shape is slower than tinygrad")
  def test_gemm2(self): verify_asm_gemm(8, 8192, 128256, 4096, dtype=self.dtype, gpus=8, allow_scale=True)
  def test_gemm3(self): verify_asm_gemm(8, 8192, 14336, 4096, dtype=self.dtype, gpus=8, allow_scale=True)
  def test_gemm4(self): verify_asm_gemm(8, 4096, 14336, 4096, dtype=self.dtype, gpus=8, allow_scale=True)
  def test_gemm5(self): verify_asm_gemm(8, 4096, 4096, 14336, dtype=self.dtype, gpus=8, allow_scale=True)
  def test_gemm6(self): verify_asm_gemm(16, 4096, 4096, 14336, dtype=self.dtype, gpus=8, allow_scale=True)
  # @unittest.skip("disabled, asm in this shape is slower than tinygrad")
  def test_gemm7(self): verify_asm_gemm(1, 8192, 128256, 4096, dtype=self.dtype, allow_scale=True)
  def test_gemm8(self): verify_asm_gemm(1, 4096, 14336, 8192, dtype=self.dtype, allow_scale=True)
  def test_gemm9(self): verify_asm_gemm(8, 4096, 14336, 8192, dtype=self.dtype, gpus=8, allow_scale=True)
  def test_gemm10(self): verify_asm_gemm(1, 4096, 8192, 4096, dtype=self.dtype, allow_scale=True)
  def test_gemm_previously_unsupported(self): verify_asm_gemm(8, 1024, 1024, 4096, gpus=8, allow_scale=True)
  def test_k_sharded_1(self): verify_asm_gemm_k_sharded(14336, 4096, 8*8192, dtype=self.dtype, gpus=8, allow_scale=True)
  def test_k_sharded_2(self): verify_asm_gemm_k_sharded(4096, 14336, 8*8192, dtype=self.dtype, gpus=8, allow_scale=True)
  def test_k_sharded_3(self): verify_asm_gemm_k_sharded(4096, 4096, 8*8192, dtype=self.dtype, gpus=8, allow_scale=True)

  # M-sharded 2D
  def test_m_sharded_1(self): verify_asm_gemm_m_sharded(8*8192, 4096, 4096, dtype=self.dtype, gpus=8, allow_scale=True)
  def test_m_sharded_2(self): verify_asm_gemm_m_sharded(8*4096, 14336, 4096, dtype=self.dtype, gpus=8, allow_scale=True)

  # N-sharded 2D
  def test_n_sharded_2d_1(self): verify_asm_gemm_n_sharded_2d(8192, 8*4096, 4096, dtype=self.dtype, gpus=8, allow_scale=True)
  def test_n_sharded_2d_2(self): verify_asm_gemm_n_sharded_2d(4096, 8*14336, 4096, dtype=self.dtype, gpus=8, allow_scale=True)

  # tensor parallel shapes (Llama 8B, MP=8)
  def test_tp_n_sharded_wq(self): verify_asm_gemm_n_sharded(1, 8192, 4096, 4096, dtype=self.dtype, gpus=8, allow_scale=True)
  def test_tp_n_sharded_w1(self): verify_asm_gemm_n_sharded(1, 8192, 14336, 4096, dtype=self.dtype, gpus=8, allow_scale=True)
  def test_tp_k_sharded_wo(self): verify_asm_gemm_k_sharded_3d(1, 8192, 4096, 4096, dtype=self.dtype, gpus=8, allow_scale=True)
  def test_tp_k_sharded_w2(self): verify_asm_gemm_k_sharded_3d(1, 8192, 4096, 14336, dtype=self.dtype, gpus=8, allow_scale=True)

  # more shapes: vary M, N, K independently
  def test_shape_small_square(self): verify_asm_gemm(1, 256, 256, 256, allow_scale=True)
  def test_shape_small_rect_m(self): verify_asm_gemm(1, 512, 256, 256, allow_scale=True)
  def test_shape_small_rect_n(self): verify_asm_gemm(1, 256, 512, 256, allow_scale=True)
  def test_shape_small_rect_k(self): verify_asm_gemm(1, 256, 256, 512, allow_scale=True)
  def test_shape_tall(self): verify_asm_gemm(1, 2048, 256, 256, allow_scale=True)
  def test_shape_wide(self): verify_asm_gemm(1, 256, 2048, 256, allow_scale=True)
  def test_shape_deep(self): verify_asm_gemm(1, 256, 256, 4096, allow_scale=True)
  def test_shape_non_square(self): verify_asm_gemm(1, 1024, 2048, 512, allow_scale=True)
  def test_shape_batched_small(self): verify_asm_gemm(2, 256, 256, 256, allow_scale=True)
  def test_shape_batched_rect(self): verify_asm_gemm(2, 512, 1024, 256, allow_scale=True)
  # K edge cases: iters=1,2,3 exercise different loop paths
  def test_shape_k64(self): verify_asm_gemm(1, 256, 256, 64, allow_scale=True)
  def test_shape_k128(self): verify_asm_gemm(1, 256, 256, 128, allow_scale=True)
  def test_shape_k192(self): verify_asm_gemm(1, 256, 256, 192, allow_scale=True)

  def test_llama3_out1(self): verify_asm_gemm(1, 8192, 128256, 4096, dtype=self.dtype, allow_scale=True)
  def test_llama3_out2(self): verify_asm_gemm(1, 8192, 4096, 128256, dtype=self.dtype, allow_scale=True)
  def test_llama3_out3(self): verify_asm_gemm(1, 4096, 128256, 8192, dtype=self.dtype, allow_scale=True)

def has_hipcc():
  try: system("hipcc --version")
  except Exception: return False
  return True

# @unittest.skipUnless(has_hipcc(), "FP8 gemm requires hipcc to compile")
class TestGemmLlamaFP8(TestGemmLlama): dtype = FP8_DTYPE

class TestMagicGu(unittest.TestCase):
  def test_magicgu_matches_old(self):
    from extra.gemm.cdna_asm_gemm import _magicgu_mulhi, TILE_M, TILE_N, TILE_K
    old_iters_args = {64: (67108864, 0), 128: (33554432, 0), 224: (613566757, 2147483656)}
    old_gemm_shapes = [
      (8192, 4096, 4096), (8192, 14336, 4096), (8192, 4096, 14336),
      (8192, 8192, 8192), (4096, 4096, 4096), (4096, 14336, 4096),
      (4096, 14336, 8192), (4096, 4096, 14336), (14336, 4096, 8192),
      (4096, 8192, 14336), (4096, 4096, 8192), (4096, 8192, 4096),
    ]
    for M, N, K in old_gemm_shapes:
      iters = K // TILE_K
      total = (M // TILE_M) * (N // TILE_N) * iters
      for batch in [1, 2]:
        magic, shift = _magicgu_mulhi(iters, total * batch)
        old_magic, old_shift = old_iters_args[iters]
        self.assertEqual((magic, shift), (old_magic, old_shift), f"mismatch for ({M},{N},{K}) batch={batch} iters={iters}")

if __name__ == "__main__":
  unittest.main()
