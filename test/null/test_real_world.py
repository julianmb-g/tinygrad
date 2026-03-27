import gc
import math
import time
import unittest

import numpy as np

from extra.lr_scheduler import OneCycleLR
from test.helpers import slow
from tinygrad import Device, GlobalCounters, Tensor, Variable, dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import Context
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters


def derandomize_model(model):
  for p in get_parameters(model):
    size = int(math.prod(p.shape))
    arr = (np.arange(size, dtype=np.float32) % 10) * 0.01
    p.replace(Tensor(arr.reshape(p.shape), device=p.device, dtype=p.dtype).contiguous())
    p.realize()

from examples.gpt2 import Transformer as GPT2Transformer
from examples.hlb_cifar10 import SpeedyResNet, hyp
from examples.stable_diffusion import UNetModel, unet_params
from extra.models.bert import BertForPretraining
from extra.models.llama import Transformer as LLaMaTransformer
from extra.models.unet import ResBlock

global_mem_used = 0
def helper_test(nm, gen, model, max_memory_allowed, max_kernels_allowed, expected_out=None, all_jitted=False):
  with Context(JIT=2):
    tms = []
    out = None
    for _ in range(2):
      early_gen = [x.realize() if isinstance(x, Tensor) else x for x in gen()]
      GlobalCounters.reset()
      Device[Device.DEFAULT].synchronize()
      st = time.perf_counter_ns()
      out = model(*early_gen)
      Device[Device.DEFAULT].synchronize()
      tms.append(time.perf_counter_ns() - st)
    mem_used = (GlobalCounters.mem_used - global_mem_used) / 1e9

    # TODO: jit should expose this correctly with graph
    kernels_used = len(model.jit_cache) if hasattr(model, "jit_cache") else None
    print(f"{nm}: used {mem_used:.5f} GB and {kernels_used} kernels in {min(tms)/1e6:.2f} ms")
    if expected_out is not None and out is not None:
      val = out.numpy()
      if val.size > 10:
        val = np.array([val.mean(), val.max(), val.min()])
      try:
        expected_arr = np.array(expected_out, dtype=np.float32) if not isinstance(expected_out, list) and np.isscalar(expected_out) else np.array(expected_out)  # noqa: E501
        assert val.shape == expected_arr.shape, f"Shape mismatch: val shape {val.shape} != expected_out shape {expected_arr.shape}"
        np.testing.assert_allclose(val, expected_arr, atol=1e-4, rtol=1e-4)
        print(f"{nm} output OK!")
      except Exception as e:
        print(f"ACTUAL OUT ({nm}):", repr(val))
        raise e
    elif out is not None:
      val = out.numpy()
      if val.size > 10:
        val = np.array([val.mean(), val.max(), val.min()])
      print(f"ACTUAL OUT ({nm}):", repr(val))
    assert mem_used < max_memory_allowed, f"{nm} used more than {max_memory_allowed:.3f} GB - {mem_used:.3f} GB used"
    assert (max_memory_allowed - mem_used) / max_memory_allowed < 0.2, f"{max_memory_allowed:.3f} GB is too far from {mem_used:.3f} GB used"
    if kernels_used:
      assert kernels_used <= max_kernels_allowed, f"{nm} used more than {max_kernels_allowed} kernels, it used {kernels_used}"
      assert (max_kernels_allowed - kernels_used) / max_kernels_allowed < 0.2, f"{max_kernels_allowed=} is too far from {kernels_used=} used"
    if all_jitted:
      assert kernels_used is not None and (kernels_used > 0 and kernels_used == GlobalCounters.kernel_count or (kernels_used <= GlobalCounters.kernel_count and getattr(Device[Device.DEFAULT], "graph", None))), f"only {kernels_used} out of {GlobalCounters.kernel_count} were jitted"  # noqa: E501

class TestRealWorld(unittest.TestCase):
  def setUp(self):
    gc.collect()
    global global_mem_used
    global_mem_used = GlobalCounters.mem_used
    self.old_float = dtypes.default_float
    np.random.seed(2002)

  def tearDown(self):
    dtypes.default_float = self.old_float

  @slow
  def test_stable_diffusion(self):
    params = unet_params
    params["model_ch"] = 8
    params["ctx_dim"] = 8
    params["num_res_blocks"] = 1
    params["n_heads"] = 2
    model = UNetModel(**params)
    derandomize_model(model)
    @TinyJit
    def test(t, t2): return model(t, Tensor([801]), t2).realize()
    helper_test("test_sd", lambda: (((Tensor.arange(1*4*32*32)%10)*0.1).reshape(1, 4, 32, 32).cast(dtypes.float32), ((Tensor.arange(1*77*params["ctx_dim"])%10)*0.1).reshape(1, 77, params["ctx_dim"]).cast(dtypes.float32)), test, 0.0105, 460, expected_out=[0.07505496, 0.18881409, -0.05983206])  # noqa: E501
  def test_unet_resblock(self):
    model = [ResBlock(16, 24, 16) for _ in range(4)]
    derandomize_model(model)
    @TinyJit
    def test(t, t2):
      for l in model: t = l(t, t2)
      return t.realize()
    helper_test("test_unet_resblock", lambda: (((Tensor.arange(4*16*8*8)%10)*0.1).reshape(4, 16, 8, 8).cast(dtypes.float32), ((Tensor.arange(1*24)%10)*0.1).reshape(1, 24).cast(dtypes.float32)), test, 0.0002, 37, expected_out=[1.1154178, 2.4336586, -0.01451139])  # noqa: E501

  def test_llama(self):
    dtypes.default_float = dtypes.float16

    args_tiny = {"dim": 1024, "hidden_dim": 2048, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-05, "vocab_size": 1000}
    model = LLaMaTransformer(**args_tiny)
    derandomize_model(model)
    @TinyJit
    def test(t): return model(t, 0).realize()
    # TODO: test first token vs rest properly
    helper_test("test_llama", lambda: (Tensor([[1,2,3,4]]),), test, 0.21, 118, expected_out=0, all_jitted=True)
  def test_gpt2(self):
    dtypes.default_float = dtypes.float16

    args_tiny = {"dim": 1024, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-5, "vocab_size": 1000}
    model = GPT2Transformer(**args_tiny)
    derandomize_model(model)
    @TinyJit
    def test(t, v):
      with Context(JIT=0): return model(t, v).realize()
    helper_test("test_gpt2", lambda: (Tensor([[1,]]),Variable("pos", 1, 100).bind(1)), test, 0.22, 168, expected_out=[0], all_jitted=True)
  @slow
  def test_train_mnist(self):
    from examples.beautiful_mnist import Model
    with Tensor.train():
      model = Model()
      derandomize_model(model)
      optimizer = optim.Adam(get_parameters(model))
      BS = 32

      @TinyJit
      def train(X):
        out = model(X)
        loss = out.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.realize()

      helper_test("train_mnist", lambda: (((Tensor.arange(BS*1*28*28)%10)*0.1).reshape(BS, 1, 28, 28).cast(dtypes.float32),), train, 0.0105, 116, expected_out=1.8422501)  # noqa: E501

  @slow
  def test_forward_cifar(self):
    BS = 32
    # with training batchnorm still though
    with Tensor.train():
      model = SpeedyResNet(Tensor.ones((12,3,2,2)))
      derandomize_model(model)
      @TinyJit
      def run(X): return model(X).realize()
      helper_test("forward_cifar", lambda: (((Tensor.arange(BS*3*32*32)%10)*0.1).reshape(BS, 3, 32, 32).cast(dtypes.float32),), run, 0.0325, 30, expected_out=[0.3623826, 0.51155293, 0.2839802])  # noqa: E501

  @slow
  def test_train_cifar(self):
    with Tensor.train():
      model = SpeedyResNet(Tensor.ones((12,3,2,2)))
      derandomize_model(model)
      optimizer = optim.SGD(get_parameters(model), lr=0.01, momentum=0.8, nesterov=True, weight_decay=0.15)
      BS = 32

      @TinyJit
      def train(X):
        out = model(X)
        loss = out.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.realize()

      helper_test("train_cifar", lambda: (((Tensor.arange(BS*3*32*32)%10)*0.1).reshape(BS, 3, 32, 32).cast(dtypes.float32),), train, 0.110, 159, expected_out=0.35809016)  # noqa: E501

  def test_train_cifar_hyp(self):
    dtypes.default_float = dtypes.float16
    with Tensor.train():
      model = SpeedyResNet(Tensor.ones((12,3,2,2)))
      optimizer = optim.SGD(get_parameters(model), lr=0.01, momentum=hyp['opt']['momentum'], nesterov=True, weight_decay=hyp['opt']['bias_decay'])
      initial_div_factor = hyp['opt']['initial_div_factor']
      final_lr_ratio = hyp['opt']['final_lr_ratio']
      pct_start = hyp['opt']['percent_start']
      lr_scheduler = OneCycleLR(optimizer, max_lr=hyp['opt']['bias_lr'], pct_start=pct_start, div_factor=initial_div_factor,
                                final_div_factor=1./(initial_div_factor*final_lr_ratio), total_steps=4)
      assert not np.isnan(lr_scheduler.min_lr), "lr too small or initial_div_facotr too big for half"
  @slow
  def test_bert(self):
    with Tensor.train():
      args_tiny = {"attention_probs_dropout_prob": 0.0, "hidden_dropout_prob": 0.0, "vocab_size": 30522, "type_vocab_size": 2,
                  "max_position_embeddings": 512, "hidden_size": 128, "intermediate_size": 512, "num_attention_heads": 2, "num_hidden_layers": 2}
      model = BertForPretraining(**args_tiny)
      derandomize_model(model)
      optimizer = optim.LAMB(get_parameters(model))

      @TinyJit
      def train(input_ids:Tensor, segment_ids:Tensor, attention_mask:Tensor,
                masked_positions:Tensor, masked_lm_ids:Tensor, masked_lm_weights:Tensor, next_sentence_labels:Tensor):
        lm_logits, seq_relationship_logits = model(input_ids, attention_mask, masked_positions, segment_ids)
        loss = model.loss(lm_logits, seq_relationship_logits, masked_lm_ids, masked_lm_weights, next_sentence_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.realize()

      from examples.mlperf.helpers import get_fake_data_bert
      data = get_fake_data_bert(BS=4)
      for v in data.values(): v.to_(Device.DEFAULT)

      helper_test("train_bert", lambda: (data["input_ids"], data["segment_ids"], data["input_mask"], data["masked_lm_positions"], \
          data["masked_lm_ids"], data["masked_lm_weights"], data["next_sentence_labels"]), train, 0.22, 425, expected_out=2.1521404)

if __name__ == '__main__':
  unittest.main()
