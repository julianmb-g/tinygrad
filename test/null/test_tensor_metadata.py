import unittest

from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import capturing
from tinygrad.engine.schedule import linear_to_schedule
from tinygrad.helpers import Context
from tinygrad.tensor import _METADATA


class TestTensorMetadata(unittest.TestCase):
  def setUp(self) -> None:
    _METADATA.set(None)
    from tinygrad.uop.ops import UOpMetaClass, all_metadata
    UOpMetaClass.ucache.clear()
    all_metadata.clear()
    self._ctx = Context(SCACHE=0)
    self._ctx.__enter__()
  def tearDown(self) -> None:
    self._ctx.__exit__(None, None, None)

  def test_exclude_noop_metadata(self):
    a = Tensor.rand(4, 4)*1
    assert a.uop.metadata is not None
    self.assertEqual(a.uop.metadata[0].name, "__mul__")
    k = a.schedule()[-1]
    self.assertEqual(len(k.metadata), 2)
  def test_exclude_const_metadata(self):
    a = Tensor.arange(4)
    b = Tensor.full((4,), -1, dtype=dtypes.int).contiguous()
    sched = Tensor.schedule(a, b)
    self.assertEqual(len(sched[0].metadata), 1)
    self.assertEqual(len(sched[1].metadata), 1)
  def test_matmul(self):
    x = Tensor.rand(3, requires_grad=True)
    W = Tensor.rand(3, 3, requires_grad=True)
    out = x.matmul(W)
    assert out.uop.metadata is not None
    self.assertEqual(out.uop.metadata[0].name, "matmul")
    si = out.schedule()[-1]
    self.assertEqual(len(si.metadata), 2)
    self.assertEqual(si.metadata[0].name, "matmul")
  def test_relu(self):
    x = Tensor.rand(3, requires_grad=True)
    out = x.relu()
    assert out.uop.metadata is not None
    self.assertEqual(out.uop.metadata[0].name, "relu")
    si = out.schedule()[-1]
    self.assertEqual(len(si.metadata), 2)
    self.assertEqual(si.metadata[0].name, "relu")
  def test_assign(self):
    x = Tensor.empty(10, 10).realize()
    x.assign(Tensor.ones(10, 10).contiguous())
    si = x.schedule()[-1]
    self.assertEqual(len(si.metadata), 2)
    self.assertEqual(si.metadata[0].name, "assign")
  def test_complex(self):
    x = Tensor.rand(3, requires_grad=True)
    y = Tensor.rand(3, requires_grad=True)
    out = x.relu() * y.sigmoid()
    assert out.uop.metadata is not None
    assert out.uop.src[0].metadata is not None
    assert out.uop.src[1].metadata is not None
    self.assertEqual(out.uop.metadata[0].name, "__mul__")
    self.assertEqual(out.uop.src[0].metadata[0].name, "relu")
    self.assertEqual(out.uop.src[1].metadata[0].name, "sigmoid")
    si = out.schedule()[-1]
    self.assertEqual(len(si.metadata), 4)
    self.assertEqual(set(m.name for m in si.metadata), {"relu", "sigmoid", "__mul__", "rand"})
  def test_complex_backward(self):
    x = Tensor.rand(3, requires_grad=True).realize()
    y = Tensor.rand(3, requires_grad=True).realize()
    out = (x.relu() * y.sigmoid()).sum()
    assert out.uop.metadata is not None
    self.assertEqual(out.uop.metadata[0].name, "sum")
    out.backward()
    assert x.grad is not None and y.grad is not None
    assert x.grad.uop.metadata is not None
    assert y.grad.uop.metadata is not None
    self.assertEqual(x.grad.uop.metadata[0].name, "relu")
    self.assertTrue(x.grad.uop.metadata[0].backward)
    self.assertEqual(y.grad.uop.metadata[0].name, "sigmoid")
    self.assertTrue(y.grad.uop.metadata[0].backward)
    si = Tensor.schedule(out, x.grad, y.grad)[-1]
    self.assertEqual(len(si.metadata), 4)
    # skip numpy, this is schedule cache
    self.assertSetEqual(set(m.name for m in si.metadata if m.name != "numpy"), {"sigmoid", "relu", "sum"})
    bw = [m for m in si.metadata if getattr(m, 'backward', False)]
    self.assertEqual(len(bw), 2)
    self.assertEqual(set(m.name for m in bw), {"sigmoid", "sum"})

  def test_tracemeta_0(self):
    with Context(TRACEMETA=0):
      x = Tensor.rand(3, requires_grad=True)
      y = Tensor.rand(3, requires_grad=True)
      out = (x.relu() * y.sigmoid()).sum()
      self.assertIsNone(out.uop.metadata)
      self.assertIsNone(out.uop.src[0].metadata)
      si = out.schedule()[-1]
      self.assertEqual(si.metadata, ())
  def _has_metadata(self, h, name):
    linears = []
    capturing.append(type("", (), {"add_linear": lambda _, linear, var_vals: linears.append(linear)})())
    try: h.realize()
    finally: capturing.clear()
    items = [ei for linear in linears for ei in linear_to_schedule(linear)]
    return any(m.name == name for ei in items for m in ei.metadata)

  def test_metadata_survives_realize_pending_assign(self):
    shared = Tensor.rand(4)
    c = Tensor.zeros(8).contiguous().realize()
    c[:4].assign(shared)
    self.assertTrue(self._has_metadata(c[:4].relu(), "relu"))
  def test_metadata_lost_realize_pending_assign(self):
    shared = Tensor.rand(4)
    c = Tensor.zeros(8).contiguous().realize()
    c[:4].assign(shared)
    self.assertTrue(self._has_metadata((c[:4] + shared).relu(), "relu"))

  def test_metadata_assertions(self):
    x = Tensor.rand(3)
    out = x.relu()
    self.assertIsNotNone(out.uop.metadata)
    self.assertEqual(out.uop.metadata[0].name, "relu")

if __name__ == '__main__':
  unittest.main()
