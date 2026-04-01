import multiprocessing.shared_memory as shared_memory
import unittest
import numpy as np

from tinygrad.helpers import CI, WIN
from tinygrad.tensor import Device, Tensor

class TestRawShmBuffer(unittest.TestCase):
  @unittest.skipIf(WIN and CI, "only fails on CI windows instance")
  def test_e2e(self):
    t = ((Tensor.arange(2*2*2) % 10) * 0.1).reshape(2, 2, 2).realize()

    # copy to shm
    s = shared_memory.SharedMemory(create=True, size=t.nbytes())
    shm_name = s.name
    view = memoryview(s.buf) # type: ignore
    view.release()
    s.close()
    t_shm = t.to(f"disk:shm:{shm_name}").realize()

    # copy from shm
    t2 = t_shm.to(Device.DEFAULT).realize()

    assert np.allclose(t.numpy(), t2.numpy())
    s.unlink()

  def test_e2e_big(self):
    # bigger than this doesn't work on Linux, maybe this is a limit somewhere?
    t = ((Tensor.arange(2048*128*8) % 10) * 0.1).reshape(2048, 128, 8).realize()

    # copy to shm
    s = shared_memory.SharedMemory(create=True, size=t.nbytes())
    shm_name = s.name
    view = memoryview(s.buf) # type: ignore
    view.release()
    s.close()
    t_shm = t.to(f"disk:shm:{shm_name}").realize()

    # copy from shm
    t2 = t_shm.to(Device.DEFAULT).realize()

    assert np.allclose(t.numpy(), t2.numpy())
    s.unlink()
if __name__ == "__main__":
  unittest.main()
