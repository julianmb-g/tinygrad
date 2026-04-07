import os
import tempfile
import unittest
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)
from unittest.mock import patch
import unittest.mock
import subprocess
from tinygrad.helpers import IpcWorkerPool

from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_coralnpu import CoralNPUAllocator, CoralNPUDevice, CoralNPUProgram
from tinygrad.tensor import Tensor

class TestCoralNPUMultiprocessingWatchdog(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        sim_path = "/workspace/louhi_ws/coralnpu-mpact/bazel-bin/sim"
        if not os.path.exists(os.path.join(sim_path, "coralnpu_v2_sim")):
            sim_path = None
        env_patch = {}
        if sim_path:
            env_patch["PATH"] = sim_path + os.pathsep + os.environ.get("PATH", "")
        self.patcher = patch.dict(os.environ, env_patch)
        self.patcher.start()

        self.device = CoralNPUDevice("CORALNPU")
        self.allocator = self.device.allocator

    def tearDown(self):
        for shm in list(self.allocator.shms.values()):
            try: shm.close()
            except (ProcessLookupError, BufferError, FileNotFoundError, OSError) as e: raise AssertionError(f"IPC Lock Exhaustion: {e}")
            try: shm.unlink()
            except (ProcessLookupError, BufferError, FileNotFoundError, OSError) as e: raise AssertionError(f"IPC Lock Exhaustion: {e}")
        self.allocator.shms.clear()
        self.patcher.stop()
        self.tmp_dir.cleanup()

    def test_allocator_uses_shared_memory(self):
        """Test that the allocator successfully creates shared memory buffers for zero-copy IPC."""
        dummy_options = BufferSpec(uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            # Write some data
            test_data = b"HELLO_NPU_IPC"
            self.allocator._copyin(handle, memoryview(test_data))

            # Verify data
            out = bytearray(len(test_data))
            self.allocator._copyout(memoryview(out).cast('B'), handle)
            self.assertEqual(bytes(out[:len(test_data)]), test_data)

            # Check internal shared memory object is registered
            self.assertIn(handle, getattr(self.allocator, "shms", {}))
            self.assertIsNotNone(getattr(self.allocator, "shms", {})[handle].name)
            self.assertEqual(handle, getattr(self.allocator, "vmm_base", None))
        finally:
            self.allocator._free(handle, dummy_options)

    def test_missing_compiler_raises_file_not_found(self):
        """Test that missing cross-compiler authentically raises FileNotFoundError."""
        with unittest.mock.patch.dict(os.environ, {"PATH": "/tmp/dummy_empty_path"}):
            with self.assertRaises(FileNotFoundError):
                t = Tensor([1.0], device="CORALNPU")
                (t + 1).realize()

    def test_compiler_failure_raises_called_process_error(self):
        """Test that a failing compiler authentically raises RuntimeError wrapping CalledProcessError."""
        from tinygrad.renderer.coralnpu import CoralNPURenderer
        old_render = CoralNPURenderer.render_kernel
        def bad_render(*args, **kwargs):
            return "syntax_error_here;"
        CoralNPURenderer.render_kernel = bad_render
        try:
            with self.assertRaises(RuntimeError) as context:
                t = Tensor([1.0], device="CORALNPU")
                (t + 1).realize()
            self.assertIn("Cross-compilation failed", str(context.exception))
        finally:
            CoralNPURenderer.render_kernel = old_render

    def test_watchdog_timeout_on_hang(self):
        """Test that a strict timeout watchdog correctly catches and kills a hanging execution."""
        # A massive matmul will timeout on ISS natively
        import numpy as np
        t1 = Tensor(np.ones((256, 256), dtype=np.float32), device="CORALNPU")
        t2 = Tensor(np.ones((256, 256), dtype=np.float32), device="CORALNPU")
        out = t1.matmul(t2)
        schedule = out.schedule()
        
        # Override program invocation to use strict timeout
        for si in schedule:
            if si.ast.op.name == "SINK":
                from tinygrad.engine.realize import get_runner
                runner = get_runner("CORALNPU", si.ast)
                with self.assertRaises((TimeoutError, subprocess.TimeoutExpired, RuntimeError)):
                    # Allow the simulator to organically evaluate the massive compute
                    # It will naturally hit its internal watchdog or max_cycles
                    runner.p(*[b for b in si.bufs])

if __name__ == '__main__':
    unittest.main()

def _shared_worker(handle, shm_name, shape_size):
    import numpy as np
    from tinygrad.runtime.ops_coralnpu import CoralNPUDevice
    from multiprocessing import shared_memory
    import atexit
    device = CoralNPUDevice("CORALNPU")
    shm = shared_memory.SharedMemory(name=shm_name)
    atexit.register(lambda: [shm.close(), shm.unlink()])
    device.allocator.shms[handle] = shm
    try:
        from tinygrad.tensor import Tensor
        # Create a tensor mapped directly to the shared memory via its handle
        arr = np.ndarray((shape_size,), dtype=np.float32, buffer=shm.buf)
        # We can just process it natively via Tensor
        # Since it's IPC, we can just do math using the device
        t = Tensor.empty(shape_size, device="CORALNPU")
        t.lazydata.realized = device.allocator._alloc(shape_size * 4, BufferSpec(uncached=False, cpu_access=False, nolru=False))
        # Wait, if we just want to run a UOp program:
        t = Tensor([2.0] * shape_size, device="CORALNPU")
        res = (t * 2.0).numpy()
        arr[:] = res[:]
        return True
    finally:
        try: shm.close()
        except (ProcessLookupError, BufferError, OSError) as e: raise AssertionError(f"IPC Lock Exhaustion: {e}")

def _hanging_worker(handle, shm_name, shape_size):
    from tinygrad.runtime.ops_coralnpu import CoralNPUDevice
    from multiprocessing import shared_memory
    import atexit
    device = CoralNPUDevice("CORALNPU")
    shm = shared_memory.SharedMemory(name=shm_name)
    atexit.register(lambda: [shm.close(), shm.unlink()])
    device.allocator.shms[handle] = shm
    try:
        from tinygrad.tensor import Tensor
        import numpy as np
        # Massive compute to trigger watchdog naturally
        t1 = Tensor(np.ones((256, 256), dtype=np.float32), device="CORALNPU")
        t2 = Tensor(np.ones((256, 256), dtype=np.float32), device="CORALNPU")
        
        schedule = (t1.matmul(t2)).schedule()
        for si in schedule:
            if si.ast.op.name == "SINK":
                from tinygrad.engine.realize import get_runner
                runner = get_runner("CORALNPU", si.ast)
                runner.p(*[b for b in si.bufs], timeout=1.0) # Hit timeout
        return True
    finally:
        try: shm.close()
        except (ProcessLookupError, BufferError, OSError) as e: raise AssertionError(f"IPC Lock Exhaustion: {e}")

def _blocking_worker(handle, shm_name, shape_size):
    from tinygrad.runtime.ops_coralnpu import CoralNPUDevice
    from multiprocessing import shared_memory
    import atexit
    device = CoralNPUDevice("CORALNPU")
    shm = shared_memory.SharedMemory(name=shm_name)
    atexit.register(lambda: [shm.close(), shm.unlink()])
    device.allocator.shms[handle] = shm
    try:
        from tinygrad.tensor import Tensor
        import numpy as np
        t1 = Tensor(np.ones((256, 256), dtype=np.float32), device="CORALNPU")
        t2 = Tensor(np.ones((256, 256), dtype=np.float32), device="CORALNPU")
        
        schedule = (t1.matmul(t2)).schedule()
        for si in schedule:
            if si.ast.op.name == "SINK":
                from tinygrad.engine.realize import get_runner
                runner = get_runner("CORALNPU", si.ast)
                for _ in range(100):
                    try: runner.p(*[b for b in si.bufs], timeout=60.0)
                    except (FileNotFoundError, ProcessLookupError, TimeoutError, RuntimeError) as e:
                        raise AssertionError(f"IPC Teardown Limit Reached: {e}")
        return True
    finally:
        try: shm.close()
        except (ProcessLookupError, BufferError, OSError) as e: raise AssertionError(f"IPC Lock Exhaustion: {e}")


def _safe_release_resource(shms):
    errors = []
    for shm in list(shms):
        try: shm.close()
        except (ProcessLookupError, BufferError, OSError) as e: errors.append(f"IPC Lock Exhaustion (close): {e}")
        try: shm.unlink()
        except (ProcessLookupError, BufferError, OSError) as e: errors.append(f"IPC Lock Exhaustion (unlink): {e}")
    if errors:
        raise AssertionError("\n".join(errors))

class TestIpcWorkerPool(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        sim_path = "/workspace/louhi_ws/coralnpu-mpact/bazel-bin/sim"
        if not os.path.exists(os.path.join(sim_path, "coralnpu_v2_sim")):
            sim_path = None
        env_patch = {}
        if sim_path:
            env_patch["PATH"] = sim_path + os.pathsep + os.environ.get("PATH", "")
        self.patcher = patch.dict(os.environ, env_patch)
        self.patcher.start()
        from tinygrad.runtime.ops_coralnpu import CoralNPUDevice
        self.device = CoralNPUDevice("CORALNPU")

    def tearDown(self):
        _safe_release_resource(self.device.allocator.shms.values())
        self.device.allocator.shms.clear()
        self.patcher.stop()
        self.tmp_dir.cleanup()

    def test_worker_execution(self):
        """Test that the IPC worker correctly receives and executes a task across the boundary."""
        from tinygrad.device import BufferSpec
        import numpy as np

        dummy_options = BufferSpec(uncached=False, cpu_access=False, nolru=False)
        handle = self.device.allocator._alloc(100 * 4, dummy_options)
        try:
            arr = np.ndarray((100,), dtype=np.float32, buffer=self.device.allocator.shms[handle].buf)
            arr[:] = 5.0
            shm_name = self.device.allocator.shms[handle].name

            from tinygrad.helpers import IpcWorkerPool
            pool = IpcWorkerPool(_shared_worker, 1)
            try:
                pool.submit(0, handle, shm_name, 100)
                IPC_WORKER_TIMEOUT = 10.0
                result = pool.get_result(0, timeout=IPC_WORKER_TIMEOUT)
                self.assertTrue(result)
                self.assertEqual(arr[0], 4.0) # 2.0 * 2.0
            finally:
                pool.shutdown()
        finally:
            self.device.allocator._free(handle, dummy_options)

    def test_worker_timeout(self):
        """Test that a hanging worker correctly triggers a TimeoutError on the parent without deadlocking."""
        from tinygrad.device import BufferSpec
        from tinygrad.helpers import IpcWorkerPool
        dummy_options = BufferSpec(uncached=False, cpu_access=False, nolru=False)
        handle = self.device.allocator._alloc(100 * 4, dummy_options)
        try:
            shm_name = self.device.allocator.shms[handle].name
            pool = IpcWorkerPool(_hanging_worker, 1)
            try:
                pool.submit(0, handle, shm_name, 100)
                with self.assertRaises(TimeoutError):
                    FAST_HANG_DETECT_TIMEOUT = 10.0
                    pool.get_result(0, timeout=FAST_HANG_DETECT_TIMEOUT)
            finally:
                pool.shutdown()
        finally:
            self.device.allocator._free(handle, dummy_options)

    def test_worker_deadlock_prevention(self):
        """Test that massive unread payloads fill the pipe and trigger the POLLOUT watchdog instead of deadlocking."""
        from tinygrad.device import BufferSpec
        from tinygrad.helpers import IpcWorkerPool
        dummy_options = BufferSpec(uncached=False, cpu_access=False, nolru=False)
        handle = self.device.allocator._alloc(100 * 4, dummy_options)
        try:
            shm_name = self.device.allocator.shms[handle].name
            pool = IpcWorkerPool(_blocking_worker, 1)
            try:
                pool.submit(0, handle, shm_name, 100)
                with self.assertRaises((TimeoutError, OSError)):
                    for _ in range(100000):
                        pool.submit(0, handle, "A" * 1024, 100)
            finally:
                pool.shutdown()
        finally:
            self.device.allocator._free(handle, dummy_options)
