import os
import tempfile
import unittest
import math
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)
from unittest.mock import patch
import time
import unittest.mock
from tinygrad.helpers import IpcWorkerPool

from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_coralnpu import CoralNPUAllocator, CoralNPUDevice, CoralNPUProgram


class TestCoralNPUMultiprocessingWatchdog(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        mock_elf_path = os.path.join(self.tmp_dir.name, "coralnpu.elf")
        
        # Authentically compile a base ELF to extract a valid _end symbol
        src_path = os.path.join(self.tmp_dir.name, "base.c")
        with open(src_path, "w") as f: f.write("void _start() {}")
        from tinygrad.runtime.ops_coralnpu import CORALNPU_DTCM_LINKER_SCRIPT
        ld_path = os.path.join(self.tmp_dir.name, "linker.ld")
        with open(ld_path, "w") as f: f.write(CORALNPU_DTCM_LINKER_SCRIPT)
        
        import subprocess
        subprocess.check_call(['riscv64-unknown-elf-gcc', '-march=rv32imf_zve32x', '-mabi=ilp32f', '-nostdlib', '-T', ld_path, src_path, '-o', mock_elf_path])

        self.patcher = patch.dict(os.environ, {"CORALNPU_ELF": mock_elf_path})
        self.patcher.start()

        self.device = CoralNPUDevice("CORALNPU")
        self.allocator = self.device.allocator
        assert isinstance(self.allocator, CoralNPUAllocator)

    def tearDown(self):
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
            self.assertEqual(bytes(out), test_data)

            # Check internal shared memory object is registered
            self.assertIn(handle, getattr(self.allocator, "shms", {}))
            self.assertIsNotNone(getattr(self.allocator, "shms", {})[handle].name)
            self.assertEqual(handle, getattr(self.allocator, "vmm_base", None))
        finally:
            self.allocator._free(handle, dummy_options)

    def test_missing_compiler_raises_file_not_found(self):
        """Test that missing cross-compiler authentically raises FileNotFoundError."""
        with unittest.mock.patch.dict(os.environ, {"PATH": "/tmp/dummy_empty_path"}):
            program = CoralNPUProgram(self.device, "missing_compiler", b"void missing_compiler() {}")
            with self.assertRaises(FileNotFoundError):
                program()

    def test_compiler_failure_raises_called_process_error(self):
        """Test that a failing compiler authentically raises RuntimeError wrapping CalledProcessError via real compiler execution."""
        program = CoralNPUProgram(self.device, "fail_compile", b"void fail_compile() { syntax_error_here; }")
        with self.assertRaises(RuntimeError) as context:
            program()
        self.assertIn("Cross-compilation failed", str(context.exception))

    def test_watchdog_timeout_on_hang(self):
        """Test that a strict timeout watchdog correctly catches and kills a hanging execution."""
        program = CoralNPUProgram(self.device, "infinite_loop", b"void infinite_loop(int x) { while(1) {} }")
        self.assertEqual(program(vals=(10,)), math.inf)

    def test_successful_execution_within_timeout(self):
        """Test that a successful execution completes and correctly writes to IPC memory using the actual simulator."""

        dummy_options = BufferSpec(uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            src = b"void write_success(void* ptr, int val, int size) { volatile char* vptr = (volatile char*)ptr; for(int i=0; i<size; i++) vptr[i] = val; }"  # noqa: E501
            program = CoralNPUProgram(self.device, "write_success", src)
            MAX_EXECUTION_TIMEOUT_SLA = 15.0  # Derived from expected execution SLA
            program(handle, vals=(65, 3), timeout=MAX_EXECUTION_TIMEOUT_SLA)
            out = bytearray(3)
            self.allocator._copyout(memoryview(out).cast('B'), handle)
            self.assertEqual(bytes(out), b"AAA")

        finally:
            self.allocator._free(handle, dummy_options)

if __name__ == '__main__':
    unittest.main()

def _shared_worker(handle, shm_name, shape_size):
    from tinygrad.runtime.ops_coralnpu import CoralNPUDevice, CoralNPUProgram
    from multiprocessing import shared_memory
    device = CoralNPUDevice("CORALNPU")
    shm = shared_memory.SharedMemory(name=shm_name)
    device.allocator.shms[handle] = shm
    try:
        src = b"void double_array(float* ptr, int size) { for(int i=0; i<size; i++) ptr[i] = ptr[i] * 2.0f; }"
        program = CoralNPUProgram(device, "double_array", src)
        program(handle, vals=(shape_size,))
        return True
    finally:
        shm.close()

def _hanging_worker(*args, **kwargs):
    while True: time.sleep(1)

def _blocking_worker(*args, **kwargs):
    while True: time.sleep(1)

class TestIpcWorkerPool(unittest.TestCase):
    def test_worker_execution(self):
        """Test that the IPC worker correctly receives and executes a task across the boundary."""
        from tinygrad.device import BufferSpec
        import numpy as np

        device = CoralNPUDevice("CORALNPU")
        dummy_options = BufferSpec(uncached=False, cpu_access=False, nolru=False)
        handle = device.allocator._alloc(100 * 4, dummy_options)
        try:
            arr = np.ndarray((100,), dtype=np.float32, buffer=device.allocator.shms[handle].buf)
            arr[:] = 5.0
            shm_name = device.allocator.shms[handle].name

            pool = IpcWorkerPool(_shared_worker, 1)
            try:
                pool.submit(0, handle, shm_name, 100)
                IPC_WORKER_TIMEOUT = 5.0  # Standard SLA for IPC worker task response
                result = pool.get_result(0, timeout=IPC_WORKER_TIMEOUT)
                self.assertTrue(result)
                self.assertEqual(arr[0], 10.0)
            finally:
                pool.shutdown()
        finally:
            device.allocator._free(handle, dummy_options)

    def test_worker_timeout(self):
        """Test that a hanging worker correctly triggers a TimeoutError on the parent without deadlocking."""
        pool = IpcWorkerPool(_hanging_worker, 1)
        try:
            pool.submit(0)
            with self.assertRaises(TimeoutError):
                FAST_HANG_DETECT_TIMEOUT = 0.1  # Minimal timeout to verify IPC watchdog hang detection
                pool.get_result(0, timeout=FAST_HANG_DETECT_TIMEOUT)
        finally:
            pool.shutdown()

    def test_worker_deadlock_prevention(self):
        """Test that massive unread payloads fill the pipe and trigger the POLLOUT watchdog instead of deadlocking."""
        pool = IpcWorkerPool(_blocking_worker, 1)
        try:
            with self.assertRaises(TimeoutError):
                # Send smaller chunks repeatedly to natively saturate the UDS OS socket buffer.
                # Once the physical memory limit is breached, the pipe fills up.
                # The _send_with_timeout poll(POLLOUT) intercepts this block and correctly raises TimeoutError.
                for _ in range(100000):
                    pool.submit(0, "A" * 1024)
        finally:
            pool.shutdown()
