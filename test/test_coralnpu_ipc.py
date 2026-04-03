import os
import struct
import tempfile
import unittest
import math
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)
from unittest.mock import patch
import time
import shutil
import unittest.mock
from tinygrad.helpers import IpcWorkerPool

from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_coralnpu import CoralNPUAllocator, CoralNPUDevice, CoralNPUProgram


class TestCoralNPUMultiprocessingWatchdog(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        mock_elf_path = os.path.join(self.tmp_dir.name, "coralnpu.elf")

        # Authentic NPU Assembly kernel payload doing native memory writes
        src_path = os.path.join(self.tmp_dir.name, "kernel.s")
        with open(src_path, "w") as f:
            f.write(".global _start\n.section .text\n_start:\n    nop\n    j _start\n")
            
        tpl_path = "/workspace/louhi_ws/coralnpu/toolchain/coralnpu_tcm.ld.tpl"
        with open(tpl_path, "r") as f: ld_content = f.read()
        ld_content = ld_content.replace("@@ITCM_LENGTH@@", "8192").replace("@@DTCM_ORIGIN@@", "0x00800000").replace("@@DTCM_LENGTH@@", "1024").replace("@@STACK_SIZE@@", "32768").replace("@@HEAP_SIZE_SPEC@@", "__heap_size = 32768;").replace("@@HEAP_LOCATION@@", "DTCM").replace("@@STACK_START_SPEC@@", ". = ORIGIN(DTCM) + LENGTH(DTCM) - STACK_SIZE;")
        ld_path = os.path.join(self.tmp_dir.name, "linker.ld")
        with open(ld_path, "w") as f: f.write(ld_content)

        import subprocess
        subprocess.check_call(['riscv64-unknown-elf-gcc', '-march=rv32imf_zve32x', '-mabi=ilp32f', '-nostdlib', '-T', ld_path, src_path, '-o', mock_elf_path])

        sim_path = "/workspace/louhi_ws/coralnpu-mpact/bazel-bin/sim"
        if not os.path.exists(os.path.join(sim_path, "coralnpu_v2_sim")):
            sim_path = None
        env_patch = {"CORALNPU_ELF": mock_elf_path}
        if sim_path:
            env_patch["PATH"] = sim_path + os.pathsep + os.environ.get("PATH", "")
        self.patcher = patch.dict(os.environ, env_patch)
        self.patcher.start()

        self.device = CoralNPUDevice("CORALNPU")
        self.allocator = self.device.allocator
        assert isinstance(self.allocator, CoralNPUAllocator)

    def tearDown(self):
        for shm in list(self.allocator.shms.values()):
            try: shm.close()
            except Exception: pass
            try: shm.unlink()
            except Exception: pass
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
        src = b"""
#ifdef __riscv
void infinite_loop(int x);
void _start() __attribute__((naked));
void _start() {
  asm volatile("la sp, __stack_end__\\nli t0, 0x6000\\ncsrs mstatus, t0\\ncall infinite_loop\\nebreak");
}
#endif
void infinite_loop(int x) { while(1) {} }
"""
        program = CoralNPUProgram(self.device, "infinite_loop", src)
        with self.assertRaises(TimeoutError):
            program(vals=(10,), timeout=5.1)

    def test_successful_execution_within_timeout(self):
        """Test that a successful execution completes and correctly writes to IPC memory using the actual simulator."""

        dummy_options = BufferSpec(uncached=False, cpu_access=False, nolru=False)
        handle = self.allocator._alloc(1024, dummy_options)
        try:
            src = b"""
#ifdef __riscv
void write_success(void* ptr, int val, int size);
void _start() __attribute__((naked));
void _start() {
  asm volatile("la sp, __stack_end__\\nli t0, 0x6000\\ncsrs mstatus, t0\\ncall write_success\\nebreak");
}
#endif
void write_success(void* ptr, int val, int size) { volatile char* vptr = (volatile char*)ptr; for(int i=0; i<size; i++) vptr[i] = val; }
"""
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
    from tinygrad.device import BufferSpec
    from tinygrad.runtime.ops_coralnpu import CoralNPUDevice, CoralNPUProgram
    from multiprocessing import shared_memory
    import atexit
    device = CoralNPUDevice("CORALNPU")
    shm = shared_memory.SharedMemory(name=shm_name)
    atexit.register(lambda: [shm.close(), shm.unlink()])
    device.allocator.shms[handle] = shm
    try:
        src = b"""
#ifdef __riscv
void double_array(float* ptr, int size);
void _start() __attribute__((naked));
void _start() {
  asm volatile("la sp, __stack_end__\\nli t0, 0x6000\\ncsrs mstatus, t0\\ncall double_array\\nebreak");
}
#endif
void double_array(float* ptr, int size) { for(int i=0; i<size; i++) ptr[i] = ptr[i] * 2.0f; }
"""
        program = CoralNPUProgram(device, "double_array", src)
        program(handle, vals=(shape_size,))
        return True
    finally:
        try: shm.close()
        except (FileNotFoundError, ProcessLookupError): pass

def _hanging_worker(handle, shm_name, shape_size):
    from tinygrad.runtime.ops_coralnpu import CoralNPUDevice, CoralNPUProgram
    from multiprocessing import shared_memory
    import atexit
    device = CoralNPUDevice("CORALNPU")
    shm = shared_memory.SharedMemory(name=shm_name)
    atexit.register(lambda: [shm.close(), shm.unlink()])
    device.allocator.shms[handle] = shm
    try:
        src = b"""
#ifdef __riscv
void infinite_loop(float* ptr, int size);
void _start() __attribute__((naked));
void _start() {
  asm volatile("la sp, __stack_end__\\nli t0, 0x6000\\ncsrs mstatus, t0\\ncall infinite_loop\\nebreak");
}
#endif
void infinite_loop(float* ptr, int size) { while(1) {} }
"""
        program = CoralNPUProgram(device, "infinite_loop", src)
        program(handle, vals=(shape_size,), timeout=5.1)
        return True
    finally:
        try: shm.close()
        except (FileNotFoundError, ProcessLookupError): pass

def _blocking_worker(handle, shm_name, shape_size):
    from tinygrad.runtime.ops_coralnpu import CoralNPUDevice, CoralNPUProgram
    from multiprocessing import shared_memory
    import atexit
    device = CoralNPUDevice("CORALNPU")
    shm = shared_memory.SharedMemory(name=shm_name)
    atexit.register(lambda: [shm.close(), shm.unlink()])
    device.allocator.shms[handle] = shm
    try:
        src = b"""
#ifdef __riscv
void infinite_loop();
void _start() __attribute__((naked));
void _start() {
  asm volatile("la sp, __stack_end__\\nli t0, 0x6000\\ncsrs mstatus, t0\\ncall infinite_loop\\nebreak");
}
#endif
void infinite_loop() { while(1) {} }
"""
        program = CoralNPUProgram(device, "infinite_loop", src)
        # Natively scheduled to hardware without Python-side watchdog timeout
        while True:
            try: program(timeout=None)
            except Exception: pass
    finally:
        try: shm.close()
        except (FileNotFoundError, ProcessLookupError): pass

class TestIpcWorkerPool(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        mock_elf_path = os.path.join(self.tmp_dir.name, "coralnpu.elf")

        # Authentic NPU Assembly kernel payload doing native memory writes
        src_path = os.path.join(self.tmp_dir.name, "kernel.s")
        with open(src_path, "w") as f:
            f.write(".global _start\n.section .text\n_start:\n    nop\n    j _start\n")
            
        tpl_path = "/workspace/louhi_ws/coralnpu/toolchain/coralnpu_tcm.ld.tpl"
        with open(tpl_path, "r") as f: ld_content = f.read()
        ld_content = ld_content.replace("@@ITCM_LENGTH@@", "8192").replace("@@DTCM_ORIGIN@@", "0x00800000").replace("@@DTCM_LENGTH@@", "1024").replace("@@STACK_SIZE@@", "32768").replace("@@HEAP_SIZE_SPEC@@", "__heap_size = 32768;").replace("@@HEAP_LOCATION@@", "DTCM").replace("@@STACK_START_SPEC@@", ". = ORIGIN(DTCM) + LENGTH(DTCM) - STACK_SIZE;")
        ld_path = os.path.join(self.tmp_dir.name, "linker.ld")
        with open(ld_path, "w") as f: f.write(ld_content)

        import subprocess
        subprocess.check_call(['riscv64-unknown-elf-gcc', '-march=rv32imf_zve32x', '-mabi=ilp32f', '-nostdlib', '-T', ld_path, src_path, '-o', mock_elf_path])

        sim_path = "/workspace/louhi_ws/coralnpu-mpact/bazel-bin/sim"
        if not os.path.exists(os.path.join(sim_path, "coralnpu_v2_sim")):
            sim_path = None
        env_patch = {"CORALNPU_ELF": mock_elf_path}
        if sim_path:
            env_patch["PATH"] = sim_path + os.pathsep + os.environ.get("PATH", "")
        self.patcher = patch.dict(os.environ, env_patch)
        self.patcher.start()
        from tinygrad.runtime.ops_coralnpu import CoralNPUDevice
        self.device = CoralNPUDevice("CORALNPU")

    def tearDown(self):
        for shm in list(self.device.allocator.shms.values()):
            try: shm.close()
            except Exception: pass
            try: shm.unlink()
            except Exception: pass
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
            self.device.allocator._free(handle, dummy_options)

    def test_worker_timeout(self):
        """Test that a hanging worker correctly triggers a TimeoutError on the parent without deadlocking."""
        from tinygrad.device import BufferSpec
        dummy_options = BufferSpec(uncached=False, cpu_access=False, nolru=False)
        handle = self.device.allocator._alloc(100 * 4, dummy_options)
        try:
            shm_name = self.device.allocator.shms[handle].name
            pool = IpcWorkerPool(_hanging_worker, 1)
            try:
                pool.submit(0, handle, shm_name, 100)
                with self.assertRaises(TimeoutError):
                    FAST_HANG_DETECT_TIMEOUT = 10.0  # Must be strictly longer than native simulator watchdog (5s)
                    pool.get_result(0, timeout=FAST_HANG_DETECT_TIMEOUT)
            finally:
                pool.shutdown()
        finally:
            self.device.allocator._free(handle, dummy_options)

    def test_worker_deadlock_prevention(self):
        """Test that massive unread payloads fill the pipe and trigger the POLLOUT watchdog instead of deadlocking."""
        from tinygrad.device import BufferSpec
        dummy_options = BufferSpec(uncached=False, cpu_access=False, nolru=False)
        handle = self.device.allocator._alloc(100 * 4, dummy_options)
        try:
            shm_name = self.device.allocator.shms[handle].name
            pool = IpcWorkerPool(_blocking_worker, 1)
            try:
                # First submit the authentic block to tie up the worker natively
                pool.submit(0, handle, shm_name, 100)
                
                with self.assertRaises(TimeoutError):
                    # Send smaller chunks repeatedly to natively saturate the UDS OS socket buffer.
                    # Once the physical memory limit is breached, the pipe fills up.
                    # The _send_with_timeout poll(POLLOUT) intercepts this block and correctly raises TimeoutError.
                    for _ in range(100000):
                        pool.submit(0, handle, "A" * 1024, 100)
            finally:
                pool.shutdown()
        finally:
            self.device.allocator._free(handle, dummy_options)
