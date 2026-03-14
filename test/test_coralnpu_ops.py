import unittest
import os
import struct
import tempfile
import subprocess
from unittest.mock import patch, MagicMock
import math
from tinygrad.runtime.ops_coralnpu import CoralNPUAllocator, CoralNPUProgram, CoralNPUDevice
from tinygrad.device import BufferSpec

def create_dummy_elf(path, padding=0x2000):
    import struct
    elf = bytearray(b'\x7fELF\x01\x01\x01\x00' + b'\x00'*8)
    elf += struct.pack("<2H I 3I I 6H",
        2, 0xf3, 1,
        0, 0, 52,
        0, 52, 0, 0, 40, 3, 2
    )
    elf += b'\x00' * 40
    elf += struct.pack("<10I", 1, 2, 0, 0, 172, 16, 2, 0, 4, 16)
    elf += struct.pack("<10I", 9, 3, 0, 0, 188, 22, 0, 0, 1, 0)
    
    dynamic_base_addr = 0x80000000 + len(elf) + padding
    
    elf += struct.pack("<IIIBBH", 17, dynamic_base_addr, 0, 0, 0, 0)
    elf += b'\x00.symtab\x00.strtab\x00_end\x00'
    with open(path, "wb") as f: f.write(elf)
    return dynamic_base_addr

class BaseCoralNPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.elf_fd, cls.elf_path = tempfile.mkstemp(suffix='.elf')
        create_dummy_elf(cls.elf_path)
        cls.env_patcher = patch.dict(os.environ, {"CORALNPU_ELF": cls.elf_path})
        cls.env_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.env_patcher.stop()
        os.close(cls.elf_fd)
        os.unlink(cls.elf_path)

class TestCoralNPUAllocator(BaseCoralNPUTest):
    def setUp(self):
        self.device = MagicMock()
        self.allocator = CoralNPUAllocator(self.device)

    def test_elf_vmm_parsing(self):
        from unittest.mock import patch
        from tinygrad.runtime.ops_coralnpu import CoralNPUAllocator
        
        # Test dynamically computed boundaries by varying padding offsets
        for padding in [0x1000, 0x2000, 0x3000]:
            fd, path = tempfile.mkstemp(suffix='.elf')
            try:
                expected_base = create_dummy_elf(path, padding=padding)
                with patch.dict(os.environ, {"CORALNPU_ELF": path}):
                    alloc = CoralNPUAllocator(self.device)
                    self.assertEqual(alloc.vmm_base, expected_base)
            finally:
                os.close(fd)
                os.unlink(path)

    def test_alloc_and_free(self):
        handle = self.allocator._alloc(100, BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False))
        self.assertIn(handle, self.allocator.mem)
        self.assertEqual(len(self.allocator.mem[handle]), 100)
        
        self.allocator._free(handle, BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False))
        self.assertNotIn(handle, self.allocator.mem)

    def test_copyin_copyout(self):
        handle = self.allocator._alloc(10, BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False))
        src_data = b"0123456789"
        
        self.allocator._copyin(handle, memoryview(src_data))
        
        dest_data = bytearray(10)
        self.allocator._copyout(memoryview(dest_data), handle)
        
        self.assertEqual(bytes(dest_data), src_data)

    def test_invalid_handles(self):
        with self.assertRaises(ValueError):
            self.allocator._copyin(999, memoryview(b"123"))
            
        with self.assertRaises(ValueError):
            dest = bytearray(3)
            self.allocator._copyout(memoryview(dest), 999)
            
    def test_invalid_tensor_extmem_boundary(self):
        # Assert that the allocator preserves EXTMEM before space and handles NaN during execution
        handle = self.allocator._alloc(4, BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False))
        
        # Pack a NaN float into 4 bytes
        nan_bytes = struct.pack('f', float('nan'))
        self.allocator._copyin(handle, memoryview(nan_bytes))
        
        # Allocate adjacent tensor for the execution to write to, ensuring it doesn't overflow to handle
        handle2 = self.allocator._alloc(4, BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False))
        
        # Execute a dummy program that does nothing (or writes to handle2) to see if simulator clobbers handle's memory space
        # We need a dummy simulator!
        with tempfile.TemporaryDirectory() as tmp_bin:
            gcc_path = os.path.join(tmp_bin, "riscv64-unknown-elf-gcc")
            sim_path = os.path.join(tmp_bin, "coralnpu_v2_sim")
            with open(gcc_path, 'w') as f:
                f.write("#!/usr/bin/env python3\nimport sys\nwith open(sys.argv[-1], 'w') as out: out.write('dummy elf')\n")
            with open(sim_path, 'w') as f:
                f.write("#!/usr/bin/env python3\nimport sys, multiprocessing.shared_memory\nshm_name = sys.argv[sys.argv.index('--shm')+1]\nshm = multiprocessing.shared_memory.SharedMemory(name=shm_name)\nshm.close()\n")
            os.chmod(gcc_path, 0o755)
            os.chmod(sim_path, 0o755)
            
            with patch.dict(os.environ, {"PATH": f"{tmp_bin}:{os.environ.get('PATH', '')}"}):
                self.allocator.device.allocator = self.allocator
                prog = CoralNPUProgram(self.allocator.device, "kernel", b"void kernel(float* a) { a[0] = 0.0f; }")
                prog(handle2, wait=False)

        dest = bytearray(4)
        self.allocator._copyout(memoryview(dest), handle)
        out_val = struct.unpack('f', dest)[0]
        self.assertTrue(math.isnan(out_val), "EXTMEM before space clobbered by simulator execution")
        self.allocator._free(handle, BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False))
        self.allocator._free(handle2, BufferSpec(image=None, uncached=False, cpu_access=False, nolru=False))

class TestCoralNPUProgram(BaseCoralNPUTest):
    @patch.dict(os.environ, {"BEAM": "1"})
    def test_beam_cost_parsing(self):
        # Validate realistic C++ byte structure
        lib = b'''
        #include <stdint.h>
        // BEAM_COST: 142.75
        void kernel() {
            int a = 0;
        }
        '''
        prog = CoralNPUProgram(None, "kernel", lib)
        self.assertTrue(prog.is_beam)
        self.assertEqual(prog.beam_cost, 142.75)
        
        cost = prog(wait=True)
        self.assertEqual(cost, 142.75)

    def test_compile_error(self):
        with tempfile.TemporaryDirectory() as tmp_bin:
            gcc_path = os.path.join(tmp_bin, "riscv64-unknown-elf-gcc")
            with open(gcc_path, 'w') as f:
                f.write("#!/usr/bin/env bash\necho 'syntax error' >&2\nexit 1\n")
            os.chmod(gcc_path, 0o755)
            with patch.dict(os.environ, {"PATH": f"{tmp_bin}:{os.environ.get('PATH', '')}"}):
                prog = CoralNPUProgram(None, "kernel", b"void kernel() {}")
                with self.assertRaises(RuntimeError) as ctx:
                    prog(wait=False)
                self.assertIn("Cross-compilation failed", str(ctx.exception))

    def test_missing_compiler(self):
        with tempfile.TemporaryDirectory() as tmp_bin:
            with patch.dict(os.environ, {"PATH": tmp_bin}):
                prog = CoralNPUProgram(None, "kernel", b"void kernel() {}")
                with self.assertRaises(FileNotFoundError) as ctx:
                    prog(wait=False)
                self.assertIn("Missing cross-compiler", str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
