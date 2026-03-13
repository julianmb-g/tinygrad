import unittest
import os
import struct
import tempfile
from unittest.mock import patch, MagicMock
from tinygrad.runtime.ops_coralnpu import CoralNPUAllocator, CoralNPUProgram, CoralNPUDevice
from tinygrad.device import BufferSpec

def create_dummy_elf(path, base_addr=0x80004000):
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
    elf += struct.pack("<IIIBBH", 17, base_addr, 0, 0, 0, 0)
    elf += b'\x00.symtab\x00.strtab\x00_end\x00'
    with open(path, "wb") as f: f.write(elf)

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
        import tempfile, os
        from unittest.mock import patch
        from tinygrad.runtime.ops_coralnpu import CoralNPUAllocator
        
        # Test multiple deterministic golden values
        golden_bases = [0x80010000, 0x80040000, 0x80080000]
        for base in golden_bases:
            fd, path = tempfile.mkstemp(suffix='.elf')
            try:
                create_dummy_elf(path, base)
                with patch.dict(os.environ, {"CORALNPU_ELF": path}):
                    alloc = CoralNPUAllocator(self.device)
                    self.assertEqual(alloc.vmm_base, base)
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

if __name__ == '__main__':
    unittest.main()
