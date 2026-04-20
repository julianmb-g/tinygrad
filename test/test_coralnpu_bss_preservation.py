import os
import subprocess
import unittest

from tinygrad.dtype import dtypes
from tinygrad.renderer.coralnpu import CoralNPURenderer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.runtime.ops_coralnpu import CoralNPUDevice, CoralNPUProgram

class TestCoralNPUBssPreservation(unittest.TestCase):
    def test_global_io_buffers_noinit_attribute(self):
        device = CoralNPUDevice("CORALNPU")
        renderer = CoralNPURenderer()

        p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), 0)
        p1 = UOp(Ops.PARAM, dtypes.float16.ptr(), (), 1)
        p2 = UOp(Ops.PARAM, dtypes.int32.ptr(), (), 2)

        idx = UOp(Ops.CONST, dtypes.int, (), 0)
        val = UOp(Ops.CONST, dtypes.float32, (), 42.0)

        bidx = UOp(Ops.INDEX, dtypes.float32.ptr(), (p0, idx))
        st = UOp(Ops.STORE, dtypes.void, (bidx, val))

        sink = UOp(Ops.SINK, dtypes.void, (p0, p1, p2, st))
        uops = [p0, p1, p2, idx, bidx, val, st, sink]

        name, kernel, bufs = renderer._render(uops)
        src = renderer.render_kernel(name, kernel, bufs, uops)

        prog = CoralNPUProgram(device, name, src.encode('utf-8'))
        
        elf_path = None
        try:
            elf_path = prog._compile_on_host(src)
            self.assertTrue(os.path.exists(elf_path))

            # Check readelf output
            output = subprocess.check_output(["riscv64-unknown-elf-readelf", "-S", elf_path]).decode("utf-8")
            
            found_noinit = False
            for line in output.split('\n'):
                if ".noinit" in line:
                    found_noinit = True
                    parts = line.split()
                    # find the index of .noinit
                    addr_idx = parts.index(".noinit") + 2
                    addr = parts[addr_idx]
                    
                    # EXTMEM is mapped to 0x20000000
                    addr_int = int(addr, 16)
                    self.assertTrue(addr_int >= 0x20000000, f"Expected .noinit to be in EXTMEM (>= 0x20000000), but found it at {hex(addr_int)}")
                    break
                    
            self.assertTrue(found_noinit, "Could not find .noinit section in compiled ELF")
            
        except FileNotFoundError as e:
            self.fail(f"Cross-compiler riscv64-unknown-elf-gcc is missing from the environment. Authentic testing requires the compiler: {e}")
        except subprocess.CalledProcessError as e:
            self.fail(f"readelf execution failed: {e}")
        finally:
            if elf_path:
                if os.path.exists(elf_path):
                    os.unlink(elf_path)
                src_path = elf_path[:-4]
                if os.path.exists(src_path):
                    os.unlink(src_path)

if __name__ == '__main__':
    unittest.main()
