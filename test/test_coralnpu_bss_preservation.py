import unittest
import os
import tempfile
import unittest.mock
from tinygrad.renderer.coralnpu import CoralNPURenderer, CoralNPUCompiler
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp

class TestCoralNPUBssPreservation(unittest.TestCase):
    def test_global_io_buffers_noinit_attribute(self):
        renderer = CoralNPURenderer()
        
        # 1. Create an organic UOp graph utilizing Ops.PARAM to represent global arrays
        # data0 (float32 pointer), data1 (half pointer), and an output int32 array.
        p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), 0)
        p1 = UOp(Ops.PARAM, dtypes.float16.ptr(), (), 1)
        p2 = UOp(Ops.PARAM, dtypes.int32.ptr(), (), 2)
        
        # We need a SINK or STORE to ensure they are parsed as writable if needed,
        # but just parsing them through `_render` organically creates `bufs`.
        idx = UOp(Ops.CONST, dtypes.int, (), 0)
        val = UOp(Ops.CONST, dtypes.float32, (), 42.0)
        
        bidx = UOp(Ops.INDEX, dtypes.float32.ptr(), (p0, idx))
        st = UOp(Ops.STORE, dtypes.void, (bidx, val))
        
        sink = UOp(Ops.SINK, dtypes.void, (p0, p1, p2, st))
        uops = [p0, p1, p2, idx, bidx, val, st, sink]
        
        # Organically extract the name, kernel text, and parsed buffers via the renderer
        name, kernel, bufs = renderer._render(uops)
        
        # Pass to the target specific compiler stage
        src = renderer.render_kernel(name, kernel, bufs, uops)
        
        # Assert that the .noinit section attribute was applied to each global buffer
        # In tinygrad renderer data0 is float32, data1 is half, data2 is int32_t.
        self.assertIn('__attribute__((section(".noinit"))) float data0[32768 / sizeof(float)];', src)
        self.assertIn('__attribute__((section(".noinit"))) half data1[32768 / sizeof(half)];', src)
        self.assertIn('__attribute__((section(".noinit"))) int32_t data2[32768 / sizeof(int32_t)];', src)
        
        # 2. Check linker script generation hermetically using mock.patch.dict
        compiler = CoralNPUCompiler()
        with tempfile.TemporaryDirectory() as temp_dir:
            with unittest.mock.patch.dict(os.environ, {"SAVE_BEAM_DIR": temp_dir}):
                # This should organically write the kernel.cc and kernel.ld files
                compiler.compile(src)
                
                # Check that .noinit (NOLOAD) is present in the linker script
                ld_path = os.path.join(temp_dir, f"kernel_{compiler.kernel_counter - 1}.ld")
                self.assertTrue(os.path.exists(ld_path))
                
                with open(ld_path, "r") as f:
                    ld_content = f.read()
                    
                self.assertIn(".noinit (NOLOAD) : { . = ALIGN(16); *(.noinit*) } > EXTMEM", ld_content)

if __name__ == '__main__':
    unittest.main()
