import unittest
import struct
import math
import tempfile
import subprocess
import ctypes
import os
import re
from tinygrad.renderer.coralnpu import CoralNPURenderer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import dtypes

class TestCoralNPUMemory(unittest.TestCase):
    def test_dtcm_28kb_hard_limit_tiling(self):
        renderer = CoralNPURenderer()
        
        # 1. Assert 4KB base address offset & exceed limit
        # We will create two local buffers.
        local1 = UOp(Ops.DEFINE_LOCAL, dtypes.float32.ptr(), (), ("buf1", 1024)) # 1024 floats = 4096 bytes
        local2 = UOp(Ops.DEFINE_LOCAL, dtypes.float32.ptr(), (), ("buf2", 1024)) # 4096 bytes
        
        idx = UOp(Ops.CONST, dtypes.int, (), 0)
        
        # In tinygrad, STORE writes to an INDEX pointer
        bidx1 = UOp(Ops.INDEX, dtypes.float32.ptr(), (local1, idx))
        bidx2 = UOp(Ops.INDEX, dtypes.float32.ptr(), (local2, idx))
        val = UOp(Ops.CONST, dtypes.float32, (), 42.0)
        
        st1 = UOp(Ops.STORE, dtypes.void, (bidx1, val))
        st2 = UOp(Ops.STORE, dtypes.void, (bidx2, val))
        sink = UOp(Ops.SINK, dtypes.void, (st1, st2))
        
        uops = [local1, local2, idx, bidx1, bidx2, val, st1, st2, sink]
        
        # To get local_offsets populated, call render_kernel with an empty body
        renderer.render_kernel("test_kernel", [], [], uops)
        
        # 1a. Check offsets
        self.assertEqual(renderer.local_offsets[local1], 4096)
        self.assertEqual(renderer.local_offsets[local2], 8192)
        
        # 1b. Test exceeding 28KB limit
        local_huge = UOp(Ops.DEFINE_LOCAL, dtypes.float32.ptr(), (), ("huge", 8000)) # 32000 bytes > 28672
        bidx_huge = UOp(Ops.INDEX, dtypes.float32.ptr(), (local_huge, idx))
        st_huge = UOp(Ops.STORE, dtypes.void, (bidx_huge, val))
        sink_huge = UOp(Ops.SINK, dtypes.void, (st_huge,))
        uops_huge = [local_huge, idx, bidx_huge, val, st_huge, sink_huge]
        
        with self.assertRaises(RuntimeError) as context:
            renderer.render_kernel("test_huge", [], [], uops_huge)
        self.assertTrue("DTCM Tiling exceeded 28KB limit" in str(context.exception))
        
        # 2. Organic Execution Testing of Memory Boundaries & NaN Preservation
        # We now generate the real C string using _render
        # renderer._render uses renderer.local_offsets and renderer.dtcm_bump which were populated
        name, kernel, bufs = renderer._render(uops)
        src = renderer.render_kernel(name, kernel, bufs, uops)
        sig_match = re.search(r'extern "C" void ([a-zA-Z0-9_]+)\(\)', src)
        func_name = sig_match.group(1) if sig_match else "compiled"
        
        src = src.replace(f'extern "C" void {func_name}() {{', f'extern "C" void {func_name}(void* base_ptr) {{')
        src = src.replace(' = (float*)(', ' = (float*)((char*)base_ptr + ')
        
        with tempfile.NamedTemporaryFile(suffix=".cc", delete=False) as f:
            dummy_includes = "#define CORAL_DMA_ASYNC(dest, src, size)\n#define CORAL_DMA_WAIT()\n#include <string.h>\n"
            f.write((dummy_includes + src).encode())
            f.flush()
            so_path = f.name + ".so"
            subprocess.check_call(["g++", "-shared", "-fPIC", "-O2", f.name, "-o", so_path])
            
        try:
            lib = ctypes.CDLL(so_path)
            
            # 3. Explicitly encode and preserve NaN values using struct.pack('f', float('nan'))
            nan_bytes = struct.pack('f', float('nan'))
            nan_val = struct.unpack('f', nan_bytes)[0]
            self.assertTrue(math.isnan(nan_val))
            
            # Create a 20000 byte array initialized entirely to NaN
            mem = bytearray(nan_bytes * 5000) # 5000 * 4 = 20000 bytes
            c_mem = (ctypes.c_char * len(mem)).from_buffer(mem)
            
            # Execute the C++ kernel organically
            func = getattr(lib, func_name)
            func(ctypes.byref(c_mem))
            
            # Check boundaries around buf1 (starts at 4096)
            before_bytes = mem[4096-4:4096]
            before_val = struct.unpack('f', before_bytes)[0]
            self.assertTrue(math.isnan(before_val), "EXTMEM boundary before allocation was clobbered")
            
            # The kernel should have written 42.0 at 4096
            written_bytes = mem[4096:4096+4]
            written_val = struct.unpack('f', written_bytes)[0]
            self.assertEqual(written_val, 42.0, "Kernel did not write the expected value")
            
            # Check boundary after written value
            after_bytes = mem[4096+4:4096+8]
            after_val = struct.unpack('f', after_bytes)[0]
            self.assertTrue(math.isnan(after_val), "EXTMEM boundary after allocation was clobbered")
            
        finally:
            os.unlink(f.name)
            os.unlink(so_path)

if __name__ == '__main__':
    unittest.main()
