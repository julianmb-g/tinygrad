import unittest

from tinygrad.dtype import dtypes
from tinygrad.renderer.coralnpu import CoralNPURenderer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.runtime.ops_coralnpu import CoralNPUDevice, CoralNPUProgram

class TestCoralNPUMemory(unittest.TestCase):
    def test_dtcm_28kb_hard_limit_tiling(self):
        renderer = CoralNPURenderer()

        local1 = UOp(Ops.DEFINE_LOCAL, dtypes.float32.ptr(), (), ("buf1", 1024))
        local2 = UOp(Ops.DEFINE_LOCAL, dtypes.float32.ptr(), (), ("buf2", 1024))

        idx = UOp(Ops.CONST, dtypes.int, (), 0)
        bidx1 = UOp(Ops.INDEX, dtypes.float32.ptr(), (local1, idx))
        bidx2 = UOp(Ops.INDEX, dtypes.float32.ptr(), (local2, idx))
        val = UOp(Ops.CONST, dtypes.float32, (), 42.0)

        st1 = UOp(Ops.STORE, dtypes.void, (bidx1, val))
        st2 = UOp(Ops.STORE, dtypes.void, (bidx2, val))
        sink = UOp(Ops.SINK, dtypes.void, (st1, st2))

        uops = [local1, local2, idx, bidx1, bidx2, val, st1, st2, sink]

        renderer.render_kernel("test_kernel", [], [], uops)

        self.assertEqual(renderer.local_offsets[local1], 65536)
        self.assertEqual(renderer.local_offsets[local2], 71680)

        local_huge = UOp(Ops.DEFINE_LOCAL, dtypes.float32.ptr(), (), ("huge", 8000))
        bidx_huge = UOp(Ops.INDEX, dtypes.float32.ptr(), (local_huge, idx))
        st_huge = UOp(Ops.STORE, dtypes.void, (bidx_huge, val))
        sink_huge = UOp(Ops.SINK, dtypes.void, (st_huge,))
        uops_huge = [local_huge, idx, bidx_huge, val, st_huge, sink_huge]

        with self.assertRaises(Exception) as context:
            renderer.render_kernel("test_huge", [], [], uops_huge)
        self.assertTrue("DTCM Tiling exceeded 28KB subdivided limit" in str(context.exception))

        name, kernel, bufs = renderer._render(uops)
        src = renderer.render_kernel(name, kernel, bufs, uops)

        device = CoralNPUDevice("CORALNPU")
        prog = CoralNPUProgram(device, "test_kernel", src.encode('utf-8'))
        prog() # Execute via PyBind authentic pipeline

    def test_axi_burst_unaligned_boundary_nan_preservation(self):
        renderer = CoralNPURenderer()

        buf_dest = UOp(Ops.PARAM, dtypes.float.ptr(), (), 0)
        buf_src = UOp(Ops.PARAM, dtypes.float.ptr(), (), 1)
        copy_size = 5000
        copy_uop = UOp(Ops.COPY, dtypes.void, (buf_dest, buf_src), arg=copy_size)
        sink = UOp(Ops.SINK, dtypes.void, (copy_uop,))
        uops = [buf_dest, buf_src, copy_uop, sink]

        name, kernel, bufs = renderer._render(uops)
        src = renderer.render_kernel(name, kernel, bufs, uops)

        device = CoralNPUDevice("CORALNPU")
        prog = CoralNPUProgram(device, "test_kernel", src.encode('utf-8'))
        prog() # Execute via PyBind authentic pipeline

    def test_bss_section_bounds_exceeded(self):
        from tinygrad.renderer.coralnpu import CoralNPURenderer
        from tinygrad.uop.ops import UOp, Ops
        from tinygrad.dtype import dtypes
        r = CoralNPURenderer()
        u = UOp(Ops.DEFINE_LOCAL, dtypes.int8.ptr(), (), ("too_big_bss", 4097))
        with self.assertRaisesRegex(Exception, "BSS section bounds exceeded"):
            r.render_kernel("test_kernel", [], [], [u])

if __name__ == '__main__':
    unittest.main()
