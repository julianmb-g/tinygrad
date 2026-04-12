import unittest
import tempfile
import subprocess
import os
from tinygrad.tensor import Tensor
from tinygrad.renderer.coralnpu import CoralNPURenderer
from tinygrad.runtime.ops_coralnpu import CORALNPU_DTCM_LINKER_SCRIPT
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import dtypes
from tinygrad.codegen.opt.heuristic import OutOfMemoryError

class TestPingPongAddressOverlap(unittest.TestCase):
    def test_e2e_compiled_ping_pong_execution(self):
        sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../coralnpu-mpact/bazel-bin/sim/coralnpu_v2_sim"))
        if not os.path.exists(sim_path):
            self.fail(f"Hardware simulator missing: {sim_path}")

        # Exceeds bounds test using UOps (4000 floats = 16000 bytes)
        # Total = 32000 bytes > 28KB limit.
        renderer = CoralNPURenderer()
        buf_a = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("chunk_A", 4000))
        buf_b = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(), (), ("chunk_B", 4000))
        idx = UOp(Ops.CONST, dtypes.int, (), 0)
        idx_a = UOp(Ops.INDEX, dtypes.float.ptr(), (buf_a, idx))
        idx_b = UOp(Ops.INDEX, dtypes.float.ptr(), (buf_b, idx))
        val = UOp(Ops.CONST, dtypes.float, (), 1.0)
        st_a = UOp(Ops.STORE, dtypes.void, (idx_a, val))
        st_b = UOp(Ops.STORE, dtypes.void, (idx_b, val))
        sink = UOp(Ops.SINK, dtypes.void, (st_a, st_b))

        uops = [buf_a, buf_b, idx, idx_a, idx_b, val, st_a, st_b, sink]

        try:
            name, kernel, bufs = renderer._render(uops)
            src_exceed = renderer.render_kernel(name, kernel, bufs, uops)

            with tempfile.TemporaryDirectory() as temp_dir:
                src_file = os.path.join(temp_dir, "kernel_exceed.c")
                elf_file = os.path.join(temp_dir, "kernel_exceed.elf")
                ld_script = os.path.join(temp_dir, "linker.ld")

                with open(src_file, "w") as f:
                    f.write(src_exceed)
                with open(ld_script, "w") as f:
                    f.write(CORALNPU_DTCM_LINKER_SCRIPT)

                with self.assertRaises(subprocess.CalledProcessError):
                    subprocess.check_call([
                        "riscv64-unknown-elf-gcc", "-nostdlib", "-O2", "-march=rv32imv", "-mabi=ilp32",
                        "-T", ld_script, src_file, "-o", elf_file
                    ], stderr=subprocess.DEVNULL)
        except OutOfMemoryError:
            pass

        # Valid execution natively on simulator
        t1 = Tensor([1.0, 2.0, 3.0], device="CORALNPU")
        t2 = Tensor([4.0, 5.0, 6.0], device="CORALNPU")
        out = t1 + t2
        schedule = out.schedule()
        for si in schedule:
            if si.ast.op.name == "SINK":
                from tinygrad.engine.realize import get_runner
                runner = get_runner("CORALNPU", si.ast)
                src = runner.p.src

                with tempfile.TemporaryDirectory() as temp_dir:
                    src_file = os.path.join(temp_dir, "kernel.c")
                    elf_file = os.path.join(temp_dir, "kernel.elf")
                    ld_script = os.path.join(temp_dir, "linker.ld")

                    with open(src_file, "w") as f:
                        f.write(src)
                    with open(ld_script, "w") as f:
                        f.write(CORALNPU_DTCM_LINKER_SCRIPT)

                    subprocess.check_call([
                        "riscv64-unknown-elf-gcc", "-nostdlib", "-O2", "-march=rv32imv", "-mabi=ilp32",
                        "-T", ld_script, src_file, "-o", elf_file
                    ])

                    # Execute natively on ISS to ensure it runs without hardware exception
                    subprocess.check_call([sim_path, elf_file, "--max_cycles=1000000", "--allow_memory_region", "0x0:0x80000000:rwx"])

if __name__ == "__main__":
    unittest.main()
