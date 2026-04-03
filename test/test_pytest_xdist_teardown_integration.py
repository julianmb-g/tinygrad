import unittest
import subprocess
import os
import tempfile
import sys
import multiprocessing

try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

class TestPytestXdistTeardown(unittest.TestCase):
    def test_xdist_no_oserror_on_teardown(self):
        """
        Executes a dummy pytest suite using xdist (-n 2) to ensure that the
        multiprocessing IPC thread synchronization works properly and does NOT
        leak 'OSError: cannot send (already closed?)' during the hookwrapper teardown.
        """
        # We need pytest-xdist installed. If not, skip organically.
        try:
            import xdist
        except ImportError:
            self.skipTest("pytest-xdist not installed, skipping IPC teardown test.")

        # Create an authentic test file that executes NPU payload
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_test_file = os.path.join(tmpdir, "test_dummy.py")
            with open(dummy_test_file, "w") as f:
                f.write("""
import pytest
import multiprocessing
from tinygrad.tensor import Tensor

try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

def test_dummy_1():
    t = Tensor([1.5, 4.2], device="CORALNPU").sqrt()
    assert t.numpy().tolist() != []

def test_dummy_2():
    t = Tensor([3.1, 2.8], device="CORALNPU").max()
    assert t.numpy().tolist() != []
""")

            # Run pytest with xdist
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-n", "2", dummy_test_file],
                    capture_output=True,
                    text=True
                )
            except (FileNotFoundError, ProcessLookupError):
                pass

            # Combine stdout and stderr
            output = result.stdout + result.stderr

            # The critical Tier 1 Pipeline Crash: Ensure OSError is NOT raised in teardown.
            self.assertNotIn(
                "OSError: cannot send (already closed?)",
                output,
                "Teardown Deadlock Detected: pytest-xdist workers threw an OSError on IPC channel closure."
            )

            # Ensure we didn't have broad exception swallowing hiding other plugin teardown errors
            self.assertNotIn(
                "PluggyTeardownRaisedWarning",
                output,
                "Teardown Warning Detected: A pytest plugin raised an exception during teardown."
            )

if __name__ == '__main__':
    unittest.main()
