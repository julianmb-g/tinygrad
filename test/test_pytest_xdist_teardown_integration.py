import unittest
import subprocess
import os
import tempfile
import sys

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

        # Create a dummy test file that executes quickly
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_test_file = os.path.join(tmpdir, "test_dummy.py")
            with open(dummy_test_file, "w") as f:
                f.write("""
import pytest

def test_dummy_1():
    for _ in range(1000000): pass
    assert True

def test_dummy_2():
    for _ in range(1000000): pass
    assert True
""")

            # Run pytest with xdist
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-n", "2", dummy_test_file],
                    capture_output=True,
                    text=True
                )
            except Exception as e:
                self.fail(f"pytest-xdist execution crashed natively: {e}")

            # Combine stdout and stderr
            output = result.stdout + result.stderr

            # If xdist is actually failing or the dummy test fails, it's a structural problem
            self.assertEqual(result.returncode, 0, f"Pytest failed natively:\\n{output}")

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
