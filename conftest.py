import os
import glob
import shutil

def pytest_sessionfinish(session, exitstatus):
    """
    Aggressively garbage collect /dev/shm blocks upon suite termination
    to prevent SharedMemory exhaustion and ProcessLookupError IPC deadlocks.
    """
    for f in glob.glob("/dev/shm/*"):
        try:
            if os.path.isfile(f):
                os.unlink(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
        except Exception:
            pass