import os
import atexit
import signal

def pytest_configure(config):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None:
        os.setpgrp()
        atexit.register(lambda: os.killpg(0, signal.SIGKILL))
