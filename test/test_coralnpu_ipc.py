import unittest
import time
import multiprocessing
import multiprocessing.shared_memory
import ctypes
import os
import numpy as np

def simulator_worker(shm_name, hang=False, use_heartbeat=False):
    """Simulates an out-of-band execution (like the CoralNPU simulator)."""
    try:
        shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return

    try:
        if use_heartbeat:
            # First byte is heartbeat counter
            heartbeat_array = np.ndarray((1,), dtype=np.uint8, buffer=shm.buf)
            for i in range(5):
                if hang and i == 2:
                    time.sleep(10.0) # Simulate a hang mid-execution
                heartbeat_array[0] = i + 1
                time.sleep(0.1)
        else:
            if hang:
                time.sleep(10.0)
            else:
                time.sleep(0.1)
                
        # Write success flag to second byte
        shm.buf[1] = 1
    finally:
        shm.close()

class TestCoralNPUIPC(unittest.TestCase):
    def test_zero_copy_shared_memory(self):
        """Test that we can establish a zero-copy shared memory IPC channel."""
        shm = multiprocessing.shared_memory.SharedMemory(create=True, size=1024)
        try:
            # Write some data
            test_data = b"HELLO_NPU"
            shm.buf[:len(test_data)] = test_data
            
            # Verify data
            self.assertEqual(bytes(shm.buf[:len(test_data)]), test_data)
        finally:
            shm.close()
            shm.unlink()

    def test_watchdog_timeout_on_hang(self):
        """Test that a strict timeout watchdog correctly catches and kills a hanging simulator."""
        shm = multiprocessing.shared_memory.SharedMemory(create=True, size=1024)
        try:
            p = multiprocessing.Process(target=simulator_worker, args=(shm.name, True, False))
            p.start()
            
            # Watchdog timeout
            p.join(timeout=0.5)
            
            if p.is_alive():
                p.terminate()
                p.join()
                timeout_triggered = True
            else:
                timeout_triggered = False
                
            self.assertTrue(timeout_triggered, "Watchdog failed to trigger on hanging process.")
            self.assertEqual(shm.buf[1], 0, "Process completed despite hang.")
        finally:
            shm.close()
            shm.unlink()

    def test_successful_execution_within_timeout(self):
        """Test that a successful execution completes within the watchdog timeout."""
        shm = multiprocessing.shared_memory.SharedMemory(create=True, size=1024)
        try:
            p = multiprocessing.Process(target=simulator_worker, args=(shm.name, False, False))
            p.start()
            
            # Watchdog timeout
            p.join(timeout=1.0)
            
            if p.is_alive():
                p.terminate()
                p.join()
                timeout_triggered = True
            else:
                timeout_triggered = False
                
            self.assertFalse(timeout_triggered, "Watchdog falsely triggered on successful process.")
            self.assertEqual(shm.buf[1], 1, "Process failed to write success flag.")
        finally:
            shm.close()
            shm.unlink()

    def test_heartbeat_mechanism(self):
        """Test a shared memory heartbeat mechanism to detect deadlocks earlier than the global timeout."""
        import numpy as np
        shm = multiprocessing.shared_memory.SharedMemory(create=True, size=1024)
        try:
            shm.buf[0] = 0 # Initialize heartbeat
            
            p = multiprocessing.Process(target=simulator_worker, args=(shm.name, True, True))
            p.start()
            
            last_heartbeat = 0
            heartbeat_stalls = 0
            deadlock_detected = False
            
            # Poll heartbeat
            for _ in range(20):
                time.sleep(0.1)
                current_heartbeat = shm.buf[0]
                if current_heartbeat == last_heartbeat:
                    heartbeat_stalls += 1
                    if heartbeat_stalls >= 3: # 300ms without heartbeat = deadlock
                        deadlock_detected = True
                        break
                else:
                    heartbeat_stalls = 0
                    last_heartbeat = current_heartbeat
                    
            if p.is_alive():
                p.terminate()
                p.join()
                
            self.assertTrue(deadlock_detected, "Heartbeat mechanism failed to detect deadlock.")
        finally:
            shm.close()
            shm.unlink()

if __name__ == '__main__':
    unittest.main()
