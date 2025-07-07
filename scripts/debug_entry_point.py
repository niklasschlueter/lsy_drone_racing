import sys
import os
import multiprocessing # For context, though not strictly needed for printing sys.path

# IMPORTANT: Ensure this import path is correct and resolves your worker function
# It will be relative to wherever `debug_worker_entry_point.py` is imported from.
# If debug_worker_entry_point.py is in the same directory as multi_ctrl_async.py:
from lsy_drone_racing.control.multi_ctrl_async import _controller_worker_process_target

# If debug_worker_entry_point.py is in your project root and multi_ctrl_async.py is in lsy_drone_racing/control/:
# from lsy_drone_racing.control.multi_ctrl_async import _controller_worker_process_target

def debug_worker_wrapper(*args, **kwargs):
    """
    A wrapper function to print sys.path and then call the actual worker function.
    This function will be the *target* for the multiprocessing.Process.
    """
    print(f"\n--- Child process PID: {os.getpid()} ---")
    print("--- Child sys.path (DEBUG WRAPPER) ---")
    for p in sys.path:
        print(p)
    print("--------------------------------------\n")

    # Now, call the actual worker function
    _controller_worker_process_target(*args, **kwargs)

if __name__ == "__main__":
    # This block won't be executed by the spawned process itself,
    # but it's good practice for general script execution.
    print("This is debug_worker_entry_point.py being run directly (not as a spawned target).")
    print(f"PID: {os.getpid()}")
    print("sys.path at direct execution:")
    for p in sys.path:
        print(p)
