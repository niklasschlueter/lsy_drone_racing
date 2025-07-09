import multiprocessing
import json
import time
import numpy as np
import sys
import os
import struct # Import for packing/unpacking data length
from multiprocessing import shared_memory, Event, Queue
#from faster_fifo import Queue
from typing import Dict, Any

# --- Configuration ---
NUM_ITERATIONS = 5000  # Number of round-trips to measure for averaging
DICTIONARY_TARGET_SIZE_KB = 40 # Target size of the dictionary payload

# Shared memory protocol: [4-byte length][JSON data]
# So, MAX_PAYLOAD_SIZE_BYTES must account for the 4-byte length prefix.
LENGTH_PREFIX_BYTES = 4
MAX_JSON_PAYLOAD_BYTES = int(DICTIONARY_TARGET_SIZE_KB * 1024 * 1.2) # 20% buffer for JSON data
MAX_PAYLOAD_SIZE_BYTES = LENGTH_PREFIX_BYTES + MAX_JSON_PAYLOAD_BYTES


# --- Helper to create a dictionary of a target size ---
def create_test_dictionary(target_size_kb: int) -> Dict[str, Any]:
    """Creates a dictionary with a payload roughly matching the target size."""
    target_bytes = target_size_kb * 1024
    test_dict = {
        "timestamp": time.time(),
        "sequence": 0,
        "payload_data": []
    }

    # Add strings to the payload_data list until we reach roughly the target size
    # This is an approximation as actual memory/pickled size varies.
    # We'll use json.dumps for size estimation as it's what we'll use for serialization.
    current_size = sys.getsizeof(json.dumps(test_dict)) # Initial size of the dictionary
    string_chunk_size = 100 # Characters per string
    
    # Estimate bytes per string after JSON encoding and Python object overhead
    # This is a rough heuristic; actual size depends on content.
    # A 100-char string in JSON is 100 bytes, plus quotes. Python object overhead.
    # Let's assume ~120 bytes per 100-char string for estimation.
    string_byte_approx = 120 

    while current_size < target_bytes:
        test_dict["payload_data"].append("X" * string_chunk_size)
        current_size = sys.getsizeof(json.dumps(test_dict)) # Re-calculate size after adding

    # Add a unique identifier for each dictionary
    test_dict["unique_id"] = os.urandom(8).hex()
    return test_dict

# --- Worker for Queue-based IPC ---
def queue_worker(input_queue: Queue, output_queue: Queue):
    """Worker process that receives from one queue and sends back to another."""
    # print(f"Queue Worker: PID {os.getpid()} started.") # Commented out for cleaner output during benchmark
    try:
        while True:
            # Get data from the input queue
            data = input_queue.get()
            if data == "STOP":
                break
            
            # Simulate some minimal work (e.g., controller processing)
            # time.sleep(0.0001) # 0.1 ms - keep it very small to focus on IPC

            # Send data back to the output queue
            output_queue.put(data)
    except Exception as e:
        print(f"Queue Worker Error: {e}")
    # finally: # Commented out for cleaner output during benchmark
    #     print("Queue Worker: Exiting.")

# --- Worker for Shared Memory-based IPC ---
def shared_memory_worker(shm_name: str, parent_ready_event: Event, child_ready_event: Event):
    """Worker process that communicates via shared memory."""
    # print(f"Shared Memory Worker: PID {os.getpid()} started.") # Commented out for cleaner output during benchmark
    
    shm = None # Initialize shm to None
    try:
        # Attach to the existing shared memory segment
        shm = shared_memory.SharedMemory(name=shm_name)
        buffer = shm.buf
        # print(f"Shared Memory Worker: Attached to shared memory '{shm_name}'.") # Commented out for cleaner output during benchmark

        t1 = time.perf_counter()
        while True:
            # Wait for the parent to signal data is ready
            parent_ready_event.wait()
            parent_ready_event.clear() # Reset the event for the next cycle
            t2 = time.perf_counter()
            print(f"time wainting for signal :{t2-t1}")

            # Read data from shared memory
            # First, read the 4-byte length prefix
            json_len = struct.unpack('<I', buffer[0:LENGTH_PREFIX_BYTES])[0] # Unpack as unsigned int
            
            if json_len == 0: # Check for explicit stop signal (empty message)
                break 

            # Read the JSON bytes based on the length
            json_bytes = buffer[LENGTH_PREFIX_BYTES : LENGTH_PREFIX_BYTES + json_len].tobytes()
            
            received_dict = json.loads(json_bytes.decode('utf-8'))
            
            if received_dict.get("STOP_SIGNAL"): # Check for explicit stop signal from dict content
                break

            # Simulate some minimal work
            # time.sleep(0.0001) # 0.1 ms

            # Prepare response (just send the same dict back for round-trip)
            response_json_bytes = json.dumps(received_dict).encode('utf-8')
            
            # Write response back to shared memory
            if len(response_json_bytes) > MAX_JSON_PAYLOAD_BYTES:
                 raise ValueError("Response data too large for shared memory in worker.")
            
            # Write length prefix
            buffer[0:LENGTH_PREFIX_BYTES] = struct.pack('<I', len(response_json_bytes))
            # Write JSON data
            buffer[LENGTH_PREFIX_BYTES : LENGTH_PREFIX_BYTES + len(response_json_bytes)] = response_json_bytes
            
            # Signal to parent that response is ready
            child_ready_event.set()
            t1 = time.perf_counter()

    except Exception as e:
        print(f"Shared Memory Worker Error: {e}")
    finally:
        if shm:
            shm.close() # Detach from shared memory
        # print("Shared Memory Worker: Exiting.") # Commented out for cleaner output during benchmark


# --- Main Comparison Logic ---
if __name__ == "__main__":
    # Ensure 'spawn' start method for robustness across OS
    try:
        multiprocessing.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Could not set multiprocessing start method (already set or error): {e}")
        pass

    # --- Generate Test Data ---
    test_dict = create_test_dictionary(DICTIONARY_TARGET_SIZE_KB)
    initial_json_size_bytes = sys.getsizeof(json.dumps(test_dict))
    print(f"\n--- Test Data ---")
    print(f"Dictionary target size: {DICTIONARY_TARGET_SIZE_KB} KB")
    print(f"Actual JSON serialized size: {initial_json_size_bytes / 1024:.3f} KB")
    if initial_json_size_bytes > MAX_JSON_PAYLOAD_BYTES:
        print(f"WARNING: Initial JSON size ({initial_json_size_bytes} bytes) exceeds MAX_JSON_PAYLOAD_BYTES ({MAX_JSON_PAYLOAD_BYTES} bytes). Adjust DICTIONARY_TARGET_SIZE_KB or MAX_PAYLOAD_SIZE_BYTES.")
        sys.exit(1)


    # --- Queue Test ---
    print(f"\n--- Running Queue Test ({NUM_ITERATIONS} iterations) ---")
    q_input = Queue(maxsize=1)
    q_input2 = Queue(maxsize=1)
    q_output = Queue(maxsize=1)
    q_output2 = Queue(maxsize=1)
    queue_proc = multiprocessing.Process(target=queue_worker, args=(q_input, q_output))
    queue_proc_2 = multiprocessing.Process(target=queue_worker, args=(q_input2, q_output2))
    queue_proc.start()
    queue_proc_2.start()

    queue_times = []
    try:
        for i in range(NUM_ITERATIONS):
            print("t"*1000)
            current_dict = test_dict.copy() # Use a copy to avoid modification issues
            current_dict["sequence"] = i

            start_time = time.perf_counter()
            q_input.put(current_dict) # Send to worker
            received_dict = q_output.get() # Get from worker

            q_input2.put(current_dict) # Send to worker
            received_dict2 = q_output2.get() # Get from worker

            end_time = time.perf_counter()
            queue_times.append(end_time - start_time)

            if received_dict["sequence"] != i:
                print(f"Queue Test: Sequence mismatch! Expected {i}, got {received_dict['sequence']}")
    except Exception as e:
        print(f"Queue Test Error in main process: {e}")
    finally:
        q_input.put("STOP") # Signal worker to stop
        queue_proc.join()
        avg_queue_time_ms = (sum(queue_times) / NUM_ITERATIONS) * 1000
        avg_queue_std_ms = np.std(queue_times*1000)
        print(f"Average Queue round-trip time: {avg_queue_time_ms:.3f} ms, std: {avg_queue_std_ms}")


    # --- Shared Memory Test ---
    print(f"\n--- Running Shared Memory Test ({NUM_ITERATIONS} iterations) ---")
    shm = None # Initialize shm to None
    try:
        # Create shared memory segment
        shm = shared_memory.SharedMemory(create=True, size=MAX_PAYLOAD_SIZE_BYTES)
        shm_buffer = shm.buf

        shm2 = shared_memory.SharedMemory(create=True, size=MAX_PAYLOAD_SIZE_BYTES)
        shm_buffer2 = shm2.buf
        
        # Create events for synchronization
        parent_ready_event = Event()
        child_ready_event = Event()

        parent_ready_event2 = Event()
        child_ready_event2 = Event()

        # Start shared memory worker
        shm_proc = multiprocessing.Process(
            target=shared_memory_worker,
            args=(shm.name, parent_ready_event, child_ready_event)
        )

        shm_proc2 = multiprocessing.Process(
            target=shared_memory_worker,
            args=(shm2.name, parent_ready_event2, child_ready_event2)
        )
        shm_proc.start()
        shm_proc2.start()

        shm_times = []
        shm_times_no_encode = []
        for i in range(NUM_ITERATIONS):
            print("t"*1000)
            current_dict = test_dict.copy()
            current_dict["sequence"] = i

            start_time = time.perf_counter()
            
            # Serialize the dictionary
            json_bytes = json.dumps(current_dict).encode('utf-8')
            if len(json_bytes) > MAX_JSON_PAYLOAD_BYTES:
                raise ValueError(f"Serialized data ({len(json_bytes)} bytes) too large for "
                                 f"MAX_JSON_PAYLOAD_BYTES ({MAX_JSON_PAYLOAD_BYTES} bytes) in main process.")

            start_time_2 = time.perf_counter()

            
            # Write length prefix
            shm_buffer[0:LENGTH_PREFIX_BYTES] = struct.pack('<I', len(json_bytes))
            # Write JSON data
            shm_buffer[LENGTH_PREFIX_BYTES : LENGTH_PREFIX_BYTES + len(json_bytes)] = json_bytes
            
            # Signal worker that data is ready
            parent_ready_event.set()

            # Wait for worker to signal response is ready
            child_ready_event.wait()
            child_ready_event.clear() # Reset for next cycle

            # Read response from shared memory
            response_json_len = struct.unpack('<I', shm_buffer[0:LENGTH_PREFIX_BYTES])[0]
            if response_json_len == 0: # Should not happen unless worker sent STOP_SIGNAL
                break 
            received_json_bytes = shm_buffer[LENGTH_PREFIX_BYTES : LENGTH_PREFIX_BYTES + response_json_len].tobytes()
            
            received_dict = json.loads(received_json_bytes.decode('utf-8'))

            #2

            #            json_bytes = json.dumps(current_dict).encode('utf-8')
            #            if len(json_bytes) > MAX_JSON_PAYLOAD_BYTES:
            #                raise ValueError(f"Serialized data ({len(json_bytes)} bytes) too large for "
            #                                 f"MAX_JSON_PAYLOAD_BYTES ({MAX_JSON_PAYLOAD_BYTES} bytes) in main process.")
            #
            
            # Write length prefix
            shm_buffer2[0:LENGTH_PREFIX_BYTES] = struct.pack('<I', len(json_bytes))
            # Write JSON data
            shm_buffer2[LENGTH_PREFIX_BYTES : LENGTH_PREFIX_BYTES + len(json_bytes)] = json_bytes
            
            # Signal worker that data is ready
            parent_ready_event2.set()

            # Wait for worker to signal response is ready
            child_ready_event2.wait()
            child_ready_event2.clear() # Reset for next cycle

            # Read response from shared memory
            response_json_len = struct.unpack('<I', shm_buffer2[0:LENGTH_PREFIX_BYTES])[0]
            if response_json_len == 0: # Should not happen unless worker sent STOP_SIGNAL
                break 
            received_json_bytes = shm_buffer2[LENGTH_PREFIX_BYTES : LENGTH_PREFIX_BYTES + response_json_len].tobytes()
            
            received_dict = json.loads(received_json_bytes.decode('utf-8'))

            # timing
            end_time = time.perf_counter()
            shm_times.append(end_time - start_time)
            shm_times_no_encode.append(end_time - start_time_2)

            if received_dict["sequence"] != i:
                print(f"Shared Memory Test: Sequence mismatch! Expected {i}, got {received_dict['sequence']}")

    except Exception as e:
        print(f"Shared Memory Test Error in main process: {e}")
    finally:
        # Signal worker to stop by sending an empty message (length 0)
        shm_buffer[0:LENGTH_PREFIX_BYTES] = struct.pack('<I', 0) 
        parent_ready_event.set() # Signal worker to check for new data (which is now empty)
        shm_proc.join() # Wait for worker to exit

        if shm:
            shm.close() # Detach from shared memory
            shm.unlink() # Unlink (release OS resource)
            print(f"Shared memory segment '{shm.name}' unlinked.")
        avg_shm_time_ms = (sum(shm_times) / NUM_ITERATIONS) * 1000
        avg_shm_std_ms = np.std(shm_times*1000)
        print(f"Average Shared Memory round-trip time: {avg_shm_time_ms:.3f} ms, std: {avg_shm_std_ms}")

        avg_shm_time_ms = (sum(shm_times_no_encode) / NUM_ITERATIONS) * 1000
        avg_shm_std_ms = np.std(shm_times_no_encode*1000)
        print(f"Average Shared Memory no encode round-trip time: {avg_shm_time_ms:.3f} ms, std: {avg_shm_std_ms}")


    # --- Comparison ---
    print(f"\n--- Comparison ---")
    print(f"Average Queue round-trip time:       {avg_queue_time_ms:.3f} ms")
    print(f"Average Shared Memory round-trip time: {avg_shm_time_ms:.3f} ms")

    if avg_shm_time_ms < avg_queue_time_ms:
        print(f"Shared Memory was faster by: {(avg_queue_time_ms - avg_shm_time_ms):.3f} ms")
        print(f"Shared Memory is approximately {(avg_queue_time_ms / avg_shm_time_ms):.2f}x faster.")
    else:
        print(f"Queue was faster by: {(avg_shm_time_ms - avg_queue_time_ms):.3f} ms")
        print(f"Queue is approximately {(avg_shm_time_ms / avg_queue_time_ms):.2f}x faster.")
