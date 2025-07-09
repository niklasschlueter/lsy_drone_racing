import time
import pickle
import sys

# ----------------------------------------------------------------------
# 1. Define your sample dictionary
#    Replace this with a representative dictionary structure and size
#    that you actually pass through your multiprocessing Queue.
# ----------------------------------------------------------------------

# Example 1: Simple dictionary (less pickling overhead)
sample_dict_simple = {
    "timestamp": time.time(),
    "data_x": [1.0, 2.0, 3.0, 4.0, 5.0],
    "data_y": [6.0, 7.0, 8.0, 9.0, 10.0]
}

# Example 2: More complex/nested dictionary (more pickling overhead for same size)
sample_dict_complex = {
    "sensor_readings": {
        "imu": {"accel": {"x": 0.1, "y": 0.2, "z": 9.8}, "gyro": {"x": 0.01, "y": 0.02, "z": 0.03}},
        "gps": {"lat": 48.137, "lon": 11.576, "alt": 520.0},
        "camera": {"image_id": 12345, "status": "ok", "properties": {"res_x": 640, "res_y": 480, "fps": 30}}
    },
    "controller_state": {
        "mode": "tracking",
        "setpoint": [10.0, 20.0, 3.0, 0.0],
        "gains": {"kp": 1.0, "ki": 0.1, "kd": 0.05}
    },
    "timestamp": time.time(),
    "sequence_num": 1001
}

# To make it ~3KB, you'd need to add more data. Let's create a larger one for illustration:
# Example 3: Larger dictionary (simulating 3KB)
def create_large_dict(size_kb):
    base_dict = {
        "timestamp": time.time(),
        "sequence": 0,
        "payload": []
    }
    # Each float takes about 24 bytes in Python object, + list overhead
    # Let's target roughly 3000 bytes.
    # A string of 100 chars is ~150 bytes.
    # A list of 100 floats is ~800 bytes.
    # To reach 3KB (3072 bytes), let's fill it with strings.
    # A 1-char string is around 50 bytes. A 20-char string around 70 bytes.
    # Let's say 50 key-value pairs of 50-char strings.
    # Approx size of 'key': 'value' = 50 + 50 + dict overhead per entry.
    # A good way to estimate is to actually measure it.
    
    # Let's fill with some data to get it close to 3KB
    # Appending large strings to a list will grow its size
    string_len = 100 # Adjust this to fine-tune size
    num_entries = int((size_kb * 1024) / (string_len + 50)) # Roughly estimate string + dict entry overhead
    
    large_dict = {f"key_{i}": "A" * string_len for i in range(num_entries)}
    large_dict["meta"] = {"timestamp": time.time(), "id": "test_payload"}
    return large_dict

# Use the dictionary that best represents your actual data:
# sample_data = sample_dict_simple
# sample_data = sample_dict_complex
sample_data = create_large_dict(3) # This will create a dictionary *around* 3KB. Actual size varies by content.


# Get rough size in memory (useful for conceptual check, not exact pickled size)
print(f"Sample data in-memory size (approx): {sys.getsizeof(sample_data) / 1024:.2f} KB")

# ----------------------------------------------------------------------
# 2. Measure Pickling Time (pickle.dumps)
# ----------------------------------------------------------------------
num_iterations = 1000 # Run multiple times for more stable average

# Warm-up (important for micro-benchmarking)
for _ in range(100):
    _ = pickle.dumps(sample_data)

start_time = time.perf_counter()
for _ in range(num_iterations):
    pickled_data = pickle.dumps(sample_data)
end_time = time.perf_counter()

avg_pickle_time_ms = ((end_time - start_time) / num_iterations) * 1000
pickled_size_kb = sys.getsizeof(pickled_data) / 1024

print(f"\n--- Pickling (pickle.dumps) ---")
print(f"Average pickling time: {avg_pickle_time_ms:.3f} ms")
print(f"Size of pickled data: {pickled_size_kb:.3f} KB")

# ----------------------------------------------------------------------
# 3. Measure Unpickling Time (pickle.loads)
# ----------------------------------------------------------------------

# Warm-up
for _ in range(100):
    _ = pickle.loads(pickled_data)

start_time = time.perf_counter()
for _ in range(num_iterations):
    unpickled_data = pickle.loads(pickled_data)
end_time = time.perf_counter()

avg_unpickle_time_ms = ((end_time - start_time) / num_iterations) * 1000

print(f"\n--- Unpickling (pickle.loads) ---")
print(f"Average unpickling time: {avg_unpickle_time_ms:.3f} ms")

# ----------------------------------------------------------------------
# 4. Total Round-Trip IPC Overhead Estimate
# ----------------------------------------------------------------------
total_ipc_overhead_ms = avg_pickle_time_ms + avg_unpickle_time_ms
print(f"\n--- Total Estimated IPC Overhead (Pickle + Unpickle) ---")
print(f"Total average IPC overhead per element: {total_ipc_overhead_ms:.3f} ms")

print("\n--- Context for 200 Hz Controller ---")
cycle_time_budget_ms = 1000 / 200
print(f"Your 200 Hz controller has a budget of {cycle_time_budget_ms:.2f} ms per cycle.")
print(f"IPC overhead is {total_ipc_overhead_ms / cycle_time_budget_ms * 100:.2f}% of your cycle budget.")
