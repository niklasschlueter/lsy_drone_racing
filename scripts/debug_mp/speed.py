
from multiprocessing import Process, Event, shared_memory
import numpy as np
import time

N = 10_000  # 10k floats ~40KB

def worker(shm_name, event_in, event_out):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray((N,), dtype=np.float32, buffer=shm.buf)
    for _ in range(1000):
        event_in.wait()
        event_in.clear()
        # Do something with arr
        arr.sum()
        event_out.set()
    shm.close()

def test_shared_numpy():
    shm = shared_memory.SharedMemory(create=True, size=N * 4)
    arr = np.ndarray((N,), dtype=np.float32, buffer=shm.buf)
    event_in = Event()
    event_out = Event()
    p = Process(target=worker, args=(shm.name, event_in, event_out))
    p.start()

    t0 = time.perf_counter()
    for _ in range(1000):
        arr[:] = np.random.rand(N).astype(np.float32)
        event_in.set()
        event_out.wait()
        event_out.clear()
    p.join()
    t1 = time.perf_counter()
    shm.close()
    shm.unlink()
    print(f"SharedMemory numpy took: {t1 - t0:.4f} s")

if __name__ == "__main__":
    test_shared_numpy()
