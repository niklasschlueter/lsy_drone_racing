from multiprocessing import Process, Queue, Pipe, Event
from multiprocessing.shared_memory import SharedMemory
import time
import pickle

# Make a ~40KB dict
big_dict = {'key_' + str(i): 'x' * 400 for i in range(100)}  # ~40 KB

def worker_pipe(conn_recv, conn_send):
    for _ in range(1000):
        data = conn_recv.recv()
        assert data == big_dict
        conn_send.send(data)

def worker_queue(q):
    for _ in range(1000):
        data = q.get()

def worker_shared(shm_name, size, event_in, event_out):
    shm = SharedMemory(name=shm_name)
    for _ in range(1000):
        event_in.wait()
        event_in.clear()
        # Read from shared memory
        data_bytes = bytes(shm.buf[:size])
        data = pickle.loads(data_bytes)
        event_out.set()
    shm.close()

def test_pipe():
    parent_conn, child_conn = Pipe()
    parent_conn2, child_conn2 = Pipe()
    p = Process(target=worker_pipe, args=(child_conn, parent_conn2,))
    p.start()
    t_toal = 0
    for _ in range(1000):
        print(f"test")

        t0 = time.perf_counter()
        parent_conn.send(big_dict)
        child_conn2.recv()
        t1 = time.perf_counter()
        t_toal += t1-t0
    p.join()
    print(f"Pipe took: {t_toal:.4f} s")
    exit()


def test_queue():
    q = Queue()
    p = Process(target=worker_queue, args=(q,))
    p.start()
    t0 = time.perf_counter()
    for _ in range(1000):
        q.put(big_dict)
    p.join()
    t1 = time.perf_counter()
    print(f"Queue took: {t1 - t0:.4f} s")

def test_shared_memory():
    pickled = pickle.dumps(big_dict)
    size = len(pickled)
    shm = SharedMemory(create=True, size=size)
    event_in = Event()
    event_out = Event()

    p = Process(target=worker_shared, args=(shm.name, size, event_in, event_out))
    p.start()
    t0 = time.perf_counter()
    for _ in range(1000):
        shm.buf[:size] = pickled
        event_in.set()
        event_out.wait()
        event_out.clear()
    p.join()
    t1 = time.perf_counter()
    shm.close()
    shm.unlink()
    print(f"SharedMemory took: {t1 - t0:.4f} s")

if __name__ == "__main__":
    print("Starting benchmark...")
    test_pipe()
    test_queue()
    test_shared_memory()
