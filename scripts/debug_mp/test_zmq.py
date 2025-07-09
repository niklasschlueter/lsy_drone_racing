
import zmq
import pickle
import numpy as np
from multiprocessing import Process
import time

N = 10_000  # ~40KB of floats

def worker():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PULL)
    socket.bind("ipc:///tmp/zmqtest")

    for _ in range(1000):
        msg = socket.recv()
        arr = np.frombuffer(msg, dtype=np.float32)

def test_zmq():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.connect("ipc:///tmp/zmqtest")

    arr = np.random.rand(N).astype(np.float32)

    p = Process(target=worker)
    p.start()

    t0 = time.perf_counter()
    for _ in range(1000):
        socket.send(arr.tobytes())
    p.join()
    t1 = time.perf_counter()
    print(f"ZeroMQ took: {t1 - t0:.4f} s")

if __name__ == "__main__":
    test_zmq()
