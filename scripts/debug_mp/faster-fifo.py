from faster_fifo import Queue
from queue import Full, Empty

q = Queue(1000 * 1000)  # specify the size of the circular buffer in the ctor

# any pickle-able Python object can be added to the queue
py_obj = dict(a=42, b=33, c=(1, 2, 3), d=[1, 2, 3], e='123', f=b'kkk')
q.put(py_obj)
assert q.qsize() == 1

retrieved = q.get()
assert q.empty()
assert py_obj == retrieved

for i in range(100):
    try:
        q.put(py_obj, timeout=0.1)
    except Full:
        log.debug('Queue is full!')

num_received = 0
while num_received < 100:
    # get multiple messages at once, returns a list of messages for better performance in many-to-few scenarios
    # get_many does not guarantee that all max_messages_to_get will be received on the first call, in fact
    # no such guarantee can be made in multiprocessing systems.
    # get_many() will retrieve as many messages as there are available AND can fit in the pre-allocated memory
    # buffer. The size of the buffer is increased gradually to match demand.
    messages = q.get_many(max_messages_to_get=100)
    num_received += len(messages)

try:
    q.get(timeout=0.1)
    assert True, 'This won\'t be called'
except Empty:
    log.debug('Queue is empty')
