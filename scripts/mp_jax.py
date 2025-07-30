import multiprocessing as mp
import os


def jax_worker(proc_id):
    # Import JAX inside the subprocess
    import jax
    import jax.numpy as jnp

    print(f"[Process {proc_id}] PID: {os.getpid()}")
    print(f"[Process {proc_id}] Devices: {jax.devices()}")

    # Perform a simple JAX computation on the GPU
    x = jnp.arange(1_000_000, dtype=jnp.float32)
    result = jnp.sum(jnp.sin(x) * jnp.cos(x))

    print(f"[Process {proc_id}] Computation result: {result}")


if __name__ == "__main__":
    # Set start method to 'spawn' (safest for CUDA/JAX)
    mp.set_start_method("spawn", force=True)

    # Create two processes
    processes = []
    for i in range(2):
        p = mp.Process(target=jax_worker, args=(i,))
        p.start()
        processes.append(p)

    # Wait for both to finish
    for p in processes:
        p.join()
