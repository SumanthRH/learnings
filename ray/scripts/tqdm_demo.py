import ray
from ray.experimental import tqdm_ray
import os

ray.init()
tqdm_remote = ray.remote(tqdm_ray.tqdm)

def generate_fibonacci(sequence_size):
    fibonacci = []
    for i in range(0, sequence_size):
        if i < 2:
            fibonacci.append(i)
            continue
        fibonacci.append(fibonacci[i-1]+fibonacci[i-2])
    return len(fibonacci)

# Function for remote Ray task with just a wrapper
@ray.remote
def generate_fibonacci_distributed(sequence_size, tqdm_ref):
    nums  = generate_fibonacci(sequence_size)
    tqdm_ref.update.remote(1)
    return nums

if __name__ == "__main__":

    sequence_size = 300_000
    tqdm_ref = tqdm_remote.remote(total=40, desc="Fibonacci")
    results = ray.get([generate_fibonacci_distributed.remote(sequence_size, tqdm_ref) for _ in range(40)])
    tqdm_ref.close.remote()
    ray.shutdown()
