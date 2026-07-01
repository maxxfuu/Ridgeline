import numpy as np 
from time import perf_counter
from typing import Callable
from .sync import sync

def benchmark(make_op: Callable, device: str, warmup: int = 5, iters: int = 100) -> float:
  times = []
  # warm up 
  for _ in range(warmup):
    make_op()

  sync(device) # block until the warmup is complete
  for _ in range(iters):
    start_time = perf_counter()
    make_op() 
    sync(device) # block until this iterations make_op() is complete    
    end_time = perf_counter()
    times.append(end_time - start_time)

  return np.median(times)
