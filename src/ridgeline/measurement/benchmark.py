import numpy as np 
from time import perf_counter
from typing import Callable
from .sync import sync

def benchmark(make_op: Callable, device: str, warmup: int = 5, iters: int = 100, setup=None) -> float:
  times = []
  for _ in range(warmup):
    if setup: setup()
    make_op()

  # block until the warmup is complete
  sync(device)   
  for _ in range(iters):
    if setup: setup()
    start_time = perf_counter()
    make_op() 
    sync(device) # block until this iterations make_op() is complete    
    end_time = perf_counter()
    times.append(end_time - start_time)

  return np.median(times)
