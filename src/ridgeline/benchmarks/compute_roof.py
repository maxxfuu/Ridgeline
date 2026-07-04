import torch
from ridgeline.measurement.benchmark import benchmark

def compute_roof(N: int, device: str) -> float:
  # pre-allocate tensors
  a = torch.randn(N, N, dtype=torch.float16, device=device)
  b = torch.randn(N, N, dtype=torch.float16, device=device)
  c = torch.empty(N, N, dtype=torch.float16, device=device)

  # only measure matmul compute time, not memory bandwidth
  make_op = lambda: torch.matmul(a, b, out=c)
  median_time = benchmark(make_op, device)

  print(f"FLOPs: {2 * N**3}")
  print(f"median time: {median_time} seconds")
  print(f"FLOPs/time: {(2 * N**3 / median_time).round(2)} FLOPs/second \n") 

  return (2 * N**3) / median_time  


""" 
dtype = float16, 2 bytes per element 
fp16 NxN, 2 * N**2 bytes per matrix 

3 live tensors: (a, b, c) 

total = 3 * (2 * N**2) = 6 * N**2 bytes

Suppose N = 8192
total = ~400MB
"""

def main():
  device, sizes = "mps", [2048, 4096, 6144, 8192]
  results = list()

  for N in sizes:
    achieved = compute_roof(N, device)
    results.append((N, achieved))

  return results

if __name__ == "__main__":
  main()