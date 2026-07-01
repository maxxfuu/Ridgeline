import torch 
import ridgeline.measurement.timing as timing

def compute_roof(N: int, device: str) -> float:
  # pre-allocate tensors 
  a = torch.randn(N, N, dtype=torch.float16, device=device)
  b = torch.randn(N, N, dtype=torch.float16, device=device)
  c = torch.empty(N, N, dtype=torch.float16, device=device)

  # only measure matmul compute time, not memory bandwidth 
  make_op = lambda: torch.matmul(a, b, out=c)
  median_time = timing.benchmark(make_op, device)

  print(f"FLOPs: {2 * N**3}")
  print(f"median time: {median_time} seconds")
  print(f"FLOPs/time: {2 * N**3 / median_time} FLOPs/second") 

  return (2 * N**3) / median_time  


""" 
dtype = float16, 2 bytes per element 
fp16 NxN, 2 * N**2 bytes per matrix 

3 live tensors: (a, b, c) 

total = 3 * (2 * N**2) = 6 * N**2 bytes

Suppose N = 8192
total = ~100MB 
"""