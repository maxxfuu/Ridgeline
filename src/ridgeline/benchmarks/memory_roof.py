import torch 
import ridgeline.measurement.timing as timing

def memory_roof(N: int, device: str) -> float:
  # pre-allocate tensors 
  x = torch.randn(N, dtype=torch.float32, device=device)
  y = torch.empty(N, dtype=torch.float32, device=device)

  # measure time it takes to move data from x to y
  make_op = lambda: torch.add(x, 1, out=y)
  median_time = timing.benchmark(make_op, device)

  print(f"Bytes: {2 * 4 * N}")
  print(f"Median Time: {median_time} seconds")
  print(f"GBs/second: {(2 * 4 * N / median_time / 1e9).round(2)} GBs/second \n") 

  return (2 * 4 * N) / median_time  

def main():
  device, N_List = "mps", [20000000, 50000000, 100000000, 200000000, 400000000, 800000000]
  results = list()

  for N in N_List:
    achieved = memory_roof(N, device)
    results.append((N, achieved))

  return results

if __name__ == "__main__":
  main()