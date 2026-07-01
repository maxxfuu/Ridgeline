import torch

def sync(device: str):
  if device == "mps":
    torch.mps.synchronize()
  elif device == "cuda":
      torch.cuda.synchronize()
  elif device == "cpu":
      pass


