import torch
from time import perf_counter
from typing import Callable
from .sync import sync

def peak_memory(op: Callable, device: str) -> int:
  """Bytes of memory footprint used by one call to op(). Not bandwidth."""
  if device == "mps":
    torch.mps.synchronize()
    before = torch.mps.current_allocated_memory()
    op()
    torch.mps.synchronize()
    after = torch.mps.current_allocated_memory()
    return after - before

  elif device == "cuda":
    torch.cuda.reset_peak_memory_stats()
    op()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()

  else:
    op()
    return 0


def trace_ops(model: torch.nn.Module, op: Callable, device: str) -> dict:
  timings = {}
  starts = {}
  handles = []

  def make_pre_hook(name):
    def pre_hook(module, args):
      sync(device)
      starts[name] = perf_counter()
    return pre_hook

  def make_post_hook(name):
    def post_hook(module, args, output):
      sync(device)
      timings[name] = perf_counter() - starts[name]
    return post_hook

  for name, module in model.named_modules():
    name = name or "model"
    handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
    handles.append(module.register_forward_hook(make_post_hook(name)))

  op()

  for h in handles:
    h.remove()

  return timings
