# Ridgeline

Ridgeline is a small roofline profiler for LLM inference. You point it at a model and it puts each phase (prefill and decode) on a measured roofline so you can see where the time is actually going.

The whole objective is to measure the hardwares real FLOPs and bandwidth, not the spec sheet numbers. Learning decode being memory bound wasn't enough, I had to prove it to myself. I'm glad I did. 

## What it does

- builds a gpt3 model from scratch with kvcache
- splits generate into prefill and decode so we can measure them seperate
- has a flop byte model so we can predict flops/second and flops/bytes (arithmetic intensity)
- measures the compute roof and memory roof of your hardware
- plots both phases on the roofline and tells you if your compute bound or memory bound

## Why

Decode is memory bound at batch 1 and i wanted to actually prove it instead of just reading it somewhere. the roofline is a predictive tool so the real goal is to predict where the dots land before running anything, then check.

## Layout

- `src/ridgeline/model` - The gpt3 model and the kvcache notebook that I built from scratch
- `src/ridgeline/analytical` - Flop byte model + roofline math
- `src/ridgeline/benchmarks` - Compute roof and memory roof sweeps
- `src/ridgeline/measurement` - Benchmarking, Tracing and Sync 
