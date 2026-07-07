"""Roofline calculator.

Ties the analytical layer (predicted FLOPs/bytes from flop_byte_model) to the
measurement layer (device ceilings from benchmarks/compute_roof.py and
benchmarks/memory_roof.py) to answer, per operation: compute-bound or
memory-bound, and how close to the ceiling it runs.

Units convention (keep everything raw inside the math, convert only for display):
  - flops:          FLOPs (count)
  - byte counts:    bytes
  - peak_flops:     FLOPs / second   (from compute_roof)
  - peak_bandwidth: bytes / second   (from memory_roof)
  - intensity:      FLOPs / byte
Sanity check: peak_bandwidth * intensity = (bytes/s) * (FLOPs/byte) = FLOPs/s.
"""

import matplotlib.pyplot as plt

def arithmetic_intensity(flops: int, byte_count: int) -> float:
    # FLOPs performed per byte moved: flops / byte_count.
    return flops/byte_count


def ridge_point(peak_flops: float, peak_bandwidth: float) -> float:
    """The arithmetic intensity where the slanted (memory) roof meets the flat
    (compute) roof: peak_flops / peak_bandwidth.

    An op with intensity below this is memory-bound; above it, compute-bound.
    """
    raise NotImplementedError


def attainable_flops(intensity: float, peak_flops: float, peak_bandwidth: float) -> float:
    """Height of the roof directly above a given intensity: the best FLOPs/s
    achievable there = min(peak_flops, peak_bandwidth * intensity).

    Below the ridge the memory term wins; above it the compute term does.
    """
    raise NotImplementedError


def classify(intensity: float, peak_flops: float, peak_bandwidth: float) -> str:
    """Return "compute-bound" or "memory-bound" by comparing intensity against
    the ridge point.
    """
    raise NotImplementedError


def plot_roofline(points, peak_flops: float, peak_bandwidth: float, ax=None):
    """Draw the roofline and scatter each measured op onto it. (Plumbing — done.)

    points: iterable of (label, intensity, achieved_flops) where
      - intensity      = arithmetic_intensity(predicted_flops, predicted_bytes)
      - achieved_flops = predicted_flops / measured_time   (from benchmark)
    peak_flops:     flat ceiling, FLOPs/s   (max achieved from compute_roof)
    peak_bandwidth: slope of the ceiling, bytes/s (max achieved from memory_roof)
    ax: optional matplotlib Axes to draw into; a new figure is made if None.
    """
    points = list(points)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    ridge = peak_flops / peak_bandwidth

    # x-range spans the ops and the ridge, with padding on each side (log scale)
    intensities = [i for _, i, _ in points] + [ridge]
    x_lo = min(intensities) / 4
    x_hi = max(intensities) * 4

    # roof: slanted (bandwidth-limited) below the ridge, flat (compute-limited) above
    xs = [x_lo, ridge, x_hi]
    ys = [min(peak_flops, peak_bandwidth * x) for x in xs]
    ax.plot(xs, ys, color="black", linewidth=2, label="roofline")
    ax.axvline(ridge, color="grey", linestyle="--", linewidth=1)
    ax.text(ridge, peak_flops * 1.1, f"ridge\n{ridge:.1f} FLOP/byte",
            ha="center", va="bottom", fontsize=8, color="grey")

    # each measured op as a dot, annotated with its label
    for label, intensity, achieved in points:
        ax.scatter(intensity, achieved, zorder=3)
        ax.annotate(label, (intensity, achieved),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("performance (FLOPs / s)")
    ax.set_title("Roofline")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    return ax
