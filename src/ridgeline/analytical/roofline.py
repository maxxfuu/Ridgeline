import matplotlib.pyplot as plt

def arithmetic_intensity(flops: int, byte_count: int) -> float:
    # FLOPs performed per byte moved: flops / byte_count.
    return flops/byte_count


def ridge_point(peak_flops: float, peak_bandwidth: float) -> float:
    # the arithmetic intensity where the slanted (memory) roof meets the flat
    return peak_flops / peak_bandwidth


def attainable_flops(intensity: float, peak_flops: float, peak_bandwidth: float) -> float:
    # Height of the roof directly above a given intensity: the best FLOPs/s
    return min(peak_flops, peak_bandwidth * intensity)


def classify(intensity: float, peak_flops: float, peak_bandwidth: float) -> str:
    return "memory-bound" if intensity < ridge_point(peak_flops, peak_bandwidth) else "compute-bound"

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

    # each measured op as a dot, annotated with its label and (x, y) coords.
    # x = arithmetic intensity (FLOP/byte), y = achieved performance (GFLOP/s).
    colors = {"prefill": "tab:blue", "decode": "tab:orange"}
    for label, intensity, achieved in points:
        ax.scatter(intensity, achieved, zorder=3, color=colors.get(label))
        ax.annotate(f"{label}\n({intensity:.2f} FLOP/byte, {achieved / 1e9:.1f} GFLOP/s)",
                    (intensity, achieved), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color=colors.get(label))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("performance (FLOPs / s)")
    ax.set_title("Roofline")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    return ax
