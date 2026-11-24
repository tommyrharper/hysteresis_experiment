"""Capture a transformer run, auto-ranged, and save CSV + plots to data/ without overwriting.

Run: uv run python capture.py
"""

import os
import re

# ARM Mac driver path; must be set before picosdk is imported
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/Library/Frameworks/PicoSDK.framework/Libraries/libps2000a"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from picosdk.device import ChannelConfig, TimebaseOptions
from picosdk.ps2000a import ps2000a as ps

OUT_DIR = "data"
TOTAL_T = 1.0          # Capture length /s
MAX_DELTA_T = 1e-5     # Max point spacing after downsampling /s
COUPLING = "DC"
RANGES = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]   # Allowed +/- ranges on the PicoScope (V, before x10 probe)
GAIN_ACC = 0.03        # Scope DC gain accuracy per channel (check PicoScope 2000 series datasheet)


def measure(A_range, B_range):
    """Returns downsampled t, V_a, V_b (scope volts, NOT x10) and full-rate per-channel peak |V| and clipped flag.
    Clipping is detected from the data because picosdk's overflow flag never reports channel B (bit-shift bug)."""
    timebase = TimebaseOptions(min_collection_time=TOTAL_T)
    channels = [ChannelConfig("A", True, COUPLING, A_range),
                ChannelConfig("B", True, COUPLING, B_range)]

    with ps.open_unit() as scope:
        times, voltages, _ = scope.capture_block(timebase, channels)

    ranges = {"A": A_range, "B": B_range}
    peak = {ch: float(np.max(np.abs(voltages[ch]))) for ch in ranges}
    clipped = {ch: peak[ch] >= 0.999 * ranges[ch] for ch in ranges}   # Clipped samples read exactly +/- range

    downsample = max(1, int(MAX_DELTA_T / (times[1] - times[0])))
    return times[::downsample], voltages["A"][::downsample], voltages["B"][::downsample], peak, clipped


def measure_autorange(headroom=0.8, max_tries=10, start=None):
    """Recaptures, moving each channel to the finest range with full scale above peak/headroom, until settled.
    A range that ever clipped is never used again, so it cannot oscillate.
    start: initial ranges, e.g. the last settled ones, so repeated captures settle in one try."""
    ranges = dict(start or {"A": 1, "B": 1})
    too_small = {"A": 0, "B": 0}   # Largest range seen to clip, per channel

    for attempt in range(max_tries):
        t, Va, Vb, peak, clipped = measure(ranges["A"], ranges["B"])
        print(f"Try {attempt + 1}: ranges {ranges}, peak A {peak['A']:.3g} V, B {peak['B']:.3g} V, "
              f"clipped A {clipped['A']}, B {clipped['B']}")

        new = {}
        for ch in ranges:
            if clipped[ch]:
                too_small[ch] = max(too_small[ch], ranges[ch])
            ok = [r for r in RANGES if r > too_small[ch] and (clipped[ch] or peak[ch] < headroom * r)]
            if not ok:
                raise RuntimeError(f"Channel {ch} clips even on the largest range")
            new[ch] = ok[0]

        if new == ranges and not any(clipped.values()):
            return t, Va, Vb, ranges
        ranges = new

    raise RuntimeError("Auto-range did not settle; check the signal is stable")


def next_run_name(material):
    """First '<material>_<n>' with no existing file of that prefix in OUT_DIR."""
    n = 1
    while any(f.startswith(f"{material}_{n}_") or f == f"{material}_{n}.csv" for f in os.listdir(OUT_DIR)):
        n += 1
    return f"{material}_{n}"


def save_fig(fig, path):
    if os.path.exists(path):
        raise FileExistsError(path)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_traces(t, Va, Vb, title, path):
    fig, (ax1a, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1b = ax1a.twinx()
    ax1a.plot(t, Va, color="darkblue", linewidth=0.5)
    ax1b.plot(t, Vb, color="darkred", linewidth=0.5)
    ax1a.set_xlim(0, 0.1)   # First 5 cycles at 50 Hz
    ax1a.set_xlabel("t /s")
    ax1a.set_ylabel("V_x /V", color="darkblue")
    ax1b.set_ylabel("V_y /V", color="darkred")

    ax2.plot(Va, Vb, color="darkgreen", linewidth=0.5)
    ax2.set_xlabel("V_x /V")
    ax2.set_ylabel("V_y /V")

    fig.suptitle(title)
    fig.tight_layout()
    save_fig(fig, path)


def plot_fit(Va, Vb, title, path):
    """Straight-line fit of V_y against V_x with stat + systematic uncertainty. Returns the gradient string."""
    (m, c), cov = np.polyfit(Va, Vb, 1, cov=True)
    m_err_stat, c_err = np.sqrt(np.diag(cov))
    m_err_sys = abs(m) * np.sqrt(2) * GAIN_ACC   # A and B gain errors add in quadrature
    m_err = np.hypot(m_err_stat, m_err_sys)

    # Round gradient to the decimal place of the uncertainty's first significant figure
    dp = max(0, -int(np.floor(np.log10(m_err))))
    gradient_str = f"{m:.{dp}f} ± {m_err:.{dp}f}"
    print(f"Gradient  = {gradient_str}  (stat ± {m_err_stat:.2g}, sys ± {m_err_sys:.2g})")
    print(f"Intercept = {c:.5g} ± {c_err:.2g} V")

    x_fit = np.array([Va.min(), Va.max()])
    fig, ax = plt.subplots()
    ax.plot(Va, Vb, color="darkgreen", linewidth=0.5, label="Data")
    ax.plot(x_fit, m * x_fit + c, color="black", label=f"Fit: gradient = {gradient_str}")
    ax.set_title(title)
    ax.set_xlabel("V_x /V")
    ax.set_ylabel("V_y /V")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, path)
    return gradient_str


def main():
    material = ""
    while not material:
        material = re.sub(r"[^a-z0-9]+", "_", input("Material name: ").strip().lower()).strip("_")

    os.makedirs(OUT_DIR, exist_ok=True)

    t, Va, Vb, ranges = measure_autorange()
    Va, Vb = 10 * Va, 10 * Vb   # 10x probe

    run = next_run_name(material)
    base = os.path.join(OUT_DIR, run)
    title = f"{run}  (ranges A ±{ranges['A']} V, B ±{ranges['B']} V before x10)"

    pd.DataFrame({"t /s": t, "V_x /V": Va, "V_y /V": Vb}).to_csv(f"{base}.csv", mode="x")
    plot_traces(t, Va, Vb, title, f"{base}_traces.png")
    saved = [f"{base}.csv", f"{base}_traces.png"]

    # Straight-line fit only means something for a linear (air core) response, not a hysteresis loop
    if "air" in material:
        plot_fit(Va, Vb, title, f"{base}_fit.png")
        saved.append(f"{base}_fit.png")

    print("Saved:\n  " + "\n  ".join(saved))


if __name__ == "__main__":
    main()
