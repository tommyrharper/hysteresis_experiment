"""Capture hysteresis loops repeatedly over time (e.g. core warming as ice melts). Stop with Ctrl+C.

Each capture is saved as it happens, so nothing is lost if the run is interrupted:
  data/<material>_timelapse_<n>/capture_<k>.csv   raw data per capture
  data/<material>_timelapse_<n>/log.csv           elapsed time, ranges and loop area per capture
  data/<material>_timelapse_<n>/area_vs_time.png  refreshed after every capture
  data/<material>_timelapse_<n>/loops.png         loops coloured by time, written on exit

Run: uv run python timelapse.py
"""

import os
import re
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capture import OUT_DIR, measure_autorange
from loop_area import loop_area

LOOPS_SHOWN = 8   # Captures overlaid in loops.png, evenly spaced in time


def ask(prompt, default):
    reply = input(f"{prompt} [{default}]: ").strip()
    return type(default)(reply) if reply else default


def new_run_dir(material):
    n = 1
    while os.path.exists(os.path.join(OUT_DIR, f"{material}_timelapse_{n}")):
        n += 1
    path = os.path.join(OUT_DIR, f"{material}_timelapse_{n}")
    os.makedirs(path)
    return path


def plot_area(log, title, path):
    fig, ax = plt.subplots()
    ax.errorbar(log["elapsed /s"] / 60, log["area /V^2"], yerr=log["area stat err /V^2"],
                fmt="o-", color="darkgreen", markersize=3, linewidth=0.8, capsize=2)
    ax.set_xlabel("Elapsed time /min")
    ax.set_ylabel("Loop area /V²")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_loops(run_dir, log, title, path):
    """Overlay the loop from evenly spaced captures, coloured by elapsed time."""
    picks = np.unique(np.linspace(0, len(log) - 1, min(LOOPS_SHOWN, len(log))).round().astype(int))
    cmap = plt.get_cmap("coolwarm")
    t_max = max(log["elapsed /s"].iloc[-1], 1)

    fig, ax = plt.subplots()
    for i in picks:
        row = log.iloc[i]
        data = pd.read_csv(os.path.join(run_dir, row["file"]))
        n = len(data) // 10   # ~5 cycles is plenty to show the loop
        ax.plot(data["V_x /V"][:n], data["V_y /V"][:n], linewidth=0.5,
                color=cmap(row["elapsed /s"] / t_max), label=f"{row['elapsed /s'] / 60:.1f} min")
    ax.set_xlabel("V_x /V")
    ax.set_ylabel("V_y /V")
    ax.set_title(title)
    ax.legend(fontsize=7, title="Elapsed", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    material = ""
    while not material:
        material = re.sub(r"[^a-z0-9]+", "_", input("Material name: ").strip().lower()).strip("_")
    interval = ask("Seconds between captures", 30.0)

    os.makedirs(OUT_DIR, exist_ok=True)
    run_dir = new_run_dir(material)
    title = os.path.basename(run_dir)
    log_path = os.path.join(run_dir, "log.csv")
    print(f"Saving to {run_dir}. Press Ctrl+C to stop.\n")

    rows = []
    ranges = None
    t0 = time.monotonic()
    k = 0
    try:
        while True:
            # Fixed schedule from t0, so capture time doesn't make the interval drift
            time.sleep(max(0.0, t0 + k * interval - time.monotonic()))

            elapsed = time.monotonic() - t0
            wall = datetime.now().isoformat(timespec="seconds")
            t, Va, Vb, ranges = measure_autorange(start=ranges)
            Va, Vb = 10 * Va, 10 * Vb   # 10x probe

            file = f"capture_{k + 1:04d}.csv"
            pd.DataFrame({"t /s": t, "V_x /V": Va, "V_y /V": Vb}).to_csv(os.path.join(run_dir, file), mode="x")

            try:
                area, area_err, cycles = loop_area(Va, Vb)
            except (ValueError, IndexError):   # No clean cycles (e.g. signal off)
                area, area_err, cycles = np.nan, np.nan, 0

            rows.append({"capture": k + 1, "elapsed /s": round(elapsed, 2), "wall time": wall, "file": file,
                         "A range /V": ranges["A"], "B range /V": ranges["B"], "cycles": cycles,
                         "area /V^2": area, "area stat err /V^2": area_err})
            log = pd.DataFrame(rows)
            log.to_csv(log_path, index=False)
            plot_area(log, title, os.path.join(run_dir, "area_vs_time.png"))

            print(f"#{k + 1}  {elapsed / 60:6.2f} min  area {area:.4g} ± {area_err:.2g} V²  ranges {ranges}\n")
            k += 1
    except KeyboardInterrupt:
        print("\nStopped.")

    if rows:
        plot_loops(run_dir, pd.DataFrame(rows), title, os.path.join(run_dir, "loops.png"))
        print(f"{len(rows)} captures saved in {run_dir}")


if __name__ == "__main__":
    main()
