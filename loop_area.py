"""Hysteresis loop area (V_x * V_y, in V^2) per run, from CSVs saved by capture.py.

Run: uv run python loop_area.py
"""

import glob
import os
import re

import numpy as np
import pandas as pd

OUT_DIR = "data"
MATERIALS = ["mild_steel", "iron", "cu_ni"]
GAIN_ACC = 0.03   # Scope DC gain accuracy per channel (check PicoScope 2000 series datasheet)


def cycle_starts(Va):
    """Indices where V_x crosses zero upwards, one per cycle. Schmitt trigger: a crossing only counts
    after V_x has dropped below -h, so noise jitter around zero cannot add extra cycles."""
    h = 0.2 * np.max(np.abs(Va))
    up = np.flatnonzero((Va[:-1] < 0) & (Va[1:] >= 0)) + 1
    below = np.flatnonzero(Va < -h)
    keep = []
    for i in up:
        # Must have been below -h since the previous kept crossing (or ever, for the first)
        j = np.searchsorted(below, keep[-1] if keep else 0)
        if j < len(below) and below[j] < i:
            keep.append(i)
    return np.array(keep)


def shoelace(x, y):
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def loop_area(Va, Vb):
    """Mean area over whole cycles, with standard error of the mean across cycles."""
    starts = cycle_starts(Va)
    areas = np.array([shoelace(Va[a:b], Vb[a:b]) for a, b in zip(starts[:-1], starts[1:])])
    return areas.mean(), areas.std(ddof=1) / np.sqrt(len(areas)), len(areas)


def main():
    rows = []
    for material in MATERIALS:
        for path in sorted(glob.glob(os.path.join(OUT_DIR, f"{material}_*.csv"))):
            run = os.path.basename(path)[:-4]
            if not re.fullmatch(rf"{material}_\d+", run):
                continue
            data = pd.read_csv(path)
            area, err_stat, n = loop_area(data["V_x /V"].to_numpy(), data["V_y /V"].to_numpy())
            err_sys = area * np.sqrt(2) * GAIN_ACC   # A and B gain errors add in quadrature
            rows.append({"run": run, "cycles": n, "area /V^2": area,
                         "stat err /V^2": err_stat, "sys err /V^2": err_sys,
                         "total err /V^2": np.hypot(err_stat, err_sys)})

    results = pd.DataFrame(rows)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    out = os.path.join(OUT_DIR, "loop_areas.csv")
    results.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
