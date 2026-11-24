"""Convert V_y vs V_x runs to B vs H using config.toml, and compute hysteresis energy loss per cycle.

  H = N1 V_x / (R1 L)
  B = [R2 C V_y / N2 - mu0 H (A_coil - A)] / A
The second term removes flux through the air gap between sample and coil. It is linear in H, so it does not
change the loop area: energy loss per cycle per volume = (N1 R2 C / (R1 L N2 A)) * loop area in V^2.

Run:
  uv run python bh.py                          all single runs in data/
  uv run python bh.py data/iron_1.csv          specific runs
  uv run python bh.py data/cu_ni_timelapse_1   a timelapse folder (energy loss vs time)
"""

import glob
import os
import re
import sys
import tomllib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loop_area import cycle_starts, loop_area

MU0 = 4e-7 * np.pi
DATA_DIR = "data"

with open("config.toml", "rb") as f:
    CFG = tomllib.load(f)


def rel(pair):
    return pair[1] / pair[0]


def material_of(run):
    """Longest config area key that prefixes the run name."""
    matches = [m for m in CFG["area"] if run == m or run.startswith(m + "_")]
    if not matches:
        raise ValueError(f"No area in config.toml for run '{run}' (keys: {list(CFG['area'])})")
    return max(matches, key=len)


def convert(Va, Vb, material):
    """Returns H /A m^-1, B /T, loop energy loss /J m^-3 per cycle, and its stat and total uncertainty."""
    c = {k: v[0] for k, v in CFG["circuit"].items()}
    A, A_coil = CFG["area"][material][0], CFG["area"]["air"][0]

    # Integration constant is arbitrary and V_x should be AC, so centre both on whole cycles
    s = cycle_starts(Va)
    Va = Va - Va[s[0]:s[-1]].mean()
    Vb = Vb - Vb[s[0]:s[-1]].mean()

    H = c["N1"] * Va / (c["R1"] * c["L"])
    B = (c["R2"] * c["C"] * Vb / c["N2"] - MU0 * H * (A_coil - A)) / A

    k = c["N1"] * c["R2"] * c["C"] / (c["R1"] * c["L"] * c["N2"] * A)
    area, area_err, _ = loop_area(Va, Vb)
    rel_sys = np.sqrt(sum(rel(v) ** 2 for v in CFG["circuit"].values())
                      + rel(CFG["area"][material]) ** 2
                      + 2 * CFG["scope"]["gain_acc"] ** 2)
    energy = k * area
    energy_stat = k * area_err
    return H, B, energy, energy_stat, np.hypot(energy_stat, energy * rel_sys)


def fmt(value, err):
    """Value and error rounded to the error's first significant figure."""
    dp = -int(np.floor(np.log10(err)))
    return f"{round(value, dp):.{max(dp, 0)}f} ± {round(err, dp):.{max(dp, 0)}f}"


def plot_bh(ax, H, B, **kw):
    ax.plot(H / 1e3, B, linewidth=0.5, **kw)
    ax.set_xlabel("H /kA m⁻¹")
    ax.set_ylabel("B /T")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)


def do_run(csv_path):
    run = os.path.basename(csv_path)[:-4]
    material = material_of(run)
    data = pd.read_csv(csv_path)
    H, B, energy, _, energy_err = convert(data["V_x /V"].to_numpy(), data["V_y /V"].to_numpy(), material)

    base = csv_path[:-4]
    pd.DataFrame({"t /s": data["t /s"], "H /A m^-1": H, "B /T": B}).to_csv(f"{base}_BH.csv", index=False)

    fig, ax = plt.subplots()
    plot_bh(ax, H, B, color="darkgreen")
    ax.set_title(f"{run}: loss = {fmt(energy, energy_err)} J m⁻³ per cycle")
    fig.tight_layout()
    fig.savefig(f"{base}_BH.png", dpi=200)
    plt.close(fig)

    print(f"{run:24s} B_max {np.max(np.abs(B)):.3g} T  H_max {np.max(np.abs(H)) / 1e3:.3g} kA/m  "
          f"loss {fmt(energy, energy_err)} J/m^3/cycle")


def do_timelapse(folder):
    """Reads only captures already listed in log.csv, so it is safe to run while the timelapse is recording."""
    run = os.path.basename(os.path.normpath(folder))
    material = material_of(run)
    log = pd.read_csv(os.path.join(folder, "log.csv"))

    rows, loops = [], []
    for _, r in log.iterrows():
        data = pd.read_csv(os.path.join(folder, r["file"]))
        H, B, energy, energy_stat, energy_err = convert(data["V_x /V"].to_numpy(), data["V_y /V"].to_numpy(), material)
        rows.append({"capture": r["capture"], "elapsed /s": r["elapsed /s"], "wall time": r["wall time"],
                     "B_max /T": np.max(np.abs(B)), "H_max /A m^-1": np.max(np.abs(H)),
                     "loss /J m^-3": energy, "loss stat err /J m^-3": energy_stat, "loss err /J m^-3": energy_err})
        loops.append((r["elapsed /s"], H[:len(H) // 10], B[:len(B) // 10]))   # ~5 cycles is plenty to draw

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(folder, "BH_log.csv"), index=False)

    # Error bars are stat only: the systematic scale error moves every point together
    fig, ax = plt.subplots()
    ax.errorbar(out["elapsed /s"] / 60, out["loss /J m^-3"], yerr=out["loss stat err /J m^-3"],
                fmt="o-", color="darkgreen", markersize=3, linewidth=0.8, capsize=2)
    ax.set_xlabel("Elapsed time /min")
    ax.set_ylabel("Hysteresis loss /J m⁻³ per cycle")
    ax.set_title(f"{run} (±{100 * out['loss err /J m^-3'].iloc[0] / out['loss /J m^-3'].iloc[0]:.0f}% common scale error)")
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "BH_loss_vs_time.png"), dpi=200)
    plt.close(fig)

    picks = np.unique(np.linspace(0, len(loops) - 1, min(8, len(loops))).round().astype(int))
    t_max = max(loops[-1][0], 1)
    fig, ax = plt.subplots()
    for i in picks:
        t, H, B = loops[i]
        plot_bh(ax, H, B, color=plt.get_cmap("coolwarm")(t / t_max), label=f"{t / 60:.1f} min")
    ax.set_title(run)
    ax.legend(fontsize=7, title="Elapsed", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "BH_loops.png"), dpi=200)
    plt.close(fig)

    print(f"{run}: {len(out)} captures -> {folder}/BH_log.csv, BH_loss_vs_time.png, BH_loops.png")


def main():
    targets = sys.argv[1:] or sorted(p for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))
                                     if re.fullmatch(r"[a-z0-9_]+_\d+", os.path.basename(p)[:-4]))
    for t in targets:
        do_timelapse(t) if os.path.isdir(t) else do_run(t)


if __name__ == "__main__":
    main()
