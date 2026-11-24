import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ARM Mac driver path (also set via .env; harmless if both)
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/Library/Frameworks/PicoSDK.framework/Libraries/libps2000a"

from picosdk.ps2000a import ps2000a as ps
from picosdk.device import ChannelConfig, TimebaseOptions

RANGES = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]   # Allowed +/- ranges on the PicoScope (V, before x10 probe)


def measure(total_t, max_delta_t, A_range, B_range, A_coupling="DC", B_coupling="DC"):
    """
    Captures both channels. Returns downsampled t, V_a, V_b (scope volts, NOT x10) and per-channel
    peak |V| and clipped flag, both computed on the full-rate data.
    Clipping is detected from the data because picosdk's overflow flag never reports channel B (bit-shift bug).
    """
    timebase = TimebaseOptions(min_collection_time = total_t)
    channels = [ChannelConfig("A", True, A_coupling, A_range),
                ChannelConfig("B", True, B_coupling, B_range)]

    with ps.open_unit() as scope:
        times, voltages, _ = scope.capture_block(timebase, channels)

    ranges = {"A": A_range, "B": B_range}
    peak = {ch: float(np.max(np.abs(voltages[ch]))) for ch in ranges}
    clipped = {ch: peak[ch] >= 0.999*ranges[ch] for ch in ranges}   # Clipped samples read exactly +/- range

    downsample = max(1, int(max_delta_t/(times[1]-times[0])))
    return times[::downsample], voltages["A"][::downsample], voltages["B"][::downsample], peak, clipped


def measure_autorange(total_t, max_delta_t, A_coupling="DC", B_coupling="DC", headroom=0.8, max_tries=10):
    """
    Repeats the capture, moving each channel to the finest range whose full scale is above peak/headroom,
    until the ranges settle. A range that ever clipped is never used again, so it cannot oscillate.
    """
    ranges = {"A": 1, "B": 1}
    too_small = {"A": 0, "B": 0}   # Largest range seen to clip, per channel

    for attempt in range(max_tries):
        t, Va, Vb, peak, clipped = measure(total_t, max_delta_t, ranges["A"], ranges["B"], A_coupling, B_coupling)
        print(f"Try {attempt+1}: ranges {ranges}, peak {{A: {peak['A']:.3g}, B: {peak['B']:.3g}}} V, clipped {clipped}")

        new = {}
        for ch in ranges:
            if clipped[ch]:
                too_small[ch] = max(too_small[ch], ranges[ch])
            ok = [r for r in RANGES if r > too_small[ch] and (clipped[ch] or peak[ch] < headroom*r)]
            if not ok:
                raise RuntimeError(f"Channel {ch} clips even on the largest range")
            new[ch] = ok[0]

        if new == ranges and not any(clipped.values()):
            return t, Va, Vb, ranges
        ranges = new

    raise RuntimeError("Auto-range did not settle; check the signal is stable")


def next_free_csv(material):
    n = 1
    while os.path.exists(f"transformer_{material}_{n}.csv"):
        n += 1
    return f"transformer_{material}_{n}.csv"

material = "iron"   # e.g. "air", "iron", "mild_steel", "cu_ni"

t, Va, Vb, ranges = measure_autorange(total_t = 1.0, max_delta_t = 1e-5,   # Capture for 1s with point spacing at most 10µs
                                      A_coupling = "DC", B_coupling = "DC")

# Compensate for use of 10x probe setting
Va = 10*Va
Vb = 10*Vb

# Saved straight away to a new numbered file; mode="x" guarantees nothing is overwritten
saved_file = next_free_csv(material)
pd.DataFrame({"t /s": t, "V_x /V": Va, "V_y /V": Vb}).to_csv(saved_file, mode="x")
print(f"Final ranges {ranges} -> saved {saved_file}")

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = max(glob.glob("transformer_*.csv"), key=os.path.getmtime)   # Newest capture; or set e.g. "transformer_iron_2.csv"

data = pd.read_csv(csv_file)
t = data["t /s"].to_numpy()
Va = data["V_x /V"].to_numpy()
Vb = data["V_y /V"].to_numpy()

# Infer each channel's range from its ADC step: 8-bit scope gives step = range/127 (x10 for the probe)
def scope_range(v):
    step = np.median(np.diff(np.unique(v)))
    return min([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20], key=lambda r: abs(np.log(r/(step*12.7))))
A_range, B_range = scope_range(Va), scope_range(Vb)

# Clipping pins a channel at +/- its range (x10 for the probe); 0.999 absorbs float rounding
clip_a, clip_b = abs(Va) >= 0.999*10*A_range, abs(Vb) >= 0.999*10*B_range
print(f"{csv_file}: ranges A = ±{A_range} V, B = ±{B_range} V (scope setting); clipped samples: {np.sum(clip_a | clip_b)} of {len(Va)}")

fig1, (ax1a, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1b = ax1a.twinx()
ax1a.plot(t, Va, color="darkblue", linewidth=0.5)
ax1b.plot(t, Vb, color="darkred", linewidth=0.5)
ax1a.set_xlim(0, 0.1)   # First 5 cycles at 50 Hz; remove to see the full capture
ax1a.set_xlabel("t /s")
ax1a.set_ylabel("V_x /V", color="darkblue")
ax1b.set_ylabel("V_y /V", color="darkred")

ax2.plot(Va, Vb, color="darkgreen", linewidth=0.5)
ax2.set_xlabel("V_x /V")
ax2.set_ylabel("V_y /V")

fig1.suptitle(csv_file)
fig1.tight_layout()

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = max(glob.glob("transformer_*.csv"), key=os.path.getmtime)   # Newest capture; or set e.g. "transformer_iron_2.csv"

data = pd.read_csv(csv_file)
t = data["t /s"].to_numpy()
Va = data["V_x /V"].to_numpy()
Vb = data["V_y /V"].to_numpy()

# Infer each channel's range from its ADC step: 8-bit scope gives step = range/127 (x10 for the probe)
def scope_range(v):
    step = np.median(np.diff(np.unique(v)))
    return min([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20], key=lambda r: abs(np.log(r/(step*12.7))))
A_range, B_range = scope_range(Va), scope_range(Vb)

# Clipping pins a channel at +/- its range (x10 for the probe); 0.999 absorbs float rounding
clip_a, clip_b = abs(Va) >= 0.999*10*A_range, abs(Vb) >= 0.999*10*B_range
print(f"{csv_file}: ranges A = ±{A_range} V, B = ±{B_range} V (scope setting); clipped samples: {np.sum(clip_a | clip_b)} of {len(Va)}")

# Where a channel clips, drop the whole V_x span it clips over, else the noise band is cut one-sided and biases the fit
ok = ~clip_a
for clipped in (clip_b & (Vb < 0), clip_b & (Vb > 0)):
    if clipped.any():
        ok &= ~((Va >= Va[clipped].min()) & (Va <= Va[clipped].max()))
print(f"Excluded {np.sum(~ok)} of {len(ok)} samples in clipped regions")

(m, c), cov = np.polyfit(Va[ok], Vb[ok], 1, cov=True)
m_err_stat, c_err = np.sqrt(np.diag(cov))

# Scope DC gain accuracy per channel (check PicoScope 2000 series datasheet); A and B errors add in quadrature
gain_acc = 0.03
m_err_sys = abs(m) * np.sqrt(2) * gain_acc
m_err = np.hypot(m_err_stat, m_err_sys)

# Round gradient to the decimal place of the uncertainty's first significant figure
dp = max(0, -int(np.floor(np.log10(m_err))))
gradient_str = f"{m:.{dp}f} ± {m_err:.{dp}f}"

print(f"Gradient  = {gradient_str}  (stat ± {m_err_stat:.2g}, sys ± {m_err_sys:.2g})")
print(f"Intercept = {c:.5g} ± {c_err:.2g} V")

x_fit = np.array([Va[ok].min(), Va[ok].max()])

fig3, ax3 = plt.subplots()
ax3.plot(Va[ok], Vb[ok], color="darkgreen", linewidth=0.5, label="Data")
ax3.plot(x_fit, m*x_fit + c, color="black", label=f"Fit: gradient = {gradient_str}")
ax3.set_title(csv_file)
ax3.set_xlabel("V_x /V")
ax3.set_ylabel("V_y /V")
ax3.legend()
fig3.tight_layout()
