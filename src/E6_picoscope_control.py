#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------
# ---------------------------------------------------------
# ### **Example Code:**
# ---------------------------------------------------------

# ##### **Importing libraries**
# 
# You will need to uncomment lines relevant to your operating system!

# In[1]:


# =============================================================================
# Importing basic libraries

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import os


# =============================================================================
# Path to PicoScope drivers
# TODO: UNCOMMENT THE LINES RELEVANT TO YOUR LAPTOP OPERATING SYSTEM

# Uncomment these three lines for Windows
# pico_dir = r"C:\Program Files\Pico Technology\PicoScope 7 T&M Stable"  # Change this to your PicoScope 7 install location, if needed
# os.environ["PATH"] = pico_dir + os.pathsep + os.environ["PATH"]
# os.add_dll_directory(pico_dir)

# Uncomment this line for Macs with intel chips (i.e. NOT the M1, M2 etc. chips)
# os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/Applications/PicoScope 7 T&M.app/Contents/Resources"

# Uncomment this line for Mac with ARM64 architecture (M1, M2, etc. chips). Make sure you have installed the ARM64 compatible SDK via the link above!
# os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/Library/Frameworks/PicoSDK.framework/Libraries/libps2000a"

# Our PicoScope model
from picosdk.ps2000a import ps2000a as ps
from picosdk.device import ChannelConfig, TimebaseOptions



# ##### **Defining a function to capture data**

# In[2]:


# =============================================================================
# Defining a measurement function

def measure(total_t,         max_delta_t,
            A_range,         B_range,           # Allowed values are 20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02 (Possible +/- ranges on PicoScope)
            A_coupling="DC", B_coupling="DC",   # Allowed values are "DC" or "AC"
            ):
    """
    Measures both channels in the PicoScope
    Returns the (downsampled) times, channel voltages, and any overflow warnings that may have arisen (will warn you of any clipping!)
    Note: This does NOT compensate for use of a 10x probe! If you are using the 10x setting, you need to multiply the output voltages by 10 to get the true voltage

    total_t: The (minimum) total measurment time in seconds. True value may end up slightly larger!

    max_delta_t: The maximum time spacing between samples in seconds. PicoScope will capture as fast as it can, then the data will be downsampled until it has a time spacing close to the value you specify here

    A_range, B_range: The voltage ranges in channels A and B in Volts, which can only take discrete values as listed above

    A_coupling, B_coupling: DC or AC coupling in each channel; defaults to "DC"

    """

    timebase = TimebaseOptions(min_collection_time = total_t)
    
    channels = [ChannelConfig("A", True, A_coupling, A_range),
                ChannelConfig("B", True, B_coupling, B_range)]


    with ps.open_unit() as scope:

        times, voltages, overflow_warnings = scope.capture_block(timebase, channels)


    delta_t = times[1]-times[0]
    downsample = int(max_delta_t/delta_t)

    t   = times[::downsample]
    V_a = voltages["A"][::downsample]   # Assumes using a 10x probe
    V_b = voltages["B"][::downsample]

    return t, V_a, V_b, overflow_warnings



# ##### **Example capture and plots**

# In[ ]:


# =============================================================================
# Capture data

A_range, B_range = 1, 0.05   # Channel A range is +/- 1V; Channel B range is +/- 50mV

t, Va, Vb, overflow = measure(total_t = 1.0, max_delta_t = 1e-5,     # Capture for 1s with point spacing at most 10µs
                              A_range = A_range, B_range = B_range,
                              A_coupling = "DC", B_coupling = "DC")  # Channels A and B both DC-coupled

print(f"Overflow warnings: {overflow}")


# Compensate for use of 10x probe setting
Va = 10*Va
Vb = 10*Vb


# In[9]:


# =============================================================================
# Plot time traces

fig1, ax1a = plt.subplots()

ax1b = ax1a.twinx()

ax1a.plot(t, Va, color="darkblue", linewidth=0.5)
ax1b.plot(t, Vb, color="darkred", linewidth=0.5)


# In[10]:


# =============================================================================
# Plot X-Y

fig2, ax2 = plt.subplots()

ax2.plot(Va, Vb, color="darkgreen", linewidth=0.5)


# ##### **Save data**

# In[ ]:


# =============================================================================
# Saves data to a dataframe, then records it in a CSV file

transformer_iron = pd.DataFrame({"t /s":   t,
                                 "V_x /V": Va,   # Assumes V_x is in Channel A and V_y is in Channel B
                                 "V_y /V": Vb})  


# CSV will be saved to your current active directory; to choose this yourself, open a directory using "File" >> "Open Folder"
# Will fail if file already exists, to prevent accidentally overwriting saved data. If you want to overwrite, remove the mode="x" argument
transformer_iron.to_csv("transformer_air_1.csv", mode="x")  


# ---------------------------------------------------------
# ---------------------------------------------------------
# ### **Your Code Below:**
# ---------------------------------------------------------
# 

# In[7]:


# =============================================================================
# Gradient of X-Y plot from a saved CSV (least-squares straight-line fit of V_y against V_x)
# Self-contained: run this cell alone, no need to recapture

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = "transformer_air_1.csv"

data = pd.read_csv(csv_file)
Va = data["V_x /V"].to_numpy()
Vb = data["V_y /V"].to_numpy()

# Infer each channel's range from its ADC step: 8-bit scope gives step = range/127 (x10 for the probe)
def scope_range(v):
    step = np.median(np.diff(np.unique(v)))
    return min([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20], key=lambda r: abs(np.log(r/(step*12.7))))
A_range, B_range = scope_range(Va), scope_range(Vb)
print(f"Inferred ranges: A = ±{A_range} V, B = ±{B_range} V (scope setting, before x10)")

# Exclude clipping: a channel pins at +/- its range (x10 for the probe); 0.999 absorbs float rounding.
# Where V_y clips, drop the whole V_x span it clips over, else the noise band is cut one-sided and biases the fit.
lim_a, lim_b = 0.999*10*A_range, 0.999*10*B_range
ok = abs(Va) < lim_a
for clipped in (Vb <= -lim_b, Vb >= lim_b):
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


# In[8]:


# =============================================================================
# Hysteresis loop X-Y plot from a saved CSV (iron, mild steel, ...) (hysteresis loop, so no straight-line fit)
# Self-contained: run this cell alone, no need to recapture

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = "transformer_mild_steel_1.csv"

data = pd.read_csv(csv_file)
Va = data["V_x /V"].to_numpy()
Vb = data["V_y /V"].to_numpy()

# Infer each channel's range from its ADC step: 8-bit scope gives step = range/127 (x10 for the probe)
def scope_range(v):
    step = np.median(np.diff(np.unique(v)))
    return min([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20], key=lambda r: abs(np.log(r/(step*12.7))))
A_range, B_range = scope_range(Va), scope_range(Vb)
print(f"Inferred ranges: A = ±{A_range} V, B = ±{B_range} V (scope setting, before x10)")

# Clipping pins a channel at +/- its range (x10 for the probe); 0.999 absorbs float rounding
n_clipped = np.sum((abs(Va) >= 0.999*10*A_range) | (abs(Vb) >= 0.999*10*B_range))
print(f"Clipped samples: {n_clipped} of {len(Va)}")

fig4, ax4 = plt.subplots()
ax4.plot(Va, Vb, color="darkgreen", linewidth=0.5)
ax4.set_title(csv_file)
ax4.set_xlabel("V_x /V")
ax4.set_ylabel("V_y /V")
fig4.tight_layout()


plt.show()
