#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------
# ---------------------------------------------------------
# ### **Example Code:**
# ---------------------------------------------------------

# ##### **Importing libraries**
# 
# You will need to uncomment lines relevant to your operating system!

# In[20]:


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

# In[21]:


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

# In[22]:


# =============================================================================
# Capture data

A_range, B_range = 1, 1   # Channel A range is +/- 1V; Channel B range is +/- 50mV

t, Va, Vb, overflow = measure(total_t = 1.0, max_delta_t = 1e-5,     # Capture for 1s with point spacing at most 10µs
                              A_range = A_range, B_range = B_range,
                              A_coupling = "DC", B_coupling = "DC")  # Channels A and B both DC-coupled

print(f"Overflow warnings: {overflow}")


# Compensate for use of 10x probe setting
Va = 10*Va
Vb = 10*Vb


# In[23]:


# =============================================================================
# Plot time traces

fig1, ax1a = plt.subplots()

ax1b = ax1a.twinx()

ax1a.plot(t, Va, color="darkblue", linewidth=0.5)
ax1b.plot(t, Vb, color="darkred", linewidth=0.5)


# In[24]:


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
transformer_iron.to_csv("transformer_iron_1.csv", mode="x")  


# ---------------------------------------------------------
# ---------------------------------------------------------
# ### **Your Code Below:**
# ---------------------------------------------------------
# 

# In[48]:


# =============================================================================
# Mild steel X-Y plot from a saved CSV (hysteresis loop, so no straight-line fit)
# Self-contained: run this cell alone, no need to recapture

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = "transformer_iron_1.csv"
A_range, B_range = 1, 1   # Ranges used when THIS file was captured (needed to spot clipping)

data = pd.read_csv(csv_file)
Va = data["V_x /V"].to_numpy()
Vb = data["V_y /V"].to_numpy()

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
