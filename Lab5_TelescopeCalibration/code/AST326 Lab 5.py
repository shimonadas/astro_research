#!/usr/bin/env python
# coding: utf-8

# In[3]:


# importing the libraries 
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

# defining the current location 
obs_loc = EarthLocation(lat=43.7*u.deg, lon=-79.4*u.deg, height=76*u.m)

# create variable for observation time (9:30 AM EST = 14:30 UTC on February 24, 2025)
obs_start = Time('2025-02-24 14:30:00')

# defininf observation interval (around 5 mins each)
obs_interval = 1.5 * u.min

# define the Galactic coordinates (l, b) 
l_values = np.arange(3, 84, 3)  # every 3 degrees
b_values = np.zeros_like(l_values)  # all at b = 0

# now initialise lists for storing results
ra_list, dec_list, alt_list, az_list, time_list = [],[],[],[],[]

# converting Galactic to Equatorial and then to Horizon coordinates
for i, (l, b) in enumerate(zip(l_values, b_values)):
    # Galactic to Equatorial
    galactic_coord = SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic')
    equatorial_coord = galactic_coord.transform_to('icrs')  # to RA, Dec

    # the time for each pointing
    obs_time = obs_start + i * obs_interval
    # conversion to Horizon coordinates
    altaz_frame = AltAz(obstime=obstime, location=obs_loc)
    horizon_coord = equatorial_coord.transform_to(altaz_frame)
    # store results in empty lists 
    ra_list.append(equatorial_coord.ra.deg)
    dec_list.append(equatorial_coord.dec.deg)
    alt_list.append(horizon_coord.alt.deg)
    az_list.append(horizon_coord.az.deg)
    time_list.append(observation_time.iso)

# creating df
df_updated = pd.DataFrame({
    'Time (UTC)': time_list,
    'Galactic l (deg)': l_values,
    'Galactic b (deg)': b_values,
    'RA (deg)': ra_list,
    'Dec (deg)': dec_list,
    'Alt (deg)': alt_list,
    'Az (deg)': az_list
})
# Display the updated data
print(df_updated)


# In[5]:


import numpy as np
import struct

start_str = "output"
mid_str_1 = "000"
mid_str_2 = "00"
end_str = ".dat"
filenames = []

for i in range(28):
    if i < 10:
        filename = start_str + mid_str_1 + str(i) + end_str
    else:
        filename = start_str + mid_str_2 + str(i) + end_str
    filenames.append(filename)

for file in filenames:
    with open(file, 'rb') as f:
        nf, nt, alt, az = struct.unpack('iiff', f.read(16))
        lf = np.fromfile(f, dtype=np.float32, count=nf)
        lt = np.fromfile(f, dtype=np.float64, count=nt)  # Fix count
        sp = np.fromfile(f, dtype=np.float32, count=nf)
        bl = np.fromfile(f, dtype=np.float32, count=nf)
        wf = np.fromfile(f, dtype=np.float32, count=nf * nt)


# In[15]:


import numpy as np
import struct
import matplotlib.pyplot as plt

# Constants
c = 299792.458  # Speed of light in km/s
f0 = 1420.4  # Rest frequency of HI line in MHz

# File name format
start_str = "output"
mid_str_1 = "000"
mid_str_2 = "00"
end_str = ".dat"
filenames = []

for i in range(28):
    if i < 10:
        filename = start_str + mid_str_1 + str(i) + end_str
    else:
        filename = start_str + mid_str_2 + str(i) + end_str
    filenames.append(filename)

# Plot all spectra on the same plot
plt.figure(figsize=(10, 6))

for file in filenames:
    with open(file, 'rb') as f:
        nf, nt, alt, az = struct.unpack('iiff', f.read(16))
        lf = np.fromfile(f, dtype=np.float32, count=nf)  # Frequency list
        lt = np.fromfile(f, dtype=np.float64, count=nt)  # Time list
        sp = np.fromfile(f, dtype=np.float32, count=nf)  # Spectrum
        bl = np.fromfile(f, dtype=np.float32, count=nf)  # Baseline
        wf = np.fromfile(f, dtype=np.float32, count=nf * nt)  # Waterfall data

    # Remove polynomial baseline
    spectrum_corrected = sp - bl

    # Plot spectrum
    plt.plot(lf, spectrum_corrected, label=f'Alt: {alt:.1f}, Az: {az:.1f}')

# Plot settings
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (arbitrary units)")
plt.title("21cm Spectra from Different Pointings")
plt.ylim(-1.5,2.5)
plt.xlim(1415.5,1426)
# plt.legend(loc='upper right', fontsize='small', ncol=2)  # Adjust legend for readability
plt.grid()
# plt.gca().invert_xaxis()  # Invert x-axis so approaching gas is on the left
plt.show()



# In[13]:


import numpy as np
import struct
import matplotlib.pyplot as plt

# format of the file name 
start_str = "output"
mid_str_1 = "000"
mid_str_2 = "00"
end_str = ".dat"
filenames = []

# constants 
c = 299792.458  # light speed in km/s
f0 = 1420.4  # rest frequency of HI line in MHz

for i in range(28):
    if i < 10:
        filename = start_str + mid_str_1 + str(i) + end_str
    else:
        filename = start_str + mid_str_2 + str(i) + end_str
    filenames.append(filename)

# plot spectra on the same plot
plt.figure(figsize=(10, 6))
for file in filenames:
    with open(file, 'rb') as f:
        nf, nt, alt, az = struct.unpack('iiff', f.read(16))
        lf = np.fromfile(f, dtype=np.float32, count=nf)  # Frequency list
        lt = np.fromfile(f, dtype=np.float64, count=nt)  # Time list
        sp = np.fromfile(f, dtype=np.float32, count=nf)  # Spectrum
        bl = np.fromfile(f, dtype=np.float32, count=nf)  # Baseline
        wf = np.fromfile(f, dtype=np.float32, count=nf * nt)  # Waterfall data
    # removing polynomial fit baseline
    spectrum_corrected = sp-bl
    # convert frequency to radial velocity
    velocity = c*(f0-lf)/f0  # the doppler shift equation
    # plotting spectrum
    plt.plot(velocity, spectrum_corrected, label=f'Alt: {alt:.1f}, Az: {az:.1f}')

# plotting settings
plt.xlabel("Radial Velocity (km/s)")
plt.ylabel("Power (arbitrary units)")
plt.ylim(-2,2.5)
plt.title("21cm Spectra from Different Pointings")
plt.grid()
plt.gca().invert_xaxis()  # inverting x-axis so approaching gas comes from left
plt.show()


# In[1]:


import numpy as np
import struct
import matplotlib.pyplot as plt

# Constants
c = 299792.458  # Speed of light in km/s
f0 = 1420.4  # Rest frequency of HI line in MHz

# File name format
start_str = "output"
mid_str_1 = "000"
mid_str_2 = "00"
end_str = ".dat"
filenames = []

# Store data for plotting
velocity_list = []
latitude_list = []
flux_data = []

for i in range(28):
    if i < 10:
        filename = start_str + mid_str_1 + str(i) + end_str
    else:
        filename = start_str + mid_str_2 + str(i) + end_str
    filenames.append(filename)

# Read data and store for heatmap
for file in filenames:
    with open(file, 'rb') as f:
        nf, nt, alt, az = struct.unpack('iiff', f.read(16))
        lf = np.fromfile(f, dtype=np.float32, count=nf)  # Frequency list
        lt = np.fromfile(f, dtype=np.float64, count=nt)  # Time list
        sp = np.fromfile(f, dtype=np.float32, count=nf)  # Spectrum
        bl = np.fromfile(f, dtype=np.float32, count=nf)  # Baseline
        wf = np.fromfile(f, dtype=np.float32, count=nf * nt)  # Waterfall data

    # Remove polynomial baseline
    spectrum_corrected = sp - bl

    # Convert frequency to radial velocity
    velocity = c * (f0 - lf) / f0  # Doppler shift equation

    # Store values for heatmap
    velocity_list = velocity  # Store once (same for all observations)
    latitude_list.append(alt)  # Assuming altitude represents Galactic latitude
    flux_data.append(spectrum_corrected)  # Append corrected flux

# Convert to numpy arrays for plotting
latitude_list = np.array(latitude_list)
flux_data = np.array(flux_data)  # Shape: (num_latitudes, num_velocities)

# Plot 2D heatmap
plt.figure(figsize=(10, 6))
plt.imshow(flux_data, aspect='auto', cmap='inferno', 
           extent=[velocity_list.min(), velocity_list.max(), latitude_list.min(), latitude_list.max()],
           origin='lower')

# Plot settings
plt.colorbar(label="Flux (arbitrary units)")
plt.xlabel("Doppler Velocity (km/s)")
plt.ylabel("Galactic Latitude (°)")
plt.title("HI 21cm Flux as a Function of Doppler Velocity and Galactic Latitude")
plt.gca().invert_xaxis()  # Invert x-axis for standard convention
plt.show()


# In[16]:


import numpy as np
import struct
import matplotlib.pyplot as plt

# Constants
c = 299792.458  # Speed of light in km/s
f0 = 1420.4  # Rest frequency of HI line in MHz

# File naming pattern
start_str = "output"
mid_str_1 = "000"
mid_str_2 = "00"
end_str = ".dat"
filenames = []
galactic_longitudes = np.linspace(3, 90, 28)  # Example Galactic longitudes (modify as needed)

# Data storage
velocity_list = None
flux_data = []

for i in range(28):
    if i < 10:
        filename = start_str + mid_str_1 + str(i) + end_str
    else:
        filename = start_str + mid_str_2 + str(i) + end_str
    filenames.append(filename)

# Read data and prepare for plotting
for file in filenames:
    with open(file, 'rb') as f:
        nf, nt, alt, az = struct.unpack('iiff', f.read(16))
        lf = np.fromfile(f, dtype=np.float32, count=nf)  # Frequency list
        lt = np.fromfile(f, dtype=np.float64, count=nt)  # Time list
        sp = np.fromfile(f, dtype=np.float32, count=nf)  # Spectrum
        bl = np.fromfile(f, dtype=np.float32, count=nf)  # Baseline
        wf = np.fromfile(f, dtype=np.float32, count=nf * nt)  # Waterfall data

    # Remove baseline
    spectrum_corrected = sp - bl

    # Convert frequency to velocity
    velocity = c * (f0 - lf) / f0  # Doppler shift equation

    # Store the first velocity list (same for all)
    if velocity_list is None:
        velocity_list = velocity

    # Store flux values
    flux_data.append(spectrum_corrected)

# Convert to numpy array
flux_data = np.array(flux_data)  # Shape: (num_longitudes, num_velocities)

# Plot 2D heatmap
plt.figure(figsize=(10, 6))
plt.imshow(flux_data, aspect='auto', cmap='turbo',
           extent=[galactic_longitudes.min(), galactic_longitudes.max(), velocity_list.min(), velocity_list.max()],
           origin='lower')

# Labels and colorbar
plt.colorbar(label="Flux (arbitrary units)")
plt.xlabel("Galactic Longitude (°)")
plt.ylabel("Velocity (km/s)")
plt.title("21cm HI Emission: Doppler Velocity vs. Galactic Longitude")
plt.gca().invert_yaxis()  # Invert y-axis to match convention
plt.show()


# In[6]:


# import numpy as np
# import struct
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# import pandas as pd

# # Constants
# c = 299792.458  # Speed of light in km/s
# f0 = 1420.4  # Rest frequency of HI line in MHz

# # File name format
# start_str = "output"
# mid_str_1 = "000"
# mid_str_2 = "00"
# end_str = ".dat"
# filenames = []
# galactic_longitudes = np.linspace(3, 84, 28)  # Example Galactic longitudes (modify as needed)

# # Data storage
# peak_data = []

# # Generate file names
# for i in range(28):
#     if i < 10:
#         filename = start_str + mid_str_1 + str(i) + end_str
#     else:
#         filename = start_str + mid_str_2 + str(i) + end_str
#     filenames.append(filename)

# # Plot all spectra on the same plot
# plt.figure(figsize=(10, 6))

# for i, file in enumerate(filenames):
#     with open(file, 'rb') as f:
#         nf, nt, alt, az = struct.unpack('iiff', f.read(16))
#         lf = np.fromfile(f, dtype=np.float32, count=nf)  # Frequency list
#         lt = np.fromfile(f, dtype=np.float64, count=nt)  # Time list
#         sp = np.fromfile(f, dtype=np.float32, count=nf)  # Spectrum
#         bl = np.fromfile(f, dtype=np.float32, count=nf)  # Baseline
#         wf = np.fromfile(f, dtype=np.float32, count=nf * nt)  # Waterfall data

#     # Remove polynomial baseline
#     spectrum_corrected = sp - bl

#     # Convert frequency to radial velocity
#     velocity = c * (f0 - lf) / f0  # Doppler shift equation

#     # Find peaks in the spectrum
#     peaks, _ = find_peaks(spectrum_corrected, height=0.2)  # Adjust height threshold if needed

#     # Store peak data
#     for peak in peaks:
#         peak_data.append({
#             "Galactic Longitude": galactic_longitudes[i],
#             "Altitude": alt,
#             "Azimuth": az,
#             "Radial Velocity (km/s)": velocity[peak],
#             "Power": spectrum_corrected[peak]
#         })

#     # Plot spectrum
#     plt.plot(velocity, spectrum_corrected, label=f'Alt: {alt:.1f}, Az: {az:.1f}')

# # Plot settings
# plt.xlabel("Radial Velocity (km/s)")
# plt.ylabel("Power (arbitrary units)")
# plt.ylim(-2, 2.5)
# plt.title("21cm Spectra from Different Pointings")
# plt.grid()
# plt.gca().invert_xaxis()  # Invert x-axis so approaching gas is on the left
# plt.show()

# # Convert peak data to DataFrame
# df_peaks = pd.DataFrame(peak_data)

# # Save to CSV (optional, useful for further analysis)
# df_peaks.to_csv("HI_21cm_Peak_Data.csv", index=False)

# # Display the first few rows
# print(df_peaks.head())  # Shows the first few peaks found

# # If using Jupyter Notebook, uncomment the line below for a nicer display:
# # display(df_peaks)


# In[20]:


import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd

# ---------------------------
# 1. Constants & Configuration
# ---------------------------
c = 299792.458        # Speed of light in km/s
f0 = 1420.4           # Rest frequency of HI line in MHz

R0 = 8.2              # kpc, Sun's distance from Galactic center
V0 = 220.0            # km/s, local circular speed at R0

# Adjust these as needed
VEL_MIN = -300.0      # km/s
VEL_MAX =  300.0      # km/s

# Example file naming scheme & galactic longitudes
start_str = "output"
mid_str_1 = "000"
mid_str_2 = "00"
end_str = ".dat"

n_files = 28
galactic_longitudes = np.linspace(3, 84, n_files)  # 0°, 3°, 6°, ... 84° (example)
filenames = []
for i in range(n_files):
    if i < 10:
        filename = start_str + mid_str_1 + str(i) + end_str
    else:
        filename = start_str + mid_str_2 + str(i) + end_str
    filenames.append(filename)

# ---------------------------
# 2. Data Storage
# ---------------------------
all_peaks_data = []   # Will store all peaks, if you want multiple arms
tp_data = []          # Will store only the tangent-point solution (max velocity) for 0<l<90

# ---------------------------
# 3. Plotting Setup for Spectra
# ---------------------------
plt.figure(figsize=(10, 6))

for i, file in enumerate(filenames):
    # Read the file
    with open(file, 'rb') as f:
        nf, nt, alt, az = struct.unpack('iiff', f.read(16))
        lf = np.fromfile(f, dtype=np.float32, count=nf)  # freq array
        lt = np.fromfile(f, dtype=np.float64, count=nt)  # time array
        sp = np.fromfile(f, dtype=np.float32, count=nf)  # average spectrum
        bl = np.fromfile(f, dtype=np.float32, count=nf)  # baseline
        wf = np.fromfile(f, dtype=np.float32, count=nf*nt)  # waterfall

    # Subtract baseline
    spectrum_corrected = sp - bl

    # Convert frequencies to velocities
    velocity = c * (f0 - lf) / f0  # km/s

    # ---------------------------
    # 3A. Restrict to ±300 km/s
    # ---------------------------
    mask = (velocity >= VEL_MIN) & (velocity <= VEL_MAX)
    velocity_masked = velocity[mask]
    spectrum_masked = spectrum_corrected[mask]

    # ---------------------------
    # 3B. Find peaks in masked region
    # ---------------------------
    # Adjust height, prominence, distance as needed
    peaks, peak_props = find_peaks(
        spectrum_masked,
        height=1.0,         # e.g., min peak height
        prominence=0.5,     # e.g., how "sharp" the peak must be
        distance=5          # e.g., min separation in index units
    )

    # Store all peaks (optional)
    l_deg = galactic_longitudes[i]
    l_rad = np.radians(l_deg)
    for pk in peaks:
        v_los = velocity_masked[pk]
        power = spectrum_masked[pk]

        all_peaks_data.append({
            "Galactic Longitude (deg)": l_deg,
            "Altitude": alt,
            "Azimuth": az,
            "Radial Velocity (km/s)": v_los,
            "Power": power
        })

    # ---------------------------
    # 3C. Tangent-point method (first quadrant)
    # ---------------------------
    # For 0° < l < 90°, pick the maximum velocity peak as the tangent point.
    # (If l=0 or l=90 exactly, or if no valid peak, skip.)
    if 0 < l_deg < 90 and len(peaks) > 0:
        # Find the index of the maximum velocity among the detected peaks
        max_peak_idx = peaks[np.argmax(velocity_masked[peaks])]
        v_los_tangent = velocity_masked[max_peak_idx]

        # Tangent-point radius and rotation speed
        R_tangent = R0 * np.sin(l_rad)
        V_tangent = v_los_tangent + V0 * np.sin(l_rad)

        tp_data.append({
            "Galactic Longitude (deg)": l_deg,
            "Altitude": alt,
            "Azimuth": az,
            "Radial Velocity (km/s)": v_los_tangent,
            "R_kpc": R_tangent,
            "V_km_s": V_tangent
        })

    # ---------------------------
    # 3D. Plot the masked spectrum
    # ---------------------------
#     plt.plot(velocity_masked, spectrum_masked,
#              label=f'l={l_deg:.1f}°, alt={alt:.1f}, az={az:.1f}')

# ---------------------------
# 4. Save the All-Peaks Data
# ---------------------------
# df_all = pd.DataFrame(all_peaks_data)
# df_all.to_csv("HI_21cm_All_Peaks.csv", index=False)
# print("All peaks data (first few rows):")
# print(df_all.head())

# ---------------------------
# 5. Tangent-Point Rotation Curve
# ---------------------------
df_tp = pd.DataFrame(tp_data)
df_tp.sort_values(by="R_kpc", inplace=True)  # sort by radius if you like
# df_tp.to_csv("HI_21cm_TangentPoint.csv", index=False)
# print("\nTangent-point data (first few rows):")
# print(df_tp.head())

# Plot the rotation curve
plt.figure(figsize=(8, 6))
plt.scatter(df_tp["R_kpc"], df_tp["V_km_s"],
            color='red', edgecolor='k', alpha=0.7, label='Tangent-Point Data')
plt.xlabel("Galactocentric Radius (kpc)")
plt.ylabel("Rotation Speed (km/s)")
plt.title("Galactic Rotation Curve (Tangent-Point Method)")
plt.grid(True)
plt.legend()
plt.show()


# In[3]:


# import the libraries 
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd

R0 = 8.2 # kpc, distance of Sun from Galactic center
V0 = 220.0 # km/s, local circular speed at R0
VEL_MIN = -300.0
VEL_MAX =  300.0

# naming scheme & galactic longitudes
start_str = "output"
mid_str_1 = "000"
mid_str_2 = "00"
end_str = ".dat"
n_files = 28
galactic_longitudes = np.linspace(3, 84, n_files)  # e.g. 3° to 84° in steps
filenames = []
for i in range(n_files):
    if i < 10:
        filename = f"{start_str}{mid_str_1}{i}{end_str}"
    else:
        filename = f"{start_str}{mid_str_2}{i}{end_str}"
    filenames.append(filename)

# storing the data 
tp_data = []

# processing each file and apply Tangent Point Method
for i, file in enumerate(filenames):
    with open(file, 'rb') as f:
        nf, nt, alt, az = struct.unpack('iiff', f.read(16))
        lf = np.fromfile(f, dtype=np.float32, count=nf)
        lt = np.fromfile(f, dtype=np.float64, count=nt)
        sp = np.fromfile(f, dtype=np.float32, count=nf)
        bl = np.fromfile(f, dtype=np.float32, count=nf)
        wf = np.fromfile(f, dtype=np.float32, count=nf * nt)
    # subtraction of polynomial baseline 
    spec2 = sp-bl
    # convert frequency to velocity (km/s)
    velocity = c*(f0-lf)/f0

    # Restrict to ±300 km/s
    mask = (velocity >= VEL_MIN) & (velocity <= VEL_MAX)
    velocity_masked = velocity[mask]
    spec_masked = spec2[mask]

    # Find peaks in the masked region
    peaks, peak_props = find_peaks(
        spec_masked,
        height=1.0,
        prominence=0.5,
        distance=5
    )
    # For tangent-point method (first quadrant): choose maximum velocity peak
    l_deg = galactic_longitudes[i]
    l_rad = np.radians(l_deg)
    if 0 < l_deg < 90 and len(peaks) > 0:
        max_peak_idx = peaks[np.argmax(velocity_masked[peaks])]
        v_los_tangent = velocity_masked[max_peak_idx]
        R_tangent = R0 * np.sin(l_rad)
        V_tangent = v_los_tangent + V0 * np.sin(l_rad)
        tp_data.append({
            "Galactic Longitude (deg)": l_deg,
            "Altitude": alt,
            "Azimuth": az,
            "Radial Velocity (km/s)": v_los_tangent,
            "R_kpc": R_tangent,
            "V_km_s": V_tangent
        })
        
# build df and sort it 
df_tp = pd.DataFrame(tp_data)
df_tp.sort_values(by="R_kpc", inplace=True)

# plot Rotation Curve and Fit Polynomial
plt.figure(figsize=(8, 6))
plt.scatter(df_tp["R_kpc"], df_tp["V_km_s"], color='red', edgecolor='k', alpha=0.7, label='Tangent Point Data')
plt.xlabel("Galactocentric Radius (kpc)")
plt.ylabel("Rotation Speed (km/s)")
plt.title("Galactic Rotation Curve (Tangent-Point Method)")
plt.grid()
# connecting the lines 
plt.plot(df_tp["R_kpc"], df_tp["V_km_s"], color='red', alpha=0.4, linestyle='--', label='Connected Points')

# plonomial cubic fit 
x = df_tp["R_kpc"].values
y = df_tp["V_km_s"].values
if len(x) > 3:
    coeffs = np.polyfit(x, y, deg=3)  # cubic fit
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = np.polyval(coeffs, x_fit)
    plt.plot(x_fit, y_fit, 'b-', label='Polynomial Fit (deg=3)')
plt.legend()
plt.show()

# making the Residual Graph
if len(x) > 3:
    # observed - predicted 
    residuals = y - np.polyval(coeffs, x)
    plt.figure(figsize=(8, 6))
    plt.scatter(x, residuals, color='blue', label='Residuals', edgecolor='k')
    plt.axhline(0, color='gray', linestyle='--', label='Zero Residual')
    plt.xlabel("Galactocentric Radius (kpc)")
    plt.ylabel("Residual (km/s)")
    plt.title("Residuals of the Polynomial Fit to the Rotation Curve")
    plt.grid()
    plt.legend()
    plt.show()


# In[3]:


import numpy as np
import struct
import matplotlib.pyplot as plt

# ---------------------------
# 1. Constants & Helper Functions
# ---------------------------
c = 299792.458  # Speed of light in km/s
f0 = 1420.4     # Rest frequency of the HI line in MHz

def freq_to_vel(freq_mhz):
    """
    Convert Frequency (MHz) to Radial Velocity (km/s),
    using v = c * (f0 - f) / f0.
    """
    return c * (f0 - freq_mhz) / f0

def vel_to_freq(vel_km_s):
    """
    Inverse of freq_to_vel:
    Convert Velocity (km/s) back to Frequency (MHz).
    """
    return f0 - (vel_km_s * f0 / c)

# ---------------------------
# 2. File Naming & Longitude Array
# ---------------------------
start_str = "output"
mid_str_1 = "000"
mid_str_2 = "00"
end_str = ".dat"

n_files = 28
filenames = []

for i in range(n_files):
    if i < 10:
        filename = f"{start_str}{mid_str_1}{i}{end_str}"
    else:
        filename = f"{start_str}{mid_str_2}{i}{end_str}"
    filenames.append(filename)

# Example: 28 pointings from l=0° to l=84° in steps of 3°
galactic_longitudes = np.linspace(0, 84, n_files)

# ---------------------------
# 3. Read & Stack Spectra
# ---------------------------
all_spectra = None
freq_array = None

for i, file in enumerate(filenames):
    with open(file, 'rb') as f:
        # Read header info
        nf, nt, alt, az = struct.unpack('iiff', f.read(16))
        lf = np.fromfile(f, dtype=np.float32, count=nf)  # Frequency array
        lt = np.fromfile(f, dtype=np.float64, count=nt)  # Time array
        sp = np.fromfile(f, dtype=np.float32, count=nf)  # Averaged spectrum
        bl = np.fromfile(f, dtype=np.float32, count=nf)  # Baseline
        wf = np.fromfile(f, dtype=np.float32, count=nf * nt)  # Waterfall data (unused here)

    # Subtract baseline from the averaged spectrum
    spectrum_corrected = sp - bl

    # On first file, initialize the 2D array
    if all_spectra is None:
        all_spectra = np.zeros((n_files, nf), dtype=np.float32)
        freq_array = lf  # Store the frequency axis from the first file
    else:
        # Optionally check that lf == freq_array if you want to ensure consistency
        pass

    # Store this pointing's corrected spectrum
    all_spectra[i, :] = spectrum_corrected

# ---------------------------
# 4. 2D Waterfall Plot
# ---------------------------
# We'll plot: X = galactic_longitudes, Y = freq_array, Color = power
# Then we add a secondary Y-axis for velocity.

fig, ax_freq = plt.subplots(figsize=(8, 6))

mesh = ax_freq.pcolormesh(
    galactic_longitudes,      # X-axis
    freq_array,               # Y-axis (Frequency)
    all_spectra.T,            # Z = power, shape = (nf, n_files)
    shading='auto',
    cmap='viridis'
)

# Primary y-axis in Frequency
ax_freq.set_xlabel("Galactic Longitude [deg]")
ax_freq.set_ylabel("Frequency [MHz]")
ax_freq.set_ylim(1419.5, 1422.0)

# Secondary y-axis in Velocity
ax_vel = ax_freq.secondary_yaxis('right', functions=(freq_to_vel, vel_to_freq))
ax_vel.set_ylabel("Velocity [km/s]")

# Colorbar for the power scale
cbar = plt.colorbar(mesh, ax=ax_freq, pad=0.01)
cbar.set_label("Power (arbitrary units)")

plt.title("21cm Waterfall Plot")
plt.tight_layout()
plt.show()


# In[ ]:




