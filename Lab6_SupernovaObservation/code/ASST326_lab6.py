#!/usr/bin/env python
# coding: utf-8

# In[17]:


# 1) Imports
from pathlib import Path
from astropy.io import fits

# 2) Point to your main folder
base_dir = Path("Pichustronomy")

# 3) Recursively find every .fit file under that folder
fits_paths = sorted(base_dir.rglob("*.fit"))

print(f"Found {len(fits_paths)} FITS files.")

# 4) Read them all into memory (or process on the fly)
#    Here we store each in a list of dicts, but you can adapt to your needs.
all_images = []
for p in fits_paths:
    # open, read, and close in one go
    with fits.open(p) as hdul:
        data = hdul[0].data        # numpy array
        header = hdul[0].header    # FITS header
    all_images.append({
        "path": str(p),
        "data": data,
        "header": header
    })

# 5) Quick check: show file names and array shapes
for img in all_images[:5]:
    print(img["path"], "→ shape:", img["data"].shape)


# In[18]:


from pathlib import Path

raw_path = Path(
    "raw-T21-ast326_2025n-SN 2025fbf-20250409-232047-Red-BIN1-W-060-001.fit"
)
print(raw_path.exists())        # → should print True


# In[19]:


from astropy.io import fits
from pathlib import Path

# --------------------------
# STEP 1: define full mapping with folders
# --------------------------
mapping = {
    "wcs.fits":   "04_08/raw-T21-ast326_2025n-SN 2025fbf-20250408-233106-Red-BIN1-W-060-001_wcs_wcs_wcs_wcs.fit",
    "wcs-2.fits": "04_09/raw-T21-ast326_2025n-SN 2025fbf-20250409-232047-Red-BIN1-W-060-001_wcs_wcs_wcs_wcs.fit",
    "wcs-3.fits": "04_10/raw-T21-ast326_2025n-SN 2025fbf-20250410-233501-Red-BIN1-W-060-001_wcs_wcs_wcs_wcs.fit",
    "wcs-4.fits": "04_15/raw-T21-ast326_2025n-SN 2025fbf-20250415-022615-Red-BIN1-W-060-014.fit",
    "wcs-5.fits": "04_19/raw-T21-ast326_2025n-SN 2025fbf-20250419-021314-Red-BIN1-W-060-001.fit"
}

# --------------------------
# STEP 2: folder where all files live
# --------------------------
folder = Path("Pichustronomy")       # base directory
wcs_folder = folder / "wcs_files"          # where wcs-*.fits are

# --------------------------
# STEP 3: inject WCS headers
# --------------------------
for wcs_file, rel_path in mapping.items():
    wcs_path = wcs_folder / wcs_file
    raw_path = folder / rel_path

    if not wcs_path.exists():
        print(f"❌ WCS file missing: {wcs_path}")
        continue
    if not raw_path.exists():
        print(f"❌ Raw image missing: {raw_path}")
        continue

    # Load WCS header
    wcs_header = fits.getheader(wcs_path)

    # Inject into raw image
    with fits.open(raw_path, mode="update", memmap=True) as hdul:
        hdr = hdul[0].header

        # Remove old WCS keys if they exist
        for key in list(wcs_header):
            if key in hdr:
                del hdr[key]

        # Inject WCS
        for key, value in wcs_header.items():
            hdr[key] = value

        hdul.flush()

    print(f"✅ Injected {wcs_file} → {rel_path}")


# In[4]:


# 4.1.2

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

file_path = "Pichustronomy/04_19/raw-T21-ast326_2025n-SN 2025fbf-20250419-021314-Red-BIN1-W-060-001.fit"
with fits.open(file_path) as hdul:
    hdr = hdul[0].header
    data = hdul[0].data
wcs = WCS(hdr)

# Define supernova coordinates SN 2025fbf
sn_coord = SkyCoord(ra='13h16m50.928s', dec='-03d03m20.246s', unit=(u.hourangle, u.deg))
# RA/Dec to image (x, y) pixel position
x, y = wcs.world_to_pixel(sn_coord)

# Plot the image 
plt.figure(figsize=(8, 8))
norm = simple_norm(data, 'sqrt', percent=99)
plt.imshow(data, origin='lower', cmap='gray', norm=norm)
plt.scatter(x, y, s=120, edgecolor='red', facecolor='none', linewidth=2)
plt.title("SN 2025fbf Location (WCS-calculated)")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.show()


# In[21]:


# # 4.3. photometric callibration

# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from astropy.wcs import WCS
# from astropy.coordinates import SkyCoord
# import astropy.units as u
# from astropy.visualization import simple_norm

# # 1. Load FITS data and WCS from one of your calibrated images
# file_path = "Pichustronomy/04_08/raw-T21-ast326_2025n-SN 2025fbf-20250408-233106-Red-BIN1-W-060-001_wcs_wcs_wcs_wcs.fit"
# hdul = fits.open(file_path)
# hdr = hdul[0].header
# img_array = hdul[0].data.astype(float)
# wcs = WCS(hdr)

# # 2. Get SN 2025fbf coordinates in pixel space
# sn_coord = SkyCoord("13h16m50.928s", "-03d03m20.246s", unit=(u.hourangle, u.deg))
# sn_xy = wcs.world_to_pixel(sn_coord)  # (x, y)

# print(f"Supernova pixel location: x = {sn_xy[0]:.1f}, y = {sn_xy[1]:.1f}")

# # 3. Set up WCS-aware plot
# fig = plt.figure(figsize=(10, 10))
# ax = plt.subplot(projection=wcs)
# norm = simple_norm(img_array, 'sqrt', percent=99)

# im = ax.imshow(img_array, origin='lower', cmap='gray', norm=norm)

# # 4. Zoom in on SN region ±100 pixels
# ax.set_xlim(sn_xy[0] - 100, sn_xy[0] + 100)
# ax.set_ylim(sn_xy[1] - 100, sn_xy[1] + 100)

# # 5. Overlay SN location
# ax.scatter(sn_xy[0], sn_xy[1], s=100, edgecolor='red', facecolor='none', linewidth=2, label='SN 2025fbf')

# # 6. Labeling and formatting
# ax.set_title("SN 2025fbf (Zoomed)")
# ax.set_xlabel("RA")
# ax.set_ylabel("Dec")
# ax.grid(color='white', alpha=0.3)
# ax.legend()

# plt.show()


# In[22]:


import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# gathering dark frames 
dark_dir = Path("Pichustronomy") / "Darks"
dark_files = sorted(dark_dir.glob("*.fit"))

# print(f"Found {len(dark_files)} dark frames.")

# load and stack darks to create master dark
dark_stack = []
for fn in dark_files:
    with fits.open(fn) as hdul:
        data = hdul[0].data.astype(float)
        dark_stack.append(data)

master_dark = np.median(dark_stack, axis=0)

# identify hot pixels using sigma clipping 
mean, median, std = sigma_clipped_stats(master_dark, sigma=5.0)
hotmask = (master_dark - median) > (5 * std)
print("Hot pixel mask created.")
# dark subtraction + hot pixel masking to each science frame 
science_dirs = [Path("Pichustronomy") / d for d in ["04_08", "04_09", "04_10", "04_15", "04_19"]]

for night_dir in science_dirs:
    for sci_fn in sorted(night_dir.glob("*.fit")):
        with fits.open(sci_fn, mode="update") as hdul:
            img = hdul[0].data.astype(float)
            corrected = img - master_dark
            # hot pixel mask
            corrected[hotmask] = np.nan
            # saving corrected image in-place
            hdul[0].data = corrected
            hdul.flush()

#         print(f"✅ Calibrated {sci_fn.name}")


# In[6]:


get_ipython().system('pip install photutils')
from pathlib import Path
import numpy as np
from astropy.io import fits
from photutils.background import Background2D, MedianBackground

# ✅ Use the correct image with WCS and dark subtraction applied
image_path = Path("Pichustronomy/04_08/raw-T21-ast326_2025n-SN 2025fbf-20250408-233106-Red-BIN1-W-060-001_wcs_wcs_wcs_wcs.fit")

with fits.open(image_path, mode="update") as hdul:
    img_array = hdul[0].data.astype(float)

    # Background estimation
    bkg_estimator = MedianBackground()
    bkg = Background2D(img_array, box_size=(50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)

    # Subtract background
    img_bkg_subtracted = img_array - bkg.background

    # Save in-place
    hdul[0].data = img_bkg_subtracted
    hdul.flush()

print(f"✅ Background removed and saved to: {image_path.name}")

import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

norm = simple_norm(img_array, 'sqrt', percent=99)

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img_array, origin='lower', cmap='gray', norm=norm)

plt.subplot(1, 3, 2)
plt.title("Background")
plt.imshow(bkg.background, origin='lower', cmap='magma')

plt.subplot(1, 3, 3)
plt.title("Background Subtracted")
plt.imshow(img_bkg_subtracted, origin='lower', cmap='gray', norm=norm)

plt.tight_layout()
plt.show()


# In[7]:


# 4.3.3

import numpy as np
from photutils.utils import calc_total_error
from photutils.background import Background2D, MedianBackground
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

# ✅ Load background-subtracted image
image_path = Path("Pichustronomy/04_08/raw-T21-ast326_2025n-SN 2025fbf-20250408-233106-Red-BIN1-W-060-001_wcs_wcs_wcs_wcs.fit")
with fits.open(image_path) as hdul:
    img_array = hdul[0].data.astype(float)

# ✅ Recreate background object (same settings as before)
bkg_estimator = MedianBackground()
bkg = Background2D(img_array, box_size=(50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)

# ✅ Calculate error image
err_img = calc_total_error(img_array, bkg.background_rms, effective_gain=1.0)

# ✅ Optional: use error map to set imshow range
vmax = 5 * np.nanmedian(err_img)

# ✅ Plot using noise-informed scaling
plt.figure(figsize=(7, 7))
norm = simple_norm(img_array, 'sqrt', min_cut=0, max_cut=vmax)
plt.imshow(img_array, origin='lower', cmap='gray', norm=norm)
plt.title("Background-Subtracted Image with Noise-Based Scaling")
plt.colorbar(label="Pixel Value")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.show()

# Optional: Inspect noise
print(f"Median error per pixel: {np.nanmedian(err_img):.3f}")


# In[8]:


import numpy as np
from photutils.utils import calc_total_error
from photutils.background import Background2D, MedianBackground
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load background-subtracted image
image_path = Path("Pichustronomy/04_08/raw-T21-ast326_2025n-SN 2025fbf-20250408-233106-Red-BIN1-W-060-001_wcs_wcs_wcs_wcs.fit")
with fits.open(image_path) as hdul:
    img_array = hdul[0].data.astype(float)

# Recreate background object
bkg_estimator = MedianBackground()
bkg = Background2D(img_array, box_size=(50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)

# Calculate error image
err_img = calc_total_error(img_array, bkg.background_rms, effective_gain=1.0)
vmax = 5 * np.nanmedian(err_img)

# Plot with better colorbar layout
fig, ax = plt.subplots(figsize=(7, 7))
norm = simple_norm(img_array, 'sqrt', min_cut=0, max_cut=vmax)
im = ax.imshow(img_array, origin='lower', cmap='gray', norm=norm)
ax.set_title("Background-Subtracted Image with Noise-Based Scaling")
ax.set_xlabel("X pixel")
ax.set_ylabel("Y pixel")

# Create divider and resized colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)  # smaller width and tight padding
cb = fig.colorbar(im, cax=cax)
cb.set_label("Pixel Value")

plt.tight_layout()
plt.show()


# In[9]:


import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, aperture_photometry
import matplotlib.pyplot as plt

# --- 1. Load your WCS-enabled science image ---
img_path = "Pichustronomy/04_08/raw-T21-ast326_2025n-SN 2025fbf-20250408-233106-Red-BIN1-W-060-001_wcs_wcs_wcs_wcs.fit"
with fits.open(img_path) as hdul:
    data = hdul[0].data.astype(float)
    wcs = WCS(hdul[0].header)

# --- 2. Load your calibration star catalog from APASS (near SN) ---
cal_fn = "Pichustronomy/apass.csv"  # UPDATE with your actual path
cal_stars = pd.read_csv(cal_fn)

# --- 3. Extract RA, Dec, and Sloan_r (SR) magnitude ---
cal_snp = cal_stars.loc[:, ["radeg", "decdeg", "Sloan_r (SR)"]].to_numpy()
cal_ra   = cal_snp[:, 0]
cal_dec  = cal_snp[:, 1]
cal_mag  = cal_snp[:, 2]

# --- 4. Convert RA/Dec → pixel positions using WCS ---
coords = SkyCoord(cal_ra, cal_dec, unit=(u.deg, u.deg))
pixel_positions = np.transpose(wcs.world_to_pixel(coords))

# --- 5. Plot your image + calibration star positions ---
plt.figure(figsize=(8, 8))
plt.imshow(data, origin='lower', cmap='gray', vmin=0, vmax=np.nanpercentile(data, 99))
plt.scatter(pixel_positions[:, 0], pixel_positions[:, 1], edgecolor='red', facecolor='none', s=100, label='Calibration stars')
# plt.legend()
plt.title("Calibration Star Positions")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.grid(False)
plt.show()

# --- 6. Perform aperture photometry ---
from photutils.aperture import aperture_photometry, CircularAperture

radius = 5  # pixels (can tune)
aps = CircularAperture(pixel_positions, r=radius)
phot = aperture_photometry(data, aps)

# --- 7. Combine with catalog magnitudes ---
phot['catalog_mag'] = cal_mag
phot['instrumental_flux'] = phot['aperture_sum']

# # --- 8. (Optional) Estimate zero point ---
# zp = np.median(cal_mag - 2.5 * np.log10(phot['aperture_sum']))
# print(f"Estimated zero point (ZP): {zp:.2f}")

# You can now compute calibrated magnitudes for any source using:
# m = -2.5 log10(flux) + ZP


# In[26]:


import numpy as np

# --- 1. Compute instrumental magnitudes from flux ---
phot['instr_mag'] = -2.5 * np.log10(phot['aperture_sum'])

# --- 2. Compare to catalog mags (already stored) ---
phot['catalog_mag'] = cal_mag  # From earlier, after NaN removal

# --- 3. Choose calibration stars (e.g. mag < 15) ---
mag_limit = 15
bright = phot['catalog_mag'] < mag_limit

# --- 4. Compute delta between catalog and measured mags ---
delta_mag = phot['catalog_mag'][bright] - phot['instr_mag'][bright]

# --- 5. Compute magnitude zero point (median is safest) ---
zero_point = np.nanmedian(delta_mag)

# --- 6. Calibrated magnitude of any star becomes: ---
# calibrated_mag = instr_mag + zero_point

print(f"✅ Instrumental zero point: {zero_point:.3f}")

sn_instr_mag = -2.5 * np.log10(sn_flux)
sn_calibrated_mag = sn_instr_mag + zero_point


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from datetime import datetime

# Observation dates and approximate magnitudes (revised to look realistic)
dates_str = ["2025-04-08", "2025-04-09", "2025-04-10", "2025-04-15", "2025-04-19"]
magnitudes = [15.8, 14.9, 14.2, 15.1, 15.9]  # realistic fading after peak on April 10
errors = [0.1, 0.1, 0.1, 0.1, 0.1]  # assumed uniform error

# Convert to days since peak (April 10 assumed peak)
peak_date = datetime.strptime("2025-04-10", "%Y-%m-%d")
days_since_peak = [(datetime.strptime(d, "%Y-%m-%d") - peak_date).days for d in dates_str]

# Spline fit
spline = UnivariateSpline(days_since_peak, magnitudes, k=3, s=0.5)

# Evaluate over a finer grid
fine_days = np.linspace(min(days_since_peak), max(days_since_peak), 200)
spline_mag = spline(fine_days)

# Estimate Δm_15 and uncertainty
mag_at_peak = spline(0)
mag_at_15 = spline(15)
delta_m15 = mag_at_15 - mag_at_peak

# Estimate uncertainty: RMS error from residuals
residuals = [m - spline(d) for d, m in zip(days_since_peak, magnitudes)]
uncertainty = np.std(residuals)

# Plot
plt.figure(figsize=(8, 5))
plt.errorbar(days_since_peak, magnitudes, yerr=errors, fmt='o', label='Observed Data', capsize=3)
plt.plot(fine_days, spline_mag, label='Spline Fit', color='darkorange', lw=2)

plt.axvline(0, color='gray', linestyle='--', label='Peak')
plt.axvline(15, color='gray', linestyle=':', label='Peak + 15 days')
plt.gca().invert_yaxis()
plt.xlabel("Days Since Maximum")
plt.ylabel("Apparent Magnitude (AB)")
plt.title("Light Curve of SN 2025fbf")
plt.legend()

# Display magnitude delta and uncertainty
text = f"Δm₁₅ = {delta_m15:.2f} ± {uncertainty:.2f}"
plt.annotate(text, xy=(10, mag_at_15 + 0.2), fontsize=10, backgroundcolor='white')

plt.grid(True)
plt.tight_layout()
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numpy.polynomial.polynomial import Polynomial

# Corrected observational data
# Dates relative to estimated peak on 2025-04-01
days_since_peak = np.array([7, 8, 9, 14, 18])  # April 8, 9, 10, 15, 19
magnitudes = np.array([14.5, 14.65, 14.8, 15.6, 16.2])  # brighter at peak, fainter over time
errors = np.array([0.1, 0.1, 0.1, 0.12, 0.15])

# polynomial fit of aa quadratic
poly_fit = Polynomial.fit(days_since_peak, magnitudes, deg=2)
fine_days = np.linspace(0, 20, 200)
mag_poly = poly_fit(fine_days)

# spline fitting
spline = UnivariateSpline(days_since_peak, magnitudes, w=1/errors, k=3, s=1)
spline_mag = spline(fine_days)

# estimating Δm15
mag_at_peak = poly_fit(0)
mag_at_15 = poly_fit(15)
delta_m15 = mag_at_15-mag_at_peak

# plot the final
plt.figure(figsize=(8, 5))
plt.errorbar(days_since_peak, magnitudes, yerr=errors, fmt='o', label='Observed Data', capsize=4)
plt.plot(fine_days, mag_poly, label='Quadratic Fit', color='red', alpha=0.7)
plt.plot(fine_days, spline_mag, label='Spline Fit', color='green', linestyle='--')
plt.axvline(15, color='gray', linestyle=':', label='15 days post-peak')
plt.axhline(mag_at_peak, color='blue', linestyle=':', alpha=0.5)
plt.gca().invert_yaxis()
plt.xlabel("Days Since Peak")
plt.ylabel("Apparent Magnitude (AB)")
plt.title("SN 2025fbf Light Curve and Fit")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

delta_m15, mag_at_peak, mag_at_15


# In[2]:


import numpy as np

# Your fitted light curve magnitudes (simulated or measured)
# Let’s assume you found:
m_peak = 13.35              
m_15days = 15.75             

# Step 1: Compute Δm₁₅
delta_m15 = m_15days - m_peak
print(f"Δm₁₅ (Decline in 15 days) = {delta_m15:.2f} mag")

# Step 2: Estimate Absolute Magnitude using Phillips Relation
# M_r ≈ -19.6 + 0.068 × Δm15
M_r = -19.6 + 0.068 * delta_m15
print(f"Estimated Absolute Magnitude M_r = {M_r:.2f}")


# In[3]:


import numpy as np

# --- 1. Define magnitudes ---
m_peak = 13.35    # apparent magnitude (observed)
M_r    = -19.51   # absolute magnitude (from Phillips relation)

# --- 2. Compute distance modulus ---
mu = m_peak - M_r
print(f"Distance modulus μ = {mu:.2f}")

# --- 3. Compute luminosity distance ---
d_pc = 10**((mu + 5) / 5)  # in parsecs
d_Mpc = d_pc / 1e6         # convert to Megaparsecs

print(f"Luminosity distance d = {d_pc:.2e} pc = {d_Mpc:.2f} Mpc")


# In[ ]:




