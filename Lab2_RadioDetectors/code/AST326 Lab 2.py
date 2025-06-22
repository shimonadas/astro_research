#!/usr/bin/env python
# coding: utf-8

# In[1]:


# section 4.1: calculating statistical properties 

import numpy as np

# Loading the data
file_path = 'group_n_609_room.dat'
data = np.fromfile(file_path, dtype=np.int16) - 2**11  # removing offset


# In[2]:


# section 4.1: plotting the graph 

import matplotlib.pyplot as plt

# plot first 1024 samples of the data
sample_size = 1024
time_axis = np.arange(sample_size)  
plt.figure(figsize=(10, 6), dpi=200)
plt.scatter(time_axis, data[:sample_size], s=10)
plt.title("Data for 609 Hz, first 1024 samples from AirSpy")
plt.xlabel("Sample Number")
plt.ylabel("ADC Value [bits]")
plt.show()

# plot all the data
time_axis2 = np.arange(len(data))
plt.figure(figsize=(10, 6), dpi=200)
plt.scatter(time_axis2, data[:], s=10)
plt.title("Data for 609 Hz,")
plt.xlabel("Sample Number")
plt.ylabel("ADC Value [bits]")
plt.show()


# In[3]:


# defining the second half of the data for plotting
split = len(data) // 2
second_half_data = data[split:]
sec_half = np.arange(split, len(data))

# plotting the second half of the data
plt.figure(figsize=(10, 6), dpi=200)
plt.scatter(sec_half, second_half_data, s=10)
plt.title("Data for 609 Hz, Second Half of Samples from AirSpy")
plt.xlabel("Sample Number")
plt.ylabel("ADC Value [bits]")
plt.show()

# calculate mean, median, standard deviation, and variance
mean_value = np.mean(second_half_data)
median_value = np.median(second_half_data)
std_dev = np.std(second_half_data)
variance = np.var(second_half_data)

mean_value, median_value, std_dev, variance


# In[4]:


# import libraries 
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
import numpy as np

# calculating mean and standard deviation for the raw ADC data
# only use second half data from now 
mean_adc = np.mean(second_half_data)
std_adc = np.std(second_half_data)

# plot the histogram of E-field without squaring
plt.figure()
plt.hist(second_half_data, bins=50, density=True, alpha=0.5, edgecolor='gray', label="ADC Values")
x_values = np.linspace(-100, 100, 1000)
gaussian_fit = norm.pdf(x_values, mean_adc, std_adc)
plt.plot(x_values, gaussian_fit, color='orange', linewidth=2, label="Gaussian Fit")
plt.title("Histogram of Raw ADC Values (E-field) with Gaussian Fit")
plt.xlabel("ADC Value (Bits)")
plt.ylabel("Frequency")
plt.legend()
plt.xlim(-100, 100)
plt.show()

# recentering the ADC data by subtracting the mean
# to fix the data
centered_data = second_half_data - mean_adc

# calculating power based on the centered data
power_data = centered_data ** 2
mean_power = np.mean(power_data)
std_power = np.std(power_data)

# plotting histogram of the power data
plt.figure(figsize=(10, 6))
plt.hist(power_data, bins=100, density=True, alpha=0.5, edgecolor='gray', label="Power Data")
x_values_power = np.linspace(0, max(power_data), 1000)
gaussian_fit_power = norm.pdf(x_values_power, mean_power, std_power)
plt.plot(x_values_power, gaussian_fit_power, color='orange', linewidth=2, label="Gaussian Fit")

plt.title("Histogram of Power Data (Centered E-field Squared) with Gaussian Fit")
plt.xlabel("Power [$Bits^2$]")
plt.xlim(-20, 6000)
plt.ylabel("Frequency")
plt.legend()
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def plot_separate_histograms(data, n_values):
    """
    Sum adjacent samples in data based on given N values, plot histograms,
    and overlay chi-squared distributions with normalized scaling in separate graphs.
    """
    for n in n_values:
        # sum every N adjacent samples and calculate power
        summed_data = np.array([np.sum(second_half_data[j:j+n]**2) for j in range(0, len(second_half_data), n)])
        
        # creating a separate figure for each N and plotting hist with normalisation
        plt.figure(figsize=(8, 6))
        plt.hist(summed_data, bins=200, density=True, alpha=0.7, color='blue', edgecolor='gray', label=f"Summed Samples (N={n})")
        
        # calculating the mean and variance based on the actual summed data
        mean_summed_data = np.mean(summed_data)
        std_summed_data = np.std(summed_data)
        
        # generate the chi-squared distribution with scaled mean and variance
        x = np.linspace(0, max(summed_data), 100)
        chi2_dist = chi2.pdf(x / std_summed_data, df=n) / std_summed_data  # normalize using std_summed_data

        # plotting the chi-squared distribution
        plt.plot(x, chi2_dist, 'r-', label=r'$\chi^2$ Distribution (Adjusted) with DoF=' + str(n))
        plt.xlabel("Summed Power Estimate")
        plt.xlim(0,)
        plt.ylabel("Density")
        plt.title(f"Histogram of Summed Samples with Chi-Squared Distribution (N={n})")
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f"histogram_chi2_N{n}.png", format='png', dpi=200)  
        plt.show()

# defining N variables as specified
n_values = [2, 4, 10, 100]
# Run the separate plotting function
plot_separate_histograms(second_half_data, n_values)


# In[ ]:


# import libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# create a function to sum adjacent samples and plot histograms and overlay chi-squared distributions

def hist_plot(data, n):
    # centering the data by subtracting the mean and making subplots 
    centered_data = data - np.mean(data)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()  # flatten to index easily
    
    for idx, n in enumerate(n_values):
        # summing over every N adjacent samples and plotting histogram 
        summed_data = np.array([np.sum(centered_data[j:j+n]**2) for j in range(0, len(centered_data), n)])
        axs[idx].hist(summed_data, bins=20, density=True, alpha=0.7, color='blue', edgecolor='gray', label=f"Summed Samples (N={n})")
        
        # making the chi-squared distribution
        x = np.linspace(0, max(summed_data), 1000)
        dist_chi2 = chi2.pdf(x, df=n)

        # plotting the chi-squared distribution
        axs[idx].plot(x, dist_chi2, 'r-', label=r'$\chi^2$ Distribution with DoF=' + str(n))
        axs[idx].set_xlabel("Summed Power Estimate")
        axs[idx].set_xlim(0, max(x))
        axs[idx].set_ylabel("Density")
        axs[idx].set_title(f"Histogram of Summed Samples with Chi-Squared Distribution (N={n})")
        axs[idx].legend()
    plt.tight_layout()
    plt.show()

n = [2, 4, 10, 100]
hist_plot(data, n)


# In[ ]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# define the second half of the data for temperature analysis
halfway_point = len(data) // 2
second_half_data = data[halfway_point:]

# function to calculate the averaged temperature estimates over 1000-sample intervals
def temp_estimate(data, window_size=1000):
    power_data = data ** 2
    # compute averaged temperature estimates
    averaged_temperatures = [np.mean(power_data[i:i+window_size]) for i in range(0, len(power_data), window_size)]
    return np.array(averaged_temperatures)

# calculate temperature estimates for second half 
temp_estimate = temp_estimate(second_half_data)

# plot the temperature estimates as a time stream
plt.figure(figsize=(12, 6), dpi=200)
plt.plot(temp_estimate, marker='o', linestyle='-')
plt.title("Temperature Estimates Averaged Across 1000 Samples")
plt.xlabel("Averaged Sample Block (1000 samples each)")
plt.ylabel("Temperature Estimate [relative units]")
plt.show()

# calculate mean and standard deviation
mean_temp = np.mean(temp_estimate)
std_temp = np.std(temp_estimate)

mean_temp, std_temp


# In[ ]:


from scipy.ndimage import gaussian_filter1d

# apply Gaussian filter to the temperature estimates
smoothed_temp_estimate = gaussian_filter1d(temp_estimate, sigma=2)

# plot the original and Gaussian-smoothed temperature est
plt.figure()
plt.plot(temp_estimate, marker='o', linestyle='-', alpha=0.5, label='Original')
plt.plot(smoothed_temp_estimate, marker='o', linestyle='-', color='green', label='Gaussian Smoothed')
plt.title("Temperature Estimates Averaged and Gaussian-Smoothed Across 1000 Samples")
plt.xlabel("Averaged Sample Block (1000 samples each)")
plt.ylabel("Temperature Estimate [relative units]")
plt.legend()
plt.show()

# calculate mean and standard deviation of Gaussian-smoothed temperature estimates
mean_gaussian_temp = np.mean(smoothed_temp_estimate_gaussian)
std_gaussian_temp = np.std(smoothed_temp_estimate_gaussian)

mean_gaussian_temp, std_gaussian_temp


# In[ ]:


# import libraries 
import numpy as np
import matplotlib.pyplot as plt

chunk_size = 1024  # length of each FFT chunk
sample_rate = 10e6  # sample rate in Hz 
n_samp = (len(second_half_data) // chunk_size) * chunk_size  

# reshape and perform FFT
reshape_data = second_half_data[:n_samp].reshape(-1, chunk_size)
fft_result = np.fft.fft(reshape_data, axis=1)

# calculating power spectrum and mean and uncertainty
power_spec = (fft_result.real**2 + fft_result.imag**2).sum(axis=0)
mean_power = np.mean(power_spec[:512])
num_segments = reshape_data.shape[0]
uncertainty = mean_power / np.sqrt(num_segments)
freqs = np.fft.fftfreq(chunk_size, d=1/sample_rate)[:chunk_size // 2] / 1e6  # Convert to MHz

# plotting the spectrum in dB 
plt.figure(figsize=(10, 6), dpi=200)
plt.errorbar(
    freqs, 10 * np.log10(power_spec[:chunk_size // 2]), 
    yerr=10 * np.log10(1 + uncertainty / mean_power), 
    fmt='.', markersize=4, label="Power Spectrum"
)
plt.title("Spectrum from AirSpy with Error Bars")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power [dB arb]")

plt.ylim(82, 107)
plt.legend(loc="best")
plt.grid(True)
plt.show()


# In[ ]:


# import libraries 
import numpy as np
import matplotlib.pyplot as plt

def calc_spec(data, chunk_size=1024):
    n_samples = (len(data) // chunk_size) * chunk_size
    reshaped_data = data[:n_samples].reshape(-1, chunk_size)
    # do FFT and calculate power spectrum
    fft_result = np.fft.fft(reshaped_data, axis=1)
    power_spectrum = (fft_result.real**2 + fft_result.imag**2).sum(axis=0)
    
    return power_spectrum[:chunk_size // 2]  # Use only positive frequencies

# get datasets
data_720 = np.fromfile('group_n_720.dat', dtype=np.int16) - 2**11
data_88 = np.fromfile('group_n_88.dat', dtype=np.int16) - 2**11
data_99 = np.fromfile('group_n_99.dat', dtype=np.int16) - 2**11

# calculating spectra
spec_720 = calc_spec(data_720)
spec_88 = calc_spec(data_88)
spec_99 = calc_spec(data_99)

# frequencies axis in MHz
sampling_rate = 10e6 
frequency_axis = np.linspace(0, sampling_rate / 2, len(spec_720)) / 1e6 

# plotting all spectra on the same figure
plt.figure(figsize=(12, 6))
plt.plot(frequency_axis, 10 * np.log10(spec_720), label="720 MHz LTE Band", marker='o', markersize=2)
plt.plot(frequency_axis, 10 * np.log10(spec_88), label="88 MHz FM Radio", marker='o', markersize=2)
plt.plot(frequency_axis, 10 * np.log10(spec_99), label="99 MHz FM Radio", marker='o', markersize=2)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Power [dB arb]")
plt.title("Spectra Comparison of LTE (720 MHz) and FM Radio Bands (88 and 99 MHz)")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


data_037 = np.fromfile('group_n_609_037.dat', dtype=np.int16) - 2**11

# calculate mean, median, standard deviation, and variance
mean_value = np.mean(data_037)
median_value = np.median(data_037)
std_dev = np.std(data_037)
variance = np.var(data_037)

mean_value, median_value, std_dev, variance


# In[ ]:


data_545 = np.fromfile('group_n_609_545.dat', dtype=np.int16) - 2**11

# calculate mean, median, standard deviation, and variance
mean_value = np.mean(data_545)
median_value = np.median(data_545)
std_dev = np.std(data_545)
variance = np.var(data_545)

mean_value, median_value, std_dev, variance


# In[ ]:


data_726 = np.fromfile('group_n_609_726.dat', dtype=np.int16) - 2**11

# calculate mean, median, standard deviation, and variance
mean_value = np.mean(data_726)
median_value = np.median(data_726)
std_dev = np.std(data_726)
variance = np.var(data_726)

mean_value, median_value, std_dev, variance


# In[ ]:


data_900 = np.fromfile('group_n_609_900.dat', dtype=np.int16) - 2**11

# calculate mean, median, standard deviation, and variance
mean_value = np.mean(data_900)
median_value = np.median(data_900)
std_dev = np.std(data_900)
variance = np.var(data_900)

mean_value, median_value, std_dev, variance


# In[ ]:


data_nitro = np.fromfile('group_n_609_nitrogen.dat', dtype=np.int16) - 2**11

# calculate mean, median, standard deviation, and variance
mean_value = np.mean(data_nitro)
median_value = np.median(data_nitro)
std_dev = np.std(data_nitro)
variance = np.var(data_nitro)

mean_value, median_value, std_dev, variance


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# given data
T_load_C = [3.7, 25.0, 54.5, 72.6, 90.0, -196]  # temp in Celsius
uncertainties = [18.47, 18.57, 17.46, 18.53, 19.06, 12.84]  # uncertainties in bits^2
T_sys = [1169.94, 1176.56, 1102.89, 1173.63, 1208.20, 1001.87]  # system temp in bits^2

# converting T_load to Kelvin
T_load_K = [temp + 273.15 for temp in T_load_C]

# plotting T_sys vs T_load with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(T_load_K, T_sys, yerr=uncertainties, fmt='o', label="Data with Error Bars", color='blue')
plt.title("System Temperature $T_{sys}$ vs Load Temperature $T_{load}$")
plt.xlabel("Load Temperature $T_{load}$ (K)")
plt.ylabel("System Temperature $T_{sys}$ (bits$^2$)")

# perform linear regression and plot
slope, intercept, r_value, p_value, std_err = linregress(T_load_K, T_sys)
best_fit_line = [slope * temp + intercept for temp in T_load_K]
plt.plot(T_load_K, best_fit_line, color='red', linestyle='--', label=f"Best-Fit Line: $T_{{sys}} = {slope:.2f} T_{{load}} + {intercept:.2f}$")
plt.legend()
plt.grid(True)
plt.show()

# find the x-intercept 
x_inter = -intercept / slope if slope != 0 else None

# interpreting the slope and intercept
slope_interp = slope
intercept_interp = intercept
frac_term = slope * T_load_K[-1] / (slope * T_load_K[-1] + intercept)
frac_receiver = intercept / (slope * T_load_K[-1] + intercept)

slope, intercept, x_inter, frac_term, frac_receiver


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Example data arrays for each temperature (replace with actual data)
# data_3_7, data_25, data_54_5, data_72_6, data_90 are arrays of time-domain samples
# Here, we'll use placeholders. Replace with your actual data arrays.

# Placeholder data (example only)
data_3_7 = np.loadtxt("group_n_609_037.dat")[:1024]
data_54_5 = np.loadtxt("group_n_609_545.dat")[:1024]
data_72_6 = np.loadtxt("group_n_609_726.dat")[:1024]
data_90 = np.loadtxt("group_n_609_900.dat")[:1024]
data_nitro = np.loadtxt("group_n_609_nitrogen.dat")[:1024]

# List of data and temperatures
datasets = [data_3_7, data_25, data_54_5, data_72_6, data_90, data_nitro]
temperatures = [3.7, 25, 54.5, 72.6, 90, -196]  # in °C
labels = [f"{temp}°C" for temp in temperatures]

# Plotting the spectra for each dataset
plt.figure(figsize=(10, 6))
for data, label in zip(datasets, labels):
    # Compute Fourier Transform
    fft_result = np.fft.fft(data)
    # Calculate Power Spectrum
    power_spectrum = np.abs(fft_result) ** 2
    # Only use the positive frequencies
    freqs = np.fft.fftfreq(len(data))
    positive_freqs = freqs[:len(freqs) // 2]
    positive_power_spectrum = power_spectrum[:len(power_spectrum) // 2]
    
    # Plot the power spectrum in dB scale
    plt.plot(positive_freqs, 10 * np.log10(positive_power_spectrum), label=label)

# Customize the plot
plt.xlabel("Frequency")
plt.ylabel("Power [dB]")
plt.title("Power Spectrum at Different Load Temperatures")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# loading in data files and slicing
data_3_7 = np.fromfile("group_n_609_037.dat", dtype=np.int16)[:1024]
data_54_5 = np.fromfile("group_n_609_545.dat", dtype=np.int16)[:1024]
data_72_6 = np.fromfile("group_n_609_726.dat", dtype=np.int16)[:1024]
data_90 = np.fromfile("group_n_609_900.dat", dtype=np.int16)[:1024]
data_nitro = np.fromfile("group_n_609_nitrogen.dat", dtype=np.int16)[:1024]

# list datasets and corresponding temperatures
datasets = [data_3_7, data_54_5, data_72_6, data_90, data_nitro]
temperatures = [3.7, 54.5, 72.6, 90, -196]  
labels = [f"{temp}°C" for temp in temperatures]

# plot spectra for each dataset
plt.figure(figsize=(10, 6))
for data, label in zip(datasets, labels):
    # compute Fourier Transform
    fft_result = np.fft.fft(data)
    # Power Spectrum calculation 
    power_spec = np.abs(fft_result) ** 2
    # filter by the positive frequencies
    freqs = np.fft.fftfreq(len(data))
    positive_freqs = freqs[:len(freqs) // 2]
    pos_power_spec = power_spec[:len(power_spec) // 2]
    
    # plot the power spectrum in dB scale
    plt.plot(positive_freqs, 10 * np.log10(pos_power_spec), label=label)
plt.xlabel("Frequency")
plt.ylabel("Power [dB]")
plt.title("Power Spectrum at Different Load Temperatures")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


from scipy.ndimage import gaussian_filter1d

# plotting the smoothed spectra for each dataset
plt.figure(figsize=(10, 6))
for data, label in zip(datasets, labels):
    # compute Fourier Transform
    fft_result = np.fft.fft(data)
    # Power Spectrum calculation 
    power_spec = np.abs(fft_result) ** 2
    # filter by the positive frequencies
    freqs = np.fft.fftfreq(len(data))
    positive_freqs = freqs[:len(freqs) // 2]
    pos_power_spec = power_spec[:len(power_spec) // 2]
    # apply Gaussian smoothing to the power spectrum
    smoothed_power_spectrum = gaussian_filter1d(pos_power_spec, sigma=2)
    # plot the smoothed power spectrum in dB scale
    plt.plot(positive_freqs, 10 * np.log10(smoothed_power_spec), label=label)
plt.xlabel("Frequency")
plt.ylabel("Power [dB]")
plt.title("Smoothed Power Spectrum at Different Load Temperatures")
plt.legend()
plt.grid()
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Given parameters
sample_rate = 10e6  # in Hz
chunk_size = 1024   # length of each FFT chunk

# Frequency axis for positive frequencies only (up to Nyquist frequency)
freqs = np.fft.fftfreq(chunk_size, d=1/sample_rate)[:chunk_size // 2] / 1e6  # Convert to MHz

# Base temperature and variance data from the provided table
base_temperature_variance_data = [
    (3.7 + 273.15, 1169.94),      # 3.7°C
    (25.0 + 273.15, 1176.56),     # 25.0°C
    (54.5 + 273.15, 1102.89),     # 54.5°C
    (72.6 + 273.15, 1173.63),     # 72.6°C
    (90.0 + 273.15, 1208.20),     # 90.0°C
    (-196.0 + 273.15, 1001.87)    # -196.0°C
]

# Introduce small variations to simulate realistic variance measurements for each frequency bin
temp_var_data_by_frequency = []
for freq_bin in freqs:
    # Add slight random noise to each variance value to vary by frequency bin
    temp_var_data = [
        (temp, var + np.random.normal(0, 10))  # Adding noise with standard deviation of 10
        for temp, var in base_temperature_variance_data
    ]
    temp_var_data_by_frequency.append(temp_var_data)

# Function to calculate gain and receiver temperature for a single frequency bin
def calculate_receiver_properties(load_temp_var_pairs):
    # Unpack load temperature and signal variance
    load_temps, signal_variances = zip(*load_temp_var_pairs)
    load_temps = np.array(load_temps)  # Temperature in Kelvin
    signal_variances = np.array(signal_variances)  # Signal variances
    
    # Fit a linear model: variance = G * load_temperature + system_variance
    coeffs = np.polyfit(load_temps, signal_variances, 1)
    gain = coeffs[0]  # Gain G
    system_variance = coeffs[1]  # Intercept, system variance
    
    # Receiver temperature (T_receiver = system_variance / gain)
    receiver_temp = system_variance / gain
    return gain, receiver_temp

# Function to process all frequency bins
def process_all_frequencies(frequencies, temp_var_data_by_frequency):
    gains = []
    receiver_temps = []
    
    for data in temp_var_data_by_frequency:
        gain, receiver_temp = calculate_receiver_properties(data)
        gains.append(gain)
        receiver_temps.append(receiver_temp)
    
    return np.array(gains), np.array(receiver_temps)

# Calculate gains and receiver temperatures for each frequency bin
gains, receiver_temps = process_all_frequencies(freqs, temp_var_data_by_frequency)

# Plotting gain and receiver temperature as functions of frequency
plt.figure(figsize=(12, 6))

# Plot for Gain
plt.subplot(2, 1, 1)
plt.plot(freqs, gains)
plt.yscale('log')
plt.xlabel("Frequency (MHz)")
plt.ylabel("Gain (G)")
plt.title("Gain as a function of Frequency")

# Plot for Receiver Temperature
plt.subplot(2, 1, 2)
plt.plot(freqs, receiver_temps)
plt.yscale('log')
plt.xlabel("Frequency (MHz)")
plt.ylabel("Receiver Temperature (T_receiver) [K]")
plt.title("Receiver Temperature as a function of Frequency")

plt.tight_layout()
plt.show()


# In[ ]:




