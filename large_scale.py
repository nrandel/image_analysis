# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
#1. pick a certain time window after the first stimulus onset and compute the peak DF/F and the average DF/F in that time window for each cell body
#2. Compute the mean peak and the mean average across all cell bodies, and the SD.
#3. Find all neurons whose responses are 1.5 SD higher than the overall mean
# %%
# Read the CSV file into a DataFrame
# raw fluorescence 1099
#raw_fluorescence = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_klb.csv') 

# dff long sliding window 1099
#dff_long = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_t_stacks.csv')

#%%
#calculate cell specific dff from raw fluorescence
#TODO

# %%

def plot_neuron_activity(df, neurons, time_range, title):
    plt.figure(figsize=(12, 8))
    for neuron in neurons:
        plt.plot(df.index[time_range[0]:time_range[1]+1], df[neuron][time_range[0]:time_range[1]+1], label=neuron)
    plt.title(title)
    plt.xlabel('Timepoint')
    plt.ylabel('ΔF/F')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()

# Load the dataset from the CSV file
df = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_t_stacks.csv')

# Drop the 'timepoint' column and use the index as the new timepoint
df.drop(columns=['timepoint'], inplace=True)

# Define the time window (102-117, inclusive)
time_window = list(range(102, 118))

# Extract the relevant rows for the time window using the index as timepoint
time_window_data = df.iloc[time_window]

# Compute the peak ΔF/F and the average ΔF/F in the time window for each neuron
peak_dff = time_window_data.max()
average_dff = time_window_data.mean()

# Compute the mean and standard deviation for the peak and average ΔF/F across all neurons
mean_peak_dff = peak_dff.mean()
mean_average_dff = average_dff.mean()

sd_peak_dff = peak_dff.std()
sd_average_dff = average_dff.std()

# Define the threshold for identifying responsive neurons
threshold_peak = mean_peak_dff + 1.5 * sd_peak_dff
threshold_average = mean_average_dff + 1.5 * sd_average_dff

# Identify neurons whose responses are 1.5 SD higher than the overall mean
responsive_neurons_peak = peak_dff[peak_dff > threshold_peak]
responsive_neurons_average = average_dff[average_dff > threshold_average]

# Find the intersection of responsive neurons
responsive_neurons_intersection = responsive_neurons_peak.index.intersection(responsive_neurons_average.index)

# Find all neurons
all_neurons = df.columns

# Find neurons that are not responsive for both peak and average ΔF/F
non_responsive_neurons = all_neurons.difference(responsive_neurons_intersection)

# Print the results
print("Responsive neurons based on peak ΔF/F (count: {}):".format(len(responsive_neurons_peak)))  # Number of neurons
print(responsive_neurons_peak)

print("\nResponsive neurons based on average ΔF/F (count: {}):".format(len(responsive_neurons_average)))  # Number of neurons
print(responsive_neurons_average)

print("\nNeurons that are responsive based on both peak and average ΔF/F (count: {}):".format(len(responsive_neurons_intersection)))  # Number of neurons
print(responsive_neurons_intersection)

print("\nNeurons that are not responsive for both peak and average ΔF/F (count: {}):".format(len(non_responsive_neurons)))  # Number of neurons
print(non_responsive_neurons)

# User input for timepoint range
start_time = int(input("Enter start timepoint for plotting: "))
end_time = int(input("Enter end timepoint for plotting: "))
time_range = (start_time, end_time)

# Plot results
plot_neuron_activity(df, responsive_neurons_peak.index, time_range, 'Responsive neurons based on peak ΔF/F')
plot_neuron_activity(df, responsive_neurons_average.index, time_range, 'Responsive neurons based on average ΔF/F')
plot_neuron_activity(df, responsive_neurons_intersection, time_range, 'Neurons responsive based on both peak and average ΔF/F')
plot_neuron_activity(df, non_responsive_neurons, time_range, 'Neurons not responsive for both peak and average ΔF/F')
# %%
