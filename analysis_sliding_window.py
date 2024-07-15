# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
#Stimulus: 101-106/ 681-695

# %%
"Important: behaviour.csv starts with '1', activity.csv with '0' - fixed for MY dff, not here (I think)"

#%%
#1. Pick a certain time window and read out the peak DF/F and the average DF/F in that time window for each cell body
#2. Compute the mean peak and the mean average across all cell bodies, and the SD.
#3. Find all neurons whose responses are 1.5 or 2 SD higher than the overall mean

# Input: df == activity traces for each cell body in the brain, 
# read out from tif stack generated with sliding window for dff
# Output: Neuron names and activity that meet the statistic requirements. Specifically for each event

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

def save_neurons_to_csv(neurons, filename, path=''):  # Add a path parameter with default value ''
    filepath = os.path.join(path, filename)  # Construct the full file path
    pd.DataFrame(neurons).to_csv(filepath, header=False)

# Load the dataset from the CSV file
df = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_t_stacks-dff_long-slidingWindow.csv')

# Drop the 'timepoint' column and use the index as the new timepoint
df.drop(columns=['timepoint'], inplace=True)

# Define the time window for statistical analysis (Note: Time window for plotting is below)

time_window = list(range(100, 110))

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
#threshold_peak = mean_peak_dff + 1.5 * sd_peak_dff
#threshold_average = mean_average_dff + 1.5 * sd_average_dff

threshold_peak = mean_peak_dff + 2 * sd_peak_dff
threshold_average = mean_average_dff + 2 * sd_average_dff

# Identify neurons whose responses are 1.5 or 2 SD higher than the overall mean
responsive_neurons_peak = peak_dff[peak_dff > threshold_peak]
responsive_neurons_average = average_dff[average_dff > threshold_average]

# Find the intersection of responsive neurons
responsive_neurons_intersection = responsive_neurons_peak.index.intersection(responsive_neurons_average.index)

# Find all neurons
all_neurons = df.columns

# Find neurons that are not responsive for both peak and average ΔF/F
non_responsive_neurons = all_neurons.difference(responsive_neurons_intersection)


"""
# Print the results
print("Responsive neurons based on peak ΔF/F (count: {}):".format(len(responsive_neurons_peak)))  # Number of neurons
print(responsive_neurons_peak)

print("\nResponsive neurons based on average ΔF/F (count: {}):".format(len(responsive_neurons_average)))  # Number of neurons
print(responsive_neurons_average)

# Compute the difference between responsive_neurons_peak and responsive_neurons_average
difference_neurons = responsive_neurons_peak.index.difference(responsive_neurons_average.index)
print("\nDifference in neurons between peak and average ΔF/F (count: {}):".format(len(difference_neurons)))
print(difference_neurons)

print("\nNeurons that are responsive based on both peak and average ΔF/F (count: {}):".format(len(responsive_neurons_intersection)))  # Number of neurons
print(responsive_neurons_intersection)

print("\nNeurons that are not responsive for both peak and average ΔF/F (count: {}):".format(len(non_responsive_neurons)))  # Number of neurons
print(non_responsive_neurons)
"""

# Define the directory where you want to save the files
save_dir = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/'

# Save neurons to CSV files with specified path
"change name"
save_neurons_to_csv(responsive_neurons_peak, "responsive_neurons_sd_2_peak_100-110.csv", path=save_dir)
save_neurons_to_csv(responsive_neurons_average, "responsive_neurons_sd_2_average_100-110.csv", path=save_dir)
#save_neurons_to_csv(difference_neurons, "difference_neurons_682-699.csv", path=save_dir)
#save_neurons_to_csv(responsive_neurons_intersection, "responsive_neurons_intersectio_682-699.csv", path=save_dir)
#save_neurons_to_csv(non_responsive_neurons, "non_responsive_neurons_682-699.csv", path=save_dir)

# Save the original CSV file with only the columns corresponding to the responsive neurons
"change name"
df_responsive_neurons_peak = df[list(responsive_neurons_peak.index)]
df_responsive_neurons_average = df[list(responsive_neurons_average.index)]
#df_difference_neurons = df[list(difference_neurons)]
#df_responsive_neurons_intersection = df[list(responsive_neurons_intersection)]
#df_non_responsive_neurons = df[list(non_responsive_neurons)]

df_responsive_neurons_peak.to_csv(os.path.join(save_dir, "activity_of_responsive_neurons_sd_2_peak_100-110.csv"), index=False)
df_responsive_neurons_average.to_csv(os.path.join(save_dir, "activity_of_responsive_neurons_sd_2_average_100-110.csv"), index=False)
#df_difference_neurons.to_csv(os.path.join(save_dir, "activity_of_difference_neurons_682-699.csv"), index=False)
#df_responsive_neurons_intersection.to_csv(os.path.join(save_dir, "activity_of_responsive_neurons_intersection_682-699.csv"), index=False)
#df_non_responsive_neurons.to_csv(os.path.join(save_dir, "activity_of_non_responsive_neurons_682-699.csv"), index=False)

# Define start and end times for plotting 80-150/ 660-740
# Note: stimulus window 101-106/ 681-695
start_time = 80
end_time = 150
time_range = (start_time, end_time)

# Plot results
#plot_neuron_activity(df, responsive_neurons_peak.index, time_range, 'Responsive neurons based on peak ΔF/F')
plot_neuron_activity(df, responsive_neurons_average.index, time_range, 'Responsive neurons based on average ΔF/F')
#plot_neuron_activity(df, responsive_neurons_intersection, time_range, 'Neurons responsive based on both peak and average ΔF/F')
#plot_neuron_activity(df, non_responsive_neurons, time_range, 'Neurons not responsive for both peak and average ΔF/F') #For now to many datapoints to plot



#%%
# Find intersection between neurons that respond to first and sec action 
# Input: Neuron names that meet the statistic requirements for each event.
# Output: Neuron names! that meet statistic requirements for both events.

# Load the two CSV files
df1 = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/responsive_neurons_sd_2_average_100-110.csv', header=None)
df2 = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/responsive_neurons_sd_2_average_680-699.csv', header=None)

# Extract the first column from each dataframe
df1_first_col = df1[[0]]
df2_first_col = df2[[0]]

# Find the intersection of the first columns
intersection = pd.merge(df1_first_col, df2_first_col, how='inner')

# Add a header to the intersection dataframe
intersection.columns = ['cell coordinates']

# Save the result to a new CSV file
"change file name"
intersection.to_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/intersection_responsive_neurons_sd_2_average_100-110-AND-680-699.csv', index=False)

print("Intersection saved to output.csv")


#%%
# Get neuron activity that meet statistic for actions

# Input 1: inntersection_df == Neuron names! that meet statistic requirements for both events.
# Input 2: df == activity traces for each cell body in the brain, 
# read out from tif stack generated with sliding window for dff
# Output: Neuron traces that meet statistic requirements for both events (csv and plot)


import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_neuron_activity(df, neurons, time_range, title, show_legend=True):
    plt.figure(figsize=(12, 8))
    for neuron in neurons:
        plt.plot(df.index[time_range[0]:time_range[1]+1], df[neuron][time_range[0]:time_range[1]+1], label=neuron)
    plt.title(title)
    plt.xlabel('Timepoint')
    plt.ylabel('ΔF/F')
    if show_legend:
        plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()

def save_neurons_to_csv(neurons, filename, path=''):
    filepath = os.path.join(path, filename)
    pd.DataFrame(neurons).to_csv(filepath, header=False)

# Load the dataset from the CSV file
df = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_t_stacks-dff_long-slidingWindow.csv')

# Drop the 'timepoint' column and use the index as the new timepoint
df.drop(columns=['timepoint'], inplace=True)

# Load the second CSV file to find the intersection
intersection_df = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/intersection_responsive_neurons_sd_2_average_100-110-AND-680-699.csv')

# Assuming that 'intersection_df' contains the responsive neuron identifiers in the first column
responsive_neurons_new_csv = intersection_df.iloc[:, 0].values

# Save the original CSV file with only the columns corresponding to the responsive neurons
df_responsive_neurons = df[list(responsive_neurons_new_csv)]
output_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/intersection_neuronal_activity_responsive_neurons_sd_2_average_100-110-AND-680-699.csv'

df_responsive_neurons.to_csv(output_path, index=False)

print(f"Filtered DataFrame with final responsive neurons saved to {output_path}")

# Define start and end times for plotting 80-150/ 660-740
# Note: stimulus window 101-106/ 681-695
start_time = 80
end_time = 740 # adjust
time_range = (start_time, end_time)

# Plot results without legend
plot_neuron_activity(df, responsive_neurons_new_csv, time_range, 'Responsive neurons from new CSV', show_legend=False)

# %%
