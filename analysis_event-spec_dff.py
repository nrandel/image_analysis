
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
#Stimulus: 100-105/ 680-694

# %%
"Important: behaviour.csv starts with '1', activity.csv with '0' - fixed (edited file)"

#%%
#1. Pick a certain time window and read out the peak DF/F and the average DF/F in that time window for each cell body
#2. Compute the mean peak and the mean average across all cell bodies, and the SD.
#3. Find all neurons whose responses are 1.5 or 2 SD higher than the overall mean

"Input dff from dff_calculation.py"
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

# dff from dff_calculation.py
"Select"
df = pd.read_csv('/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/output_dff_S_680.csv')

# Drop the 'timepoint' column and use the index as the new timepoint
df.drop(columns=['timepoint'], inplace=True)

# Select the time window for statistical analysis (SD) 
# to define neurons that correlate with behaviour 

"Adjust time window for statistic analysis SD"
"Stimulus: Ft, action end + 5 frames"
time_window = list(range(680, 699))

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
"Choose SD"
threshold_peak = mean_peak_dff + 1.5 * sd_peak_dff
threshold_average = mean_average_dff + 1.5 * sd_average_dff

#threshold_peak = mean_peak_dff + 2 * sd_peak_dff
#threshold_average = mean_average_dff + 2 * sd_average_dff

# Identify neurons whose responses are XX SD higher than the overall mean
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

# Save to directory
save_dir = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff'


# Save neurons to CSV files with specified path
"change name: Event_F0_adjust_SD_peak or avg_eventstart"
#save_neurons_to_csv(responsive_neurons_peak, "Stimulus_F0_15_adjust_0_responsive_neurons_sd_1-5_peak_680.csv", path=save_dir)
save_neurons_to_csv(responsive_neurons_average, "Stimulus_F0_15_adjust_0_responsive_neurons_sd_1-5_average_680.csv", path=save_dir)
#save_neurons_to_csv(difference_neurons, "difference_neurons_682-699.csv", path=save_dir)
#save_neurons_to_csv(responsive_neurons_intersection, "responsive_neurons_intersectio_682-699.csv", path=save_dir)
#save_neurons_to_csv(non_responsive_neurons, "non_responsive_neurons_682-699.csv", path=save_dir)

# Save the original CSV file with only the columns corresponding to the responsive neurons
#df_responsive_neurons_peak = df[list(responsive_neurons_peak.index)]
df_responsive_neurons_average = df[list(responsive_neurons_average.index)]
#df_difference_neurons = df[list(difference_neurons)]
#df_responsive_neurons_intersection = df[list(responsive_neurons_intersection)]
#df_non_responsive_neurons = df[list(non_responsive_neurons)]

"change name: Event_F0_adjust_SD_peak or avg_eventstart"
#df_responsive_neurons_peak.to_csv(os.path.join(save_dir, "Stimulus_F0_15_adjust_0_activity_of_responsive_neurons_sd_1-5_peak_680.csv"), index=False)
df_responsive_neurons_average.to_csv(os.path.join(save_dir, "Stimulus_F0_15_adjust_0_activity_of_responsive_neurons_sd_1-5_average_680.csv"), index=False)
#df_difference_neurons.to_csv(os.path.join(save_dir, "activity_of_difference_neurons_682-699.csv"), index=False)
#df_responsive_neurons_intersection.to_csv(os.path.join(save_dir, "activity_of_responsive_neurons_intersection_682-699.csv"), index=False)
#df_non_responsive_neurons.to_csv(os.path.join(save_dir, "activity_of_non_responsive_neurons_682-699.csv"), index=False)

# Define start and end times for plotting 80-150/ 660-740
# Note: stimulus window 100-105/ 680-694
start_time = 660
end_time = 740
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

# List of file paths
file_paths = [
    '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/Stimulus_F0_15_adjust_0_responsive_neurons_sd_1-5_average_100.csv',
    '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/Stimulus_F0_15_adjust_0_responsive_neurons_sd_1-5_average_680.csv',
    # Add more file paths as needed
]

# Read the CSV files into a list of dataframes
dfs = [pd.read_csv(file_path, header=None) for file_path in file_paths]

# Extract the first column from each dataframe
first_columns = [df[[0]] for df in dfs]

# Find the intersection of the first columns
intersection = first_columns[0]
for col in first_columns[1:]:
    intersection = pd.merge(intersection, col, how='inner')

# Add a header to the intersection dataframe
intersection.columns = ['cell coordinates']

# Save the result to a new CSV file
"change name"
output_file_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/Intersection_Stimulus_F0_15_adjust_0_responsive_neurons_sd_1-5_average_100_AND_680.csv'
intersection.to_csv(output_file_path, index=False)

print("Intersection saved to", output_file_path)


#%%
"very restricted - needs to be more generic to handle multiple events"
# Get neuron activity that meet statistic for actions fron the intersectionn of responsive neurons

# Input 1: inntersection_df == Neuron names! that meet statistic requirements for both events.
# Input 2: df == activity traces for each cell body in the brain, 
# read out from tif stack generated with sliding window for dff. 
# Important: because the dff is calculated separately for each action, 
# the loaded activity file must be correspond to each action
# Output: Neuron traces that meet statistic requirements for both events (csv and plot)

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

# Load the activity for each action (csv)
"change name"
"needs to be run separately for action 1 and action 2"
# Neuronal activity for action 1
#df = pd.read_csv('/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/Stimulus_F0_15_adjust_0_activity_of_responsive_neurons_sd_1-5_average_100.csv')

# Neuronal activity for action 2
df = pd.read_csv('/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/Stimulus_F0_15_adjust_0_activity_of_responsive_neurons_sd_1-5_average_680.csv')


# Clean column names to remove leading/trailing spaces and quotes
df.columns = df.columns.str.strip().str.replace('"', '')

# Load the second CSV file to find the intersection
"change name"
intersection_df = pd.read_csv('/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/Intersection_Stimulus_F0_15_adjust_0_responsive_neurons_sd_1-5_average_100_AND_680.csv')

# Assuming that 'intersection_df' contains the responsive neuron identifiers in the first column
responsive_neurons_new_csv = intersection_df.iloc[:, 0].values

# Identify the neurons present in both the dataset and the responsive neurons list
matching_neurons = [neuron for neuron in responsive_neurons_new_csv if neuron in df.columns]

# Log neurons that were not found
non_matching_neurons = [neuron for neuron in responsive_neurons_new_csv if neuron not in df.columns]
if non_matching_neurons:
    print(f"Neurons not found in the dataset: {non_matching_neurons}")

# Save the original CSV file with only the columns corresponding to the responsive neurons
df_responsive_neurons = df[matching_neurons]

"change name"
"needs to be run separately for action 1 and action 2"
# Action 1
#output_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/Intersection_Stimulus_F0_15_adjust_0_neuronal_activity_responsive_neurons_sd_1-5_average_100_AND_680_For_Action_1.csv'

# Action 2
output_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/Intersection_Stimulus_F0_15_adjust_0_neuronal_activity_responsive_neurons_sd_1-5_average_100_AND_680_For_Action_2.csv'


df_responsive_neurons.to_csv(output_path, index=False)

print(f"Filtered DataFrame with final responsive neurons saved to {output_path}")

# Define start and end times for plotting 80-150/ 660-740
start_time = 660
end_time = 740  # adjust
time_range = (start_time, end_time)

# Plot results without legend
plot_neuron_activity(df, matching_neurons, time_range, 'Responsive neurons from new CSV', show_legend=False)




# %%
