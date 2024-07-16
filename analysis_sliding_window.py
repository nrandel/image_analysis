# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
"Actions for 1099, using 0-1000 frames with threshold 3 == frames between actions"
# Stimulus: 100-105/ 680-694 (2)
# Forward: (26)
# Turn: (14)
# HP: (27)

# %%
"Important: behaviour.csv starts with '1', activity.csv with '0' - fixed for MY dff (edited) and sliding window (raw)"

#%%
#1. Pick a certain time window and read out the peak DF/F and the average DF/F in that time window for each cell body
#2. Compute the mean average across all cell bodies, and the SD.
#3. Find all neurons whose responses are 1.5 SD higher than the overall mean

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

def save_neurons_to_csv(neurons, filename, path=''):  
    filepath = os.path.join(path, filename)  
    pd.DataFrame(neurons).to_csv(filepath, header=False)

# Load the dataset from the CSV file (dff sliding window)
df = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_t_stacks-dff_long-slidingWindow.csv')

# Drop the 'timepoint' column and use the index as the new timepoint
df.drop(columns=['timepoint'], inplace=True)

# Define fixed window length for calculating SD (adjust as needed)
window_length = 10

# Define start adjustment (set as needed). e.g., -9
adjustment = 0

# Read time windows from CSV and adjust start if needed
time_windows_file = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/action/Turn_threshold_3_18-02-15L1-behavior-ol_filtered_1-1000.csv'
time_windows_df = pd.read_csv(time_windows_file)

save_dir = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/T_adjust-9'

for index, row in time_windows_df.iterrows():
    start_time = int(row['start'])
    end_time = start_time + window_length
    
    start_time_adjusted = start_time + adjustment
    
    time_range = (start_time_adjusted, end_time)
    time_range_plot = (start_time_adjusted - 30, start_time_adjusted + 100)  # Adjusted for plotting
    
    # Compute average ΔF/F in the time window for each neuron
    time_window_data = df.iloc[time_range[0]:time_range[1]+1]
    average_dff = time_window_data.mean()
    
    # Compute mean and standard deviation for average ΔF/F
    mean_average_dff = average_dff.mean()
    sd_average_dff = average_dff.std()
    
    # Define the threshold for identifying responsive neurons
    threshold_average = mean_average_dff + 1.5 * sd_average_dff
    
    # Identify neurons whose responses are 1.5 SD higher than the mean
    responsive_neurons_average = average_dff[average_dff > threshold_average]
    
    # Save neurons to CSV file with specified path
    filename = f"Turn_long-sliding-window_activity_of_responsive_neurons_sd_1-5_average_{start_time}.csv"
    save_neurons_to_csv(responsive_neurons_average, filename, path=save_dir)
    
    # Save the original CSV file with only the columns corresponding to the responsive neurons
    df_responsive_neurons_average = df[list(responsive_neurons_average.index)]
    df_responsive_neurons_average.to_csv(os.path.join(save_dir, filename), index=False)
    
    # Plot results
    plot_neuron_activity(df, responsive_neurons_average.index, time_range_plot, f'Responsive neurons based on average ΔF/F (Start: {start_time})')


#%%
# Find intersection between neurons that respond to first and sec action 
# Input: Neuron names that meet the statistic requirements for each event.
# Output: Neuron names! that meet statistic requirements for both events.

# List of file paths
file_paths = [
    '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/responsive_neurons_sd_2_average_100-110.csv',
    '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/responsive_neurons_sd_2_average_100-110.csv',
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
"change file name"
intersection.to_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/intersection_responsive_neurons_sd_2_average_100-110-AND-680-699.csv', index=False)

print("Intersection saved to output.csv")


#%%
# Get neuron activity that meet statistic for actions

# Input 1: inntersection_df == Neuron names! that meet statistic requirements for both events.
# Input 2: df == activity traces for each cell body in the brain, 
# read out from tif stack generated with sliding window for dff
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


# Input:
