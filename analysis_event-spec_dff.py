
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
"Actions for 1099, using 0-1000 frames with threshold 3 == frames between actions"
"Important: behaviour.csv starts with '1', activity.csv with '0' - fixed (edited file)"
"start of event == in output filename from dff_calculation.py"
# Stimulus: 100-105/ 680-694 (2)
# Forward: 51, 74 (26)
# Turn: (14)
# HP: (27)

#%%
#1. Pick a certain time window and read out the peak DF/F and the average DF/F in that time window for each cell body
#2. Compute the mean peak and the mean average across all cell bodies, and the SD.
#3. Find all neurons whose responses are 1.5 or 2 SD higher than the overall mean

"Input dff from dff_calculation.py"
# Input: df == activity traces for each cell body in the brain per event, 
# read out from tif stack generated with sliding window for dff
# Output: Neuron names and activity that meet the statistic requirements. Specifically for each event
# check for existing filename before saving


import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_neuron_activity(df, neurons, time_range, title):
    plt.figure(figsize=(12, 8))
    time_indices = df.index[time_range[0]:time_range[1]+1]
    for neuron in neurons:
        plt.plot(time_indices, df.loc[time_indices, neuron], label=neuron)
    plt.title(title)
    plt.xlabel('Timepoint')
    plt.ylabel('ΔF/F')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()

def save_neurons_to_csv(neurons, filename, path=''):  
    filepath = os.path.join(path, filename)
    pd.DataFrame(neurons).to_csv(filepath, header=False)

def process_csv_files(directory, output_dir, window_length, adjustment):
    # Iterate through each CSV file in the specified directory
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            print(f"Processing file: {file}")
            
            # Extract start time from filename (e.g., output_dff_S_100.csv -> 100)
            start_time = int(file.split('_')[-1].split('.')[0])
            
            # Load the dataset
            df = pd.read_csv(os.path.join(directory, file))
            
            # Drop the 'timepoint' column
            if 'timepoint' in df.columns:
                df.drop(columns=['timepoint'], inplace=True)
            
            # Compute adjusted start time
            start_time_adjusted = start_time + adjustment
            end_time = start_time_adjusted + window_length
            time_range = (start_time_adjusted, end_time)
            time_range_plot = (start_time_adjusted - 30, start_time_adjusted + 100)
            
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
            
            if not responsive_neurons_average.empty:
                # Save neurons to CSV file with specified path
                "change name and adjust"
                filename = f"Turn_F0_15_adjust_9_SD_1-5_average_{start_time}_timepoint_{start_time}.csv"
                save_neurons_to_csv(responsive_neurons_average, filename, path=output_dir)
                
                # Save the original CSV file with only the columns corresponding to the responsive neurons
                df_responsive_neurons_average = df[list(responsive_neurons_average.index)]
                df_responsive_neurons_average.to_csv(os.path.join(output_dir, filename), index=False)
                
                # Plot results
                plot_neuron_activity(df, responsive_neurons_average.index, time_range_plot, f'Responsive neurons based on average ΔF/F (Start: {start_time})')

# Parameters
"change name"
directory = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_9/Turn_F0_15_adjust_9'
output_dir = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_9/Turn_F0_15_adjust_9/SD_1-5'
window_length = 10
adjustment = 0

# Process CSV files
process_csv_files(directory, output_dir, window_length, adjustment)




















#%%
# Find intersection between neurons that respond to first and sec action 
# Input: Neuron names that meet the statistic requirements for each event.
# Output: Neuron names! that meet statistic requirements for both events.

# List of file paths
file_paths = [
    '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/F/forward_F0_15_adjust_9_responsive_neurons_sd_1-5_average_51.csv',
    '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/F/forward_F0_15_adjust_9_responsive_neurons_sd_1-5_average_74.csv',
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
output_file_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/F/Intersection_Forward_F0_15_adjust_9_responsive_neurons_sd_1-5_average_51_AND_74.csv'
intersection.to_csv(output_file_path, index=False)

print("Intersection saved to", output_file_path)


#%%
"very restricted - needs to be more generic to handle multiple events" #TODO
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
#df = pd.read_csv('/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/F/Forward_F0_15_adjust_9_activity_of_responsive_neurons_sd_1-5_average_51.csv')

# Neuronal activity for action 2
df = pd.read_csv('/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/F/Forward_F0_15_adjust_9_activity_of_responsive_neurons_sd_1-5_average_74.csv')


# Clean column names to remove leading/trailing spaces and quotes
df.columns = df.columns.str.strip().str.replace('"', '')

# Load the second CSV file to find the intersection
"change name"
intersection_df = pd.read_csv('/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/F/Intersection_Forward_F0_15_adjust_9_responsive_neurons_sd_1-5_average_51_AND_74.csv')

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
#output_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/F/Intersection_Forward_F0_15_adjust_9_neuronal_activity_responsive_neurons_sd_1-5_average_51_AND_74_For_Action_1.csv'

# Action 2
output_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/F/Intersection_Forward_F0_15_adjust_9_neuronal_activity_responsive_neurons_sd_1-5_average_51_AND_74_For_Action_2.csv'


df_responsive_neurons.to_csv(output_path, index=False)

print(f"Filtered DataFrame with final responsive neurons saved to {output_path}")

# Define start and end times for plotting 80-150/ 660-740
start_time = 660
end_time = 740  # adjust
time_range = (start_time, end_time)

# Plot results without legend
plot_neuron_activity(df, matching_neurons, time_range, 'Responsive neurons from new CSV', show_legend=False)



# %%
# Plot neuronal activity (intersection) per action. 
# Plot columns in chunks against index with specified x-axis range

def plot_columns_in_chunks(df, chunk_size=5, x_start=None, x_end=None):
    num_cols = df.shape[1]
    column_chunks = [range(i, min(i + chunk_size, num_cols)) for i in range(0, num_cols, chunk_size)]

    for chunk_indices in column_chunks:
        plt.figure(figsize=(12, 8))
        
        plotted_anything = False  # Flag to check if anything was actually plotted
        
        for col_idx in chunk_indices:
            col_name = df.columns[col_idx]
            plt.plot(df.index, df[col_name], label=col_name)
            plotted_anything = True  # Set flag to True if at least one plot was made
        
        if plotted_anything:
            plt.title(f'Columns {chunk_indices[0]} to {chunk_indices[-1]} vs Index')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            
            if x_start is not None and x_end is not None:
                plt.xlim(x_start, x_end)  # Set x-axis limits if provided
            
            plt.show()
        else:
            print(f"No data plotted for columns {chunk_indices[0]} to {chunk_indices[-1]}")

# Load:
#Beh 1
file_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/F/Intersection_Forward_F0_15_adjust_9_neuronal_activity_responsive_neurons_sd_1-5_average_51_AND_74_For_Action_1.csv'

#Beh 2
#file_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/F/Intersection_Forward_F0_15_adjust_9_neuronal_activity_responsive_neurons_sd_1-5_average_51_AND_74_For_Action_2.csv'


df = pd.read_csv(file_path, index_col = False)  # Assuming the first column is the index

# Specify x-axis range (optional)
x_start = 10  # Replace with your desired start index 80, 640
x_end = 110   # Replace with your desired end index 140, 740

# Print some debug information
print(f"DataFrame head:\n{df.head()}\n")
print(f"Index values:\n{df.index}\n")
print(f"Columns in DataFrame:\n{df.columns}\n")

# Plot every 20 columns against the index with specified x-axis range
plot_columns_in_chunks(df, chunk_size=5, x_start=x_start, x_end=x_end)
# %%
