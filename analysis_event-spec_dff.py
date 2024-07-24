
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
"Actions for 1099, using 0-1000 frames with threshold 3 == frames between actions: from beh-structure.py"
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
                filename = f"Stimulus_F0_15_adjust_9_SD_1-5_average_{start_time}_timepoint_{start_time}.csv"
                save_neurons_to_csv(responsive_neurons_average, filename, path=output_dir)
                
                # Save the original CSV file with only the columns corresponding to the responsive neurons
                df_responsive_neurons_average = df[list(responsive_neurons_average.index)]
                df_responsive_neurons_average.to_csv(os.path.join(output_dir, filename), index=False)
                
                # Plot results
                plot_neuron_activity(df, responsive_neurons_average.index, time_range_plot, f'Responsive neurons based on average ΔF/F (Start: {start_time})')

# Parameters
"change name"
directory = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/Stim_F0_15_adjust_0_all'
output_dir = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/Stim_F0_15_adjust_0_all/SD_1-5'
window_length = 10
adjustment = 0

# Process CSV files
process_csv_files(directory, output_dir, window_length, adjustment)


#%%
# Find intersection between neurons that respond to two consecutive "action. Meaning, neurons that respond to an action and all actions before  
# Input: Neuron activity that meet the statistic requirements for each event.
# Output: Neuron names! that meet statistic requirements for the events.

"Important: If there are no common headers, there is a error message"
"This code is strict and excludes all neurons that are not meeting the statistic, even for a single action"

import pandas as pd
import os
import re
import glob
 
# Directory containing the stimulus files
input_dir = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/Stim_F0_15_adjust_0_all/SD_1-5/'

# Save directory
save_dir = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/Stim_F0_15_adjust_0_all/SD_1-5/intersection'
os.makedirs(save_dir, exist_ok=True)

# Define the action manually
action = 'Stimulus'

# Use glob to list all CSV files in the directory
file_paths = glob.glob(os.path.join(input_dir, f'{action}_*_timepoint_*.csv'))

# Check if any files are found
if not file_paths:
    raise FileNotFoundError(f'No files found for action "{action}". Please check the action name or the directory.')

# Extract timepoints from file paths
pattern = rf'{action}_.+_timepoint_(\d+).csv'
actions_timepoints = [(re.search(pattern, os.path.basename(fp)).group(1), fp) for fp in file_paths]
timepoints = sorted([(int(tp), fp) for tp, fp in actions_timepoints])

# Check if any valid files are found after pattern matching
if not timepoints:
    raise ValueError(f'No valid files found for action "{action}" with the specified pattern. Please check the file naming convention.')


# Function to merge DataFrames with common columns only
def merge_dataframes(df1, df2):
    common_columns = set(df1.columns).intersection(df2.columns)
    merged_df = pd.merge(df1[common_columns], df2[common_columns], on=list(common_columns), how='inner')
    return merged_df

# Read the CSV files into DataFrames
dataframes = {tp: pd.read_csv(fp) for tp, fp in timepoints}

# Initialize the merged DataFrame with the first file's data
merged_df = dataframes[timepoints[0][0]].copy()

# Merge all subsequent DataFrames iteratively
for tp, fp in timepoints[1:]:
    current_df = dataframes[tp]
    merged_df = merge_dataframes(merged_df, current_df)

    # Update the filename
    merged_filename = f'{action}_timepoint_' + '_'.join(map(str, [tp for tp, _ in timepoints[:timepoints.index((tp, fp))+1]])) + '.csv'
    
    # Save the merged DataFrame
    merged_df.to_csv(os.path.join(save_dir, merged_filename), index=False)

print('Merging and saving of neuron NAMES completed.')


#%%
# Input: dff from each event
# Input: intersection.csv Neuron Names
# Containig neurons that respond to two consecutive two consecutive "action". 
# Meaning, neurons that respond to an action and all actions before.
# Output: dff for each event including only neurons fromm intersection
import os
import pandas as pd

def load_csv_files(directory):
    """Load all CSV files in the specified directory into a dictionary of DataFrames."""
    dataframes = {}
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    print(f"Directory exists: {directory}")
    print(f"Contents of the directory: {os.listdir(directory)}")
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith(".csv"):
            df = pd.read_csv(filepath)
            dataframes[filename] = df
    return dataframes

def load_subset_csv(filepath):
    """Load the subset CSV file containing a subset of neuron names."""
    subset_df = pd.read_csv(filepath)
    return subset_df.columns.tolist()

def extract_neuronal_activity(df, neuron_subset):
    """Extract the neuronal activity data for the neurons listed in the subset CSV."""
    common_neurons = [neuron for neuron in neuron_subset if neuron in df.columns]
    if common_neurons:
        return df[common_neurons]
    return pd.DataFrame()

def save_extracted_data(df, input_filename, subset_filename, output_dir):
    """Save the extracted data to a new CSV file."""
    subset_name = os.path.splitext(os.path.basename(subset_filename))[0]
    output_filename = f"selected_neurons_intersection_{subset_name}_{os.path.splitext(input_filename)[0]}.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    df.to_csv(output_filepath, index=False)
    print(f"Neuronal activity data extracted and saved to {output_filepath}")

def main(data_dir, subset_file, output_dir):
    # Load all CSV files in the directory
    dataframes = load_csv_files(data_dir)
    
    # Check the loaded CSV files
    if len(dataframes) == 0:
        raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")

    # Load the subset CSV file
    neuron_subset = load_subset_csv(subset_file)

    # Process each CSV file and save the extracted data
    for filename, df in dataframes.items():
        extracted_data = extract_neuronal_activity(df, neuron_subset)
        if not extracted_data.empty:
            save_extracted_data(extracted_data, filename, subset_file, output_dir)

# Example usage:
main('/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/Stim_F0_15_adjust_0_all', '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/Stim_F0_15_adjust_0_all/SD_1-5/intersection/Stimulus_timepoint_100_680_1260.csv', '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/Stim_F0_15_adjust_0_all/selected_neurons')



#%%
# Input csv of selected neurons. (== neuronal activity of neurons that response to multiple events)
# Output: plot average of neurons for all events. (each event is a different csv)
# set action start (== *_timepoint.csv) to 0, and plot +/- specific no. of frames
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def load_csv_files(directory):
    """Load all CSV files in the specified directory into a dictionary of DataFrames."""
    dataframes = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith(".csv"):
            df = pd.read_csv(filepath)
            dataframes[filename] = df
    return dataframes

def extract_timepoints_from_filename(filename):
    """Extract the list of timepoints from the filename using regex."""
    match = re.search(r'_S_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Timepoint not found in filename: {filename}")

def add_aligned_index(df, timepoint):
    """Add an index column aligned so that the action start is set to 0."""
    df['index'] = df.index - timepoint
    return df.set_index('index')

def plot_individual_neurons(dataframes, frames_before, frames_after):
    """Plot the activity of individual neurons and their averages for all events."""
    neuron_activity = {}

    for filename, df in dataframes.items():
        try:
            timepoint = extract_timepoints_from_filename(filename)
        except ValueError as e:
            print(e)
            continue

        aligned_df = add_aligned_index(df, timepoint)
        start = -frames_before
        end = frames_after + 1
        selected_range_df = aligned_df.loc[start:end]

        print(f"Aligned data for {filename} at timepoint {timepoint}:")
        print(selected_range_df.head())  # Print first few rows to verify alignment

        for neuron in selected_range_df.columns:
            if neuron != 'index':
                if neuron not in neuron_activity:
                    neuron_activity[neuron] = []
                neuron_activity[neuron].append(selected_range_df[neuron])

    for neuron, traces in neuron_activity.items():
        all_traces = pd.concat(traces, axis=1)
        avg_trace = all_traces.mean(axis=1)

        print(f"Collected traces for neuron {neuron}:")
        print(all_traces.head())  # Print first few rows to verify traces collection

        plt.figure(figsize=(10, 6))
        
        # Plot all individual traces
        for trace in traces:
            plt.plot(trace.index, trace.values, color='black', alpha=0.3)

        # Plot the average activity
        plt.plot(avg_trace.index, avg_trace.values, color='red', label='Average Activity')

        plt.xlabel('Frame')
        plt.ylabel('Neuronal Activity')
        plt.title(f'Neuronal Activity for {neuron}')
        plt.axvline(x=0, color='r', linestyle='--', label='Action Start')
        plt.legend()
        plt.show()

def main(selected_neurons_dir, frames_before, frames_after):
    # Load all CSV files of selected neurons
    dataframes = load_csv_files(selected_neurons_dir)
    
    # Plot the activity of individual neurons
    plot_individual_neurons(dataframes, frames_before, frames_after)

# Example usage:
main('/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/Stim_F0_15_adjust_0_all/selected_neurons', frames_before=30, frames_after=30)





















#%%

"keep this block!!!"
# Find intersection between neurons that respond to 2 action-events 
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
# the loaded activity files must be correspond to each action
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
#file_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/Stim_F0_15_adjust_0_all/SD_1-5/Stimulus_F0_15_adjust_9_SD_1-5_average_100_timepoint_100.csv'

#Beh 2
file_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/Stim_F0_15_adjust_0_all/selected_neurons/selected_neurons_intersection_Stimulus_timepoint_100_680_1260_output_dff_S_1260.csv'


df = pd.read_csv(file_path, index_col = False)  # Assuming the first column is the index

# Specify x-axis range (optional)
x_start = 1220  # Replace with your desired start index 80, 640
x_end = 1300   # Replace with your desired end index 140, 740

# Print some debug information
print(f"DataFrame head:\n{df.head()}\n")
print(f"Index values:\n{df.index}\n")
print(f"Columns in DataFrame:\n{df.columns}\n")

# Plot every 20 columns against the index with specified x-axis range
plot_columns_in_chunks(df, chunk_size=5, x_start=x_start, x_end=x_end)
# %%
