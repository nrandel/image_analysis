# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob

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

import pandas as pd
import os
import re
import glob

# Directory containing the stimulus files
input_dir = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/T_adjust-9/'

# Use glob to list all CSV files in the directory
file_paths = glob.glob(os.path.join(input_dir, 'Turn_long-sliding-window_activity_of_responsive_neurons_sd_1-5_average_*.csv'))

# Extract action and timepoints from file paths
pattern = r'Turn_(.*?)_average_(\d+).csv'
actions_timepoints = [(re.search(pattern, os.path.basename(fp)).groups()) for fp in file_paths]
actions, timepoints = zip(*actions_timepoints)
timepoints = sorted(map(int, timepoints))

# Read the CSV files into a list of dataframes and debug the content
dfs = []
for file_path in file_paths:
    try:
        df = pd.read_csv(file_path)
        dfs.append(df)
        print(f"Contents of {file_path}:")
        print(df.head())
    except pd.errors.EmptyDataError:
        print(f"Warning: {file_path} is empty or improperly formatted and will be skipped.")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Extract the headers from each dataframe
headers = [set(df.columns) for df in dfs]

# Find the common headers across all dataframes
common_headers = set.intersection(*headers)

print(f"Common headers: {common_headers}")

# Filter the dataframes to include only the common headers
filtered_dfs = [df[list(common_headers)] for df in dfs]

# Function to save the filtered dataframes with common headers
def save_intersections(filtered_dfs, timepoints, save_dir, action):
    intersection = filtered_dfs[0]
    cumulative_timepoints = [timepoints[0]]
    
    for i, df in enumerate(filtered_dfs[1:], start=1):
        intersection = pd.merge(intersection, df, how='inner', on=list(common_headers))
        cumulative_timepoints.append(timepoints[i])
        
        # Generate the filename
        filename = f"Intersection_Stimulus_{action}_average_{'_'.join(map(str, cumulative_timepoints))}.csv"
        filepath = os.path.join(save_dir, filename)
        
        # Save the result to a new CSV file
        intersection.to_csv(filepath, index=False)
        print(f"Intersection saved to {filepath}")
    
    # Return the final intersection for debugging
    return intersection

# Save directory
save_dir = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/T_adjust-9/intersection'

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Save intersections and get the final intersection for debugging
if filtered_dfs:
    final_intersection = save_intersections(filtered_dfs, timepoints, save_dir, actions[0])
    
    # Debug the final intersection dataframe
    print("Final intersection dataframe:")
    print(final_intersection.head())
else:
    print("No valid dataframes to process.")


# %%
"move later to plot/"
# Function to plot columns in chunks against index with specified x-axis range

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
file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/T/intersection/Intersection_Stimulus_long-sliding-window_activity_of_responsive_neurons_sd_1-5_average_13_33_59_120.csv'
df = pd.read_csv(file_path, index_col = False)  # Assuming the first column is the index

# Specify x-axis range (optional)
x_start = 3  # Replace with your desired start index 80, 640
x_end = 130   # Replace with your desired end index 140, 740

# Print some debug information
print(f"DataFrame head:\n{df.head()}\n")
print(f"Index values:\n{df.index}\n")
print(f"Columns in DataFrame:\n{df.columns}\n")

# Plot every 20 columns against the index with specified x-axis range
plot_columns_in_chunks(df, chunk_size=5, x_start=x_start, x_end=x_end)




# %%
