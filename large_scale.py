#%%
#TEST
import csv

# Define the path to your input text file and output CSV file
input_file_path = '/Users/nadine/Documents/Zlatic_lab/1099_spatial-filtered/neuron_centroids.txt'
output_file_path = '/Users/nadine/Documents/Zlatic_lab/1099_spatial-filtered/neuron_centroids.csv'

# Read the text file and convert the values to floating point numbers
converted_data = []
with open(input_file_path, 'r') as infile:
    for line in infile:
        # Split the line by commas and convert each value to float
        converted_line = [float(value) for value in line.split(',')]
        converted_data.append(converted_line)

# Write the converted data to a CSV file
with open(output_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(converted_data)

print("Data has been successfully converted and saved to 'converted_data.csv'")




#%%
import numpy as np

def inspect_npz(npz_path):
    # Load the NPZ file
    data = np.load(npz_path)
    
    # Print the keys and some information about each key's associated data
    print(f"Keys in the NPZ file: {data.files}")
    for key in data.files:
        print(f"\nKey: {key}")
        print(f"Shape: {data[key].shape}")
        print(f"Data Type: {data[key].dtype}")
        print(f"First few elements: {data[key].flat[:10]}")  # Print first few elements if array is large

# Path to the NPZ file
npz_path = '/Users/nadine/Documents/Zlatic_lab/1099_spatial-filtered/neuron_traces_cleaned.npz'

# Inspect the NPZ file
inspect_npz(npz_path)

# %%























#%%

import csv
import matplotlib.pyplot as plt

def get_value(file_path, column_name, row_index):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i == row_index:
                return row[column_name]
    return None

def plot_column(file_path, column_name, start_row_index, end_row_index):
    column_data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if start_row_index <= i <= end_row_index:
                column_data.append(float(row[column_name]))  # Convert to float for plotting
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(start_row_index, end_row_index + 1), column_data, label=column_name)
    plt.xlabel('Row Index')
    plt.ylabel('Value')
    plt.title(f'Plot of {column_name} from row {start_row_index} to {end_row_index}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Usage example
file_path = '/Users/nadine/Documents/Zlatic_lab/1099_spatial-filtered/analysis/dff_F0_15_adjust-9/Forward/output_dff_F_51.csv'
#column_name = '110.441156::371.654771::58.375592'
#column_name = '326.009847::473.670480::57.439819'
#column_name = '112.976473::410.183829::62.392316'
column_name = 'neuron_4620'
#row_index = 681  # 0-based index (681 corresponds to the 682nd row in the CSV)

# Get specific value
#value = get_value(file_path, column_name, row_index)
#print(f"Value in column '{column_name}' at row {row_index}:", value)

# Plot the column data from row 80 to row 140
start_row_index = 40
end_row_index = 100
plot_column(file_path, column_name, start_row_index, end_row_index)



































#%%
"test for my dff"
"works for stimulus"

import pandas as pd
import os
import re
import glob

# Directory containing the stimulus files
input_dir = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/Stim_F0_15_adjust_0/SD_1-5/'

# Save directory
save_dir = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/Stim_F0_15_adjust_0/SD_1-5/intersection'
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
actions_timepoints = [(re.search(pattern, os.path.basename(fp)), fp) for fp in file_paths]

# Filter out None values (where the pattern didn't match)
actions_timepoints = [(match.group(1), fp) for match, fp in actions_timepoints if match]

# Sort timepoints based on the numeric part
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

print("Merging and saving completed successfully.")



































# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%

#temp testing
# %%
"keep until fixed"
# test for sliding window

import pandas as pd
import os
import re
import glob

# Directory containing the stimulus files
input_dir = '/Users/nadine/Desktop/test_3/'

# Save directory
save_dir = '/Users/nadine/Desktop/test_3/intersection'
os.makedirs(save_dir, exist_ok=True)

# Define the action manually
action = 'Turn'

# Use glob to list all CSV files in the directory
file_paths = glob.glob(os.path.join(input_dir, f'{action}_*_average_*.csv'))

# Check if any files are found
if not file_paths:
    raise FileNotFoundError(f'No files found for action "{action}". Please check the action name or the directory.')

# Extract timepoints from file paths
pattern = rf'{action}_.+_average_(\d+).csv'
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

print('Merging and saving completed.')



















# %%
"keep until intersection is fixed!!"
#test for my dff 
import pandas as pd
import os
import re
import glob

# Directory containing the stimulus files
input_dir = '/Users/nadine/Desktop/test/'

# Save directory
save_dir = '/Users/nadine/Desktop/test/intersection'
os.makedirs(save_dir, exist_ok=True)

# Define the action manually
action = 'Forward'

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

print('Merging and saving completed.')
# %%
