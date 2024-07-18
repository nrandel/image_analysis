# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%

#temp testing
# %%
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
input_dir = '/Users/nadine/Desktop/test_1/'

# Save directory
save_dir = '/Users/nadine/Desktop/test_1/intersection'
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

print('Merging and saving completed.')
# %%
