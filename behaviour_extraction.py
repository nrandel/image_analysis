# %%
# Import libraries
import numpy as np
import pandas as pd
import os
import re

# %%
'''
# Generate merged csv of activity traces
# Path to the directory containing your CSV files
directory_path = '/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces'

# List all CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# Sort CSV files if necessary (change the sorting logic as needed)
#csv_files.sort()  # Change this sorting logic based on your requirement (sorts, 1, 10, 11, 2, 3,..)

# Create a sorted copy of the list of CSV file names with a custom key for numerical sorting
def extract_number(filename):
    try:
        return int(re.search(r'\d+', filename.split('_')[-1]).group())
    except AttributeError:
        return float('inf')  # Return a high value for filenames where a number isn't found

sorted_csv_files = sorted(csv_files, key=extract_number)

# Print the sorted list of CSV file names
for file in sorted_csv_files:
    print(file)

# Initialize an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

# Iterate through each sorted CSV file, read it, and append its contents to the merged_data DataFrame
for file in sorted_csv_files:
    file_path = os.path.join(directory_path, file)
    data = pd.read_csv(file_path)
    merged_data = pd.concat([merged_data, data], ignore_index=True)

# Output the merged data to a single CSV file
merged_data.to_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/merged_activity_traces.csv', index=False)
'''

# %%
# Load csvfile of behavioural events used for analysis (model)

behavioural_data = pd.read_csv('/Users/nadine/Documents/paper/single-larva/BW_included_event_annotations.csv')

beh = ["F", "H", "TL", "TR", "B", "Q", "P", "U"]
columns = ["start", "end", "beh", "subject-id", "beh_before", "beh_before_start", "beh_before_end", "beh_after", "beh_after_start", "beh_after_end", "starts_within_event", "stops_within_event"]

# %%
# Define single event type you want to extract
desired_event = 'F'

# Define columns to keep
columns_to_keep = ["start", "end", "beh"]

# Filter rows where 'beh' column matches the desired event type
filtered_behavioural_data_single_event = behavioural_data[behavioural_data['beh'] == desired_event][columns_to_keep]

# Show the filtered data
print(filtered_behavioural_data_single_event)

# %%
# Define multiple events for 'AND' condition
desired_events_and = ["TL", "TR"]  # Example events

# Define columns to keep
columns_to_keep = ["start", "end", "beh"]

# Filter rows where 'beh' column matches all desired events (AND condition)
filtered_behavioural_data_and = behavioural_data[behavioural_data['beh'].isin(desired_events_and)][columns_to_keep]

# Show the filtered data
print(filtered_behavioural_data_and)

# %%
# Define multiple events for 'OR' condition
desired_events_or = ['F', 'TL']  # Example events

# Define columns to keep
columns_to_keep = ["start", "end", "beh"]

# Filter rows where 'beh' column matches at least one desired event (OR condition)
filtered_behavioural_data_or = behavioural_data[behavioural_data['beh'].isin(desired_events_or)][columns_to_keep]

# Show the filtered data
print(filtered_behavioural_data_or)

# %%
#Filter for behavioural sequence, where preceding event is e.g., 'F' and postceding event occurs after a delay
# Define the preceding event type 
preceding_event = 'TR'

# Define the postceding event type 
postceding_event = ['B']

# Define the maximum allowable delay in frames
max_delay_frames = 6

# Calculate the delay between beh_end and beh_after_start
behavioural_data['delay'] = behavioural_data['beh_after_start'] - behavioural_data['end']

# Define columns to keep
columns_to_keep = ['start', 'end', 'beh', 'beh_after', 'beh_after_start']

# Filter rows based on conditions
filtered_data = behavioural_data[
    (behavioural_data['beh'] == preceding_event) &
    (behavioural_data['beh_after'].isin(postceding_event)) &
    (behavioural_data['delay'] <= max_delay_frames)
]

# Select specific columns from the filtered rows
filtered_behavioural_data_sequence = filtered_data[columns_to_keep]

# Show the filtered data
print(filtered_behavioural_data_sequence)

# %%
