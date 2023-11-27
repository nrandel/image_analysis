# %%
# Import libraries
import numpy as np
import pandas as pd
import os
import re
import ast

# %%
# Load csvfile of behavioural events used for analysis (model)
behavioural_data = pd.read_csv('/Users/nadine/Documents/paper/single-larva/BW_included_event_annotations.csv')

beh = ["F", "H", "TL", "TR", "B", "Q", "P", "U"]
columns = ["start", "end", "beh", "subject-id", "beh_before", "beh_before_start", "beh_before_end", "beh_after", "beh_after_start", "beh_after_end", "starts_within_event", "stops_within_event"]

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
# Read csvfile of merged activity traces
merged_activity_traces = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/2023-11-24/merged_activity_traces.csv')

# %%
# Extract start time points of filtered events
start_time_points = filtered_behavioural_data_sequence['start'].tolist()
#print(start_time_points)

# Define a range of specific time points to add/subtract
time_range = -20  # Negative value implies subtracting from the start-point

# Create a list of tuples containing event label and time point range
# Here the preceeding event is the starting point, but can be changed to postceding event
event_time_ranges = []
for event_num, start in enumerate(start_time_points, start=1):
    if time_range >= 0:
        time_points_range = [start + i for i in range(time_range)]
    else:
        time_points_range = [start - i for i in range(abs(time_range))]
    event_label = f"{preceding_event}_{event_num}"
    event_time_ranges.append((event_label, time_points_range))

# Convert the list of tuples to a DataFrame
event_time_ranges_df = pd.DataFrame(event_time_ranges, columns=['Event_Label', 'Time_Points_Range'])

# Export the DataFrame to a CSV file
#event_time_ranges_df.to_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/2023-11-24/Decision_beh-after_-20TP/sequence-TR-B-Decision-20TP.csv', index=False)
# %%

# Extract the data from the 'merged_activity_traces' CSV corresponding to the time points specified in the 'event_time_ranges_df' CSV. 

# Read the CSV files, specifying the 'Time_Points_Range' column to be converted using ast.literal_eval
merged_activity_traces = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/merged_activity_traces.csv')
event_time_ranges_df = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/event_time_ranges.csv', converters={'Time_Points_Range': ast.literal_eval})

# Iterate through each row in event_time_ranges_df to extract data
final_output = []

for _, row in event_time_ranges_df.iterrows():
    event_label = row['Event_Label']
    time_points_range = row['Time_Points_Range']
    
    # Extract data from merged_activity_traces corresponding to time points
    data_for_event = merged_activity_traces.iloc[time_points_range]
    
    # Reshape the extracted data for each time point to match the desired format
    data_for_event = data_for_event.stack().reset_index()
    data_for_event.columns = ['Time_Point', 'Neuron', 'Value']
    
    # Add event label to the extracted data
    data_for_event['Event_Number'] = event_label
    
    # Append the data to the final output list
    final_output.append(data_for_event)

# Concatenate all extracted data into a single DataFrame
result_df = pd.concat(final_output, ignore_index=True)

# Save the result to a new CSV file
result_df.to_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/final_output.csv', index=False)
