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
# Define multiple events for 'AND' condition
desired_events_and = ["TL", "TR"]  # Example events

# Define columns to keep
columns_to_keep = ["start", "end", "beh"]

# Filter rows where 'beh' column matches all desired events (AND condition)
filtered_behavioural_data_and = behavioural_data[behavioural_data['beh'].isin(desired_events_and)][columns_to_keep]

# Show the filtered data
print(filtered_behavioural_data_and)

#%%
#%%
# Determinne event duration ("end" - "start")

event_duration = filtered_behavioural_data_and['end'] - filtered_behavioural_data_and['start']
print(event_duration)

# Save the result to a new CSV file
event_duration.to_csv('/Users/nadine/Documents/paper/single-larva/generated-data/TL-TR-event_durtion.csv', index=False)


# Average duration
avg_event_duration = event_duration.mean() 
print(avg_event_duration)

# %%
# Read csvfile of merged activity traces
#merged_activity_traces = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/2023-12-05/radius_3-3-1/out_3-3-1.csv')

# %%
'''
# Extract start time points of filtered events
start_time_points = filtered_behavioural_data_and['start'].tolist()
print(start_time_points)

# Define a range of specific time points to add/subtract
'''Either positive OR negative values'''
time_range = -20  # For example, a range of 5 time points, includinng the start-point

# Create a list of tuples containing event label and time point range
event_time_ranges = []
for event_num, start in enumerate(start_time_points, start=1):
    if time_range >= 0:
        time_points_range = [start + i for i in range(time_range)]
    else:
        time_points_range = [start - i for i in range(abs(time_range))]
    event_label = f"{desired_events_and}_{event_num}"
    event_time_ranges.append((event_label, time_points_range))

# Convert the list of tuples to a DataFrame
event_time_ranges_df = pd.DataFrame(event_time_ranges, columns=['Event_Label', 'Time_Points_Range'])

# Export the DataFrame to a CSV file
''' Add timerange to file name '''
event_time_ranges_df.to_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/2023-12-05/radius_3-3-1/Action-selection/TL-TR_radius-3-3-1-20TP.csv', index=False)
'''

# %%
start_time_points = filtered_behavioural_data_and['start'].tolist()
print(start_time_points)

# Define a range of specific time points to add/subtract
'''Range of positive AND negative values'''
negative_range = range(-60, 0)  # range(-10, 0) Range from -10 to -1
positive_range = range(0, 15)    # range(0, 6) Range from 0 to 5

# Merge the ranges into a single list
time_points_range = list(negative_range) + list(positive_range)
print(time_points_range)

# Create a list of tuples containing event label and time point range
event_time_ranges = []
for event_num, start in enumerate(start_time_points, start=1):
    event_label = f"{desired_events_and}_{event_num}"
    # Generate the time points range by adding the start point to each value in the range
    event_time_ranges.append((event_label, [start + i for i in time_points_range]))

# If you want to print the event_time_ranges
print(event_time_ranges)

# Convert the list of tuples to a DataFrame
event_time_ranges_df = pd.DataFrame(event_time_ranges, columns=['Event_Label', 'Time_Points_Range'])

# Export the DataFrame to a CSV file
''' Add timerange to file name '''
event_time_ranges_df.to_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/alliged_behaviour/TL-TR-Decision-60_15TP.csv', index=False)

# %%
# Extract the data from the 'merged_activity_traces' CSV corresponding to the time points specified in the 'event_time_ranges_df' CSV. 

# Read the CSV files, specifying the 'Time_Points_Range' column to be converted using ast.literal_eval
merged_activity_traces = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/traces_3-3-1.csv')

event_time_ranges_df = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/alliged_behaviour/truncated/TL-TR-Decision-60_15TP_Event_1-15.csv', converters={'Time_Points_Range': ast.literal_eval})

# Remove the 'timepoint' column
merged_activity_traces.drop('time', axis=1, inplace=True)  # axis=1 specifies column-wise operation

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
result_df.to_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/2023-12-05/radius_3-3-1/Action_selection/final_output_Tl-TR-Decision-60_15TP_Event_1-15.csv', index=False)

# %%