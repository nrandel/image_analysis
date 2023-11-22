# %%
# Import libraries
import numpy as np
import pandas as pd

# %%
# Load csvfile

behavioural_data = pd.read_csv('/Users/nadine/Documents/paper/single-larva/BW_included_event_annotations.csv')

beh = ["F", "H", "TL", "TR", "B", "Q", "P", "U"]
columns = ["start", "end", "beh", "subject-id", "beh_before", "beh_before_start", "beh_before_end", "beh_after", "beh_after_start", "beh_after_end", "starts_within_event", "stops_within_event"]

# %%
# Define single event type you want to extract
desired_event = 'F'

# Define columns to keep
columns_to_keep = ["start", "end", "beh"]

# Filter rows where 'beh' column matches the desired event type
filtered_behavioural_data = behavioural_data[behavioural_data['beh'] == desired_event][columns_to_keep]

# Show the filtered data
print(filtered_behavioural_data)

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
postceding_event = ['B']  # You can define multiple postceding events e.g., ['TL', 'TR']

# Define the maximum allowable delay in frames
max_delay_frames = 6  # Change this value according to your requirement

# Calculate the delay between beh_end and beh_after_start
behavioural_data['delay'] = behavioural_data['beh_after_start'] - behavioural_data['end']

# Define columns to keep
columns_to_keep = ["start", "end", "beh", "beh_after", "beh_after_start"]

# Filter rows where preceding event is e.g., 'F' and postceding event occurs after a delay
filtered_behavioural_data = behavioural_data[(behavioural_data['beh'] == preceding_event) & (behavioural_data['beh_after'].isin(postceding_event)) & (behavioural_data['delay'] <= max_delay_frames)]

# Show the filtered data
print(filtered_behavioural_data)

# Test 
#print(behavioural_data[behavioural_data['beh'] == preceding_event])  # Check rows where 'beh' matches the preceding event
#print(behavioural_data[behavioural_data['beh_after'].isin(postceding_event)])  # Check rows where 'beh_after' matches postceding events
#print(behavioural_data[behavioural_data['delay'] == delay_frames])  # Check rows where delay matches delay_frames

# %%
