#%%
# Import
import pandas as pd

#%%
"""
# Adjust behaviour.cs starting with '1', to the activity.csv starts with '0'.
# Therefore, the behaviour.csv must -1 for star and end column 

import pandas as pd

def adjust_start_end_values(input_csv, output_csv):
    # Read the CSV file with a semicolon delimiter
    df = pd.read_csv(input_csv, delimiter=';')

    # Subtract 1 from each value in the START and END columns
    df['START'] = df['START'] - 1
    df['END'] = df['END'] - 1

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False, sep=';')

# Date
input_csv = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/18-02-15L1-behavior-ol.csv'
output_csv = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/18-02-15L1-behavior-ol-1.csv'
adjust_start_end_values(input_csv, output_csv)
"""

# %%
# Extract specific behaviours, during a specific time window, e.g., 0-1000 frammes
# Control for subsequent behaviours
# No threshold (gap_threshold = None): Extract behaviors without any gap threshold (== frames between actions).
# gap_threshold set with include_less_than_threshold = True: Extract behaviors where the gap between the end of one behavior and the start of the next is less than or equal to gap_threshold.
# gap_threshold set with include_greater_than_threshold = True: Extract behaviors where the gap between the end of one behavior and the start of the next is strictly greater than gap_threshold.
# Usage:
# No Gap Threshold: Set gap_threshold = None.
# Include Behaviors <= Threshold: Set gap_threshold and include_less_than_threshold = True.
# Include Behaviors > Threshold: Set gap_threshold and include_greater_than_threshold = True.


def filter_behaviors(df, start_range=(None, None), behaviors=None, gap_threshold=None, include_less_than_threshold=False, include_greater_than_threshold=False):
    # Define the behavior mapping
    beh_map = {
        'fw': 'F',
        'bw': 'B',
        'stim': 'S',
        'hunch': 'H',
        'other': 'O',
        'turn': 'T',
        'left turn': 'TL',
        'right turn': 'TR',
        'HP': 'HP'
    }

    # Prepare an empty list to collect the filtered rows
    new_rows = []

    # Sort the dataframe by the start time to ensure proper sequence
    df = df.sort_values(by='START').reset_index(drop=True)

    # Iterate through each row in the dataframe
    previous_end = None
    for _, row in df.iterrows():
        start = row['START']
        end = row['END']
        for beh in beh_map.keys():
            if row[beh] == 1:
                # Apply start range filtering
                if start_range[0] is not None and start < start_range[0]:
                    continue
                if start_range[1] is not None and start > start_range[1]:
                    continue
                # Apply behavior filtering
                if behaviors is not None and beh_map[beh] not in behaviors:
                    continue
                # Apply gap threshold filtering
                if gap_threshold is not None and previous_end is not None:
                    if include_less_than_threshold:
                        if start - previous_end <= gap_threshold:
                            new_row = [
                                start,
                                end,
                                beh_map[beh], # Behavior
                                'CW_18-02-15-L1', # Subject ID
                            ]
                            new_rows.append(new_row)
                    if include_greater_than_threshold:
                        if start - previous_end > gap_threshold:
                            new_row = [
                                start,
                                end,
                                beh_map[beh], # Behavior
                                'CW_18-02-15-L1', # Subject ID
                            ]
                            new_rows.append(new_row)
                else:
                    new_row = [
                        start,
                        end,
                        beh_map[beh], # Behavior
                        'CW_18-02-15-L1', # Subject ID
                    ]
                    new_rows.append(new_row)
                previous_end = end

    # Convert the list of rows into a DataFrame
    new_df = pd.DataFrame(new_rows, columns=[
        'start', 'end', 'beh', 'subject_id'
    ])

    return new_df

# Read csv
input_csv = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/18-02-15L1-behavior-ol-1.csv'

# Read the CSV file
df = pd.read_csv(input_csv, sep=';')

# Set the start range and behaviors to filter (optional)
start_range = (1, 10000)
behaviors = ['F']  # Sspecific behaviors to filter for 'F', 'T', 'HP'
gap_threshold = 3  # Gap threshold in the same time unit as START and END columns
include_less_than_threshold = False  # Set to True to include behaviors with gaps less than or equal to threshold
include_greater_than_threshold = True  # Set to True to include behaviors with gaps strictly greater than threshold

# Filter the DataFrame
filtered_df = filter_behaviors(df, start_range, behaviors, gap_threshold, include_less_than_threshold, include_greater_than_threshold)

# Save the filtered DataFrame to a new CSV file
output_csv = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/action/Forward_threshold_3_18-02-15L1-behavior-ol_filtered_1-10100.csv'
filtered_df.to_csv(output_csv, index=False)

# Output the filtered DataFrame for reference
print(filtered_df.head())



# %%
