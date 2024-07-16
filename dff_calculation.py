#%% 
import pandas as pd
import matplotlib.pyplot as plt
import os

#%%
"adjust 9 not working - fixing in progress"
#%%
# Calculate action specific dff with specific F0 and before action-start (Ft)
# ADAPT FO AVERAGE WINDOW START - ADD ADJUSTMENT 
# adjustment == moves F0 away from action start (Ft)
# Generic dff extraction for specific events, using behaviour.csv
# dff has to be extracted for each event and saved as separate csv

# Import 1: activity_csv_path == neuron activity extracted from raw fluorescence tiff
"behaviour_csv from beh_structure.py"
# Import 2: behaviour_csv_path == all action of the same kind 
# for specific time frame (e.g., 0-1000 frames) and threshold (== Gap between actions e.g., 3)

def calculate_dff(csv_path, F0_window, F0_start, Ft_start, problematic_neurons_path):
    # Load the CSV file
    data = pd.read_csv(csv_path)
    
    # Clean column names to remove leading/trailing spaces and quotes
    data.columns = data.columns.str.strip().str.replace('"', '')

    # Drop the 'timepoint' column and use the index as the new timepoint if it exists
    if 'timepoint' in data.columns:
        data.drop(columns=['timepoint'], inplace=True)
    
    # Reset the index to use it as the new timepoint
    data.reset_index(drop=True, inplace=True)
    
    # Extract the time points and neuronal activity
    time_points = data.index
    neuronal_activity = data  # The entire DataFrame except for the index column
    
    # Calculate ΔF/F
    dff = pd.DataFrame()
    dff['timepoint'] = time_points
    
    problematic_neurons = []

    for neuron in neuronal_activity.columns:
        if neuron == 'index':
            continue
        
        # Ensure the F0 window is within the valid range
        if F0_start + F0_window > Ft_start:
            raise ValueError("Specified F0 window overlaps with Ft start period.")
        
        # Extract the F0 window for each neuron, ignoring NaNs
        F0 = neuronal_activity[neuron].iloc[F0_start:F0_start+F0_window].mean(skipna=True)
        
        # Check if F0 is zero to avoid division by zero
        if F0 == 0:
            problematic_neurons.append(neuron)
            continue
        
        # Calculate ΔF/F for each time point
        dff[neuron] = [(ft - F0) / F0 for ft in neuronal_activity[neuron]]
    
    # Save the list of problematic neurons
    if problematic_neurons:
        pd.Series(problematic_neurons).to_csv(problematic_neurons_path, index=False, header=['problematic_neurons'])

    return dff

def plot_dff(dff, output_plot_path):
    # Plot the ΔF/F values for each neuron
    plt.figure(figsize=(15, 8))
    for neuron in dff.columns[1:]:  # Skip the 'timepoint' column
        plt.plot(dff['timepoint'], dff[neuron], label=neuron)
    
    plt.xlabel('Timepoint')
    plt.ylabel('ΔF/F')
    plt.title('ΔF/F Over Time for Neurons')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.show()


# Paths
activity_csv_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_klb-raw_data.csv'  
behaviour_csv_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/action/Stimulus_18-02-15L1-behavior-ol_filtered_1-1000.csv'  
# output file name: F0 average over 15 frames, 
# adjust == F0 window calculated 9 frames before start of behaviour (Ft)
output_dir = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_9/'  

# Parameters
F0_window = 15  # Number of points to average for F0
adjustment = 9  # Value to adjust F0 starting time. Use 0 for no adjustment.

# Load behaviour data
behaviour_data = pd.read_csv(behaviour_csv_path)

# Iterate through behaviour data
for index, row in behaviour_data.iterrows():
    start_time = int(row['start'])
    end_time = int(row['end'])
    behaviour = row['beh']
    
    # Calculate F0_start based on behaviour start time and adjustment
    F0_start = start_time - F0_window - adjustment
    
    # Ensure F0_start is not negative
    if F0_start < 0:
        print(f"Skipping row with start time {start_time} because F0_start is negative.")
        continue
    
    # Set Ft_start based on behaviour start time
    Ft_start = start_time

    # Calculate ΔF/F
    problematic_neurons_path = os.path.join(output_dir, f'problematic_neurons_{behaviour}_{start_time}.csv')
    dff = calculate_dff(activity_csv_path, F0_window, F0_start, Ft_start, problematic_neurons_path)
    
    # Save output ΔF/F data
    output_dff_path = os.path.join(output_dir, f'output_dff_{behaviour}_{start_time}.csv')
    dff.to_csv(output_dff_path, index=False)

    # Optionally plot the results (too many neurons, here)
    # output_plot_path = os.path.join(output_dir, f'dff_plot_{behaviour}_{start_time}.png')
    # plot_dff(dff, output_plot_path)

    # Print some debug information
    print(f"Processed row with start time {start_time} and behaviour {behaviour}.")
    print(f"DataFrame head:\n{dff.head()}\n")
    print(f"Columns in DataFrame:\n{dff.columns}\n")

# %%
# Test output dff with specific columns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the CSV file
file_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_0/output_dff_S_100.csv'
df = pd.read_csv(file_path)

# Check if 'timepoint' column exists and drop it
if 'timepoint' in df.columns:
    df.drop(columns=['timepoint'], inplace=True)

# Reset the index to use it as the new timepoint
df.reset_index(drop=True, inplace=True)

# Step 2: Select the first 10 columns and rows from index 10 to 110
df_first_10_columns = df.iloc[10:150, 500:600] #row 10-150, columnn 500-600

# Step 3: Plot the line plot
plt.figure(figsize=(20, 10))  # Adjust the size as needed
sns.lineplot(data=df_first_10_columns, dashes=False, markers=False, style=None)
plt.title('Line Plot of First 10 Columns (Rows 10 to 110)')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()


# %%
# Test output dff with specific header names

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the CSV file
file_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff_F0_15_adjust_9/output_dff_S_100.csv'
df = pd.read_csv(file_path)

# List of selected column headers
selected_columns = [
    "296.468774::453.023438::30.552028",
    "156.132785::418.524036::8.528740",
    "294.232986::464.458234::47.435177",
    "183.112771::448.796633::35.421674"
]

# Step 2: Select the specified columns and rows from index 10 to 200
df_selected_columns = df.loc[10:200, selected_columns]

# Step 3: Plot the line plot
plt.figure(figsize=(20, 10))  # Adjust the size as needed
sns.lineplot(data=df_selected_columns)
plt.title('Line Plot of Selected Columns (Rows 10 to 110)')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()

# %%
