#%% 
import pandas as pd

#%%
# Important: behaviour.csv starts with "1", activity.csv with "0"
# Stimulus window: 101-106
# Stimulus window: 681-695

# csv file had some zeros that resulted in NaN 

import pandas as pd
import matplotlib.pyplot as plt

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

# Load csv and set parameters
csv_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_klb-raw_data.csv'  # Replace with your CSV file path
problematic_neurons_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/raw_fluorescence/F0-8_Ft-40/problematic_neurons.csv'
output_plot_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/raw_fluorescence/F0-8_Ft-40/dff_plot.png'

F0_window = 24  # Number of points to average for F0
F0_start = 575    # Start index for F0 window
Ft_start = 680    # Start index for Ft window

dff = calculate_dff(csv_path, F0_window, F0_start, Ft_start, problematic_neurons_path)
output_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/raw_fluorescence/F0-8_Ft-40/dff_output_Ft_680_second_stimulus.csv'
dff.to_csv(output_path, index=False)

# Plot the results
#plot_dff(dff, output_plot_path)

# Print some debug information
print(f"DataFrame head:\n{dff.head()}\n")
print(f"Index values:\n{dff.index}\n")
print(f"Columns in DataFrame:\n{dff.columns}\n")


















# %%
# Generic dff extraction for specific events, using behaviour.csv
# dff has to be extracted for each event and saved as separate csv
# F0 directly before behaviour-start TODO check

import pandas as pd
import matplotlib.pyplot as plt
import os

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
activity_csv_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_klb-raw_data.csv'  # Replace with your activity CSV file path
behaviour_csv_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/action/Forward_threshold_3_18-02-15L1-behavior-ol_filtered_1-1000.csv'  # Replace with your behaviour CSV file path
output_dir = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff/'  # Replace with your desired output directory

# Parameters
F0_window = 24  # Number of points to average for F0

# Load behaviour data
behaviour_data = pd.read_csv(behaviour_csv_path)

# Iterate through behaviour data
for index, row in behaviour_data.iterrows():
    start_time = int(row['start'])
    end_time = int(row['end'])
    behaviour = row['beh']
    
    # Calculate F0_start based on behaviour start time
    F0_start = start_time - F0_window
    
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

    # Optionally plot the results (uncomment if needed)
    # output_plot_path = os.path.join(output_dir, f'dff_plot_{behaviour}_{start_time}.png')
    # plot_dff(dff, output_plot_path)

    # Print some debug information
    print(f"Processed row with start time {start_time} and behaviour {behaviour}.")
    print(f"DataFrame head:\n{dff.head()}\n")
    print(f"Columns in DataFrame:\n{dff.columns}\n")




# %%
'''
# Paths
activity_csv_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_klb-raw_data.csv'  # Replace with your activity CSV file path
behaviour_csv_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/action/Stimulus_18-02-15L1-behavior-ol_filtered_1-1000.csv'  # Replace with your behaviour CSV file path
output_dir = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff/'  # Replace with your desired output directory
'''
#%%
# ADAPT FO AVERAGE WINDOW START - ADD ADJUSTMENT
# Generic dff extraction for specific events, using behaviour.csv
# dff has to be extracted for each event and saved as separate csv

import pandas as pd
import matplotlib.pyplot as plt
import os

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
activity_csv_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_klb-raw_data.csv'  # Replace with your activity CSV file path
behaviour_csv_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/action/Forward_threshold_3_18-02-15L1-behavior-ol_filtered_1-1000.csv'  # Replace with your behaviour CSV file path
output_dir = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/dff/'  # Replace with your desired output directory

# Parameters
F0_window = 24  # Number of points to average for F0
adjustment = 0  # Value to adjust F0 starting time

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

    # Optionally plot the results (uncomment if needed)
    # output_plot_path = os.path.join(output_dir, f'dff_plot_{behaviour}_{start_time}.png')
    # plot_dff(dff, output_plot_path)

    # Print some debug information
    print(f"Processed row with start time {start_time} and behaviour {behaviour}.")
    print(f"DataFrame head:\n{dff.head()}\n")
    print(f"Columns in DataFrame:\n{dff.columns}\n")
