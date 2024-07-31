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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_dff(npz_path, F0_window, F0_start, Ft_start, problematic_neurons_path):
    # Load the NPZ file
    data = np.load(npz_path)
    
    # Extract neuronal activity data
    neuronal_activity = data['neuron_traces']
    
    # Create time points array
    time_points = np.arange(neuronal_activity.shape[0])
    
    # Calculate ΔF/F
    dff = pd.DataFrame()
    dff['timepoint'] = time_points
    
    problematic_neurons = []

    for neuron_idx in range(neuronal_activity.shape[1]):
        neuron = f'neuron_{neuron_idx}'

        # Ensure the F0 window is within the valid range
        if F0_start + F0_window > Ft_start:
            raise ValueError("Specified F0 window overlaps with Ft start period.")
        
        # Extract the F0 window for each neuron, ignoring NaNs
        F0 = np.nanmean(neuronal_activity[F0_start:F0_start+F0_window, neuron_idx])
        
        # Check if F0 is zero to avoid division by zero
        if F0 == 0:
            problematic_neurons.append(neuron)
            continue
        
        # Calculate ΔF/F for each time point
        dff[neuron] = [(ft - F0) / F0 for ft in neuronal_activity[:, neuron_idx]]
    
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
npz_path = '/Users/nadine/Documents/Zlatic_lab/1099_spatial-filtered/neuron_traces_cleaned.npz' 
# for behaviour first 1000TP, except stim (all)
behaviour_csv_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/action/Forward_threshold_3_18-02-15L1-behavior-ol_filtered_1-1000.csv'  
# output file name: F0 average over 15 frames, 
# adjust == F0 window calculated 9 frames before start of behaviour (Ft)
output_dir = '/Users/nadine/Documents/Zlatic_lab/1099_spatial-filtered/analysis/dff_F0_15_adjust-9/Forward'  

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
    dff = calculate_dff(npz_path, F0_window, F0_start, Ft_start, problematic_neurons_path)
    
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
