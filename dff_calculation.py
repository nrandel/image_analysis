#%% 
import pandas as pd

#%%
# F0 = 8s (24 frames) (Eschbach et al 5s)
# Ft = onset of stimulus or behaviour =40s (120 frame) 

# Important: behaviour.csv starts with "1", activity.csv with "0"
# Stimulus window: 101-106
# Stimulus window: 681-695

# csv file had some zeros that resulted in NaN 


import pandas as pd

def calculate_dff(csv_path, F0_window, Ft_window, F0_start, Ft_start, problematic_neurons_path):
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
        
        # Ensure the windows are within the valid range
        if F0_start + F0_window > len(neuronal_activity) or Ft_start + Ft_window > len(neuronal_activity):
            raise ValueError("Specified window is out of the range of the data.")
        
        # Extract the F0 and Ft windows for each neuron, ignoring NaNs
        F0 = neuronal_activity[neuron].iloc[F0_start:F0_start+F0_window].mean(skipna=True)
        
        # Check if F0 is zero to avoid division by zero
        if F0 == 0:
            problematic_neurons.append(neuron)
            continue
        
        Ft = neuronal_activity[neuron].iloc[Ft_start:Ft_start+Ft_window].mean(skipna=True)
        
        # Calculate ΔF/F for each time point
        dff[neuron] = (Ft - F0) / F0
    
    # Save the list of problematic neurons
    if problematic_neurons:
        pd.Series(problematic_neurons).to_csv(problematic_neurons_path, index=False, header=['problematic_neurons'])

    return dff

# Load csv and set parameters
csv_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_klb-raw_data.csv'  # Replace with your CSV file path
problematic_neurons_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/raw_fluorescence/F0-8_Ft-40/problematic_neurons.csv'

F0_window = 24  # Number of points to average for F0
Ft_window = 120   # Number of points to average for Ft
F0_start = 75    # Start index for F0 window
Ft_start = 100    # Start index for Ft window

dff = calculate_dff(csv_path, F0_window, Ft_window, F0_start, Ft_start, problematic_neurons_path)
output_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/raw_fluorescence/F0-8_Ft-40/dff_output_Ft_100_first_stimulus.csv'
dff.to_csv(output_path, index=False)

# Print some debug information
print(f"DataFrame head:\n{dff.head()}\n")
print(f"Index values:\n{dff.index}\n")
print(f"Columns in DataFrame:\n{dff.columns}\n")




# %%
