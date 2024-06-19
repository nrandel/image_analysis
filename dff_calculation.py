#%% 
import pandas as pd

#%%
# F0 = 8s (24 frames) (Eschbach et al 5s)
# Ft = onset of stimulus or behaviour =40s (120 frame) 

# Important: behaviour.csv starts with "1", activity.csv with "0"
# Stimulus window: 101-106
# Stimulus window: 681-695

def calculate_dff(csv_path, F0_window, Ft_window, F0_start, Ft_start):
    # Load the CSV file
    data = pd.read_csv(csv_path)
    
    # Extract the time points and neuronal activity
    time_points = data.iloc[:, 0]
    neuronal_activity = data.iloc[:, 1:]
    
    # Calculate ΔF/F
    dff = pd.DataFrame()
    dff['timepoint'] = time_points

    for neuron in neuronal_activity.columns:
        # Extract the F0 and Ft windows for each neuron
        F0 = neuronal_activity[neuron].iloc[F0_start:F0_start+F0_window].mean()
        Ft = neuronal_activity[neuron].iloc[Ft_start:Ft_start+Ft_window]

        # Calculate ΔF/F for each time point
        dff[neuron] = (Ft - F0) / F0

    return dff

# Load csv and set parameters
csv_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_klb-raw_data.csv'  # Replace with your CSV file path
F0_window = 24  # Number of points to average for F0
Ft_window = 120   # Number of points to average for Ft
F0_start = 75    # Start index for F0 window
Ft_start = 100    # Start index for Ft window

dff = calculate_dff(csv_path, F0_window, Ft_window, F0_start, Ft_start)
dff.to_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/raw_fluorescence/F0-8_Ft-40/dff_output_Ft_100_first_stimulus.csv', index=False)

# %%
