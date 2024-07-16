# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%

#temp testing
# %%
import pandas as pd
import os

def plot_neuron_activity(df, neurons, time_range, title):
    plt.figure(figsize=(12, 8))
    for neuron in neurons:
        plt.plot(df.index[time_range[0]:time_range[1]+1], df[neuron][time_range[0]:time_range[1]+1], label=neuron)
    plt.title(title)
    plt.xlabel('Timepoint')
    plt.ylabel('ΔF/F')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()

def save_neurons_to_csv(neurons, filename, path=''):  
    filepath = os.path.join(path, filename)  
    pd.DataFrame(neurons).to_csv(filepath, header=False)

# Load the dataset from the CSV file (dff sliding window)
df = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_t_stacks-dff_long-slidingWindow.csv')

# Drop the 'timepoint' column and use the index as the new timepoint
df.drop(columns=['timepoint'], inplace=True)

# Define fixed window length for calculating SD (adjust as needed)
window_length = 10

# Define start adjustment (set as needed). e.g., -9
adjustment = 0

# Read time windows from CSV and adjust start if needed
time_windows_file = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/action/Forward_threshold_3_18-02-15L1-behavior-ol_filtered_1-1000.csv'
time_windows_df = pd.read_csv(time_windows_file)

save_dir = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/F'

for index, row in time_windows_df.iterrows():
    start_time = int(row['start'])
    end_time = start_time + window_length
    
    start_time_adjusted = start_time + adjustment
    
    time_range = (start_time_adjusted, end_time)
    time_range_plot = (start_time_adjusted - 50, start_time_adjusted + 100)  # Adjusted for plotting
    
    # Compute average ΔF/F in the time window for each neuron
    time_window_data = df.iloc[time_range[0]:time_range[1]+1]
    average_dff = time_window_data.mean()
    
    # Compute mean and standard deviation for average ΔF/F
    mean_average_dff = average_dff.mean()
    sd_average_dff = average_dff.std()
    
    # Define the threshold for identifying responsive neurons
    threshold_average = mean_average_dff + 1.5 * sd_average_dff
    
    # Identify neurons whose responses are 1.5 SD higher than the mean
    responsive_neurons_average = average_dff[average_dff > threshold_average]
    
    # Save neurons to CSV file with specified path
    filename = f"Forward_long-sliding-window_activity_of_responsive_neurons_sd_1-5_average_{start_time}.csv"
    save_neurons_to_csv(responsive_neurons_average, filename, path=save_dir)
    
    # Save the original CSV file with only the columns corresponding to the responsive neurons
    df_responsive_neurons_average = df[list(responsive_neurons_average.index)]
    df_responsive_neurons_average.to_csv(os.path.join(save_dir, filename), index=False)
    
    # Plot results
    plot_neuron_activity(df, responsive_neurons_average.index, time_range_plot, f'Responsive neurons based on average ΔF/F (Start: {start_time})')

# %%
