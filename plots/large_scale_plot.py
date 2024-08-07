#%%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# for dff long sliding window
# Function to plot columns in chunks against index with specified x-axis range
def plot_columns_in_chunks(df, chunk_size=5, x_start=None, x_end=None):
    num_cols = df.shape[1]
    column_chunks = [range(i, min(i + chunk_size, num_cols)) for i in range(0, num_cols, chunk_size)]

    for chunk_indices in column_chunks:
        plt.figure(figsize=(12, 8))
        
        plotted_anything = False  # Flag to check if anything was actually plotted
        
        for col_idx in chunk_indices:
            col_name = df.columns[col_idx]
            plt.plot(df.index, df[col_name], label=col_name)
            plotted_anything = True  # Set flag to True if at least one plot was made
        
        if plotted_anything:
            plt.title(f'Columns {chunk_indices[0]} to {chunk_indices[-1]} vs Index')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            
            if x_start is not None and x_end is not None:
                plt.xlim(x_start, x_end)  # Set x-axis limits if provided
            
            plt.show()
        else:
            print(f"No data plotted for columns {chunk_indices[0]} to {chunk_indices[-1]}")

# Load:
file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/intersection_neuronal_activity_responsive_neurons_sd_2_average_100-110-AND-680-699.csv'
df = pd.read_csv(file_path, index_col = False)  # Assuming the first column is the index

# Specify x-axis range (optional)
x_start = 640  # Replace with your desired start index 80, 640
x_end = 740   # Replace with your desired end index 140, 740

# Print some debug information
print(f"DataFrame head:\n{df.head()}\n")
print(f"Index values:\n{df.index}\n")
print(f"Columns in DataFrame:\n{df.columns}\n")

# Plot every 20 columns against the index with specified x-axis range
plot_columns_in_chunks(df, chunk_size=5, x_start=x_start, x_end=x_end)





# %%
# for dff from raw data - stimulus specific (initial code)
# Function to plot columns in chunks against index with specified x-axis range
def plot_columns_in_chunks(df, chunk_size=5, x_start=None, x_end=None):
    num_cols = df.shape[1]
    column_chunks = [range(i, min(i + chunk_size, num_cols)) for i in range(0, num_cols, chunk_size)]

    for chunk_indices in column_chunks:
        plt.figure(figsize=(12, 8))
        
        plotted_anything = False  # Flag to check if anything was actually plotted
        
        for col_idx in chunk_indices:
            col_name = df.columns[col_idx]
            plt.plot(df.index, df[col_name], label=col_name)
            plotted_anything = True  # Set flag to True if at least one plot was made
        
        if plotted_anything:
            plt.title(f'Columns {chunk_indices[0]} to {chunk_indices[-1]} vs Index')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            
            if x_start is not None and x_end is not None:
                plt.xlim(x_start, x_end)  # Set x-axis limits if provided
            
            plt.show()
        else:
            print(f"No data plotted for columns {chunk_indices[0]} to {chunk_indices[-1]}")

# Load:
#Stimulus 1
file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/raw_fluorescence/F0-8_Ft-40/intersection_neuronal_activity_responsive_neurons_sd_2_average_100-110-AND-680-699_For_Stimulus_1.csv'

#Stimulus 2
#file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/raw_fluorescence/F0-8_Ft-40/intersection_neuronal_activity_responsive_neurons_sd_2_average_100-110-AND-680-699_For_Stimulus_2.csv'



df = pd.read_csv(file_path, index_col = False)  # Assuming the first column is the index

# Specify x-axis range (optional)
x_start = 640  # Replace with your desired start index 80, 640
x_end = 740   # Replace with your desired end index 140, 740

# Print some debug information
print(f"DataFrame head:\n{df.head()}\n")
print(f"Index values:\n{df.index}\n")
print(f"Columns in DataFrame:\n{df.columns}\n")

# Plot every 20 columns against the index with specified x-axis range
plot_columns_in_chunks(df, chunk_size=5, x_start=x_start, x_end=x_end)











# %%
# Plot behavior-dff from raw data

# for dff from raw data
# Function to plot columns in chunks against index with specified x-axis range
def plot_columns_in_chunks(df, chunk_size=5, x_start=None, x_end=None):
    num_cols = df.shape[1]
    column_chunks = [range(i, min(i + chunk_size, num_cols)) for i in range(0, num_cols, chunk_size)]

    for chunk_indices in column_chunks:
        plt.figure(figsize=(12, 8))
        
        plotted_anything = False  # Flag to check if anything was actually plotted
        
        for col_idx in chunk_indices:
            col_name = df.columns[col_idx]
            plt.plot(df.index, df[col_name], label=col_name)
            plotted_anything = True  # Set flag to True if at least one plot was made
        
        if plotted_anything:
            plt.title(f'Columns {chunk_indices[0]} to {chunk_indices[-1]} vs Index')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            
            if x_start is not None and x_end is not None:
                plt.xlim(x_start, x_end)  # Set x-axis limits if provided
            
            plt.show()
        else:
            print(f"No data plotted for columns {chunk_indices[0]} to {chunk_indices[-1]}")

# Load:
#Beh 1
#file_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/F/Forward_F0_15_adjust_0_activity_of_responsive_neurons_sd_1-5_average_51.csv'

#Beh 2
file_path = '/Users/nadine/Documents/paper/single-larva/behavior_extraction/filtered_dff/F/Forward_F0_15_adjust_9_activity_of_responsive_neurons_sd_1-5_average_51.csv'


df = pd.read_csv(file_path, index_col = False)  # Assuming the first column is the index

# Specify x-axis range (optional)
x_start = 10  # Replace with your desired start index 80, 640
x_end = 130   # Replace with your desired end index 140, 740

# Print some debug information
print(f"DataFrame head:\n{df.head()}\n")
print(f"Index values:\n{df.index}\n")
print(f"Columns in DataFrame:\n{df.columns}\n")

# Plot every 20 columns against the index with specified x-axis range
plot_columns_in_chunks(df, chunk_size=5, x_start=x_start, x_end=x_end)


















# %%
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Function to plot columns and their smoothed first derivatives in chunks against index with specified x-axis range
# my dff
def plot_columns_in_chunks(df, chunk_size=5, x_start=None, x_end=None):
    num_cols = df.shape[1]
    column_chunks = [range(i, min(i + chunk_size, num_cols)) for i in range(0, num_cols, chunk_size)]
    
    for chunk_indices in column_chunks:
        plt.figure(figsize=(12, 8))
        
        plotted_anything = False  # Flag to check if anything was actually plotted
        
        for col_idx in chunk_indices:
            col_name = df.columns[col_idx]
            values = df[col_name].values
            time_points = df.index
            
            # Apply Savitzky-Golay filter for smoothing
            smoothed_values = savgol_filter(values, window_length=5, polyorder=2)  # Adjust parameters as needed
            plt.plot(time_points, smoothed_values, label=f'{col_name} (smoothed)')
            
            # Calculate and plot first derivative using Savitzky-Golay filter
            derivative = savgol_filter(values, window_length=5, polyorder=2, deriv=1)  # Adjust parameters as needed
            plt.plot(time_points, derivative, linestyle='--', label=f'{col_name} (derivative)')
            
            plotted_anything = True  # Set flag to True if at least one plot was made
        
        if plotted_anything:
            plt.title(f'Columns {chunk_indices[0]} to {chunk_indices[-1]} vs Index')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            
            if x_start is not None and x_end is not None:
                plt.xlim(x_start, x_end)  # Set x-axis limits if provided
            
            plt.show()
        else:
            print(f"No data plotted for columns {chunk_indices[0]} to {chunk_indices[-1]}")

# Load Stimulus 1 data my dff
#file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/raw_fluorescence/F0-8_Ft-40/intersection_neuronal_activity_responsive_neurons_sd_2_average_100-110-AND-680-699_For_Stimulus_1.csv'

# Stimulus 2 my dff
# file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/raw_fluorescence/F0-8_Ft-40/intersection_neuronal_activity_responsive_neurons_sd_2_average_100-110-AND-680-699_For_Stimulus_2.csv'

# Stimulus pre-extracteed from log dff
file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/intersection_neuronal_activity_responsive_neurons_sd_2_average_100-110-AND-680-699.csv'

df = pd.read_csv(file_path, index_col=False)  # Assuming the first column is not the index

# Specify x-axis range (optional)
x_start = 80  # Replace with your desired start index 80, 640
x_end = 140    # Replace with your desired end index 140, 740

# Print some debug information
print(f"DataFrame head:\n{df.head()}\n")
print(f"Index values:\n{df.index}\n")
print(f"Columns in DataFrame:\n{df.columns}\n")

# Plot every 5 columns against the index with specified x-axis range
plot_columns_in_chunks(df, chunk_size=5, x_start=x_start, x_end=x_end)



# %%





# %%
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Function to plot columns and their smoothed first derivatives in chunks against index with specified x-axis range
# long sliding window (same code as above but 'time column' nneeds to be dropped)

def plot_columns_in_chunks(df, chunk_size=5, x_start=None, x_end=None):
    num_cols = df.shape[1]
    column_chunks = [range(i, min(i + chunk_size, num_cols)) for i in range(0, num_cols, chunk_size)]
    
    for chunk_indices in column_chunks:
        plt.figure(figsize=(12, 8))
        
        plotted_anything = False  # Flag to check if anything was actually plotted
        
        for col_idx in chunk_indices:
            col_name = df.columns[col_idx]
            values = df[col_name].values
            time_points = df.index
            
            # Apply Savitzky-Golay filter for smoothing
            smoothed_values = savgol_filter(values, window_length=5, polyorder=2)  # Adjust parameters as needed
            plt.plot(time_points, smoothed_values, label=f'{col_name} (smoothed)')
            
            # Calculate and plot first derivative using Savitzky-Golay filter
            derivative = savgol_filter(values, window_length=5, polyorder=2, deriv=1)  # Adjust parameters as needed
            plt.plot(time_points, derivative, linestyle='--', label=f'{col_name} (derivative)')
            
            plotted_anything = True  # Set flag to True if at least one plot was made
        
        if plotted_anything:
            plt.title(f'Columns {chunk_indices[0]} to {chunk_indices[-1]} vs Index')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            
            if x_start is not None and x_end is not None:
                plt.xlim(x_start, x_end)  # Set x-axis limits if provided
            
            plt.show()
        else:
            print(f"No data plotted for columns {chunk_indices[0]} to {chunk_indices[-1]}")


# Stimulus lonng sliding window
file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_t_stacks-dff_long-slidingWindow.csv'

df = pd.read_csv(file_path, index_col=False)  # Assuming the first column is not the index

# Drop the 'timepoint' column and use the index as the new timepoint
df.drop(columns=['timepoint'], inplace=True)


# Specify x-axis range (optional)
x_start = 80  # Replace with your desired start index 80, 640
x_end = 140    # Replace with your desired end index 140, 740

# Print some debug information
print(f"DataFrame head:\n{df.head()}\n")
print(f"Index values:\n{df.index}\n")
print(f"Columns in DataFrame:\n{df.columns}\n")

# Plot every 5 columns against the index with specified x-axis range
plot_columns_in_chunks(df, chunk_size=5, x_start=x_start, x_end=x_end)
# %%
