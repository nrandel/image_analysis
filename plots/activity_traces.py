# %%
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# %%
# Read the CSV file into a DataFrame
# Single behaviour
#final_output = pd.read_csv('/Users/nadine/Documents/paper/single-larva/ACardona_dff-file/final_output_F-Decision-60_20TP-ClemClam.csv') #single event-type

# truncated single behaviour
#final_output = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/2023-12-05/radius_3-3-1/Action_selection/final_output_Tl-TR-Decision-60_15TP_Event_1-15.csv') #single event-type

# multiple behaviour
final_output = pd.read_csv('/Users/nadine/Documents/paper/single-larva/ACardona_dff-file/final_output_TL-TR-Decision-50_20TP-ClemClam.csv') #multiple event-type

# %%
# Group by Event_Number and reset time point for each event to identify number of time points 
# event_time_range_max = final_output.groupby('Event_Number')['Time_Point'].max() - 1 the tiff stack starts at o, the behaviour at 1!!

event_time_range_max = final_output.groupby('Event_Number')['Time_Point'].max() -1
event_time_range_min = final_output.groupby('Event_Number')['Time_Point'].min() -1

event_time_range = event_time_range_max - event_time_range_min

# %%
# Define the neurons you want to include in the plot
neurons_to_plot = [9233, 23971]  # Replace with the specific neuron IDs you want to include, e.g., [73673, 89409]

# %%
# Print the values of the specific column
#column_name = 'Event_Number'  
#print(final_output[column_name])

#event_numbers_to_plot = ['F_1', 'F_2']  # Replace 'F_1', 'F_2', etc., with the specific values you want to filter

# Filter the DataFrame based on the specified event numbers
#filtered_data = final_output[final_output[column_name].isin(event_numbers_to_plot)]
#print(filtered_data)


# %%
# Plot values per event over time points for selected neurons

plt.figure(figsize=(10, 6))  # Adjust figure size if needed
for neuron_id in neurons_to_plot:
    plt.subplot(len(neurons_to_plot), 1, neurons_to_plot.index(neuron_id) + 1) # select if individual plots for each neuron
    for event_label, event_data in final_output[final_output['Neuron'] == neuron_id].groupby('Event_Number'):
        time_point_range = event_time_range.loc[event_label] + 1  # Fetch time point range for the event (+1 for inclusive range)
        time_points = range(time_point_range)  # Create range based on the time point range
        values = event_data[event_data['Event_Number'] == event_label]['Value'].values[:time_point_range]  # Select values corresponding to the time point range

        if len(values) != time_point_range:
            print(f"Event {event_label} has mismatched lengths! Expected: {time_point_range}, Actual: {len(values)}")
            continue  # Skip plotting this event
        
        # Plot if lengths match
        plt.plot(time_points, values, label=f'Event {event_label}')
        plt.xticks(range(time_point_range))  # Set x-axis ticks to match the time points

    plt.xlabel(f'Time (Reset for Each Event, Range: 0-{time_point_range - 1})')
    plt.ylabel(f'Neuron {neuron_id} Value')
    plt.title(f'Values per Event over Reset Time Points for Neuron {neuron_id}')

    # Set custom tick labels for x-axis
    plt.xticks(range(0, time_point_range, 10))
    #plt.legend()

plt.tight_layout()
plt.show()

# %%
# Plot values per event over time points for selected neurons, and average, either for each neuronID separately, or combined

plt.figure(figsize=(10, 6))  # Adjust figure size if needed

all_values_all_neurons = []  # To store all values for averaging
max_time_point = 0

for neuron_id in neurons_to_plot:
    all_values = []  # To store values for averaging
    for event_label, event_data in final_output[final_output['Neuron'] == neuron_id].groupby('Event_Number'):
        time_point_range = event_time_range.loc[event_label] + 1  # Fetch time point range for the event (+1 for inclusive range)
        time_points = range(time_point_range)  # Create range based on the time point range
        values = event_data[event_data['Event_Number'] == event_label]['Value'].values[:time_point_range]  # Select values corresponding to the time point range
        
        if len(values) != time_point_range:
            print(f"Event {event_label} for Neuron {neuron_id} has mismatched lengths! Expected: {time_point_range}, Actual: {len(values)}")
            continue  # Skip plotting this event
        
        # Plot individual traces
        plt.plot(time_points, values, alpha=0.5, label=f'Neuron {neuron_id}, Event {event_label}', color = 'k')
        
        # Store values for averaging
        all_values.append(values)
        all_values_all_neurons.extend(values)
        
        max_time_point = max(max_time_point, len(time_points))  # Update maximum time point

    # Calculate and plot average values per neuron
    if all_values:
        average_values = sum(all_values) / len(all_values)
        plt.plot(range(len(average_values)), average_values, label=f'Neuron {neuron_id} Average', linestyle='-', linewidth=5)
"""
# Calculate and plot average values across all neurons
if all_values_all_neurons:
    average_values_all_neurons = sum(all_values_all_neurons) / len(all_values_all_neurons)
    plt.plot(range(max_time_point), [average_values_all_neurons] * max_time_point, label='All Neurons Average', linestyle='-', linewidth=2, color = 'k')
"""
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Combined Plot of Neurons with Overall Average')
#plt.legend()
plt.show()


# %%
# Smooth data using Savitzky-Golay filter and plot derivatives

from scipy.signal import savgol_filter

plt.figure(figsize=(10, 6))

for neuron_id in neurons_to_plot:
    plt.subplot(len(neurons_to_plot), 1, neurons_to_plot.index(neuron_id) + 1)
    for event_label, event_data in final_output[final_output['Neuron'] == neuron_id].groupby('Event_Number'):
        time_point_range = event_time_range.loc[event_label] + 1  # Fetch time point range for the event (+1 for inclusive range)
        time_points = range(time_point_range)  # Create range based on the time point range
        values = event_data[event_data['Event_Number'] == event_label]['Value'].values[:time_point_range]  # Select values corresponding to the time point range

        if len(values) != time_point_range:
            print(f"Event {event_label} has mismatched lengths! Expected: {time_point_range}, Actual: {len(values)}")
            continue  # Skip plotting this event
 
        smoothed_values = savgol_filter(values, window_length=7, polyorder=3)  # Adjust window length and polynomial order as needed (e.g., window length 5)
        plt.plot(time_points, smoothed_values, label=f'Smoothed Event {event_label}')

        # Calculating the derivative using the Savitzky-Golay filter
        #derivative = savgol_filter(values, window_length=7, polyorder=3, deriv=1)  # Adjust parameters as needed (e.g., window_length=5, polyorder=2, deriv=2)
        #plt.plot(time_points, derivative, label=f'Derivative Event {event_label}')  # Plot derivative

    plt.xlabel(f'Time (Reset for Each Event, Range: 0-{time_point_range - 1})')
    plt.ylabel(f'Neuron {neuron_id} Value')
    plt.title(f'Values per Event over Reset Time Points for Neuron {neuron_id}')
    #plt.legend()

plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

# Create a linear gradient from white to black
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# Plot the gradient
plt.imshow(gradient, cmap='gray', aspect='auto')
plt.axis('off')

# Add a colorbar with greyscale values
cbar = plt.colorbar()
cbar.set_label('Intensity')

plt.savefig('/Users/nadine/Desktop/greyscale.png')

plt.show()

# %%
# Create a linear gradient for viridis
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# Plot the gradient
plt.imshow(gradient, cmap='viridis', aspect='auto')
plt.axis('off')

# Add a colorbar with viridis values
cbar = plt.colorbar()
cbar.set_label('Intensity')

plt.savefig('/Users/nadine/Desktop/viridis.png')
plt.show()

# %%
# ONLY USE FOR SPECIFIC EVENTS  

plt.figure(figsize=(10, 6))

# Define the neurons you want to include in the plot
neurons_to_plot = [73673, 89409]  # Replace with the specific neuron IDs you want to include, e.g., [73673, 89409]


# Define the event numbers you want to include in the plot
column_name = 'Event_Number'  
#print(final_output[column_name])

#event_numbers_to_plot = ['F_1', 'F_2']  # Replace 'F_1', 'F_2', etc., with the specific values you want to filter
event_numbers_to_plot = ["['TL', 'TR']_2", "['TL', 'TR']_3", "['TL', 'TR']_4"]  # Replace 'F_1', 'F_2', etc., with the specific values you want to filter


# Filter the DataFrame based on the specified event numbers
filtered_data = final_output[final_output[column_name].isin(event_numbers_to_plot)]
#print(filtered_data)


for neuron_id in neurons_to_plot:
    plt.subplot(len(neurons_to_plot), 1, neurons_to_plot.index(neuron_id) + 1)
    for event_label, event_data in final_output[(final_output['Neuron'] == neuron_id) & (final_output['Event_Number'].isin(event_numbers_to_plot))].groupby('Event_Number'):
        time_point_range = event_time_range.loc[event_label] + 1  # Fetch time point range for the event (+1 for inclusive range)
        time_points = range(time_point_range)  # Create range based on the time point range
        values = event_data['Value'].values[:time_point_range]  # Select values corresponding to the time point range

        if len(values) != time_point_range:
            print(f"Event {event_label} has mismatched lengths! Expected: {time_point_range}, Actual: {len(values)}")
            continue  # Skip plotting this event
 
        smoothed_values = savgol_filter(values, window_length=7, polyorder=3)  # Adjust window length and polynomial order as needed (e.g., window length 5)
        plt.plot(time_points, smoothed_values, label=f'Smoothed Event {event_label}')

        # Calculating the derivative using the Savitzky-Golay filter
        #derivative = savgol_filter(values, window_length=7, polyorder=3, deriv=1)  # Adjust parameters as needed (e.g., window_length=5, polyorder=2, deriv=2)
        #plt.plot(time_points, derivative, label=f'Derivative Event {event_label}')  # Plot derivative

    plt.xlabel(f'Time (Reset for Each Event, Range: 0-{time_point_range - 1})')
    plt.ylabel(f'Neuron {neuron_id} Value')
    plt.title(f'Values per Event over Reset Time Points for Neuron {neuron_id}')
    #plt.legend()

plt.tight_layout()
plt.show()

# %%
# plot fot sigle cell ca-imaging
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/measurements/a_k_3__20240330_154256.measurements.csv'
single_cell_activity = pd.read_csv(file_path)

# Remove all columns except activity traces (assuming 'timepoint' is the only non-activity trace column)
column_to_drop = 'timepoint'
single_cell_activity.drop(columns=[column_to_drop], inplace=True)

# Convert the index to integer if it's not already (assuming the index should be numeric)
single_cell_activity.index = pd.to_numeric(single_cell_activity.index, errors='coerce')

# Drop any rows where the index conversion resulted in NaN
single_cell_activity.dropna(inplace=True)

# Convert the index to integer type
single_cell_activity.index = single_cell_activity.index.astype(int)

# Select a range according to index
start_index = 0
#end_index = 2300
end_index = single_cell_activity.index.max()
selected_data = single_cell_activity.loc[start_index:end_index]

# Create the figure and set the size
fig, ax = plt.subplots(figsize=(200, 10))  # Set the plot size (width, height in inches)

# Plot the selected data
selected_data.plot(kind='line', ax=ax)

# Set the labels and title
ax.set_xlabel('Index')
ax.set_ylabel('Activity')
ax.set_title('MBON-e1-ak3')

# Set the x-axis label tick interval to 1
#ax.set_xticks(range(start_index, end_index + 1))

# Set the x-axis label tick interval to 10
ax.set_xticks(range(start_index, end_index + 1, 10))

# Display the plot
plt.show()


# %%
