# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Read the CSV file into a DataFrame
final_output = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/final_output.csv')

# %%

# Group by Event_Number and calculate the maximum time point for each event
event_time_range_max = final_output.groupby('Event_Number')['Time_Point'].max() 1
event_time_range_min = final_output.groupby('Event_Number')['Time_Point'].min() -1

event_time_range = event_time_range_max - event_time_range_min

# Define the neurons you want to include in the plot
neurons_to_plot = [89409]  # Replace with the specific neuron IDs you want to include, e.g., [89409, 48991]

# Plot values per event over reset time points for selected neurons
# ... (Previous code remains the same)

plt.figure(figsize=(10, 6))  # Adjust figure size if needed
for neuron_id in neurons_to_plot:
    plt.subplot(len(neurons_to_plot), 1, neurons_to_plot.index(neuron_id) + 1)
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
    plt.legend()

plt.tight_layout()
plt.show()

# %%
