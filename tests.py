# Identify cut-off time for activity traces
# The sample did not respond in the end of the experiment and cells stop having a change in signal towards the end

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Read csvfile of merged activity traces
merged_activity_traces = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/traces_3-3-1.csv')

# %%
# Display available columns
print("Available columns:")
print(list(merged_activity_traces.columns))

# %%
# Select columns
selected_columns = ["65408"] # e.g., ["89409", "68884"]

# %%
# Define the range of rows to plot
start_row = 1
end_row = 10300


# %%
# Plot the selected columns for the specified rows
for column in selected_columns:
    if column in merged_activity_traces.columns:
        plt.plot(merged_activity_traces.index[start_row:end_row], merged_activity_traces[column].iloc[start_row:end_row], label=column)

# Add labels and legend
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Plotting selected columns (Rows {} to {})'.format(start_row, end_row))
plt.legend()
plt.show()

# %%
