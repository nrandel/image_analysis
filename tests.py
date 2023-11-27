# Identify cut-off time for activity traces
# The sample did not respond in the end of the experiment and cells stop having a change in signal towards the end

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Read csvfile of merged activity traces
merged_activity_traces = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/2023-11-24/merged_activity_traces.csv')

# %%
# Display available columns
print("Available columns:")
print(list(merged_activity_traces.columns))

# %%
# Select columns
selected_columns = ["89409", "68884"]

# %%
# Plot the selected columns
for column in selected_columns:
    if column in merged_activity_traces.columns:
        plt.plot(merged_activity_traces.index, merged_activity_traces[column], label=column)

# Add labels and legend
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Plotting selected columns')
plt.legend()
plt.show()

# %%
