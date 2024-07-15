# Identify cut-off time for activity traces
# The sample did not respond in the end of the experiment and cells stop having a change in signal towards the end

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Read csvfile of merged activity traces
#merged_activity_traces = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/traces_3-3-1.csv')
merged_activity_traces = pd.read_csv('/Users/nadine/Documents/paper/single-larva/ACardona_dff-file/clemclam-coordinates/measurements.csv')

# %%
# Display available columns
print("Available columns:")
print(list(merged_activity_traces.columns))

# %%
# Select columns
#selected_columns = ['73673', '89409', '68884', '48991', '50620', '61198', '56311', '50663', '24746', '23967', '85110', '24958', '24006', '45819', '36672', '9233', '23971', '35672', '66113', '33923', '25217', '26236', '57617', '48584', '56730', '32240', '50631', '71849', '24010', '69798', '49290', '23935', '22652', '63136', '50764', '62188', '58054', '65408', '32111', '62653', '70584'] # e.g., ["89409", "68884"]
selected_columns = ['56311']
# %%
# Define the range of rows to plot
start_row = 1
end_row = 11000


# %%


plt.figure(figsize=(400, 30))
# Plot the selected columns for the specified rows
for column in selected_columns:
    if column in merged_activity_traces.columns:
        plt.plot(merged_activity_traces.index[start_row:end_row], merged_activity_traces[column].iloc[start_row:end_row], label=column)

# Add labels and legend
plt.xlabel('frames')
plt.ylabel('df/f')

# Set ticks every other value on the x-axis
#plt.xticks(range(start_row, end_row, 2))
# Label every 20th value
#labels = ['' if i % 20 != 0 else str(i) for i in range(start_row, end_row)]
#plt.gca().set_xticklabels(labels)

# Label every 10th value without the tick
plt.xticks(range(start_row, end_row, 20))  # Assuming your index values represent frames


plt.title('Plotting selected columns (Rows {} to {})'.format(start_row, end_row))
plt.legend()

plt.savefig('/Users/nadine/Desktop/test.png')

plt.show()

# %%
# TEMPORARY

import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def f(x):
    return x**3 - 3*x**2 + 2*x

def f_prime(x):
    return 3*x**2 - 6*x + 2

def f_double_prime(x):
    return 6*x - 6

# Generate x values
x = np.linspace(-2, 4, 100)

# Calculate y values for each function
y_f = f(x)
y_f_prime = f_prime(x)
y_f_double_prime = f_double_prime(x)

# Plotting
plt.figure(figsize=(12, 8))

# Plot the original function
plt.subplot(311)
plt.plot(x, y_f, label='f(x) = x^3 - 3x^2 + 2x')
plt.title('Original Function')
plt.legend()

# Plot the first derivative
plt.subplot(312)
plt.plot(x, y_f_prime, label="f'(x) = 3x^2 - 6x + 2")
plt.axhline(0, color='black',linewidth=0.5)
plt.title('First Derivative')
plt.legend()

# Plot the second derivative
plt.subplot(313)
plt.plot(x, y_f_double_prime, label="f''(x) = 6x - 6")
plt.axhline(0, color='black',linewidth=0.5)
plt.title('Second Derivative')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# TEST
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Assuming merged_activity_traces is your DataFrame

# Define start_row and end_row values
start_row = 0
end_row = 100

# Define selected_columns (list of columns you want to plot)
selected_columns = ['24958']  # Modify with your column names, e.g., ["89409", "68884"]

# Plot the selected columns for the specified rows after applying Savitzky-Golay filter and derivatives
for column in selected_columns:
    if column in merged_activity_traces.columns:
        # Apply the Savitzky-Golay filter and calculate derivatives
        smoothed = savgol_filter(merged_activity_traces[column].iloc[start_row:end_row], window_length=7, polyorder=2)
        first_derivative = savgol_filter(smoothed, window_length=7, polyorder=2, deriv=1)
        second_derivative = savgol_filter(smoothed, window_length=7, polyorder=2, deriv=2)

        # Plotting the original data, smoothed data, first derivative, and second derivative
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        plt.plot(merged_activity_traces.index[start_row:end_row], merged_activity_traces[column].iloc[start_row:end_row], label='Original Data')
        plt.title('Original Data - {}'.format(column))
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(merged_activity_traces.index[start_row:end_row], smoothed, label='Smoothed Data')
        plt.title('Smoothed Data - {}'.format(column))
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(merged_activity_traces.index[start_row:end_row], first_derivative, label='First Derivative')
        plt.title('First Derivative - {}'.format(column))
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(merged_activity_traces.index[start_row:end_row], second_derivative, label='Second Derivative')
        plt.title('Second Derivative - {}'.format(column))
        plt.legend()

        plt.tight_layout()
        plt.show()

# Add labels and legend for the overall plot
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Plotting selected columns (Rows {} to {})'.format(start_row, end_row))
plt.legend()
plt.show()

# %%
# Alternatve

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Assuming merged_activity_traces is your DataFrame

# Define start_row and end_row values
start_row = 0
end_row = 100

# Define selected_columns (list of columns you want to plot)
selected_columns = ['24958']  # Modify with your column names

# Initialize empty plots for the legend
plt.figure(figsize=(10, 8))
plt.plot([], [], label='Original Data', color='blue')  # Empty plot for the legend
plt.plot([], [], label='Smoothed Data', color='orange')  # Empty plot for the legend
plt.plot([], [], label='First Derivative', color='green')  # Empty plot for the legend
plt.plot([], [], label='Second Derivative', color='red')  # Empty plot for the legend

# Plot the selected columns for the specified rows after applying Savitzky-Golay filter and derivatives
for column in selected_columns:
    if column in merged_activity_traces.columns:
        # Apply the Savitzky-Golay filter and calculate derivatives
        smoothed = savgol_filter(merged_activity_traces[column].iloc[start_row:end_row], window_length=7, polyorder=2)
        first_derivative = savgol_filter(smoothed, window_length=7, polyorder=2, deriv=1)
        second_derivative = savgol_filter(smoothed, window_length=7, polyorder=2, deriv=2)

        # Plotting the original data, smoothed data, first derivative, and second derivative
        plt.plot(merged_activity_traces.index[start_row:end_row], merged_activity_traces[column].iloc[start_row:end_row], label='Original Data - {}'.format(column))
        plt.plot(merged_activity_traces.index[start_row:end_row], smoothed, label='Smoothed Data - {}'.format(column))
        plt.plot(merged_activity_traces.index[start_row:end_row], first_derivative, label='First Derivative - {}'.format(column))
        plt.plot(merged_activity_traces.index[start_row:end_row], second_derivative, label='Second Derivative - {}'.format(column))

# Add labels and legend for the overall plot
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Plotting selected columns (Rows {} to {})'.format(start_row, end_row))
plt.legend()
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'selected_columns' is a list of columns you want to include in the heatmap
columns_to_plot = [column for column in selected_columns if column in merged_activity_traces.columns]

# Transpose the DataFrame
heatmap_data = merged_activity_traces[columns_to_plot].T.iloc[:, start_row:end_row]

# Create a heatmap
sns.heatmap(heatmap_data, cmap='hot', annot=False, fmt='.2f', xticklabels=merged_activity_traces.index[start_row:end_row], yticklabels=columns_to_plot)

plt.title("Combined Heatmap for Selected Columns")
plt.xlabel('Time')  # You may need to customize this based on your data
plt.ylabel('Columns')
plt.show()

# %%



