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
selected_columns = ['24958'] # e.g., ["89409", "68884"]

# %%
# Define the range of rows to plot
start_row = 50
end_row = 100


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
