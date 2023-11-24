# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Read the CSV file into a DataFrame
df = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/average_pixel_values_with_names_5.csv')

# %%
# Specify multiple column names you want to plot
columns_to_plot = ['89409']  # Replace with desired column names, e.g., ['89409', '68884', '48991']

# Plot the selected columns based on their header names
for column in columns_to_plot:
    plt.plot(df.index, df[column], label=column)

# Add labels and legend
plt.xlabel('Frame')
plt.ylabel('Value')
plt.legend()
plt.title('Values over Frames')
plt.show()


# %%
'''
# Specify siingle column name you want to plot
column_to_plot = '89409'  # Replace 'xxx' with the desired column name

# Plot the selected column based on its header name
plt.plot(df.index, df[column_to_plot], label=column_to_plot)

# Add labels and legend
plt.xlabel('Frame')
plt.ylabel('Value')
plt.legend()
plt.title(f'{column_to_plot} over Frames')
plt.show()
'''
# %%
'''
# Plot each column based on its header name
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

# Add labels and legend
plt.xlabel('Frame')
plt.ylabel('Value')
plt.legend()
plt.title('Values over Time')
plt.show()
'''
# %%
