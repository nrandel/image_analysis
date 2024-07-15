# %%
# Import libraries
import numpy as np
import pandas as pd
import os
import re

# %%

# Generate merged csv of activity traces
# Path to the directory containing your CSV files
directory_path = '/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/2023-11-24'

# List all CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# Sort CSV files if necessary (change the sorting logic as needed)
#csv_files.sort()  # Change this sorting logic based on your requirement (sorts, 1, 10, 11, 2, 3,..)

# Create a sorted copy of the list of CSV file names with a custom key for numerical sorting
def extract_number(filename):
    try:
        return int(re.search(r'\d+', filename.split('_')[-1]).group())
    except AttributeError:
        return float('inf')  # Return a high value for filenames where a number isn't found

sorted_csv_files = sorted(csv_files, key=extract_number)

# Print the sorted list of CSV file names
for file in sorted_csv_files:
    print(file)

# Initialize an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

# Iterate through each sorted CSV file, read it, and append its contents to the merged_data DataFrame
for file in sorted_csv_files:
    file_path = os.path.join(directory_path, file)
    data = pd.read_csv(file_path)
    merged_data = pd.concat([merged_data, data], ignore_index=True)

# Output the merged data to a single CSV file
merged_data.to_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/merged_activity_traces.csv', index=False)

# %%
# Intersection dff-sliding window and my dff
# Normalize heder format

import pandas as pd

# Define a function to normalize the headers
def normalize_headers(headers):
    return [header.strip().replace('"', '').replace(' ', '') for header in headers]

# Read the CSV files into dataframes
df1 = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/activity_of_responsive_neurons_sd_2_average_100-110.csv')
df2 = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/raw_fluorescence/F0-8_Ft-40/intersection_neuronal_activity_responsive_neurons_sd_2_average_100-110-AND-680-699_For_Stimulus_2.csv')


# Normalize the headers
df1.columns = normalize_headers(df1.columns)
df2.columns = normalize_headers(df2.columns)

# Extract column names
columns_df1 = set(df1.columns)
columns_df2 = set(df2.columns)

# Find common columns
common_columns = columns_df1.intersection(columns_df2)

# Print the common columns
print("Common columns:", common_columns)

# Subset both dataframes to include only the common columns (if needed)
df1_common = df1[common_columns]
df2_common = df2[common_columns]

# Optionally, save the results to new CSV files
df1_common.to_csv('/Users/nadine/Desktop/test/WB-file1_common.csv', index=False)
df2_common.to_csv('/Users/nadine/Desktop/test/NR-file2.csv', index=False)

# Print the first few rows of the subset dataframes to verify the results
print("File1 with common columns:\n", df1_common.head())
print("File2 with common columns:\n", df2_common.head())




# %%
# Normalize header and plot specific columns

import pandas as pd
import matplotlib.pyplot as plt

# Define a function to normalize the headers
def normalize_headers(headers):
    return [header.strip().replace('"', '').replace(' ', '') for header in headers]

# Read the CSV file into a dataframe
# WB data
file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_t_stacks-dff_long-slidingWindow.csv'

# NR data
#file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_klb-raw_data.csv'

# NR dff
#file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/raw_fluorescence/F0-8_Ft-40/activity_of_responsive_neurons_sd_2_average_100-110.csv'

df = pd.read_csv(file_path)

# Normalize the headers
df.columns = normalize_headers(df.columns)

# Print headers of the first few columns to check format
print("Headers of the first few columns:")
for col in df.columns[:5]:  # Print headers of the first 5 columns
    print(col)

# Predefined columns to plot (replace with your specific column names)
#selected_columns = ["235.035805::295.642623::16.318526", "243.233741::295.602835::17.998019"]
selected_columns = ["296.468774::453.023438::30.552028", "156.132785::418.524036::8.528740", "294.232986::464.458234::47.435177", "183.112771::448.796633::35.421674"]


# Normalize predefined column names to match dataframe headers
selected_columns = [col.strip().replace('"', '').replace(' ', '') for col in selected_columns]

# Check if the selected columns are in the dataframe
for col in selected_columns:
    if col not in df.columns:
        print(f"Column '{col}' not found in the dataframe. Please check the column names.")
        exit()

# Define x-axis range
x_min = 50
x_max = 200

# Plot the selected columns with specified x-axis range
plt.figure(figsize=(10, 6))
for col in selected_columns:
    plt.plot(df.index[x_min:x_max+1], df[col].iloc[x_min:x_max+1], label=col)

plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Selected Columns Plot')
plt.legend()
plt.grid(True)
plt.show()


# %%
