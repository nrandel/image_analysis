# %%
# Import libraries
import numpy as np
import pandas as pd

# %%

# Generate merged csv of activity traces
# Path to the directory containing your CSV files
directory_path = '/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces'

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
