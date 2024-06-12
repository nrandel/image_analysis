#%%
# Import
import numpy as np
import tifffile as tiff
import pandas as pd
from tqdm import tqdm

#%%
# Find from brain segmentation, which of the segmented nuclei has at least a single node
# 1) from catmaid skeleton-ID find for each skeleton the largest node radius. 
# output: csv file containing only one node & coordinates per skeleton ID
# 2) Overlapp between segmented nuclei and skeleton-ID (next code block)--> find traced and non-traced cells

# Path to the input file
input_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/skeleton_coordinates.csv'
output_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/skeleton_coordinates_soma.csv'

# Load the input CSV file
df = pd.read_csv(input_csv_path)

# Remove the 'treenode_id' and 'parent_treenode_id' columns
df = df.drop(columns=[' treenode_id', ' parent_treenode_id'])

# Rename columns ' x', ' y', ' z' to 'x,y,z'
df.rename(columns={' skeleton_id': 'skeleton_id', ' r': 'r', ' x': 'x', ' y': 'y', ' z': 'z'}, inplace=True)

# Find the row with the largest 'r' value for each 'skeleton_id'
result_df = df.loc[df.groupby('skeleton_id')['r'].idxmax()]

# Remove the 'r' columns
df = df.drop(columns=['r'])

# Save the resulting dataframe to a new CSV file
result_df.to_csv(output_csv_path, index=False)

print(f"Processed output saved to {output_csv_path}")


#%% 
#TODO
# Input: Segmented EM nuclei and skeleton root nodes (not transformed == raw Catmaid output from previous code block)
# Output: Centroid (EM nuclei) and root node coordinates that overlap in nuclei segmentation

#Load the segmented 3D nuclei TIFF file.
#Load the CSV with xyz coordinates.
#Calculate the centroids of each segmented nucleus.
#Assign the closest xyz coordinate to each centroid.
#Save the segmented nuclei that do not have a corresponding xyz separately.
#Save the results in a new CSV file with both centroid and xyz information.

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

# Conversion factor from micrometers to nanometers
MICROMETERS_TO_NANOMETERS = 1000

# Paths to the input files
centroids_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/centroids_brain-only_z-450.csv'
xyz_coordinates_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/skeleton_coordinates_soma.csv'

# Path to save the output files
output_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/output_centroids_and_coords.csv'
no_xyz_output_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/nuclei_without_xyz.csv'

# Load the centroids CSV file
centroids_df = pd.read_csv(centroids_csv_path)
# Rename columns for clarity
centroids_df.rename(columns={'x': 'centroid_x', 'y': 'centroid_y', 'z': 'centroid_z'}, inplace=True)

# Display the headers of the centroids_df CSV file
print(f"Headers of the centroids_df CSV file: {centroids_df.columns.tolist()}")

# Load the XYZ coordinates
xyz_coordinates = pd.read_csv(xyz_coordinates_path)

# Display the headers of the XYZ coordinates file
print(f"Headers of the XYZ coordinates CSV file: {xyz_coordinates.columns.tolist()}")

# Ensure coordinates are floats
coordinates = xyz_coordinates[['x', 'y', 'z']].values

# Convert coordinates from nanometers to micrometers for matching
coordinates_micrometers = coordinates / MICROMETERS_TO_NANOMETERS

# Ensure centroids are floats and get their values
centroids = centroids_df[['centroid_z', 'centroid_y', 'centroid_x']].values

# Build a KDTree for the coordinates in micrometers
kdtree = KDTree(coordinates_micrometers)

# Find the closest XYZ coordinate for each centroid
closest_coords = []
no_xyz_coords = []

# Track used coordinates
used_indices = set()

for idx, xyz_coord in tqdm(enumerate(coordinates_micrometers), desc='Assigning closest coordinates', total=len(coordinates_micrometers)):
    closest_distance, closest_idx = np.inf, None
    for centroid_idx, centroid in enumerate(centroids):
        distance = np.linalg.norm(centroid - xyz_coord)
        if distance < closest_distance and centroid_idx not in used_indices:
            closest_distance = distance
            closest_idx = centroid_idx
    if closest_idx is not None:
        # Mark the centroid as used
        used_indices.add(closest_idx)
        closest_coords.append([xyz_coordinates.iloc[idx]['skeleton_id'], *centroids[closest_idx], *coordinates[idx]])
    else:
        no_xyz_coords.append([*xyz_coord])

# Create DataFrames for the results
columns = ['skeleton_id', 'centroid_z', 'centroid_y', 'centroid_x', 'x', 'y', 'z']
output_df = pd.DataFrame(closest_coords, columns=columns)
no_xyz_df = pd.DataFrame(no_xyz_coords, columns=['x', 'y', 'z'])

# Save the results to CSV files
output_df.to_csv(output_csv_path, index=False)
no_xyz_df.to_csv(no_xyz_output_csv_path, index=False)

print(f"Merged output saved to {output_csv_path}")
print(f"Nuclei without XYZ coordinates saved to {no_xyz_output_csv_path}")













#%%
# Because the Catmaid xyz out put is preserve the order is maintained. 
# Therefore 


# %%
#TODO
# Note: The output file from previous section contains the EM centroid with all rood nodes from the brain lobes


#Load the second CSV file.
#Identify the common columns for matching.
#Find the rows in the second CSV where the values in these common columns match those in the output_df.
#Concatenate the matching rows from the second CSV with the output_df.

# Output: EM centroid, xyz coordinates, skeleton id

# Import
import pandas as pd

# Paths to the files
output_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/output_centroids_and_coords.csv'
second_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/skeleton_coordinates.csv'

# Load the output_df from the previous script
output_df = pd.read_csv(output_csv_path)

# Load the second CSV file
second_df = pd.read_csv(second_csv_path)

# Display the headers of the second CSV file to understand the current column names
print(f"Headers of the second CSV file: {second_df.columns.tolist()}")

# Identify the expected common columns for matching
common_columns = ['x', 'y', 'z']

# Mapping of old headers to new headers if necessary
# Replace 'old_x_header', 'old_y_header', and 'old_z_header' with actual column names in the second CSV file
header_mapping = {
    ' x': 'x',
    ' y': 'y',
    ' z': 'z'
}

# Rename headers of the second CSV to ensure they match the output_df headers
second_df.rename(columns=header_mapping, inplace=True)

# Ensure common columns exist
common_columns = ['x', 'y', 'z']
for column in common_columns:
    if column not in output_df.columns or column not in second_df.columns:
        raise ValueError(f"Column '{column}' not found in both dataframes")


# Adjust scale of x,y,z values






# Merge based on common columns
merged_df = pd.merge(output_df, second_df, on=common_columns, how='inner')

# Discard extra rows in second_df
second_df_subset = second_df[second_df[common_columns].isin(output_df[common_columns]).all(axis=1)]

# Save the merged dataframe to a new CSV file
merged_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/merged_output.csv'
merged_df.to_csv(merged_csv_path, index=False)

# Save the subset of second_df to a new CSV file
#subset_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/subset_second_df_output.csv'
#second_df_subset.to_csv(subset_csv_path, index=False)

print(f"Merged output saved to {merged_csv_path}")
#print(f"Subset of second_df saved to {subset_csv_path}")



#%% Test if csv output has no data

print(f"Number of rows in merged dataframe: {len(merged_df)}")

print(f"Common columns in output_df: {common_columns}")
print(f"Columns in output_df: {output_df.columns}")
print(f"Columns in second_df: {second_df.columns}")

print(f"Number of rows in output_df: {len(output_df)}")
print(f"Number of rows in second_df: {len(second_df)}")



#%%
#TODO
# Input: merged_output (previous code block) & skeleton_names
# Output: EM centroid, xyz coordinates, skeleton id, neuron name

# Import
import numpy as np
import pandas as pd

# Paths to the files
output_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/merged_output.csv'
second_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/skeleton_names.csv'

# Load the output_df from the previous script
output_df = pd.read_csv(output_csv_path)

# Load the second CSV file
second_df = pd.read_csv(second_csv_path)

# Display the headers of the second CSV file to understand the current column names
print(f"Headers of the second CSV file: {second_df.columns.tolist()}")

# Identify the expected common columns for matching
common_columns = ['skeleton_id']

# Ensure the common columns exist in both dataframes after renaming
for column in common_columns:
    if column not in output_df.columns or column not in second_df.columns:
        raise ValueError(f"Column '{column}' not found in both dataframes")

# Merge the dataframes on the common columns
merged_df = pd.merge(output_df, second_df, on=common_columns, how='inner')

# Save the merged dataframe to a new CSV file
merged_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/output_ID-name-coordinates.csv'
merged_df.to_csv(merged_csv_path, index=False)

print(f"Merged output saved to {merged_csv_path}")

#%%
#TODO
# Fix the measurements_klb-raw_data.csv from large_scale.py that it can be used

#CSV1: Contains rows of floating-point values with headers x, y, z. (output from previous block)
#CSV2: Contains headers formatted as "x::y::z" strings and rows of data, but the order of columns in CSV2 does not correspond to the order of rows in CSV1.

#Match the columns in CSV2 to the rows in CSV1 based on the headers of CSV2.
#Rearrange CSV2 columns so that they align with the rows in CSV1.

import pandas as pd

# Paths to the files
csv1_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/output_ID-name-coordinates.csv'
csv2_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/measurements_t_stacks-dff_long-slidingWindow.csv'

# Load CSV1
csv1_df = pd.read_csv(csv1_path)

# Load CSV2
csv2_df = pd.read_csv(csv2_path)

# Display the headers of CSV2
print(f"Headers of CSV2: {csv2_df.columns.tolist()}")

# Parse and clean the headers of CSV2
csv2_headers = csv2_df.columns.tolist()
cleaned_headers = [header.strip().strip('"') for header in csv2_headers if header != 'timepoint']
parsed_headers = [tuple(map(float, header.split('::'))) for header in cleaned_headers]

# Create a DataFrame from parsed headers
headers_df = pd.DataFrame(parsed_headers, columns=['x', 'y', 'z'])

# Merge CSV1 and headers_df to get the matching order
csv1_with_headers = csv1_df.reset_index().merge(headers_df, on=['x', 'y', 'z']).set_index('index')
matching_order = csv1_with_headers.index

# Include the 'timepoint' column and reorder CSV2 columns based on the matching order
reordered_columns = ['timepoint'] + [csv2_headers[i+1] for i in matching_order]  # +1 because 'timepoint' is the first column
reordered_csv2_df = csv2_df[reordered_columns]

# Save the reordered CSV2
reordered_csv2_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/reordered_csv2.csv'
reordered_csv2_df.to_csv(reordered_csv2_path, index=False)

print(f"Reordered CSV2 saved to {reordered_csv2_path}")

# %%
