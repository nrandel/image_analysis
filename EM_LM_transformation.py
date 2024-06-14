#%%
# Import
import numpy as np
import tifffile as tiff
import pandas as pd
from tqdm import tqdm

#%%
# Find from catmaid skeleton-ID for each skeleton the largest node radius. 
# output: csv file containing only one node & coordinates per skeleton ID

# Path to the input file
input_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/skeleton_coordinates.csv'
output_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/S0-skeleton_coordinates_soma.csv'

# Load the input CSV file
df = pd.read_csv(input_csv_path)

# Remove the 'treenode_id' and 'parent_treenode_id' columns
df = df.drop(columns=[' treenode_id', ' parent_treenode_id'])

# Rename columns ' x', ' y', ' z' to 'x,y,z'
df.rename(columns={' skeleton_id': 'skeleton_id', ' r': 'r', ' x': 'x', ' y': 'y', ' z': 'z'}, inplace=True)

# Find the row with the largest 'r' value for each 'skeleton_id'
result_df = df.loc[df.groupby('skeleton_id')['r'].idxmax()]

# Sort the DataFrame by column 'r' in descending order
result_df = df.sort_values(by='r', ascending=False)

# Loop through the rows and drop rows where column 'C' is 0
rows_to_drop = []
for index, row in result_df.iterrows():
    if row['r'] == 0:
        rows_to_drop.append(index)

filtered_df = result_df.drop(rows_to_drop)

# Remove the 'r' columns
filtered_df = filtered_df.drop(columns=['r'])

# Save the resulting dataframe to a new CSV file
filtered_df.to_csv(output_csv_path, index=False)

print(f"Processed output saved to {output_csv_path}")

#%%

# Input 3D tif file with segmented areas, csv with xyz coordinates.
# Output all segmented areas that overlap with a xyz coordinate

import numpy as np
import pandas as pd
import tifffile

# Step 1: Load and Process the 3D TIFF File (Segmented Areas)
def load_tiff_stack(tiff_file):
    return tifffile.imread(tiff_file)

segmented_tiff_file = '/path/to/segmented_areas.tif'
segmented_areas = load_tiff_stack(segmented_tiff_file)

# Step 2: Load XYZ Coordinates from CSV
csv_file = '/path/to/xyz_coordinates.csv'
xyz_data = pd.read_csv(csv_file)

# Step 3: Identify Segmented Areas Overlapping with XYZ Coordinates
overlapping_segments = []

for index, row in xyz_data.iterrows():
    x, y, z = int(row['x']), int(row['y']), int(row['z'])
    
    # Check if the coordinates are within the segmented areas
    if 0 <= x < segmented_areas.shape[0] and 0 <= y < segmented_areas.shape[1] and 0 <= z < segmented_areas.shape[2]:
        if segmented_areas[x, y, z] == 1:  # Assuming segmented areas are marked as 1 in the TIFF stack
            overlapping_segments.append((x, y, z))

# Step 4: Output the Results
print("Overlapping Segmented Areas:")
for segment in overlapping_segments:
    print(f"Segmented Area at (x={segment[0]}, y={segment[1]}, z={segment[2]})")









#%% 
#TODO Output needs testing
# Input: Segmented EM nuclei (S0 or s4) and skeleton root nodes S0 or S4 (not transformed == raw Catmaid output from previous code block)
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

# Threshold distance in micrometers for all three axis
THRESHOLD_DISTANCE_MICROMETERS = 8

# Paths to the input files

# S4 resolution
centroids_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/centroids_brain-only_z-450.csv'  # in um
xyz_coordinates_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/S4-skeleton_coordinates_soma.csv'  # in um

# Path to save the output files
# S4 resolution
output_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/s4-output_centroids_and_coords_threshold_8um.csv'
no_xyz_output_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/s4-nuclei_without_xyz_threshold_8um.csv'

# Load the centroids CSV file
centroids_df = pd.read_csv(centroids_csv_path)
# Rename columns for clarity
centroids_df.rename(columns={'x': 'centroid_x', 'y': 'centroid_y', 'z': 'centroid_z'}, inplace=True)

# Load the XYZ coordinates
xyz_coordinates = pd.read_csv(xyz_coordinates_path)

# Define the subset of columns you want to keep
desired_columns = ['skeleton_id', 'x_s4', 'y_s4', 'z_s4']

# Create a new dataframe with only the desired columns
xyz_coordinates = xyz_coordinates[desired_columns]

# Ensure coordinates are floats
coordinates = xyz_coordinates[['x_s4', 'y_s4', 'z_s4']].values

# Ensure centroids are floats and get their values
centroids = centroids_df[['centroid_z', 'centroid_y', 'centroid_x']].values

# Build a KDTree for the coordinates in micrometers
kdtree = KDTree(coordinates)

# Find the closest XYZ coordinate for each centroid within the threshold distance
closest_coords = []
no_xyz_coords = []

for centroid_idx, centroid in tqdm(enumerate(centroids), desc='Assigning closest coordinates', total=len(centroids)):
    # Query the KDTree for points within threshold distance along all axes
    closest_indices = kdtree.query_ball_point(centroid, THRESHOLD_DISTANCE_MICROMETERS, p=2)
    
    if closest_indices:
        for idx in closest_indices:
            closest_coord = coordinates[idx]
            closest_coords.append([xyz_coordinates.iloc[idx]['skeleton_id'], *centroid, *closest_coord])
            
        # Mark all found XYZ coordinates as used
        coordinates[closest_indices] = np.inf
    
# Create DataFrames for the results
columns = ['skeleton_id', 'centroid_z', 'centroid_y', 'centroid_x', 'x_s4', 'y_s4', 'z_s4']
output_df = pd.DataFrame(closest_coords, columns=columns)

# Filter out rows where x_s4, y_s4, z_s4 are inf (indicating no corresponding XYZ found within threshold)
no_xyz_df = pd.DataFrame(coordinates[~np.isinf(coordinates).any(axis=1)], columns=['x_s4', 'y_s4', 'z_s4'])

# Save the results to CSV files
output_df.to_csv(output_csv_path, index=False)
no_xyz_df.to_csv(no_xyz_output_csv_path, index=False)

print(f"Merged output saved to {output_csv_path}")























#%%
# TODO Not working because the rows df1 are not df2 the same!!
# Concatenate EM centroid s4 with transformed LM s4 
# Change transformed LM s0! from nm to um

import pandas as pd

# Load the CSV files
file1_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/centroids_brain-only_z-450.csv'
file2_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/S4-skeleton_coordinates_soma.csv'

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Rename headers for df1
df1.rename(columns={'x': 'centroid_x_s4', 'y': 'centroid_y_s4', 'z': 'centroid_z_s4'}, inplace=True)

# Inspect the first few rows to determine the correct index column
print("File 1 headers:", df1.columns)
print("File 2 headers:", df2.columns)
print("File 1 preview:\n", df1.head())
print("File 2 preview:\n", df2.head())

# Convert df2 coordinates from nanometers to microns
df2[['x_s0', 'y_s0', 'z_s0']] = df2[['x_s0', 'y_s0', 'z_s0']] / 1000.0



# Concatenate the DataFrames side by side
concatenated_df = pd.concat([df1, df2], axis=1)

# Save the concatenated DataFrame to a new CSV file
concatenated_file_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/Concatenate-S4-centroid_skeleton_coordinates.csv'
concatenated_df.to_csv(concatenated_file_path, index=False)

print("Concatenated data saved to 'Concatenate-S4-centroid_skeleton_coordinates.csv''")





# %%
#TODO

# Inout csv: activtiy & concatenated file with skeleton id and centroid xyz
# Output: act

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
