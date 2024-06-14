#%%
# Import
import numpy as np
import tifffile as tiff
import pandas as pd
from tqdm import tqdm

#%%
# Find from catmaid skeleton-ID for each skeleton the largest node radius. 
# output: csv file containing only one node & coordinates per skeleton ID

import pandas as pd

# Path to the input and output files
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

# Loop through the rows and drop rows where column 'r' is 0
rows_to_drop = []
for index, row in result_df.iterrows():
    if row['r'] == 0:
        rows_to_drop.append(index)

filtered_df = result_df.drop(rows_to_drop)

# Remove the 'r' column
filtered_df = filtered_df.drop(columns=['r'])

# Sort the DataFrame by column 'skeleton_id' in descending order
filtered = df.sort_values(by='skeleton_id', ascending=False)

# Save the resulting DataFrame to a new CSV file
filtered_df.to_csv(output_csv_path, index=False)

print(f"Processed output saved to {output_csv_path}")


#%%
# Test 
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

# Threshold distance in micrometers for all three axes
# Note: To transform from s0 to s4, the scale levels (the s0, s1, s2, ...) are powers of two. 
# s4 means pow(2, 4) = 16, which means 1/16th. --> s4 = s0/16

# Threshold distance in micrometers for s0 and s4
# adjust in filenames, take transformation into account
THRESHOLD_DISTANCE_MICROMETERS_S0 = 2
THRESHOLD_DISTANCE_MICROMETERS_S4 = THRESHOLD_DISTANCE_MICROMETERS_S0 / 16

# Paths to the input files
centroids_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/S4-S0-centroids_brain-only_z-450_um.csv'  # in um
xyz_coordinates_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/S4-S0-skeleton_coordinates_soma_um.csv'  # in um

# Path to save the output files
output_csv_path_s4 = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/s4-output_centroids_and_coords_threshold_S0-2um.csv'
no_xyz_output_csv_path_s4 = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/s4-nuclei_without_xyz_threshold_S0-2um.csv'

output_csv_path_s0 = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/s0-output_centroids_and_coords_threshold_S0-2um.csv'
no_xyz_output_csv_path_s0 = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/s0-nuclei_without_xyz_threshold_S0-2um.csv'

# Load the centroids CSV file
centroids_df = pd.read_csv(centroids_csv_path) 

# Rename columns for clarity
centroids_df.rename(columns={
    'x_s4': 'centroid_x_s4', 'y_s4': 'centroid_y_s4', 'z_s4': 'centroid_z_s4',
    'x_s0': 'centroid_x_s0', 'y_s0': 'centroid_y_s0', 'z_s0': 'centroid_z_s0'
}, inplace=True)

# Load the XYZ coordinates
xyz_coordinates = pd.read_csv(xyz_coordinates_path)

# Define the subset of columns for S4 and S0
desired_columns_s4 = ['skeleton_id', 'x_s4', 'y_s4', 'z_s4']
desired_columns_s0 = ['skeleton_id', 'x_s0', 'y_s0', 'z_s0']
desired_centroids_s4 = ['centroid_x_s4', 'centroid_y_s4', 'centroid_z_s4']
desired_centroids_s0 = ['centroid_x_s0', 'centroid_y_s0', 'centroid_z_s0']

# Create new DataFrames with only the desired columns
xyz_coordinates_s4 = xyz_coordinates[desired_columns_s4].dropna()
xyz_coordinates_s0 = xyz_coordinates[desired_columns_s0].dropna()
centroids_df_s4 = centroids_df[desired_centroids_s4].dropna()
centroids_df_s0 = centroids_df[desired_centroids_s0].dropna()

# Function to find closest coordinates using KDTree
def find_closest_coordinates(xyz_coords, centroids, threshold, skeleton_id_col):
    kdtree = KDTree(xyz_coords[['x', 'y', 'z']])
    closest_coords = []
    for centroid in tqdm(centroids.values, desc='Assigning closest coordinates', total=len(centroids)):
        closest_indices = kdtree.query_ball_point(centroid, threshold, p=2)
        if closest_indices:
            for idx in closest_indices:
                closest_coord = xyz_coords.iloc[idx]
                closest_coords.append([closest_coord[skeleton_id_col], *centroid, *closest_coord[['x', 'y', 'z']].values])
            xyz_coords.iloc[closest_indices, xyz_coords.columns.get_loc('x'):xyz_coords.columns.get_loc('z')+1] = np.inf  # Mark all found XYZ coordinates as used
    return closest_coords, xyz_coords[~np.isinf(xyz_coords[['x', 'y', 'z']]).any(axis=1)]

# Adjust the column names for KDTree
xyz_coordinates_s4.rename(columns={'x_s4': 'x', 'y_s4': 'y', 'z_s4': 'z'}, inplace=True)
xyz_coordinates_s0.rename(columns={'x_s0': 'x', 'y_s0': 'y', 'z_s0': 'z'}, inplace=True)

# Find closest coordinates for S4
closest_coords_s4, remaining_coords_s4 = find_closest_coordinates(
    xyz_coordinates_s4, centroids_df_s4, THRESHOLD_DISTANCE_MICROMETERS_S4, 'skeleton_id'
)

# Create DataFrames for the results for S4
columns_s4 = ['skeleton_id', 'centroid_x_s4', 'centroid_y_s4', 'centroid_z_s4', 'x', 'y', 'z']
output_df_s4 = pd.DataFrame(closest_coords_s4, columns=columns_s4)

# Save the results to CSV files for S4
output_df_s4.to_csv(output_csv_path_s4, index=False)
remaining_coords_s4.to_csv(no_xyz_output_csv_path_s4, index=False, columns=['skeleton_id', 'x', 'y', 'z'])

print(f"S4 merged output saved to {output_csv_path_s4}")
print(f"S4 nuclei without XYZ coordinates saved to {no_xyz_output_csv_path_s4}")

# Find closest coordinates for S0
closest_coords_s0, remaining_coords_s0 = find_closest_coordinates(
    xyz_coordinates_s0, centroids_df_s0, THRESHOLD_DISTANCE_MICROMETERS_S0, 'skeleton_id'
)

# Create DataFrames for the results for S0
columns_s0 = ['skeleton_id', 'centroid_x_s0', 'centroid_y_s0', 'centroid_z_s0', 'x', 'y', 'z']
output_df_s0 = pd.DataFrame(closest_coords_s0, columns=columns_s0)

# Save the results to CSV files for S0
output_df_s0.to_csv(output_csv_path_s0, index=False)
remaining_coords_s0.to_csv(no_xyz_output_csv_path_s0, index=False, columns=['skeleton_id', 'x', 'y', 'z'])

print(f"S0 merged output saved to {output_csv_path_s0}")
print(f"S0 nuclei without XYZ coordinates saved to {no_xyz_output_csv_path_s0}")


#%%
# Concatenated csv of the EM-centroid_transforned to LM and centroid coordinate_s4-s0
# Rename columns if necessary

import pandas as pd

# Paths to the input CSV files
file1_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/transformed_LM-points-brain-only_z-450_um.csv'
file2_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/S4-S0-centroids_brain-only_z-450_um.csv'

# Load the input CSV files
LM_space = pd.read_csv(file1_path)
EM_space = pd.read_csv(file2_path)

# Rename headers 
LM_space.rename(columns={'x': 'x_LM', 'y': 'y_LM', 'z': 'z_LM'}, inplace=True)


# Concatenate the DataFrames along columns
concatenated_df = pd.concat([LM_space, EM_space], axis=1)

# Display the concatenated DataFrame
print(concatenated_df)

# Save the concatenated DataFrame to a new CSV file
output_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/LM_EM_space_centroids_um.csv'
concatenated_df.to_csv(output_csv_path, index=False)

print(f"Concatenated output saved to {output_csv_path}")

# %%
# Add index in the format x_LM::y_LM::z_LM, where x_LM, y_LM, z_LM are the floating poinnts of the respected column names
# Input /Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/
# LM_EM_space_centroids_um.csv

import pandas as pd

# Path to your input CSV file
input_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/LM_EM_space_centroids_um.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv_path)

# Ensure all columns are treated as floats (if necessary)
# df = df.astype(float)  # Uncomment if columns are not already floats

# Define the columns to use for the index
index_columns = ['x_LM', 'y_LM', 'z_LM']  # Replace with actual column names

# Create the index column formatted as x_LM::y_LM::z_LM with 6 decimal places
df['index'] = df[index_columns].apply(lambda row: '::'.join(f"{value:.6f}" for value in row), axis=1)

# Reorder columns with the new index column as the first column
columns = ['index'] + df.columns.drop('index').tolist()
df = df[columns]

# Print the DataFrame to verify the changes
print(df)

# Save the modified DataFrame back to a CSV file
output_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/LM_EM_space_centroids_um_INDEX.csv'
df.to_csv(output_csv_path, index=False)

print(f"DataFrame with index saved to {output_csv_path}")

# %%
# Add skeleton-id to LM_EM_space_centroids_um_INDEX, if no match == NA

import pandas as pd

# Paths to your input CSV files
csv1_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/LM_EM_space_centroids_um_INDEX.csv'
csv2_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/s0-output_centroids_and_coords_threshold_S0-2um.csv'

# Read only the headers of the CSV files
df1_headers = pd.read_csv(csv1_path, nrows=0)
df2_headers = pd.read_csv(csv2_path, nrows=0)

# Print the headers
print("Headers of CSV 1:")
print(df1_headers.columns.tolist())

print("\nHeaders of CSV 2:")
print(df2_headers.columns.tolist())

# Load the CSV files into DataFrames
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# Adding an empty column named 'skeleton_id' with None values to df1
df1['skeleton_id'] = None

# Rename the headers as needed
rename_dict1 = {
    'x_LM': 'x_LM', 
    'y_LM': 'y_LM',
    'z_LM': 'z_LM',
    'x_s4': 'x_s4', 
    'y_s4': 'y_s4',
    'z_s4': 'z_s4',
    'x_s0': 'x_s0', 
    'y_s0': 'y_s0',
    'z_s0': 'z_s0',
    'skeleton_id': 'skeleton_id'   
}

rename_dict2 = {
    'centroid_x_s0': 'x_s0', 
    'centroid_y_s0': 'y_s0',
    'centroid_z_s0': 'z_s0',
    'centroid_x_s4': 'x_s4', 
    'centroid_y_s4': 'y_s4',
    'centroid_z_s4': 'z_s4',
    'skeleton_id': 'skeleton_id'
}

# Rename columns in df2
df2.rename(columns=rename_dict2, inplace=True)

# Print the headers after renaming
print("\nHeaders of CSV 1 after renaming:")
print(df1.columns.tolist())

print("\nHeaders of CSV 2 after renaming:")
print(df2.columns.tolist())

# Check if the renamed columns exist in both DataFrames
merge_columns = ['x_s0', 'y_s0', 'z_s0']  # Adjust merge columns based on available columns in both DataFrames

# Check if merge columns exist in both DataFrames
missing_columns_df1 = [col for col in merge_columns if col not in df1.columns]
missing_columns_df2 = [col for col in merge_columns if col not in df2.columns]

if missing_columns_df1:
    print(f"Missing columns in df1: {missing_columns_df1}")
if missing_columns_df2:
    print(f"Missing columns in df2: {missing_columns_df2}")

# Proceed with merge only if all specified columns exist in both DataFrames
if not missing_columns_df1 and not missing_columns_df2:
    # Merge the DataFrames on the specified columns, keeping all rows
    merged_df = pd.merge(df1, df2, on=merge_columns, how='outer', suffixes=('_csv1', '_csv2'))

    # Print the merged DataFrame to verify the changes
    print(merged_df)

    # Path to save the merged CSV file
    output_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/LM_EM_space_centroids_skeleton_id_um_INDEX.csv'

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(output_csv_path, index=False)

    print(f"Merged DataFrame saved to {output_csv_path}")
else:
    print("Merge operation aborted due to missing columns.")


# %%
