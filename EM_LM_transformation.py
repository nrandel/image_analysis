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






#TODO Output needs testing OLD but WORKING
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
THRESHOLD_DISTANCE_MICROMETERS = 8 #A djust the output filename

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
