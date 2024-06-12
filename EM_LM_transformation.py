# Input: Segmented EM nuclei and skeleton root nodes (not transformed == raw Catmaid output)
# Output: Centroid (EM nuclei) and root node coordinates that overlap in nuclei segmentation

# %%
#Import
import numpy as np
import tifffile as tiff
import pandas as pd
from tqdm import tqdm

# Load the 3D segmented TIFF file
segmented_tiff_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/soma_labels.tif'
segmented_data = tiff.imread(segmented_tiff_path)

# Load the XYZ coordinates
xyz_coordinates_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/centroids_brain-only_z-450.csv'
xyz_coordinates = pd.read_csv(xyz_coordinates_path)

# Check the structure of the segmented_data and the coordinates
print(f"Segmented data shape: {segmented_data.shape}")
print(f"XYZ coordinates: \n{xyz_coordinates.head()}")

# Assuming the coordinates CSV has columns: 'x', 'y', 'z'
coordinates = xyz_coordinates[['x', 'y', 'z']].values

# Ensure coordinates are integers and within bounds
coordinates = np.round(coordinates).astype(int)

# Get the shape of the segmented data
depth, height, width = segmented_data.shape

# Filter out coordinates that are out of bounds
valid_coords = (coordinates[:, 0] >= 0) & (coordinates[:, 0] < width) & \
               (coordinates[:, 1] >= 0) & (coordinates[:, 1] < height) & \
               (coordinates[:, 2] >= 0) & (coordinates[:, 2] < depth)
coordinates = coordinates[valid_coords]

# Get segment IDs for each coordinate
segment_ids = segmented_data[coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]]

# Create a DataFrame with the coordinates and their corresponding segment IDs
coords_df = pd.DataFrame(coordinates, columns=['x', 'y', 'z'])
coords_df['segment_id'] = segment_ids

# Calculate centroids for each segment with overlapping coordinates
segment_centroids = coords_df.groupby('segment_id').apply(lambda group: group[['z', 'y', 'x']].mean()).reset_index()
segment_centroids.columns = ['segment_id', 'centroid_z', 'centroid_y', 'centroid_x']

# Merge the centroids back with the original coordinates
output_df = pd.merge(coords_df, segment_centroids, on='segment_id')

# Select and reorder the columns for the final output
output_df = output_df[['segment_id', 'centroid_z', 'centroid_y', 'centroid_x', 'x', 'y', 'z']]

# Save to CSV
output_csv_path = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/test/output_centroids_and_coords.csv'
output_df.to_csv(output_csv_path, index=False)

print(f"Output saved to {output_csv_path}")

# %%
