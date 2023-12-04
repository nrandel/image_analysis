# %%
# Import libraries
import os
import numpy as np
import tifffile as tf
import pandas as pd
from scipy.spatial.distance import cdist

# %%
# Load the CSV file containing coordinates (in nm) and et a sphere at defined coordinate in 4D image stack 
# and read out average pixel value (df/f in sphere over time)

coordinates_data = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/manual_registration_1099/dff_WB/landmarksLM-EMonly.csv')

columns = ["skid-left", "bool", "LM_x", "LM_y", "LM_z", "EM_x", "EM_y", "EM_z"]
coordinates_data.columns = columns

LM_points = coordinates_data[["LM_x", "LM_y", "LM_z"]].values

# Adjust csv: nm to pixel (use if coordinates in csv is converted to nm, but the image stack is 1x1x1 px)
# Predefined value for division
px_width = 406.5041 
px_height = 406.5041
px_depth = 1700

# Divide each float value in a column by the predefined value and convert to integers
coordinates_data['LM_x'] = (coordinates_data['LM_x'] / px_width).astype(int)
coordinates_data['LM_y'] = (coordinates_data['LM_y'] / px_height).astype(int)
coordinates_data['LM_z'] = (coordinates_data['LM_z'] / px_depth).astype(int)

# Print the updated DataFrame
#print(coordinates_data)
#print(coordinates_data["LM_x"])
#print(coordinates_data["LM_y"])
#print(coordinates_data["LM_z"])

# %%
# Extract tiff stack (sorted!)

import os

folder_path = '/Users/nadine/Desktop/dff_test_artifact/'  # Replace with your folder path

tiff_files = sorted(
    [f for f in os.listdir(folder_path) if f.endswith('.tif') or f.endswith('.tiff')],
    key=lambda x: int(x.split('_')[1].split('.')[0])  # Extract the numerical part for sorting
)

print(tiff_files)

# %%

# Load the image directly into an array, and create a stack of 3D arrays (stack) 
# representing the layers within the TIFF file.

tiff_stack = tf.imread(
            '/Users/nadine/Desktop/dff_test_artifact/t_1000.tiff')

# To open in 3D. Image will be a 3D stack-array
with tf.TiffFile('/Users/nadine/Desktop/dff_test_artifact//t_1000.tiff') as tif:
    stack = [page.asarray() for page in tif.pages]
stack = np.asarray(stack)


# %%
# Set radii for mask
radius_x = 3  # Radius for x
radius_y = 3  # Radius for y
radius_z = 1  # Radius for z

x_range = np.arange(0, stack.shape[2])  # x-axis
y_range = np.arange(0, stack.shape[1])  # y-axis
z_range = np.arange(0, stack.shape[0])  # z-axis
xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')

# %%
# Iterate through each tiff stack (sorted), extract average pixel value and save as csv
# Include column for Tiff-file name in output

# Dictionary to store average pixel values for each coordinate
average_values = {}
average_values['time'] = []

for filename in tiff_files:
    file_path = os.path.join(folder_path, filename)

# *** to open in 3D. image will be a 3D stack-array.
    with tf.TiffFile(file_path) as tif:
        image = [page.asarray() for page in tif.pages]
    image = np.asarray(image)

    print(filename) # *** to keep track of the progress

    for index, row in coordinates_data.iterrows():
        x, y, z = row['LM_x'], row['LM_y'], row['LM_z']
        
        # Generate sphere masks for each axis
        sphere_mask_x = (xx - x) ** 2 <= radius_x ** 2
        sphere_mask_y = (yy - y) ** 2 <= radius_y ** 2
        sphere_mask_z = (zz - z) ** 2 <= radius_z ** 2

        # Combine masks for x, y, and z axes
        sphere_mask = sphere_mask_z & sphere_mask_y & sphere_mask_z

        # Transpose sphere_mask dimensions to match the image stack's shape
        sphere_mask = np.transpose(sphere_mask, axes=(2, 1, 0))

        # Check sphere mask properties
        #print("Sphere Mask Shape:", sphere_mask.shape)
        #print("Number of True Values in Mask:", np.sum(sphere_mask))

        # Apply mask to image
        pixels_in_sphere = image[sphere_mask]
        #print("Pixels in Sphere:", pixels_in_sphere)

        # Calculate average pixel intensity within the sphere
        average_intensity = np.mean(pixels_in_sphere)

        #print(f"Average intensity for coordinates ({x}, {y}, {z}) "
        #      f"in image {filename}: {average_intensity}")

        # Create a key to store average values for each coordinate
        coordinate_key = f"Coordinate_{x}_{y}_{z}"
        
        # Add average intensity to the dictionary
        if coordinate_key not in average_values:
            average_values[coordinate_key] = []
        average_values[coordinate_key].append(average_intensity)

        # print(average_values)

        # Create a dictionary to map coordinates to names
        coordinate_to_name = {}

        for index, row in coordinates_data.iterrows():
            name = row["skid-left"]
            x, y, z = row['LM_x'], row['LM_y'], row['LM_z']
            coordinate_to_name[f"Coordinate_{x}_{y}_{z}"] = name
            # coordinate_to_name["time"] = t

    average_values['time'].append(filename)

# Convert dictionary to DataFrame and save as CSV
df = pd.DataFrame(average_values)
df.rename(columns=coordinate_to_name, inplace=True)  # Rename columns using the dictionary
df.to_csv(
    '/Users/nadine/Desktop/dff_test_artifact/average_pixel_values_with_names_test_radius-3-3-1.csv', index=False)

# %%
