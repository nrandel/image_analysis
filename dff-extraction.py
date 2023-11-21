# %%
# Import libraries
import os
import numpy as np
import tifffile as tf
import pandas as pd
from scipy.spatial.distance import cdist

#%%

# Load one TIFF stack to check dimension
tiff_stack = tf.imread('/Users/nadine/Documents/Zlatic_lab/manual_registration_1099/dff_WB/test_made-up_stack/t_1001.tiff')

# Check the dimensions (shape) of the TIFF stack
print("Dimensions (shape) of TIFF stack:", tiff_stack.shape) # z,y,x in px

# Other properties
print("Data type:", tiff_stack.dtype)
print("Number of images in stack:", tiff_stack.shape[0])  # Assuming 1st dimension represents the number of images
print("Image size (height x width):", tiff_stack.shape[1], "x", tiff_stack.shape[2])

"tiff_stack.shape[0] = z, tiff_stack.shape[1] = y, and tiff_stack.shape[2]) = x"

# %%

# Get tif files
# Path to the folder containing the 3D images
folder_path = '/Users/nadine/Documents/Zlatic_lab/manual_registration_1099/dff_WB/test_made-up_stack'

# Get a list of TIFF files in the folder
tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff')]

# %%
# Load the CSV file containing coordinates (in nm)

coordinates_data = pd.read_csv('/Users/nadine/Documents/Zlatic_lab/manual_registration_1099/dff_WB/landmarksLM-EMonly.csv')

columns = ["skid-left", "bool", "LM_x", "LM_y", "LM_z", "EM_x", "EM_y", "EM_z"]
coordinates_data.columns = columns

LM_points = coordinates_data[["LM_x", "LM_y", "LM_z"]].values

# Adjust csv: nm to pixel 
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
# Iterate through each tiff stack, extract average pixel value and save as csv

folder_path = '/Users/nadine/Documents/Zlatic_lab/manual_registration_1099/dff_WB/test_made-up_stack'  # Replace with your folder path
tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tif') or f.endswith('.tiff')]

# Dictionary to store average pixel values for each coordinate
average_values = {}

for filename in tiff_files:
    file_path = os.path.join(folder_path, filename)
    image = tf.imread(file_path)

    for index, row in coordinates_data.iterrows():
        x, y, z = row['LM_x'], row['LM_y'], row['LM_z']
        radius_x = 2  # Radius for x
        radius_y = 2  # Radius for y
        radius_z = 1  # Radius for z

        x_range = np.arange(0, image.shape[2])  # x-axis
        y_range = np.arange(0, image.shape[1])  # y-axis
        z_range = np.arange(0, image.shape[0])  # z-axis

        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')

        # Generate sphere masks for each axis
        sphere_mask_x = (xx - x) ** 2 <= radius_x ** 2
        sphere_mask_y = (yy - y) ** 2 <= radius_y ** 2
        sphere_mask_z = (zz - z) ** 2 <= radius_z ** 2

        # Combine masks for x, y, and z axes
        sphere_mask = sphere_mask_x & sphere_mask_y & sphere_mask_z

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

# Convert dictionary to DataFrame and save as CSV
df = pd.DataFrame(average_values)
df.to_csv('/Users/nadine/Documents/Zlatic_lab/manual_registration_1099/dff_WB/test_made-up_stack/average_pixel_values.csv', index=False)

# %%
