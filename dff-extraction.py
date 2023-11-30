# %%
# Import libraries
import os
import numpy as np
import tifffile as tf
import pandas as pd
from scipy.spatial.distance import cdist

#%%
'''
# Load one TIFF stack to check dimension
# Version 1
tiff_stack = tf.imread('/Users/nadine/Documents/Zlatic_lab/manual_registration_1099/dff_WB/test_made-up_stack/t_1001.tiff') #works for this stack were I before adjusted the image properties

# Check the dimensions (shape) of the TIFF stack
print("Dimensions (shape) of TIFF stack:", tiff_stack.shape) # z,y,x in px

# Other properties
print("Data type:", tiff_stack.dtype)
print("Number of images in stack:", tiff_stack.shape[0])  # Assuming 1st dimension represents the number of images
print("Image size (height x width):", tiff_stack.shape[1], "x", tiff_stack.shape[2])

#tiff_stack.shape[0] = z, tiff_stack.shape[1] = y, and tiff_stack.shape[2]) = x

'''
# %%
'''
# Load one TIFF stack to check dimension
# Version 2 (need to be used for 4D stack if no px adjustments have been made)

# Directory containing TIFF files
directory = '/Users/nadine/Desktop/dff_test_artifact/'

# Retrieve all TIFF files in the directory and sort them numerically
tiff_files = [file for file in os.listdir(directory) if file.endswith('.tiff')]
tiff_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Assuming filenames are in the format t_XXXX.tiff

# Load all TIFF files as 3D stacks and stack them along a new dimension to form a 4D array
stacked_images = []
for file in tiff_files:
    file_path = os.path.join(directory, file)
    with tf.TiffFile(file_path) as tif:
        image = tif.asarray()
        stacked_images.append(image)

# Convert the list of 3D stacks into a numpy array
tiff_stack_4d = np.array(stacked_images)

# Check the dimensions of the 4D stack
print("Dimensions (shape) of 4D TIFF stack:", tiff_stack_4d.shape)
print("Data type:", tiff_stack_4d.dtype)

#tiff_stack.shape[0] = z, tiff_stack.shape[1] = y, and tiff_stack.shape[2]) = x
'''
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
'''
# Iterate through each tiff stack, extract average pixel value and save as csv
# Works for version 1 of tiff stacks

folder_path = '/Users/nadine/Documents/Zlatic_lab/manual_registration_1099/dff_WB/test_made-up_stack'
# Get a list of TIFF files in the folder, and tiff files in numerical order
tiff_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif') or f.endswith('.tiff')], key=lambda x: int(x.split('_')[1].split('.')[0]))

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
        print("Sphere Mask Shape:", sphere_mask.shape)
        print("Number of True Values in Mask:", np.sum(sphere_mask))

        # Apply mask to image
        pixels_in_sphere = image[sphere_mask]
        print("Pixels in Sphere:", pixels_in_sphere)

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

        # Create a dictionary to map coordinates to names
        coordinate_to_name = {}

        for index, row in coordinates_data.iterrows():
            name = row["skid-left"]
            x, y, z = row['LM_x'], row['LM_y'], row['LM_z']
            coordinate_to_name[f"Coordinate_{x}_{y}_{z}"] = name

# Convert dictionary to DataFrame and save as CSV
df = pd.DataFrame(average_values)
df.rename(columns=coordinate_to_name, inplace=True)  # Rename columns using the dictionary
#df.to_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/average_pixel_values_TP_0-1000.csv', index=False)
df.to_csv('/Users/nadine/Desktop/dff_test_artifact/average_pixel_values.csv', index=False)
'''
# %%





# %%
# ANOTHER TEST

import tifffile as tf
import numpy as np
import os
import pandas as pd

# Load tiff stack numerical sorted
# Directory containing TIFF files
folder_path = '/Users/nadine/Desktop/dff_test_artifact/'

# Retrieve all TIFF files in the directory and sort them numerically
tiff_files = [file for file in os.listdir(folder_path) if file.endswith('.tiff')]
tiff_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Assuming filenames are in the format t_XXXX.tiff

# List to store 3D stacks
stacked_images = []

# Iterate through each TIFF file
for filename in tiff_files:
    file_path = os.path.join(folder_path, filename)
    # Open TIFF file using tifffile.TiffFile
    with tf.TiffFile(file_path) as tif:
        # Read all pages of the TIFF file and stack them along a new axis
        tiff_stack = np.stack([page.asarray() for page in tif.pages], axis=-1)
        stacked_images.append(tiff_stack)

# Convert the list of 3D stacks into a numpy array
tiff_stack_4d = np.array(stacked_images)

# Check the dimensions of the 4D stack
print("Dimensions (shape) of 4D TIFF stack:", tiff_stack_4d.shape)
print("Data type:", tiff_stack_4d.dtype)


# Dictionary to store average pixel values for each coordinate
average_values = {}

# Iterate through each 3D stack in the 4D TIFF stack
for image in tiff_stack_4d:

    # Iterate through each set of coordinates
    for index, row in coordinates_data.iterrows():
        x, y, z = row['LM_x'], row['LM_y'], row['LM_z']
        radius_x, radius_y, radius_z = 2, 2, 1  # Adjust your radius values here
        
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
        print("Sphere Mask Shape:", sphere_mask.shape)
        print("Number of True Values in Mask:", np.sum(sphere_mask))
      
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

        # Create a dictionary to map coordinates to names
        coordinate_to_name = {}

        for index, row in coordinates_data.iterrows():
            name = row["skid-left"]
            x, y, z = row['LM_x'], row['LM_y'], row['LM_z']
            coordinate_to_name[f"Coordinate_{x}_{y}_{z}"] = name

# Convert dictionary to DataFrame and save as CSV
df = pd.DataFrame(average_values)
df.rename(columns=coordinate_to_name, inplace=True)  # Rename columns using the dictionary
#df.to_csv('/Users/nadine/Documents/paper/single-larva/generated-data/Fluorescence-traces/average_pixel_values_TP_0-1000.csv', index=False)
#df.to_csv('/Users/nadine/Desktop/dff_test_artifact/average_pixel_values.csv', index=False)

# %%

# %%
