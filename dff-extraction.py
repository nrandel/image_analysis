# %%

# Import libraries
import numpy as np
import tifffile as tf
import pandas as pd
from scipy.spatial.distance import cdist

#%%

# Load the TIFF stack
tiff_stack = tf.imread('/Users/nadine/Documents/Zlatic_lab/manual_registration_1099/dff_WB/test_made-up_stack/t_1001.tiff')

# Check the dimensions (shape) of the TIFF stack
print("Dimensions (shape) of TIFF stack:", tiff_stack.shape) # z,y,x in px

# Other properties
print("Data type:", tiff_stack.dtype)
print("Number of images in stack:", tiff_stack.shape[0])  # Assuming 1st dimension represents the number of images
print("Image size (height x width):", tiff_stack.shape[1], "x", tiff_stack.shape[2])

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

# Iterate through each TIFF file in the folder
for filename in tif_files:
    file_path = os.path.join(folder_path, filename)
    image = tf.imread(file_path)

    # Iterate through each set of coordinates for each image
    for index, row in coordinates_data.iterrows():
        x, y, z = row['LM_x'], row['LM_y'], row['LM_z']
        radius = 2  # Define the radius of the sphere

        # Create a grid of coordinates that cover the sphere
        LM_x_range = np.arange(0, image.shape[1])
        LM_y_range = np.arange(0, image.shape[2])
        LM_z_range = np.arange(0, image.shape[0])

        xx, yy, zz = np.meshgrid(LM_x_range, LM_y_range, LM_z_range)
        sphere_mask = ((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) <= radius ** 2

        # Apply the mask to the current 3D image to extract the pixels within the sphere
        pixels_in_sphere = image[sphere_mask]
        print(pixels_in_sphere)

        # Calculate the average pixel intensity within the sphere
        average_intensity = np.mean(pixels_in_sphere)

        print(f"Average intensity for coordinates ({x}, {y}, {z}) "
              f"in image {filename}: {average_intensity}")


# %%

#TEST (coordinates xyz vs zyx) does not work pixel empty

# Iterate through each TIFF file in the folder
for filename in tif_files:
    file_path = os.path.join(folder_path, filename)
    image = tf.imread(file_path)

    # Iterate through each set of coordinates for each image
    for index, row in coordinates_data.iterrows():
        x, y, z = row['LM_x'], row['LM_y'], row['LM_z']
        radius = 2  # Define the radius of the sphere

        # Create a grid of coordinates that cover the sphere
        x_range = np.arange(0, image.shape[1])
        y_range = np.arange(0, image.shape[2])
        z_range = np.arange(0, image.shape[0])

        xx, yy, zz = np.meshgrid(z_range, y_range, x_range)
        sphere_mask = ((xx - z) ** 2 + (yy - y) ** 2 + (zz - x) ** 2) <= radius ** 2

        # Apply the mask to the current 3D image to extract the pixels within the sphere
        pixels_in_sphere = image[sphere_mask]
        print(pixels_in_sphere)

        # Calculate the average pixel intensity within the sphere
        average_intensity = np.mean(pixels_in_sphere)

        print(f"Average intensity for coordinates ({x}, {y}, {z}) "
              f"in image {filename}: {average_intensity}")


# %%
