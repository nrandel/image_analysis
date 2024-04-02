# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Read the CSV file into a DataFrame
ad_connectivity_matrix = pd.read_csv('/Users/nadine/Documents/paper/single-larva/suppl-science.add9330_data_s1_to_s4/Supplementary-Data-S1/ad_connectivity_matrix.csv') #single event-type

# Set the index to the values in the first column ('A')
ad_connectivity_matrix.set_index(ad_connectivity_matrix.columns[0], inplace=True)

print(ad_connectivity_matrix.columns)
print(ad_connectivity_matrix.index) #row names


# %%
# Find a specific cell using row and column names
row_name = 29
col_name = '9469519'

cell_value = ad_connectivity_matrix.loc[row_name, col_name]
print(f"The value at row '{row_name}' and column '{col_name}' is: {cell_value}")
# %%
