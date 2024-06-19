#%%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Function to plot columns in chunks against index with specified x-axis range
def plot_columns_in_chunks(df, chunk_size=5, x_start=None, x_end=None):
    num_cols = df.shape[1]
    column_chunks = [range(i, min(i + chunk_size, num_cols)) for i in range(0, num_cols, chunk_size)]

    for chunk_indices in column_chunks:
        plt.figure(figsize=(12, 8))
        
        plotted_anything = False  # Flag to check if anything was actually plotted
        
        for col_idx in chunk_indices:
            col_name = df.columns[col_idx]
            plt.plot(df.index, df[col_name], label=col_name)
            plotted_anything = True  # Set flag to True if at least one plot was made
        
        if plotted_anything:
            plt.title(f'Columns {chunk_indices[0]} to {chunk_indices[-1]} vs Index')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            
            if x_start is not None and x_end is not None:
                plt.xlim(x_start, x_end)  # Set x-axis limits if provided
            
            plt.show()
        else:
            print(f"No data plotted for columns {chunk_indices[0]} to {chunk_indices[-1]}")

# Load:
file_path = '/Users/nadine/Documents/Zlatic_lab/Nicolo_LSM-single-cell-data/20240531_Nadine_Randel_fluorescence_measurements/WillBishop/output/dff_long/intersection_neuronal_activity_responsive_neurons_sd_2_average_102-110-AND-682-699.csv'
df = pd.read_csv(file_path, index_col = False)  # Assuming the first column is the index

# Specify x-axis range (optional)
x_start = 80  # Replace with your desired start index
x_end = 740   # Replace with your desired end index

# Print some debug information
print(f"DataFrame head:\n{df.head()}\n")
print(f"Index values:\n{df.index}\n")
print(f"Columns in DataFrame:\n{df.columns}\n")

# Plot every 20 columns against the index with specified x-axis range
plot_columns_in_chunks(df, chunk_size=5, x_start=x_start, x_end=x_end)


# %%
