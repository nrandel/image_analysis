# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%

#temp testing
# %%
def save_neurons_to_csv(neurons, filename, path=''):  # Add a path parameter with default value ''
    filepath = os.path.join(path, filename)  # Construct the full file path
    pd.DataFrame(neurons).to_csv(filepath, header=False)
