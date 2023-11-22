# %%
# Import libraries
import numpy as np
import pandas as pd

# %%
# Load csvfile

behavioural_data = pd.read_csv('/Users/nadine/Documents/paper/single-larva/BW_included_event_annotations.csv')

behaviour = ["F", "H", "TL", "TR", "B", "Q", "P", "U"]
columns = ["start", "end", "beh", "subject-id", "beh_before", "beh_before_start", "beh_before_end", "beh_after", "beh_after_start", "beh_after_end", "starts_within_event", "stops_within_event"]

# %%
