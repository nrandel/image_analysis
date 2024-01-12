# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
# Read the CSV file into a DataFrame
event_duration_FW = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/FW-event_durtion.csv')
event_duration_T = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/TL-TR-event_durtion.csv') 

#Combine dataframes
event_duration_all= pd.concat([event_duration_FW, event_duration_T], axis=1)

# Adding headers
event_duration_all.columns = ['Forward', 'Turn']

# %%
# Create a plot with individual data points

# Define a color palette with different colors
custom_palette = ["cyan", "magenta"]

# Use stripplot with the specified palette
sns.boxplot(data=event_duration_all, width=0.1, color = 'white')
sns.stripplot(data=event_duration_all, palette=custom_palette, jitter=True, alpha=0.5)


# Set labels and title
plt.xlabel('Events')
plt.ylabel('No of Frames')
plt.title('Box and Whisker Plot with Individual Data Points')

# Show the plot
plt.show()

# %%
# Calculate the average for each column
averages = event_duration_all.mean()



# %%
