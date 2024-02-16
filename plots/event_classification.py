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

plt.figure(figsize=(5, 7))

# # Divide all values in the DataFrame by 3.44 (to transform frame to time Hz = 0.344)
event_duration_all = event_duration_all.mul(0.344)

# Define a color palette with different colors
custom_palette = ["cyan", "magenta"]

# Use stripplot with the specified palette
sns.boxplot(data=event_duration_all, width=0.2, color = 'white')
sns.stripplot(data=event_duration_all, palette=custom_palette, jitter=True, alpha=0.5)


# Set labels and title
plt.xlabel('Events')
plt.ylabel('Time in sec')
plt.title('Box and Whisker Plot with Individual Data Points')

# Save the plot as an image file
plt.savefig('/Users/nadine/Documents/paper/single-larva/generated-graphs/figures/drafts/event_duration.svg')

# Show the plot
plt.show()

# %%
# Calculate the average for each column
averages = event_duration_all.mean()



# %%
# Barplot

import matplotlib.pyplot as plt

# Data
categories = ['forward', 'turn']
values = [12,  42]
colors = ['cyan', 'magenta']
width = 0.5  # Width of the bars

# Create bar plot
plt.bar(categories, values, color=colors, width=width)

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot with Custom Colors and Width')

# Save the plot as an image file
plt.savefig('/Users/nadine/Documents/paper/single-larva/generated-graphs/figures/drafts/event_number.svg')


# Show plot
plt.show()