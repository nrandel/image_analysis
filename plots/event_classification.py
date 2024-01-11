# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
# Read the CSV file into a DataFrame
event_duration_FW = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/FW-event_durtion.csv')

#path needs to be updated TODO
event_duration_T = pd.read_csv('/Users/nadine/Documents/paper/single-larva/generated-data/FW-event_durtion.csv') 

#Combine dataframes
event_duration_all= pd.concat([event_duration_FW, event_duration_T], axis=1)

# Adding headers
event_duration_all.columns = ['Forward', 'Turn']

# %%
# Create a box and whisker plot with individual data points
sns.boxplot(data=event_duration_all, width=0.5)
sns.stripplot(data=event_duration_all, color='black', jitter=True, alpha=0.5)

# Set labels and title
plt.xlabel('Conditions')
plt.ylabel('Values')
plt.title('Box and Whisker Plot with Individual Data Points')

# Show the plot
plt.show()

# %%
