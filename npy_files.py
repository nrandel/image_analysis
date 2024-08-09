#%%
import os
import pandas as pd
import numpy as np
import skimage.io
from tqdm import tqdm
# from skimage.measure import regionprops # to get cell centroids
import matplotlib.pylab as plt

# Key: neuron_traces
#Shape: (6245, 10367) 
#(Neurons, TP)

# allEventEvokedActivity[event] = evokedData
# Shape: e.g., (30, 6245, 42)
#(First Dimension): Number of instances of the event.
#(Second Dimension): Number of neurons.
#(Third Dimension): Number of time points sampled around the event (pre-event + event duration + post-event).

# cell_centroids.txt


#%%
# Load data
base_path = "/Users/nadine/Documents/Zlatic_lab/1099_spatial-filtered"
acardona_fluorescence = pd.read_csv(f"{base_path}/measurements_klb.csv")
behaviour_annotations = pd.read_csv(f"{base_path}/18-02-15L1-behavior-ol.csv")

# Get metadata
timepoints = np.arange(len(acardona_fluorescence)) # this is a weird way to approach this
neuron_names = acardona_fluorescence.columns[1:]
neuron_locations = np.array([[float(s) for s in l.split('"')[1].split("::")] for l in neuron_names])

# Get fluorescence data
acardona_fluorescence_data = acardona_fluorescence.iloc[:,1:].to_numpy().T
n_neurons, n_timepoints = acardona_fluorescence_data.shape

# Load new neuron traces data
#neuron_traces_raw = np.load(f"{base_path}/neuron_traces_raw.npz")["neuron_traces"]
neuron_traces_cleaned = np.load(f"{base_path}/neuron_traces_cleaned.npz")["neuron_traces"]

# # New neuron locations
# neuron_segmentation = skimage.io.imread(f"{base_path}/lsm_cell_segmentation.tif")
# cells = regionprops(neuron_segmentation)
# cell_centroids = np.array([c.centroid for c in cells])
# np.savetxt(f"{base_path}/neuron_centroids.txt", cell_centroids, delimiter=",")
cell_centroids = np.loadtxt(f"{base_path}/neuron_centroids.txt", delimiter=",")

#%%
# ---------------------------------------------------
# Get times and durations of labelled events
# ---------------------------------------------------

# Parse behaviour annotations
behaviourData = behaviour_annotations.to_numpy()
eventNames = ["forward", "backward", "stim", "hunch", "other", "left_turn", "right_turn", "HP"]
adjust_start = 0 #'9' if nine frames prior event start; '0' no adjustment

# Get behaviour times
eventTimes = {e: [] for e in eventNames}
for b in behaviourData:
    start, end, forward, backward, stim, hunch, other, turn, left_turn, right_turn, HP = [int(c) if len(c)>0 else 0 for c in b[0].split(";")]
    cur_events = [forward, backward, stim, hunch, other, left_turn, right_turn, HP]
    assert np.sum(cur_events) == 1, "Multiple events in one timepoint"
    event_name = eventNames[cur_events.index(1)]
    eventTimes[event_name].append([start, end])

# Remove events after N timepoints
maxiumumTimepoint = 1100
for eventName in eventTimes:
    eventTimes[eventName] = [e for e in eventTimes[eventName] if e[1] < maxiumumTimepoint]

# Subtract the adjustmnet time from the start times
for eventName in eventTimes:
    # Get the current events (for the current event name)
    cur_events = eventTimes[eventName]
    # Iterate over the event times and subtract adjust_start from every start value
    adjusted_events = [] # empty list in which to put the adjusted times
    for e in cur_events:
        start, end = e
        adjusted_start = start - adjust_start
        adjusted_events.append([adjusted_start, end])
    # Set the adjusted times
    eventTimes[eventName] = adjusted_events
    

#%%
# ---------------------------------------------------
# Get time-locked activity for each event type
# ---------------------------------------------------

# Define neurons of interest
neuronTracesOfInterest = neuron_traces_cleaned
n_neurons, n_timepoints = neuronTracesOfInterest.shape

# Set windows
preWindow = 15 # average for F0
after_event_Window = 3 # frames after event STOP that are considered

# Get behaviour evoked data
allEventEvokedActivity = dict() # {} # make a new dictionary to keep all of the evoked data for each event type
for event in eventNames:
    # Get evoked data
    cur_events = eventTimes[event]
    eventDurations = np.array([np.ptp(e) for e in cur_events])
    postWindow = np.max(eventDurations) + after_event_Window
    # Get only those events that start after preWindow, and end before n_timepoint - postWindow
    cur_events = [e for e in cur_events if e[0] > preWindow and e[1] < n_timepoints-postWindow]
    # Calculate F - F0 for each instance of event 
    evokedData = np.zeros([len(cur_events), n_neurons, preWindow+postWindow])
    for i,e in enumerate(tqdm(cur_events)):
        start, end = e
        cur_evoked = neuronTracesOfInterest[:,start-preWindow:start+postWindow].copy()
        baseline_F0 = cur_evoked[:,:preWindow].mean(axis=1)[:,None] # make baseline habe shape of (n_neurons, 1)
        # # Calculate F - F0
        cur_evoked_baseline_subtracted = cur_evoked - baseline_F0
        # # Calculate delta F over F0
        # cur_evoked_deltaFOverF = (cur_evoked - baseline_F0) / baseline_F0
        evokedData[i] = cur_evoked_baseline_subtracted
    # Add the evoked data to the allEventEvokedActivity store
    allEventEvokedActivity[event] = evokedData

    # Save the evoked data for the current event
    # Update adjust_start in file name
    # Update maxiumumTimepoint in directory
    #save_dir = '/Users/nadine/Documents/Zlatic_lab/1099-nuc-seg/TP_0_1100'
    #save_path = os.path.join(save_dir, f"{event}_evoked_data_adjust_start_9.npy")
    #np.save(save_path, evokedData)


#%%
# Generate plots of top N responding neurons averaged over all events per neuron
N_responding = 200
for event in eventNames:
    # Get current evoked data
    evokedData = allEventEvokedActivity[event]
    # Get mean evoked over all event examples
    meanEvoked = evokedData.mean(axis=0)
    # Order neurons by mean evoked response
    neuronOrder = np.abs(meanEvoked).mean(axis=1).argsort()[::-1]
    # Plot top N responding neurons
    plt.close()
    fig,ax = plt.subplots(2,1)
    ax[0].plot(meanEvoked[neuronOrder[:N_responding],:].T)
    ax[1].imshow(meanEvoked[neuronOrder[:N_responding],:], aspect="auto", interpolation="nearest")
    #plt.savefig(f"./figures/{event}.png") 
    #plt.savefig(f"/Users/nadine/Documents/Zlatic_lab/1099_spatial-filtered/figures/{event}.png")
    plt.show()
# %%
#TEST
#%%
# ---------------------------------------------------
# Create a Dictionary of Neuron Centroids 
# ---------------------------------------------------

# Load centroids
cell_centroids = np.loadtxt(f"{base_path}/neuron_centroids.txt", delimiter=",")

# Create a mapping of neuron identifiers to their centroid coordinates
neuron_centroid_mapping = {f'neuron {i+1}': cell_centroids[i] for i in range(len(cell_centroids))}

#%%

#--------------------------------------------------
# Keep Track of Neurons in allEventEvokedActivity
#---------------------------------------------------

allEventEvokedActivity = dict() 

for event in eventNames:
    # Get evoked data (as in the original code)
    ...
    evokedData = np.zeros([len(cur_events), n_neurons, preWindow+postWindow])

    for i,e in enumerate(tqdm(cur_events)):
        ...
        evokedData[i] = cur_evoked_baseline_subtracted

    # Store the evoked data along with neuron identifiers
    neuron_ids = [f'neuron {j+1}' for j in range(n_neurons)]
    allEventEvokedActivity[event] = {
        'evoked_data': evokedData,
        'neuron_ids': neuron_ids
    }


# %%

#--------------------------------------------------
# Access Neuron Data Later
#--------------------------------------------------

#TODO (ok but statistic eeds to be included)

# Example: Access the data for the first event
event_data = allEventEvokedActivity['forward']  # example for the 'forward' event
evoked_data = event_data['evoked_data']
neuron_ids = event_data['neuron_ids']

# Let's say you want to analyze 'neuron 10' and 'neuron 50'
neuron_indices = [neuron_ids.index('neuron 10'), neuron_ids.index('neuron 50')]

# Access the evoked data for these neurons
subset_evoked_data = evoked_data[:, neuron_indices, :]


# %%
