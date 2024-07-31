#%%
import numpy as np

def inspect_npz(npz_path):
    # Load the NPZ file
    data = np.load(npz_path)
    
    # Print the keys and some information about each key's associated data
    print(f"Keys in the NPZ file: {data.files}")
    for key in data.files:
        print(f"\nKey: {key}")
        print(f"Shape: {data[key].shape}")
        print(f"Data Type: {data[key].dtype}")
        print(f"First few elements: {data[key].flat[:10]}")  # Print first few elements if array is large

# Path to the NPZ file
npz_path = '/Users/nadine/Documents/Zlatic_lab/1099_spatial-filtered/neuron_traces_cleaned.npz'

# Inspect the NPZ file
inspect_npz(npz_path)

# %%

#%%
import os
import pandas as pd
import numpy as np
import skimage.io
from tqdm import tqdm
# from skimage.measure import regionprops # to get cell centroids
import matplotlib.pylab as plt

#%%
# Load data
base_path = "/Users/nadine/Documents/Zlatic_lab/1099_spatial-filtered"
acardona_fluorescence = pd.read_csv(f"{base_path}/measurements_klb.csv")
behaviour_annotations = pd.read_csv(f"{base_path}/18-02-15L1-behavior-ol.csv")

# Get metadata
timepoints = np.arange(len(acardona_fluorescence))
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
adjust_start = 9

# Get behaviour times
eventTimes = {e: [] for e in eventNames}
for b in behaviourData:
    start, end, forward, backward, stim, hunch, other, turn, left_turn, right_turn, HP = [int(c) if len(c)>0 else 0 for c in b[0].split(";")]
    cur_events = [forward, backward, stim, hunch, other, left_turn, right_turn, HP]
    assert np.sum(cur_events) == 1, "Multiple events in one timepoint"
    event_name = eventNames[cur_events.index(1)]
    eventTimes[event_name].append([start, end])

# Remove events after N timepoints
maxiumumTimepoint = 1300
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
after_event_Window = 15 # frames after event stop that are considered

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

# Generate plots of top N responding neurons
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
    plt.savefig(f"./figures/{event}.png")
    # plt.show()
# %%
