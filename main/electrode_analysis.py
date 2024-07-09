# import libraries
import mne
import pathlib
import pandas as pd
import numpy as np
from gtda.time_series import SingleTakensEmbedding
from sklearn.decomposition import PCA
from gtda.homology import VietorisRipsPersistence
import argparse
# Define parameters
epoch_length = 60  # in seconds
overlap = 30  # in seconds (1 ms)
# time between the start of consecutive epochs
event_interval = epoch_length - overlap
# Generate fake events
sfreq = 256  # sampling frequency
event_duration = int(sfreq * epoch_length)  # number of samples in an epoch
# number of samples between consecutive events
event_step = int(sfreq * event_interval)
n_events = int((153344 - event_duration) / event_step) + 1
events = np.array([[int(i * event_step), 0, 1] for i in range(n_events)])
frequencies = np.arange(8, 30, 1)  # Frequencies of interest from 8 to 30 Hz
n_cycles = frequencies / 2.  # Number of cycles in Morlet wavelet
groupName = ['HTR','CTR','VIP','SNY']
groups = ['0','1','2','3']
def compute_tda(types,group_idx,file_idx):
    data_dir = pathlib.Path(f"./data/raw/{types}/")
    rawPath = []
    files = list(data_dir.glob(f"*{groups[group_idx]}_raw.fif"))
    rawPath = sorted(files)
    rawPath = rawPath[file_idx-1:file_idx]
    # create epochs
    raw = mne.io.read_raw_fif(rawPath[0], preload=True)
    epochs = mne.Epochs(raw, events, event_id=1, tmin=0,tmax=epoch_length, baseline=None, preload=True)
    num_epochs = len(epochs)
    data = epochs.get_data()
    # create objects
    embedding_dimension = 30
    embedding_time_delay = 100
    stride = 10
    embedder = SingleTakensEmbedding(
        parameters_type="search", n_jobs=6, time_delay=embedding_time_delay, dimension=embedding_dimension, stride=stride
    )
    pca = PCA(n_components=3)
    vr = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
    # output
    finaldf = pd.DataFrame()
    for epoch_idx in range(num_epochs):
        for electrode_idx in range(64):
            embeddings = embedder.fit_transform(data[epoch_idx,electrode_idx,:]) # taken's theorem
            point_cloud = pca.fit_transform(embeddings) # convert to 3 dim point cloud
            diagram = vr.fit_transform([point_cloud]) # use rips complex to compute diagram
            df = pd.DataFrame(diagram[0]) 
            finaldf = pd.concat([finaldf,df], axis=0, ignore_index=True) 
    finaldf.to_csv(f'./data/48plots/electode_ft/features/{types}_group{groupName[group_idx]}-{file_idx}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers and a string.")
    parser.add_argument("string_arg", type=str, help="A string argument")
    parser.add_argument("int_arg1", type=int, help="First integer argument")
    parser.add_argument("int_arg2", type=int, help="Second integer argument")
    args = parser.parse_args()
    compute_tda(args.string_arg, args.int_arg1, args.int_arg2)