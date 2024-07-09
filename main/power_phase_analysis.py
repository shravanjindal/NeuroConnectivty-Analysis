# import libraries
import mne
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mne.time_frequency import tfr_morlet, tfr_array_morlet
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve, PersistenceEntropy, Amplitude

import argparse

# Define parameters
epoch_length = 5  # in seconds
overlap = 4  # in seconds (1 ms)
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
    epochList = []
    for path in rawPath:
        raw = mne.io.read_raw_fif(path, preload=True)
        epochs = mne.Epochs(raw, events, event_id=1, tmin=0,tmax=epoch_length, baseline=None, preload=True)
        epochList.append(epochs)
        # get complex signals
    times = epochList[0][0].get_data().shape[2]
    num_epochs = len(epochList[0])
    # Precompute the TFR parameters to avoid recalculating them in each iteration
    data = epochList[0].get_data()
    # Precompute the TFR parameters to avoid recalculating them in each iteration
    tfr_params = {
                'sfreq': 256,
                'freqs': frequencies,
                'n_cycles': n_cycles,
                'use_fft': True,
                'output': 'complex',
                'n_jobs': 20
    }
        # Compute complex signals for all time points at once using TFR
    complex_signal = mne.time_frequency.tfr_array_morlet(
        # Add a new axis to match expected input shape (1, n_channels, n_times)
        data,
        **tfr_params
    )
    # calculate plv
    powers = np.zeros((num_epochs, len(frequencies), 64, 64), dtype=np.float64)
    phases = np.zeros((num_epochs, len(frequencies), 64, 64), dtype=np.float64)
    for epoch_idx in range(num_epochs):
        for freq_idx in range(len(frequencies)):
            phaseEpoch = np.angle(complex_signal[epoch_idx, :, freq_idx, :])
            powerEpoch = np.abs(complex_signal[epoch_idx, :, freq_idx, :])
            phaseCorr = 1 - np.corrcoef(phaseEpoch)
            powerCorr = 1 - np.corrcoef(powerEpoch)
            phases[epoch_idx, freq_idx] = phaseCorr
            powers[epoch_idx, freq_idx] = powerCorr
    ############################################################################################
        # Topological data analysis
        # Power
    persistentEntropyPower = np.zeros((2, num_epochs, len(frequencies)), dtype=np.float64)
    amplitudePower = np.zeros((2, num_epochs, len(frequencies)), dtype=np.float64)
    bettiAreaPower = np.zeros((2, num_epochs, len(frequencies)), dtype=np.float64)
        # Phase
    persistentEntropyPhase = np.zeros((2, num_epochs, len(frequencies)), dtype=np.float64)
    amplitudePhase = np.zeros((2, num_epochs, len(frequencies)), dtype=np.float64)
    bettiAreaPhase = np.zeros((2, num_epochs, len(frequencies)), dtype=np.float64)

    for epoch_idx in range(num_epochs):
        for freq_idx, freq in enumerate(frequencies):
                    # Initialize Vietoris-Rips complex
            vr = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
                    # Ensure powers is correctly structured
            data1 = powers[epoch_idx][freq_idx]
            data2 = phases[epoch_idx][freq_idx]
                    # Fit and transform the data
            diagramsPower = vr.fit_transform([data1])
            diagramsPhase = vr.fit_transform([data2])
                    # Initialize BettiCurve, PersistenceEntropy, and Amplitude
            betti = BettiCurve()
            pEntropy = PersistenceEntropy()
            ampObject = Amplitude()
                    ################################################################################### Power
                    # Compute the curves and entropies
            betti_curves1 = betti.fit_transform(diagramsPower)
            pee1 = pEntropy.fit_transform(diagramsPower)
            aa1 = ampObject.fit_transform(diagramsPower)
                    # Calculate the area under Betti curves
            area = []
            radii = np.linspace(0, 1, betti_curves1.shape[2])
            for i in range(2):
                area.append(np.trapz(betti_curves1[0][i], radii))
                    # Update the power arrays
            persistentEntropyPower[0,epoch_idx, freq_idx] = pee1[0][0]
            persistentEntropyPower[1,epoch_idx, freq_idx] = pee1[0][1]
            amplitudePower[0, epoch_idx, freq_idx] = aa1[0][0]
            amplitudePower[1, epoch_idx, freq_idx] = aa1[0][1]
            bettiAreaPower[0, epoch_idx, freq_idx] = area[0]
            bettiAreaPower[1, epoch_idx, freq_idx] = area[1]
                    ################################################################################### Phase
                    # Compute the curves and entropies
            betti_curves = betti.fit_transform(diagramsPhase)
            pee2 = pEntropy.fit_transform(diagramsPhase)
            aa2 = ampObject.fit_transform(diagramsPhase)
                    # Calculate the area under Betti curves
            area = []
            radii = np.linspace(0, 1, betti_curves.shape[2])
            for i in range(2):
                area.append(np.trapz(betti_curves[0][i], radii))
                    # Update the power arrays
            persistentEntropyPhase[0,epoch_idx, freq_idx] = pee2[0][0]
            persistentEntropyPhase[1,epoch_idx, freq_idx] = pee2[0][1]
            amplitudePhase[0, epoch_idx, freq_idx] = aa2[0][0]
            amplitudePhase[1, epoch_idx, freq_idx] = aa2[0][1]
            bettiAreaPhase[0, epoch_idx, freq_idx] = area[0]
            bettiAreaPhase[1, epoch_idx, freq_idx] = area[1]

    df1 = pd.DataFrame(persistentEntropyPhase[0])
    df2 = pd.DataFrame(persistentEntropyPhase[1])
    df3 = pd.DataFrame(amplitudePhase[0])
    df4 = pd.DataFrame(amplitudePhase[1])
    df5 = pd.DataFrame(bettiAreaPhase[0])
    df6 = pd.DataFrame(bettiAreaPhase[1])
    df7 = pd.DataFrame(persistentEntropyPower[0])
    df8 = pd.DataFrame(persistentEntropyPower[1])
    df9 = pd.DataFrame(amplitudePower[0])
    df10 = pd.DataFrame(amplitudePower[1])
    df11 = pd.DataFrame(bettiAreaPower[0])
    df12 = pd.DataFrame(bettiAreaPower[1])
    subject = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12], axis=0)
    subject.to_csv(f"./data/48plots/pearsonCorr/features/{types}_group{groupName[group_idx]}-{file_idx}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers and a string.")
    
    parser.add_argument("string_arg", type=str, help="A string argument")
    parser.add_argument("int_arg1", type=int, help="First integer argument")
    parser.add_argument("int_arg2", type=int, help="Second integer argument")
    
    args = parser.parse_args()
    
    compute_tda(args.string_arg, args.int_arg1, args.int_arg2)