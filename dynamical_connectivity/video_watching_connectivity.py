"""
This code will compute the dynamical connectivity from a time series dataset of EEG signals for each of the 25 subjects, for each frequency band that is present in the npz data files. 
This involves adjusting parameters like fmin and fmax dynamically based on cortical signals for specific frequency bands, as well as slicing data in 1s intervals and forcing the function to compute for specific frequencies within given ranges.
This is done by using the function spectral_connectivity_epochs from the MNE-Connectivity module. 
The computed connectivity is then stored back into the original dataset dictionary, under the key 'connectivity'.

The dataset we are working with is structured as 5 separate frequency bands (high_beta, wideband, theta, alpha, and low_beta) with each band data shaped as (25, 360, 21250), corresponding to 25 subjects (defined as epochs), 360 signals per epoch (ROIs), and 21250 time points per signal. 
The frequency sample rate, sfreq, is set at 125 Hz.
This data is loaded and all stored in the dictionary named "video_watching_dataset".

See https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html
Here we use PLI. For further analysis and comparison of different connectivity information, see https://mne.tools/mne-connectivity/stable/auto_examples/dpli_wpli_pli.html#sphx-glr-auto-examples-dpli-wpli-pli-py
"""
from load_npz_data import load_data
from Yeos7network import map_atlases
import numpy as np
import os

import mne
from mne_connectivity import spectral_connectivity_epochs #Use this function if data organized into distinct segments or epochs (such as separate trials, events, or stimuli in an EEG study). This function compute connectivity for each epoch separately. In our case, if the 25 subjects are treated as different epochs, this method makes sense.
from mne_connectivity import spectral_connectivity_time #Use this function instead if dealing with a continuous time series and want to investigate the temporal evolution of connectivity. This method might not be appropriate for our current scenario since the data is divided among different subjects.
from mne_connectivity import SpectralConnectivity
from mne_connectivity.viz import plot_connectivity_circle


########################################################################################################################################
#STEP 0 : DEFINE THE PARAMETERS AND DATA VARIABLES
########################################################################################################################################
n_seconds = 170 #time recording
sfreq = 125 #sample freq
n_ROI = 360 
method = 'wpli' #for computation of connectivity measures
n_times = n_seconds*sfreq  # = 21250 : The number of time points per signal (170sec at 125Hz ==> 21250 time points)
window_size = 125 #1s

########################################################################################################################################
#STEP 1 : LOAD THE DATA INTO VARIABLE : DEFINE A DATASET DICTIONARY
########################################################################################################################################

video_watching_dataset,_ = load_data() #return the video_watching_dataset dictionary
#Check content and shape of video_watching_dataset (sanitary check):
print(video_watching_dataset.keys())



# time-series of brain signals for different frequency bands
signal_alpha =video_watching_dataset['alpha']
signal_theta =video_watching_dataset['theta']
signal_low_beta =video_watching_dataset['low_beta']
signal_high_beta =video_watching_dataset['high_beta']
signal_wideband =video_watching_dataset['wideband']

# Frequency band boundaries for dynamical computation
frequency_band_limits = {'high_beta': (20, 30), 'wideband': (1, 50), 'theta': (4, 7), 'alpha': (8, 12), 'low_beta': (13, 20)}



########################################################################################################################################
#STEP 2 : Create an MNE EpochsArray. For this, we create an MNE Info object that holds information about the data, such as the sampling rate.
########################################################################################################################################


info = mne.create_info(ch_names=360, #corresponds to the 360 ROIs (in the context of MNE-Python and source-localized data, each Region of Interest (ROI) is treated as a separate channel)
                       sfreq=sfreq)  # Create an Info object.






########################################################################################################################################
#STEP 3 : Compute the spectral connectivity for each frequency band.
"""
The structure of this connectivity data ('con') as returned by spectral_connectivity_epochs is a 4D array with the shape (n_connections, n_freqs, n_times, n_epochs), where:
- n_connections is the number of ROI pairs (in the case of pairwise connectivity).
- n_freqs is the number of frequencies (or frequency bins) in the decomposed signal.
- n_times is the number of time points in each epoch.
- n_epochs is the number of epochs (in this case, it corresponds to the number of subjects, as we treat each subject's data as a separate epoch).
Note that each entry in this 4D array represents the computed PLI for a given pair of ROIs, at a given frequency and time point, for a given subject.
"""

########################################################################################################################################





def window_parallel(signal,key_freq,window_start=0):
    '''
    Compute the spectral connectivity for a given window of the signal
    For each window creates :  
    - sce : spectral connectivity object sce 
    - symmetric connectivity matrix that represents the connectivity between each pair of signals for each participant, within the current window.

    '''
    # print(signal.shape) #shape full signal 
    sliding_signal = signal[:, :, window_start:window_start+window_size] #defined so that each window represents 1 second of data
    # print(sliding_signal.shape) #shape sliding signal : corresponding to 1sec (we have sfreq time points per second)

    # Will use this to define fmin and fmax based on the current frequency band
    fmin, fmax = frequency_band_limits[key_freq]

    #computes the spectral connectivity between every pair of signals (ROIs) for each participant
    sce = spectral_connectivity_epochs(sliding_signal, sfreq = sfreq, fmin=fmin, fmax =fmax, method=method, faverage=True, verbose=False) #shape input signal : (25,360,125)
    connectivity_matrix = sce.get_data().reshape(n_ROI, n_ROI)
    connectivity_matrix_symmetric = connectivity_matrix +connectivity_matrix.T

    return sce, connectivity_matrix_symmetric




def compute_all_windows(signal, key_freq, window_size=125):
    '''
    Return two 3D arrays: 
    - all_sce containing the spectral connectivity objects for each window
    - all_connectivity_matrices containing the symmetric connectivity matrices for each window. 
    
    '''
    # Get the number of time points
    n_times = signal.shape[2]

    # Create a list to store the results for each window
    all_sce = []
    all_connectivity_matrices = []

    # Loop over possible starting points for the window
    for window_start in range(0, n_times, window_size):
        # Compute the spectral connectivity and connectivity matrix for this window
        sce, connectivity_matrix = window_parallel(signal, key_freq, window_start)
        
        # Add the results to the lists
        all_sce.append(sce)
        all_connectivity_matrices.append(connectivity_matrix)

    # Return the results as arrays
    return np.array(all_sce), np.array(all_connectivity_matrices)






def compute_all_frequency_bands(frequency_band_limits, window_size=125):
    '''
    Return a dictionary where keys are frequency bands and values are 3D arrays containing 
    the symmetric connectivity matrices for each window within that frequency band.
    
    Also saves the connectivity matrices for each frequency band to a .npy file.
    '''

    # Create a dictionary to store the results for each frequency band
    all_frequency_bands = {}

    # Loop over the possible frequency bands
    for key_freq in frequency_band_limits.keys():
        print(f"Computing connectivity for {key_freq} band")
        signal = video_watching_dataset[key_freq]
        # Compute the spectral connectivity and connectivity matrix for each window within this frequency band
        all_sce, all_connectivity_matrices = compute_all_windows(signal, key_freq, window_size)

        # Add the results to the dictionary
        all_frequency_bands[key_freq] = all_connectivity_matrices

        # Save the connectivity matrices to a .npy file
        np.save(f'generated_connectivity_data/{key_freq}_connectivity_matrices_{method}.npy', all_connectivity_matrices)

    return all_frequency_bands

# Call the function
all_frequency_bands = compute_all_frequency_bands(frequency_band_limits)








########################################################################################################################################
#STEP 4 : Mapping to Yeo's networks : for a given network, computes the average interaction of this network with all other networks for each time window.
########################################################################################################################################
"""
network_labels : an array of size 360 which maps each ROI to one of Yeo's 7 networks
time_series_data : a time series 4D array with shape (number_of_timepoints, number_of_windows, 360, 360) representing functional connectivity between ROIs for each time window
"""

#Get mapping Glasser (360 ROIs) to Yeo (7 networks)
Glasser2Yeo,match, best_overlap = map_atlases("~/brainhack/brainhack_project")



def segregation_computation(full_signal, key_freq, network):
    window_size = 125 #1s

    sce_whole_size = list()
    segregation_list = list()

    for window in (range(len(sce_whole_size))):
    
        network_indices = [np.where(np.array(match)==network)[0]][0]
        
        binmask_within = np.zeros((n_ROI, n_ROI))
        binmask_between_step1 = np.zeros((n_ROI, n_ROI))

        for indices in network_indices:
            

            binmask_within[indices, network_indices]=1
            binmask_within[network_indices, 1]=1

            binmask_between_step1[indices, :] = 1
            binmask_between_step1[:, indices] = 1

            binmask_between = binmask_between_step1-binmask_within
        
        matrix = sce_whole_size[window]
        
        average_strength_within = np.sum(matrix * binmask_within)/ (np.sum(binmask_within) - len(network_indices))
        
        average_strength_between = np.sum(matrix * binmask_between)/np.sum(binmask_between)

        segregation = (average_strength_within - average_strength_between)/average_strength_within
        
        segregation_list.append(segregation)

    return segregation_list


def compute_network_interactions(network_labels, time_series_data, target_network):
    """
    Compute the average interaction of a target network with all other networks.

    Args:
    network_labels (numpy.ndarray): an array mapping each ROI to a network
    time_series_data (numpy.ndarray): a 4D array representing functional connectivity between ROIs for each time window
    target_network (int): the target network for which to compute interactions

    Returns:
    numpy.ndarray: a 2D array where each cell represents the average interaction of the target network with another network for each time window
    """
    
    num_networks = len(np.unique(network_labels))
    num_timepoints, num_windows, _, _ = time_series_data.shape

    # Initialize an array to store the average interaction for each network pair for each time window
    avg_interactions = np.zeros((num_timepoints, num_windows, num_networks))

    for i in range(num_timepoints):
        for j in range(num_windows):
            for k in range(num_networks):
                if k == target_network:
                    continue
                target_indices = np.where(network_labels == target_network)[0]
                other_indices = np.where(network_labels == k)[0]

                target_data = time_series_data[i, j, target_indices, :]
                other_data = time_series_data[i, j, other_indices, :]

                interaction = np.mean(target_data[:, other_indices])
                avg_interactions[i, j, k] = interaction

    return avg_interactions

