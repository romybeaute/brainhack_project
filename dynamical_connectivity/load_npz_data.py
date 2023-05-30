import numpy as np
import os

########################################################################################################################################
#LOAD THE DATA & DEFINE A DATASET DICTIONARY
########################################################################################################################################


HOMEDIR = os.path.expanduser("~/brainhack/brainhack_project")
print('HOMEDIR: ',HOMEDIR)


"""
The max file size accepted is 5GB on OSF and the envelope_signal was 7GB. So had to cut down into 2 files:
- alpha, theta, low_beta are in one 
- high_beta and wideband are in another

Source-localized, bandpassed, Hilbert-transformed (video watching data). NB : do not contains RS
Each key is of shape : (n_subjects=25,n_ROIs=360,n_times=21250)
"""

npz_osf_filepath = f"{HOMEDIR}/src_data/src_data_osf" #path to NPZ (Numpy Zip) file format
envelope_signal_files = [item for item in os.listdir(npz_osf_filepath)] #get the 2 files of envelope_signal
print(envelope_signal_files)

video_watching_dataset = {}  # Dictionary to store the data from both .npz files
video_watching_files = {} # Dictionary to store the data from both .npz files

for file in envelope_signal_files:
    npz_filepath = os.path.join(npz_osf_filepath, file)
    data = np.load(npz_filepath)

    file_data = {}  # Dictionary to store data for each file
    for key in data.keys():
        print(key)
        file_data[key] = {
            'data': data[key],
            'shape': data[key].shape
        }
        variable_name = f"{key}_VW_data" # Create a variable name using the key
        video_watching_dataset[variable_name] = file_data[key]['data'] # Store data in the dictionary using the variable name as the key
        # globals()[variable_name] = video_watching_dataset[file][key]['data'] # use the globals() function to access the global namespace and create a new variable with the generated variable name.

    video_watching_files[file] = file_data

