"""This code is time & memory expensive and CPU taxing; 
depends on the CPU, but can expect 16+ hours approx. on a 16 core CPU for n_perms = 1000; 
memory footprint can touch 70GB, no bug observed, all intact and no improvement on the cards"""
import sys
sys.path.insert(0, '/Users/rb666/brainhack/brainhack_project/script/')

from util_5_CorrCA import *
from timeit import default_timer
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
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

    video_watching_dataset[file] = file_data



########################################################################################################################################
#CHECK THE CONTENT OF THE FILES
########################################################################################################################################



# # Accessing the content of the dataset
# for file, data in video_watching_dataset.items():
#     print(f"************************************\n Content of {file}:")
#     for key, value in data.items():
#         print(f"Key: {key}, Shape: {value['shape']}")

# # Accessing individual data
# file1_VW = video_watching_dataset[envelope_signal_files[0]]
# file2_VW = video_watching_dataset[envelope_signal_files[1]]

# print('Content file 1:', list(file1_VW.keys()))
# print('Content file 2:', list(file2_VW.keys()))




def check_npz_file(npz_filepath):
    '''
    If problem with the loading of the osf dataset, can check the data with this function 
    '''

    # 1. File Existence
    if not os.path.exists(npz_filepath):
        return "Error: File does not exist."

    # 2. File Format - basic check
    if not npz_filepath.endswith('.npz'):
        return "Error: File is not of type '.npz'."

    # 3. File Corruption & 4. Software Versions
    try:
        np.load(npz_filepath)
    except Exception as e:
        return f"Error: File could not be loaded. Potential file corruption or software version issue.\nException: {str(e)}"

    # 5. Pickle Allowance
    try:
        np.load(npz_filepath, allow_pickle=True)
    except Exception as e:
        return f"Error: File could not be loaded even with 'allow_pickle=True'.\nException: {str(e)}"
    
    return "No problem found with this file."





# video_watching_bundle_STC = np.load(
#     f"{HOMEDIR}/src_data/envelope_signal_bandpassed.npz")
# )["video_watching_bundle_STC_parcellated"]

sta = default_timer() # initialize the variable sta with the current time (using the default_timer() function)
NB_CPU = multiprocessing.cpu_count() # initialize the variable NB_CPU with the number of available CPU cores, respectively.

dict_of_ISC = dict()
dict_of_ISC["alpha"] = video_watching_dataset['alpha_VW_data']
dict_of_ISC["low_beta"] = video_watching_dataset['low_beta_VW_data']
dict_of_ISC["high_beta"] = video_watching_dataset['high_beta_VW_data']
dict_of_ISC["theta"] = video_watching_dataset['theta_VW_data']
dict_of_ISC["wideband"] = video_watching_dataset['wideband_VW_data']

n_subj = 25
n_perms = 1000
fs = 125
regions = 360



# W, _ = util_5_CorrCA.train_cca(dict_of_ISC)

W, _ = train_cca(dict_of_ISC) #computes the spatial filters (W matrix) - using the tran_cca function defined in util_5_CorrCA.py. When applied to the data, will maximize the correlation across conditions (or experiments)



def process(i):
    '''
    This function is defined to perform computations in parallel for each iteration
    '''
    np.random.seed(i)

    for condition in [key for key in video_watching_dataset.keys() if key.endswith('data')]:


        for subjects in range(n_subj):
            np.random.seed(subjects)
            rng = np.random.default_rng()

            rng.shuffle(
                np.swapaxes(
                    dict_of_ISC[condition][subjects, :, :].reshape(
                        regions, 34, fs * 5
                    ),  # Scrambling in 5s blocks
                    0,
                    1,
                )
            )

        # return util_5_CorrCA.apply_cca(dict_of_ISC[condition], W, fs)[1]
        return apply_cca(dict_of_ISC[condition], W, fs)[1]


isc_noise_floored = Parallel(n_jobs=NB_CPU - 1, max_nbytes=None)(
    delayed(process)(i) for i in tqdm(range(n_perms)) # executes the process function in parallel for n_perms 
)
stop = default_timer()

print(f"Whole Elapsed time: {round(stop - sta)} seconds.")
np.savez_compressed(
    f"{HOMEDIR}/generated_data/Cortical_surface_related/noise_floor_8s_window",
    isc_noise_floored=isc_noise_floored,
)
