#%%

"""
This script loads electrophysiological signals from a .npz file, standardizes the signals using z-score normalization with a certain baseline period, and then saves the standardized signals to a new .npz file.
"""
import numpy as np
import mne
import os

second_in_sample = 88
fs = 125

HOMEDIR = os.path.expanduser("~/brainhack/brainhack_project")
print('HOMEDIR: ',HOMEDIR)

DATA_DIR = f"{HOMEDIR}/src_data/src_data_osf"

def get_npz_files(directory):
    return [file for file in os.listdir(directory) if file.endswith('.npz')]

files = get_npz_files(DATA_DIR)

wideband_and_other_bands_zscored = dict()

for filename in files:
    filepath = os.path.join(DATA_DIR, filename)
    wideband_and_other_bands = np.load(filepath)

    for label, signal in wideband_and_other_bands.items():
        wideband_and_other_bands_zscored[f"{label}"] = mne.baseline.rescale(
            signal,
            times=np.array(list(range(second_in_sample))) / fs,
            baseline=(None, 0.2),
            mode="zscore",
            verbose=False,
        )



np.savez_compressed(
    f"{HOMEDIR}/generated_data/Cortical_surface_related/wideband_and_other_bands_zscored",
    **wideband_and_other_bands_zscored,
)

# %%
