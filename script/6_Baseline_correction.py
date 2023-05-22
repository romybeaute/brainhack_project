#%%
import numpy as np
import mne
import os

second_in_sample = 88
fs = 125

HOMEDIR = os.path.abspath(os.getcwd())

wideband_and_other_bands = np.load(
    f"{HOMEDIR}/Generated_data/Cortical_surface_related/wideband_and_other_bands.npz"
)

wideband_and_other_bands_zscored = dict()
for label, signal in wideband_and_other_bands.items():

    wideband_and_other_bands_zscored[f"{label}"] = mne.baseline.rescale(
        signal,
        times=np.array(list(range(second_in_sample))) / fs,
        baseline=(None, 0.2),
        mode="zscore",
        verbose=False,
    )


np.savez_compressed(
    f"{HOMEDIR}/Generated_data/Cortical_surface_related/wideband_and_other_bands_zscored",
    **wideband_and_other_bands_zscored,
)

# %%
