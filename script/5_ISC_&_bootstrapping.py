"""This code is time & memory expensive and CPU taxing; 
depends on the CPU, but can expect 16+ hours approx. on a 16 core CPU for n_perms = 1000; 
memory footprint can touch 70GB, no bug observed, all intact and no improvement on the cards"""

import util_5_CorrCA
from timeit import default_timer
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
import os

HOMEDIR = os.path.abspath(os.getcwd())

video_watching_bundle_STC = np.load(
    f"{HOMEDIR}/Generated_data/Cortical_surface_related/video_watching_bundle_STC_parcellated.npz"
)["video_watching_bundle_STC_parcellated"]

sta = default_timer()

dict_of_ISC = dict()
dict_of_ISC["condition1"] = video_watching_bundle_STC

n_subj = 25
n_perms = 1000
fs = 125
regions = 360

NB_CPU = multiprocessing.cpu_count()

W, _ = util_5_CorrCA.train_cca(dict_of_ISC)


def process(i):
    np.random.seed(i)

    for subjects in range(n_subj):
        np.random.seed(subjects)
        rng = np.random.default_rng()

        rng.shuffle(
            np.swapaxes(
                dict_of_ISC["condition1"][subjects, :, :].reshape(
                    regions, 34, fs * 5
                ),  # Scrambling in 5s blocks
                0,
                1,
            )
        )

    return util_5_CorrCA.apply_cca(dict_of_ISC["condition1"], W, fs)[1]


isc_noise_floored = Parallel(n_jobs=NB_CPU - 1, max_nbytes=None)(
    delayed(process)(i) for i in tqdm(range(n_perms))
)
stop = default_timer()

print(f"Whole Elapsed time: {round(stop - sta)} seconds.")
np.savez_compressed(
    "Generated_data/Cortical_surface_related/noise_floor_8s_window",
    isc_noise_floored=isc_noise_floored,
)
