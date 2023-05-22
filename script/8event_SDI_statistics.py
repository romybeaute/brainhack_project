#%%
import os
import nilearn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, ttest_1samp
from tqdm import tqdm

from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
from nilearn import plotting
import mne

total_no_of_events = "30_events"

HOMEDIR = os.path.abspath(os.getcwd())
condition = "differenced"

empirical_SDI = np.load(
    f"{HOMEDIR}/Generated_data/Graph_SDI_related/empirical_SDI_{condition}.npz"
)
surrogate_SDI = np.load(
    f"{HOMEDIR}/Generated_data/Graph_SDI_related/surrogate_SDI_{condition}.npz"
)


def stats(band, event_group):
    clusters = np.load(
        f"/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events_{total_no_of_events}.npz"
    )
    a = list()

    a.append(clusters["0"])
    a.append(clusters["1"])
    a.append(clusters["2"])

    index = list()
    for i in sorted(np.hstack(a)):

        if i in a[0]:
            index.append(0)
        if i in a[1]:
            index.append(1)
        if i in a[2]:
            index.append(2)

    empirical_one_band = empirical_SDI[f"{band}"][
        np.where(np.array(index) == event_group)[0]
    ]
    surrogate_one_band = surrogate_SDI[f"{band}"][
        np.where(np.array(index) == event_group)[0]
    ]

    n_subjects = 25
    n_events = len(np.where(np.array(index) == event_group)[0])
    n_roi = 360
    n_surrogate = 50

    test_stats = list()

    for subject in tqdm(range(n_subjects)):
        event_level_p = list()

        for event in range(n_events):
            roi_level_p = list()

            for roi in range(n_roi):

                data_empirical = empirical_one_band[event, subject, roi]
                data_surrogate = surrogate_one_band[event, :, subject, roi]

                stat_test = sum(
                    rankdata(np.abs(data_empirical - data_surrogate))
                    * np.sign(data_empirical - data_surrogate)
                )
                stat_test_normalized = stat_test / n_surrogate

                roi_level_p.append(stat_test_normalized)

            event_level_p.append(roi_level_p)

        test_stats.append(event_level_p)

    # Step 2 : Test for effect of events

    pvalues_step2 = list()
    tvalues_step2 = list()

    for sub in range(n_subjects):
        sub_wise_p = list()
        sub_wise_t = list()

        for roi in range(n_roi):
            data = np.array(test_stats)[sub, :, roi]

            t, p = ttest_1samp(data, popmean=0)
            if t == np.inf:
                t = 0
            if t == -np.inf:
                t = 0

            if p == np.inf:
                p = 1
            if p == -np.inf:
                p = 1

            sub_wise_p.append(p)
            sub_wise_t.append(t)

        pvalues_step2.append(sub_wise_p)
        tvalues_step2.append(sub_wise_t)

    # Step 3 : Second level Model
    secondlevel_t, secondlevel_p, _ = mne.stats.permutation_t_test(
        np.array(np.array(tvalues_step2)), n_jobs=-1, n_permutations=50000
    )

    path_Glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"
    mnitemp = fetch_icbm152_2009()

    thresholded_tvals = (secondlevel_p < 0.01) * np.array(
        secondlevel_t
    )  # (secondlevel_p < 0.05) *

    signal = np.expand_dims(thresholded_tvals, axis=(1, 2))

    U0_brain = signals_to_img_labels(signal, path_Glasser, mnitemp["mask"])

    plotting.plot_img_on_surf(
        U0_brain,
        title=f"2nd level; {band}; PO - BL; tvalues; perm-corrected; event_G : {event_group+1}",
        threshold=0.1,
        vmax=10,
    )

    U0_brain.to_filename(
        f"/users2/local/Venkatesh/Multimodal_ISC_Graph_study/Results/SDI/event_wise_spatial_map/{band}_event_{event_group+1}.nii.gz"
    )
    plt.show()


for i in range(3):
    # stats('alpha', i)
    # stats('theta', i)
    # stats('low_beta',i)
    # stats('high_beta', i)
    stats("wideband", i)
# %%
