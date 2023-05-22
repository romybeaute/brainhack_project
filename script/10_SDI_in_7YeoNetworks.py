#%%

from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
from scipy.stats import ttest_ind
import seaborn as sns
from statsmodels.stats import multitest
import matplotlib.pyplot as plt
import os
from nilearn import datasets, plotting, maskers
import numpy as np

yeo = datasets.fetch_atlas_yeo_2011()
yeomasker = NiftiLabelsMasker(yeo.thick_17, strategy="mean")
yeomasker.fit()

HOMEDIR = os.path.abspath(os.getcwd())

atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
yeo = atlas_yeo_2011.thick_7
glasser = "/users2/local/Venkatesh/Multimodal_ISC_Graph_study/src_data/Glasser_masker.nii.gz"  # f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"

masker = maskers.NiftiMasker(standardize=False, detrend=False)
masker.fit(glasser)
glasser_vec = masker.transform(glasser)

yeo_vec = masker.transform(yeo)
yeo_vec = np.round(yeo_vec)

matches = []
match = []
best_overlap = []
for i, roi in enumerate(np.unique(glasser_vec)):
    overlap = []
    for roi2 in np.unique(yeo_vec):
        overlap.append(
            np.sum(yeo_vec[glasser_vec == roi] == roi2) / np.sum(glasser_vec == roi)
        )
    best_overlap.append(np.max(overlap))
    match.append(np.argmax(overlap))
    matches.append((i + 1, np.argmax(overlap)))

bands = ["alpha", "theta", "low_beta", "high_beta", "wideband"]
conditions = ["baseline", "differenced"]


def plot_nili_bars(axbar, significant, version=1):
    """plots the results of the pairwise inferential model comparisons in the
    form of a set of black horizontal bars connecting significantly different
    models as in the 2014 RSA Toolbox (Nili et al. 2014).
    Args:
        axbar: Matplotlib axes handle to plot in
        significant: Boolean matrix of model comparisons
        version:
            - 1 (Normal Nili bars, indicating significant differences)
            - 2 (Negative Nili bars in gray, indicating nonsignificant
              comparison results)
    Returns:
        ---
    """

    k = 1
    for i in range(significant.shape[0]):
        drawn1 = False
        for j in range(i + 1, significant.shape[0]):
            if version == 1:
                if significant[i, j]:

                    axbar.plot((i, j), (k, k), "k-", linewidth=5, c="grey")
                    k += 1
                    drawn1 = True
        if drawn1:
            k += 1

    axbar.set_ylim((0, k))
    axbar.set_xlim(ax.get_xlim())
    axbar.set_xticks((np.arange(0, significant.shape[0])))
    axbar.set_axis_off()


for band in bands:
    for condition in conditions:
        plt.style.use("fivethirtyeight")
        thresholded_map = np.load(
            f"/users2/local/Venkatesh/Multimodal_ISC_Graph_study/Generated_data/Graph_SDI_related/2nd_level_test_stats/thresholded/thresholded_{condition}_{band}.npz"
        )["secondlevel_t_threholded"]

        network_data = list()
        for network in range(1, 8):
            network_data.append(
                thresholded_map[(np.argwhere(np.array(match) == network))]
            )

        networks = ["Vis", "Som", "DA", "VA", "Lim", "FP", "DMN"]

        pvals = dict()
        network_pairs = list()
        for i in range(7):
            for j in range(i + 1, 7):
                network_pairs.append((networks[i], networks[j]))
                pvals[f"{networks[i], networks[j]}"] = ttest_ind(
                    network_data[i], network_data[j]
                )[1]

        mc_corrected_pvals = multitest.multipletests(
            np.squeeze(list(pvals.values())), method="bonferroni"
        )

        boolean_matrix_significance = np.zeros((len(networks), len(networks)))
        counter = 0
        for i in range(len(networks)):
            for j in range(i + 1, len(networks)):
                boolean_matrix_significance[i, j] = mc_corrected_pvals[1][counter]
                counter += 1

        boolean_matrix_significance = (
            (boolean_matrix_significance < 0.05) & (boolean_matrix_significance > 0)
        ) * 1

        print(
            f"{band} {condition}:",
            np.array(list(pvals.keys()))[np.argwhere(mc_corrected_pvals[1] < 0.05)],
            np.array(mc_corrected_pvals[1])[np.argwhere(mc_corrected_pvals[1] < 0.05)],
        )
        l, b, w, h = 0, 0.15, 1, 0.8

        fig = plt.figure(figsize=(20, 15))
        ax = plt.axes((l, b, w, h * (1 - 0.4)))
        axbar = plt.axes((l, b + h * (1 - 0.4), w, h * 0.4 * 0.7))

        plot = sns.boxplot(network_data, ax=ax)

        plt.title(f"{band}_{condition}")
        plot.set_xticks(np.arange(7), networks, size=50)
        plt.yticks(fontsize=55)
        plot_nili_bars(axbar, boolean_matrix_significance, version=1)
        plt.show()

# %%
