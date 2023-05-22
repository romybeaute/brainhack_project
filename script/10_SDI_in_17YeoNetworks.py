#%%
from nilearn import datasets, maskers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats import multitest
import matplotlib as mpl
import os

HOMEDIR = '/users2/local/Venkatesh/Multimodal_ISC_Graph_study/'#os.path.abspath(os.getcwd())

atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
yeo = atlas_yeo_2011.thick_17
glasser = '/users2/local/Venkatesh/Multimodal_ISC_Graph_study/src_data/Glasser_masker.nii.gz' #f"{HOMEDIR}/src_data/Glasser_masker.nii.gz"

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

plt.style.use("fivethirtyeight")


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



frequencies = ["alpha", "theta", "low_beta", "high_beta", "wideband"]
networks = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
]
condition = ["baseline", "differenced"]
Vis = 1
DMN = 17

all_network = list()
for cond in condition:
    for band in frequencies:

        thresholded_map = np.load(
            f"{HOMEDIR}/Generated_data/Graph_SDI_related/2nd_level_test_stats/thresholded/thresholded_{cond}_{band}.npz"
        )["secondlevel_t_threholded"]

        network_data_median = list()
        network_data_raw = list()

        # Network-wise data
        for network in range(Vis, DMN + 1):
            network_data_median.append(
                np.median(thresholded_map[(np.argwhere(np.array(match) == network))])
            )
            network_data_raw.append(
                thresholded_map[(np.argwhere(np.array(match) == network))]
            )

        network_data_median = np.array(network_data_median)
        # binarized_data = (network_data_median > 0) * 1 + (network_data_median < 0) * -1
        all_network.append(network_data_median)

        # Stat comparison
        pvals = dict()
        network_pairs = list()
        for i in range(len(networks)):
            for j in range(i + 1, len(networks)):
                network_pairs.append((networks[i], networks[j]))
                pvals[f"{networks[i], networks[j]}"] = ttest_ind(
                    network_data_raw[i], network_data_raw[j]
                )[1]

        mc_corrected_pvals = multitest.multipletests(
            np.squeeze(list(pvals.values())), method="bonferroni"
        )
        print(f"{band} {cond}:", np.array(list(pvals.keys()))[np.argwhere(mc_corrected_pvals[1] <0.05)], np.array(mc_corrected_pvals[1])[np.argwhere(mc_corrected_pvals[1] <0.05)])

        boolean_matrix_significance = np.zeros((len(networks), len(networks)))
        counter = 0
        for i in range(len(networks)):
            for j in range(i + 1, len(networks)):
                boolean_matrix_significance[i, j] = mc_corrected_pvals[1][counter]
                counter += 1

        boolean_matrix_significance = (
            (boolean_matrix_significance < 0.05) & (boolean_matrix_significance > 0)
        ) * 1

        # Plot both barplot and stat comparisons
        l, b, w, h = 0, 0.15, 1, 0.8

        fig = plt.figure(figsize=(20, 15))
        ax = plt.axes((l, b, w, h * (1 - 0.4)))
        axbar = plt.axes((l, b + h * (1 - 0.4), w, h * 0.4 * 0.7))
        norm = mpl.colors.BoundaryNorm(
            sorted(np.linspace(0, np.min(all_network), len(networks))), 128
        )
        clrs = [plt.cm.coolwarm(norm(c)) for c in network_data_median]

        df = pd.DataFrame(np.vstack(network_data_median))
        df = df.transpose()
        df.columns = networks
        
        sns.barplot(data=df, palette=clrs, ax=ax)        
        plt.title(f"{band} / {cond}")
        plot_nili_bars(axbar, boolean_matrix_significance, version=1)
        fig.savefig(f"{HOMEDIR}/Results/Figures/{cond}/{band}.png", bbox_inches="tight")
        plt.show()

# %%
# panel D
all_network_baseline = np.array(all_network)[: len(frequencies), :]
all_network_differenced = np.array(all_network)[len(frequencies) :, :]

fig = plt.figure(figsize=(35, 25))
# norm = mpl.colors.BoundaryNorm(sorted(np.linspace(0, np.min(all_network), 17)), 128)
sns.heatmap(
    all_network_differenced.T,
    cmap="coolwarm",
    annot=True,
    vmax=-1 * np.min(all_network_differenced),
)

plt.xticks(range(len(frequencies)), frequencies, fontsize=55, rotation=45)
plt.yticks(range(len(networks)), networks, fontsize=35)
plt.ylabel("Networks", fontsize=55)
plt.xlabel("Frequency bands", fontsize=65)
fig.tight_layout()
fig.suptitle("differenced", fontsize=45)
#%%
