#%%
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
import os
import itertools
from nilearn._utils import check_niimg_3d
from matplotlib import gridspec
from nilearn.surface import vol_to_surf
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009

HOMEDIR = (
    "/users2/local/Venkatesh/Multimodal_ISC_Graph_study"  # os.path.abspath(os.getcwd())
)

bands = [
    "alpha",
    "theta",
    "low_beta",
    "high_beta",
    "wideband",
]  # , 'theta', 'low_beta', 'high_beta'
conditions = ["baseline", "differenced"]


import itertools
from nilearn._utils import check_niimg_3d
from matplotlib import gridspec
from nilearn.surface import vol_to_surf


def customized_plotting_img_on_surf(
    views, stat_map, hemispheres, vmax, threshold, file_location
):
    modes = plotting.surf_plotting._check_views(views=views)
    hemis = plotting.surf_plotting._check_hemispheres(hemispheres=hemispheres)
    surf_mesh = "fsaverage5"
    surf_mesh = plotting.surf_plotting._check_mesh(surf_mesh)

    inflate = False
    mesh_prefix = "infl" if inflate else "pial"

    surf = {
        "left": surf_mesh[mesh_prefix + "_left"],
        "right": surf_mesh[mesh_prefix + "_right"],
    }

    stat_map = check_niimg_3d(stat_map, dtype="auto")

    texture = {
        "left": vol_to_surf(stat_map, surf_mesh["pial_left"], mask_img=None),
        "right": vol_to_surf(stat_map, surf_mesh["pial_right"], mask_img=None),
    }
    gridspec_layout = gridspec.GridSpec(
        1, 4, left=0.0, right=1.0, bottom=0.0, top=1.0, hspace=0.0, wspace=0.0
    )
    fig = plt.figure(figsize=(40, 20))
    for i, (mode, hemi) in enumerate(itertools.product(modes, hemis)):
        bg_map = surf_mesh["sulc_%s" % hemi]
        ax = fig.add_subplot(gridspec_layout[i], projection="3d")
        plotting.plot_surf_stat_map(
            surf[hemi],
            texture[hemi],
            view=mode,
            hemi=hemi,
            bg_map=bg_map,
            axes=ax,
            colorbar=False,
            vmax=vmax,
            threshold=threshold,
        )
    # fig.savefig(f"{file_location}.png", transparent=True, dpi=500)


for condition in conditions:
    max = list()
    for band in bands:
        max.append(
            np.max(
                np.abs(
                    np.load(
                        f"{HOMEDIR}/Generated_data/Graph_SDI_related/2nd_level_test_stats/thresholded/thresholded_{condition}_{band}.npz"
                    )["secondlevel_t_threholded"]
                )
            )
        )

    for band in bands:
        path_to_file = np.load(
            f"{HOMEDIR}/Generated_data/Graph_SDI_related/2nd_level_test_stats/unthresholded/unthresholded_{condition}_{band}.npz"
        )["secondlevel_t"]

        path_Glasser = "/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz"

        mnitemp = fetch_icbm152_2009()

        spatial_map = path_to_file * (path_to_file > np.percentile(path_to_file, 95))
        U0_brain = signals_to_img_labels(spatial_map, path_Glasser, mnitemp["mask"])

        customized_plotting_img_on_surf(
            views=["lateral", "medial"],
            hemispheres=["left", "right"],
            stat_map=U0_brain,
            vmax=np.max(max),
            threshold=0.1,
            file_location=f"{HOMEDIR}/Results/Figures/spatial_maps/{condition}/{band}",
        )
        plt.show()

# %%
