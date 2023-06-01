
"""
This script analyzes neuroimaging data, specifically comparing overlaps between Yeo 2011 and Glasser brain atlases. 
It fetches the Yeo atlas, loads the Glasser atlas, applies a masker to isolate regions of interest in both atlases, and computes their overlap. 
The output is a detailed mapping of how these two prominent brain atlases correspond to each other in terms of spatial overlap of their defined regions of interest.
"""


from load_npz_data import load_data
import numpy as np
import os
from nilearn import datasets, plotting, maskers



def map_atlases(home_dir="~/brainhack/brainhack_project"):
    """
    This function calculates the overlap between regions of interest in the Glasser and Yeo atlases.
    """
    HOMEDIR = os.path.expanduser(home_dir)
    print('HOMEDIR: ',HOMEDIR)

    atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
    yeo = atlas_yeo_2011.thick_7
    glasser = f"{HOMEDIR}/src_data/Glasser_masker.nii"

    # Create and fit a masker
    masker = maskers.NiftiMasker(standardize=False, detrend=False)
    masker.fit(glasser) 
    glasser_vec = masker.transform(glasser)

    yeo_vec = masker.transform(yeo)
    yeo_vec = np.round(yeo_vec)

    # Calculate overlap between atlases 
    matches = []
    match = []
    best_overlap = []
    for i, roi in enumerate(np.unique(glasser_vec)):
        overlap = []
        for roi2 in np.unique(yeo_vec):
            overlap.append(
                np.sum(yeo_vec[glasser_vec == roi] == roi2) / np.sum(glasser_vec == roi)
            )
        best_overlap.append(np.max(overlap)) #Store the maximum overlap and corresponding Yeo ROI for each Glasser ROI
        match.append(np.argmax(overlap))
        matches.append((i + 1, np.argmax(overlap)))
    
    return matches, match, best_overlap


