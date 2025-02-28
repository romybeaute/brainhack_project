<!-- # Bio
Venkatesh, a student between IMT A France and UdeM, has this repo for a project at BHS 2023.

<a href="https://github.com/venkateshness">
   <img src="https://avatars.githubusercontent.com/u/24219857?s=400&u=2b8a6e3c942d7b0b8543e23837f2e6ccda566f25&v=4" width="100px;" alt=""/>
   <br /><sub><b>Venkatesh Subramani</b></sub>
</a> -->


# Structure-Function decoupling of electrophysiological brain activity during video-watching
## Tl;Dr :
* Dataset : http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/ 
* Video-watching EEG (170s) recorded using high-density 128 electrodes
* Consensus anatomical graph built using tractography
* Preprocessing adapting doi: 10.1038/sdata.2017.40 (2017) & https://doi.org/10.1016/j.neuroimage.2020.117001
* Source Reconstruction using eLORETA and BEM & Parcellation on 360 regions of HCP-MMP atlas
* Inter-Subject Correlation (ISC) using Correlated Component Analysis (CorrCCA) introduced in doi: 10.3389/fnhum.2012.00112
* Quantifying Structure-Function relationship using a Graph-Signal Processing (GSP) measure called Structure-Function Decoupling (SDI) introduced in https://doi.org/10.1038/s41467-019-12765-7


The folders above are ordered sequentially needed for the analysis: Downloading the dataset from AWS -> Loading and preprocessing -> EEG source localization -> Analysis on Graph space.