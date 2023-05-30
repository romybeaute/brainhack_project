Dynamical connectivity refers to the statistical dependencies among separate brain regions over time, typically represented by a network. Each node in the network represents a separate brain region (or a 
Source), and the edges or links represent the strength of the interaction between the regions.

From the description of the datafile (stored in stc_data_osf), we have signals from different frequency bands (Theta, Alpha, Low Beta, High Beta, and Wideband) for each region of interest (ROI). These 
signals are stored as numpy arrays in a dictionary called 'video_watching_dataset', with each array having the shape (n_subjects=25,n_ROIs=360,n_times=21250). This means that for each frequency band, we 
have a 3-dimensional numpy array where the dimensions represent the number of subjects, the number of ROIs, and the number of time points, respectively.

The number of time points in the data represents the number of discrete instances at which the electrophysiological activity was measured within each frequency band. For each of these time points, the 
signal's activity (here EEG) is measured and recorded. The frequency band (Theta, Alpha, Low Beta, High Beta, and Wideband) refers to a particular range of frequencies within the signal. Signals often 
comprise a mixture of different frequencies, and these bands are ranges in which these frequencies are grouped. Therefore, for each frequency band in the data, the number of time points (21250, as per in 
our data) represents the number of discrete measurements of electrophysiological activity within that frequency band. Put simply, the electrophysiological data was measured 21250 times for each frequency 
band, for each subject, and for each region of interest. This allows us to analyze how the signal's activity within each frequency band changes over time.

To compute dynamical connectivity (statistical dependencies among separate brain regions over time), we need to correlate the activity between different ROIs at each time point and across subjects. This 
can be done using different measures, including but not limited to coherence, phase-locking value (PLV), or phase-lag index (PLI).

To do so, we used the MNE-connectivity module in Python which is designed to help calculate these connectivity measures. Below is a step-by-step guide to use MNE-connectivity for calculating dynamical 
connectivity. In this example, we use the phase-lag index (PLI) as a measure of connectivity, but we could replace it with any other measure supported by MNE.
