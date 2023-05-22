"""Memory intensive jobs below; can spike up to 50GB of memory requirement"""

import mne
import numpy as np

from scipy.signal import butter, lfilter, hilbert
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
import os.path as op
import os

# A very nice overview of the Source Localization workflow : https://mne.tools/stable/overview/cookbook.html

HOMEDIR = os.path.abspath(os.getcwd())


def epochs_slicing(
    subject_raw, subject_events, event_list, tmin, tmax, fs, epochs_to_slice
):
    """Slicing only the epochs are in need

    Args:
        subject_raw (array): the signal containing different events
        subject_events (array): event timestamp at onset
        epochs_list (array): event labels
        tmin (int): onset
        tmax (int): onset + max time needed
        fs (int): sampling frequency
        epochs_to_slice (string): the event label

    Returns:
        _type_: _description_
    """

    epochs = mne.Epochs(
        subject_raw,
        subject_events,
        event_list,
        tmin=tmin,
        tmax=tmax,
        preload=True,
        verbose=False,
        baseline=(0, None),
    )
    epochs_resampled = epochs.resample(fs, verbose=False)  # Downsampling

    return epochs_resampled[epochs_to_slice]


# Data-loading
def noise_covariance(subject):
    """Computing Noise Covariance on the EEG signal. Oversimplifying, it is to set what level the noise is in the system

    Args:
        subject (string): Subject ID

    Returns:
        covariance : the computed noise covariance matrix
    """
    raw_resting_state, events_resting_state = (
        mne.io.read_raw_fif(
            f"{HOMEDIR}/Generated_data/Pre_processed_HBN_dataset/resting_state/{subject}/raw.fif",
            verbose=False,
        ),
        np.load(
            f"{HOMEDIR}/Generated_data/Pre_processed_HBN_dataset/resting_state/{subject}/events.npz"
        )["resting_state_events"],
    )

    epochs = mne.Epochs(
        raw_resting_state,
        events_resting_state,
        [20, 30, 90],
        tmin=0,
        tmax=20,
        preload=True,
        baseline=(0, None),
        verbose=False,
    )
    downsampled_fs = 250
    epochs_resampled = epochs.resample(downsampled_fs)  # Downsampling to 250Hz

    np.random.seed(55)

    # Destroying temporality
    rand = np.random.randint(1, downsampled_fs * 20, size=500)  # 20s
    cov = mne.EpochsArray(
        epochs_resampled["20"][0].get_data()[:, :, rand],
        info=raw_resting_state.info,
        verbose=False,
    )  # event '20' = RS eyes open

    covariance = mne.compute_covariance(cov, method="auto", verbose=False)

    return covariance


def forward_model(raw, trans, source_space, bem):
    """Forward solution; roughly modeling various compartements between scalp and cortical mesh

    Args:
        raw (raw.info): the raw data structure from MNE
        trans (string):
        source_space (fsaverage model): Freesurfer Cortical Mesh; fsaverage5 containing 10242 per hemisphere
        bem (bem model): Modeling the electromagnetic conductivity of various compartments
        such as scalp, skull, cortical space

    Returns:
        fwd_model: the forward solution
    """
    fwd_model = mne.make_forward_solution(
        raw.info, trans=trans, src=source_space, bem=bem, eeg=True, verbose=False
    )

    return fwd_model


def making_inverse_operator(raw, fwd_model, subject):

    covariance = noise_covariance(subject)
    inverse_operator = make_inverse_operator(
        raw.info, fwd_model, covariance, verbose=False
    )

    return inverse_operator


def source_locating(epochs, inverse_operator):
    method = "eLORETA"
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stcs = apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        verbose=False,
        return_generator=False,
    )
    return stcs


fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)


subject = "fsaverage"
trans = "fsaverage"
source_space = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")


subjects = [
    "NDARAD481FXF",
    "NDARBK669XJQ",
    "NDARCD401HGZ",
    "NDARDX770PJK",
    "NDAREC182WW2",
    "NDARGY054ENV",
    "NDARHP176DPE",
    "NDARLB017MBJ",
    "NDARMR242UKQ",
    "NDARNT042GRA",
    "NDARRA733VWX",
    "NDARRD720XZK",
    "NDARTR840XP1",
    "NDARUJ646APQ",
    "NDARVN646NZP",
    "NDARWJ087HKJ",
    "NDARXB704HFD",
    "NDARXJ468UGL",
    "NDARXJ696AMX",
    "NDARXU679ZE8",
    "NDARXY337ZH9",
    "NDARYM257RR6",
    "NDARYY218AGA",
    "NDARYZ408VWW",
    "NDARZB377WZJ",
]


n_chans_after_preprocessing = 91
time_in_samples = 21250
stc_bundle = dict()
eloreta_activation = dict()

for id in range(len(subjects)):

    data_video, events_list = (
        mne.io.read_raw_fif(
            f"{HOMEDIR}/Generated_data/Pre_processed_HBN_dataset/video-watching/{subjects[id]}/raw.fif",
            verbose=False,
        ),
        np.load(
            f"{HOMEDIR}/Generated_data/Pre_processed_HBN_dataset/video-watching/{subjects[id]}/events.npz"
        )["video_watching_events"],
    )
    sliced_epoch = epochs_slicing(
        data_video,
        events_list,
        [83, 103, 9999],
        tmin=0,
        tmax=170,
        fs=500,
        epochs_to_slice="83",
    )
    info_d = mne.create_info(
        data_video.info["ch_names"], sfreq=125, ch_types="eeg", verbose=False
    )
    the_epoch = mne.EpochsArray(
        sliced_epoch,
        mne.create_info(
            data_video.info["ch_names"], sfreq=500, ch_types="eeg", verbose=False
        ),
    ).resample(125)
    raw = mne.io.RawArray(
        the_epoch.get_data().reshape(n_chans_after_preprocessing, time_in_samples),
        info_d,
        verbose=False,
    )

    if id == 0:  # Reusing the fwd_model; cut down some time
        print("Forward model running")
        fwd_model = forward_model(
            data_video, trans=trans, source_space=source_space, bem=bem
        )

    print("Inverse model running....")
    inverse_operator = making_inverse_operator(raw, fwd_model, subjects[id])
    stc_bundle[id] = source_locating(the_epoch, inverse_operator)

    for _, stc in enumerate(stc_bundle[id]):
        eloreta_activation[id] = stc.data

    del the_epoch, raw, data_video, events_list, sliced_epoch, info_d

"""Now that the source localization has been performed and is in the fsaverage native space of having 20k vertices,
it is time to apply Glasser et al. 2016 parcellation on top"""
#%%

fs = 125
_200ms_in_samples = 25
_500ms_in_samples = 63
n_sub = 25

event_type = "30_events"

clusters = np.load(
    f"/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events_{event_type}.npz"
)
strong_isc = list()
for i, j in clusters.items():
    strong_isc.append(j)

strong_isc = sorted(np.hstack(strong_isc))


def slicing(stc):
    eloreta_activation_sliced = list()

    for time in strong_isc:
        sliced = stc[:, time * fs - _200ms_in_samples : time * fs + _500ms_in_samples]
        eloreta_activation_sliced.append(sliced)

    return eloreta_activation_sliced


eloreta_activation_sliced_for_all_subjects = list()

for sub in range(n_sub):
    eloreta_activation_sliced_for_all_subjects.append(slicing(eloreta_activation[sub]))

# np.savez_compressed(f"{HOMEDIR}/Generated_data/Cortical_surface_related/wideband_signal", wideband_signal = eloreta_activation_sliced_for_all_subjects)


with np.load(
    f"{HOMEDIR}/src_data/sourcespace_to_glasser_labels.npz"
) as dobj:  # shoutout to https://github.com/rcruces/2020_NMA_surface-plot.git
    atlas = dict(**dobj)


def averaging_by_parcellation(sub):
    """Aggregating the native brain surface fsaverage vertices to parcels (180 each hemi)

    Args:
        sub (array): source time course per subject

    Returns:
        source_signal_in_parcels : signal in parcels
    """
    source_signal_in_parcels = list()
    for roi in list(set(atlas["labels_R"]))[:-1]:
        source_signal_in_parcels.append(
            np.mean(sub.rh_data[np.where(roi == atlas["labels_R"])], axis=0)
        )

    for roi in list(set(atlas["labels_L"]))[:-1]:
        source_signal_in_parcels.append(
            np.mean(sub.lh_data[np.where(roi == atlas["labels_L"])], axis=0)
        )

    return source_signal_in_parcels


video_watching_bundle_STC = list()

for id in range(len(subjects)):
    video_watching_bundle_STC.append(
        np.array(averaging_by_parcellation(stc_bundle[id][0]))
    )


wideband_parcellated = list()
for id in range(len(subjects)):
    wideband_parcellated.append(slicing(video_watching_bundle_STC[id]))


# np.savez_compressed(f"{HOMEDIR}/Generated_data/Cortical_surface_related/wideband_parcellated", wideband_parcellated = wideband_parcellated)

# np.savez_compressed(
#     f"{HOMEDIR}/Generated_data/Cortical_surface_related/video_watching_bundle_STC_parcellated",
#     video_watching_bundle_STC_parcellated=video_watching_bundle_STC,
# )

#####################################
####Hilbert Envelope and Bandpassing
#####################################


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


bands = dict()


def filter_and_store(low, high, band):
    bandpassed = butter_bandpass_filter(
        video_watching_bundle_STC, lowcut=low, highcut=high, fs=125
    )
    hilberted = hilbert(bandpassed, N=None, axis=-1)
    bands[band] = np.abs(hilberted)


filter_and_store(8, 13, "alpha")
filter_and_store(13, 20, "low_beta")
filter_and_store(20, 30, "high_beta")
filter_and_store(4, 8, "theta")

# np.savez_compressed(
#     f"{HOMEDIR}/Generated_data/Cortical_surface_related/envelope_signal_bandpassed", **bands
# )


envelope_signal_bandpassed_sliced = dict()
for label, signal in bands.items():
    subwise = list()

    for sub in range(len(subjects)):
        subwise.append(slicing(signal[sub]))

    envelope_signal_bandpassed_sliced[f"{label}"] = subwise


wideband_and_other_bands = dict()
wideband_and_other_bands = envelope_signal_bandpassed_sliced
################################New way to make wideband signal
wideband_and_other_bands["wideband"] = np.sum(
    list(wideband_and_other_bands.values()), axis=0
)

np.savez_compressed(
    f"{HOMEDIR}/Generated_data/Cortical_surface_related/wideband_and_other_bands",
    **wideband_and_other_bands,
)

# %%
