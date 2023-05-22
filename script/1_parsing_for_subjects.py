#%%

import scipy
from scipy import io
import numpy as np
import pandas as pd
import ezodf
import os

HOMEDIR = "/users2/local/Venkatesh/Multimodal_ISC_Graph_study/"  # os.path.abspath(os.getcwd())

mat = scipy.io.loadmat(
    f"{HOMEDIR}/src_data/subject_inclusion/subject_list.mat")
df = pd.DataFrame(mat["good_EEG"])
df_sliced = [df.values[i][0][0] for i in range(len(df))]

df_SI = pd.read_csv(
    f"{HOMEDIR}/src_data/subject_inclusion/participants_SI.tsv")
df_RU = pd.read_csv(
    f"{HOMEDIR}/src_data/subject_inclusion/participants_RU.tsv", sep="\t")
df_CUNY = pd.read_csv(
    f"{HOMEDIR}/src_data/subject_inclusion/participants_CUNY.tsv", sep="\t")
df_CBIC = pd.read_csv(
    f"{HOMEDIR}/src_data/subject_inclusion/participants_CBIC.tsv", sep="\t")

subjects_aged = list()


def find_subject_age(which_df, age):
    """Filtering subject list based on their age

    Args:
        which_df (dataframe): the df from different HBN sites containing the totality of subj info
        age (_type_): threshold to apply filter for
    """
    
    subjects_aged.append(which_df[which_df["Age"] >= age]["participant_id"].values)


age = 16  # criteria 1
find_subject_age(df_CBIC, age)
find_subject_age(df_RU, age)
find_subject_age(df_CUNY, age)
find_subject_age(df_SI, age)
subjects_aged = np.hstack(subjects_aged)


def read_ods(filename, sheet_no=0, header=0):
    tab = ezodf.opendoc(filename=filename).sheets[sheet_no]
    return pd.DataFrame({
        col[header + 1].value: [x.value for x in col[header + 1:]]
        for col in tab.columns()
    })


# criteria 2
df_dwi = read_ods(f"{HOMEDIR}/src_data/subject_inclusion/dwi-subject_list.ods",
                  0, 0)
# criteria 3
df_freesurfer = read_ods(
    f"{HOMEDIR}/src_data/subject_inclusion/subjects_freesurfer.ods")

df_freesurfer = df_freesurfer.reset_index()

df_freesurfer_sliced = [
    df_freesurfer[df_freesurfer.columns[1]].values[i][31:-1]
    for i in range(1, len(df_freesurfer))
]

df_dwi_sliced = [
    df_dwi[df_dwi.columns[0]].values[i][31:] for i in range(0, len(df_dwi))
]

subjects_aged_sliced = [
    subjects_aged[i][4:] for i in range(0, len(subjects_aged))
]
total_subjects = list(
    set(df_dwi_sliced).intersection(
        set(df_sliced).intersection(set(subjects_aged_sliced))))

# criteria 4
final_subject_list = list()
for i in range(len(total_subjects)):
    path_to_file = (
        f"{HOMEDIR}/src_data/HBN_dataset/%s/RestingState_data.csv" %
        total_subjects[i])
    path_to_file_video = (
        f"{HOMEDIR}/src_data/HBN_dataset/%s/Video3_event.csv" %
        total_subjects[i])

    if os.path.isfile(path_to_file) and os.path.isfile(path_to_file_video):
        final_subject_list.append(total_subjects[i])

subjects_sanity_check = [
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

print("N = ",
      len(set(final_subject_list).intersection(subjects_sanity_check)))  #%%
#%%
new = df_CUNY[['participant_id', 'Age']].append(df_RU[['participant_id', 'Age']]).append(df_CBIC[['participant_id', 'Age']]).append(df_SI[['participant_id', 'Age']])

new_sublists = [new['participant_id'].values[i][4:] for i in range(len(new))]

new['subj'] = new_sublists
subjects_sanity_check = [
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

new[new['subj'].isin(subjects_sanity_check)]['Age'].max()
# %%
subjects_aged_sliced = [
    subjects_aged[i][4:] for i in range(0, len(subjects_aged))
]
# %%
