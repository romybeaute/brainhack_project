# Include the `Multimodal_ISC_Graph_study/src_data/subjects_to_download_for` in the command line while executing this script -> ex. sh Downloading_from_AWS.sh subjects_to_download_for

# The whole of HBN dataset, n = 25, requires 5GBish of space
for CURSUBJ in $(cat $1)
do
# Mac users, an app Cyberduck comes in very handy listing the files from the AWS bucket on-the-go

#Video-watching
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_data.csv src_data/HBN_dataset/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_event.csv src_data/HBN_dataset/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_chanlocs.csv src_data/HBN_dataset/${CURSUBJ}/ --no-sign-request

#Restingstate
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_data.csv src_data/HBN_dataset/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_event.csv src_data/HBN_dataset/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_chanlocs.csv src_data/HBN_dataset/${CURSUBJ}/ --no-sign-request

done
