# The whole of HBN dataset, n = 25, requires 5GBish of space
for CURSUBJ in $(cat subjects_to_download.txt)
do
# Mac users, an app Cyberduck comes in very handy listing the files from the AWS bucket on-the-go

#Video-watching
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_data.csv /Users/rb666/brainhack/brainhack_project/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_event.csv /Users/rb666/brainhack/brainhack_project/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_chanlocs.csv /Users/rb666/brainhack/brainhack_project/${CURSUBJ}/ --no-sign-request

#Restingstate
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_data.csv /Users/rb666/brainhack/brainhack_project/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_event.csv /Users/rb666/brainhack/brainhack_project/${CURSUBJ}/ --no-sign-request
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_chanlocs.csv /Users/rb666/brainhack/brainhack_project/${CURSUBJ}/ --no-sign-request

done

