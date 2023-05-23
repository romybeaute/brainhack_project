# The whole of HBN dataset, n = 25, requires 5GBish of space
# ==> Please note that GitHub provides 1 GB of free storage for Git LFS files, and additional storage can be purchased. Make sure to consider this if you're working with a large amount of data.
# We will only use 4 subjects in order to be able to use the free storage Git LFS files


#Loops over a list of subject IDs (provided in a file specified by $1).
#For each subject ID, it downloads several files from specific locations in the S3 bucket.

#for CURSUBJ in $(cat subjects_to_download.txt) to get the complete dataset (/!\ if do this don't download in repo because will be > 1GB)
for CURSUBJ in $(cat subjects_to_download_subset.txt) 
do
    echo "Processing subject: $CURSUBJ"
    mkdir -p /Users/rb666/brainhack/brainhack_project/EEG/${CURSUBJ}/
# Mac users, an app Cyberduck comes in very handy listing the files from the AWS bucket on-the-go
#The files are downloaded using the aws s3 cp command, which copies files from S3 to local machine.
#The --no-sign-request option allows you to perform the operation without AWS credentials. This works because the S3 bucket is public.
#The files are saved in a directory named after the subject ID under src_data/HBN_dataset/

#Video-watching
    echo "Downloading Video-watching data for $CURSUBJ"
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_data.csv /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/ --no-sign-request
    echo "Stored Video3_data.csv at /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/"

    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_event.csv /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/ --no-sign-request
    echo "Stored Video3_event.csv at /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/"

    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_chanlocs.csv /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/ --no-sign-request
    echo "Stored Video3_chanlocs.csv at /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/"

#Restingstate
    echo "Downloading Restingstate data for $CURSUBJ"
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_data.csv /Users/rb666/brainhack/brainhack_project/EEG/${CURSUBJ}/ --no-sign-request
    echo "Stored RestingState_data.csv at /Users/rb666/brainhack/brainhack_project/EEG/${CURSUBJ}/"

    
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_event.csv /Users/rb666/brainhack/brainhack_project/EEG/${CURSUBJ}/ --no-sign-request
    echo "Stored RestingState_event.csv at /Users/rb666/brainhack/brainhack_project/EEG/${CURSUBJ}/"


    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_chanlocs.csv /Users/rb666/brainhack/brainhack_project/EEG/${CURSUBJ}/ --no-sign-request
    echo "Stored RestingState_chanlocs.csv at /Users/rb666/brainhack/brainhack_project/EEG/${CURSUBJ}/"

done