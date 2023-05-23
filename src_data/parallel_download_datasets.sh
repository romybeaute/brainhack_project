#!/bin/bash

# List of subjects to download
subjects_file="subjects_to_download.txt" #use "subjects_to_download_subset.txt" to download a subset of the dataset (first 4 subjects)


# Function to download a file for a subject if it doesn't already exist 
download_file() {
  local source_file="$1"
  local target_dir="$2"

  local filename=$(basename "$source_file")
  local target_path="${target_dir}/${filename}"

  if [[ ! -f "${target_path}" ]]; then
    aws s3 cp "${source_file}" "${target_path}" --no-sign-request
    echo "Downloaded file: ${target_path}"
  else
    echo "File already exists: ${target_path}"
  fi
}


# Function to download files for a subject in the background
download_subject() {
  local CURSUBJ="$1"

  # Video-watching
  aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_data.csv /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/ --no-sign-request &
  aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_event.csv /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/ --no-sign-request &
  aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/Video3_chanlocs.csv /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/ --no-sign-request &

  # Restingstate
  aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_data.csv /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/ --no-sign-request &
  aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_event.csv /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/ --no-sign-request &
  aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/EEG/preprocessed/csv_format/RestingState_chanlocs.csv /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/ --no-sign-request &

  # Wait for all background downloads to complete
  wait

  # Print validation message for each downloaded file
  echo "Downloaded files for subject ${CURSUBJ}:"
  echo "Video-watching: /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/"
  echo "Restingstate: /Users/rb666/brainhack/brainhack_project/src_data/EEG/${CURSUBJ}/"
}

# Read subjects from file and process them in parallel
while read -r CURSUBJ; do
  echo "Processing subject: ${CURSUBJ}"
  download_subject "${CURSUBJ}" &
done < "${subjects_file}"

# Wait for all subject downloads to complete
wait

echo "All downloads completed."
