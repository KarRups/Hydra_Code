#!/bin/bash

# Define the bounding box (currently unused in this script)
lat_min=35
lat_max=52
lon_min=-123
lon_max=-100

# Define the URL template
url_template="https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={gauge_id}&legacy=&referred_module=sw&period=&begin_date=2000-08-30&end_date=2024-08-29"

# Define the output directory
output_dir="/data/Hydra_Work/Rodeo_Data/USGS_Extracted"
# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Define the array of gauge IDs
gauge_ids=(
    12362500 13037500 7099400 6639000 6054500 
    9361500 9251000 12301933 13202000 12105900 
    9109000 9050700 9080190 9211150 10128500 
    11251000 11266500 11446220 12409000 12451000 
    14181500 9406000 6259000 8378500 13183000
)

# Loop through each gauge ID and download data
for gauge_id in "${gauge_ids[@]}"; do
    # Replace {gauge_id} in the URL template with the actual gauge_id
    url=$(echo "$url_template" | sed "s/{gauge_id}/$gauge_id/")

    # Define the output file for this gauge_id
    output_file="$output_dir/streamflow_data_$gauge_id.txt"

    # Download the data using curl
    curl -o "$output_file" "$url"

    # Print a message to confirm the data has been saved
    echo "Data for gauge_id $gauge_id has been saved to $output_file"
done
