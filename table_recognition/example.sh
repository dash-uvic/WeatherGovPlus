#!/bin/bash

results_dir="$(pwd)/results"
data_dir="$(pwd)/WEATHERGOV_PLUS"

if [[ ! -d venv ]]; then
    echo "Creating virtualenv"
    python3 -m venv venv
    . venv/bin/activate
    python3 -m pip install -r requirements.txt
fi 

echo "Running docker models"
images="davar:latest tablemaster:latest"

for image in $images; do
    echo docker run --rm -it --runtime=nvidia --gpus all \
         --mount type=bind,source=$data_dir,target=/data,readonly \
         --mount type=bind,source=$results_dir,target=/results \
         $image 
done


