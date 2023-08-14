# WeatherGovPlus
Repository containing supplementary material for the DocEng 2023 paper "WEATHERGOV+".  Please see paper for details.

Requires:
* Python 3.8
* Docker
* NVIDIA CUDA Docker Toolkit


This has been tested and run on Linux Mint 20 (Ubuntu 20.04) and CUDA 11.6.  To use a different version of CUDA, you will need to build the docker containers for your system.


## Download pretrained recognition model files

1. Go to https://github.com/hikopensource/DAVAR-Lab-OCR.git and download to DAVAR-Lab-OCR folder:
  * maskrcnn-lgpma-pub-e12-pub.pth  
  * text_det_model.pth  
  * text_rcg_model_e8.pth 
2. Go to https://github.com/JiaquanYe/TableMASTER-mmocr.git and download to TableMASER-mmocr/checkpoints folder:
  * 200~epoch_16_0.7767.pth
  * master_epoch_6.pth
  * pse_epoch_600.pth

## Build and run the Docker images
0. Install Docker with NVIDIA Toolkit
1. Go to the recognition model folder
    cd table_recognition/[Model]
2. Build the docker image 
    nvidia-docker build -t [docker name] .
3. Run the inference.py script.  This will bind the evaluation folder to the docker image and save the results to evaluation/results
    docker run --rm -it --runtime=nvidia --gpus all --mount type=bind,source=../../evaluation/WEATHERGOV_PLUS,target=/data,readonly --mount type=bind,source=../../evaluation/results,target=/results [docker name] python inference.py 

## Run the NLP evaluation using MVP

0. Create a virtualenv and install the packages in evaluation/requirements
1. Unzip the folders in evaluations/WEATHER_PLUS 
2. Run the MVP NLP algorithm:
   python run_analysis.py --html-dir evaluaion/[davar,lgpma] [--run-teds] [--run-rouge] [--run-bleu]
