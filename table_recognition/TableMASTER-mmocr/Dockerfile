FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ARG CUDA="11.6"
ARG CUDNN="8"
ARG TORCH_CUDA_ARCH_LIST="8.6+PTX"

ENV DEBIAN_FRONTEND=noninteractive 
ENV MMCV_WITH_OPS=1
ENV FORCE_CUDA=1

RUN nvcc --version

WORKDIR /content
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    git \
    python3.8-dev \
    python-is-python3 \
    pip \
    cmake\ 
    pkg-config \
    ninja-build \ 
    ffmpeg \
    libsm6 \
    libxext6 

RUN pip install -U pip && pip install tqdm setuptools==59.5.0 torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/JiaquanYe/TableMASTER-mmocr.git 

WORKDIR /content/TableMASTER-mmocr/mmdetection-2.11.0
RUN pip install -v -e .
WORKDIR /content/TableMASTER-mmocr
RUN pip install -v -e .
RUN pip install mmcv-full==1.3.4 

COPY . .
CMD ["python", "inference.py", "/data"]

