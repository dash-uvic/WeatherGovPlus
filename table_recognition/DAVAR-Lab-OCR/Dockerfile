FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive 
ARG CUDA="11.6"
ARG CUDNN="8"
ARG TORCH_CUDA_ARCH_LIST="8.6+PTX"

RUN nvcc --version

WORKDIR /content
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    git \
    python3.8-dev \
    python-is-python3 \
    pip \
    cmake\ 
    pkg-config \
    ffmpeg \
    libsm6 \
    libxext6 \
    ninja-build 

RUN pip install -U pip && pip install setuptools==59.5.0 torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/hikopensource/DAVAR-Lab-OCR.git

ENV MMCV_WITH_OPS=1
ENV FORCE_CUDA=1
RUN git clone https://github.com/open-mmlab/mmcv.git
WORKDIR /content/mmcv
RUN git checkout v1.3.4
RUN pip install -e . -v
WORKDIR /content/DAVAR-Lab-OCR
COPY . .
RUN patch -p1 < int.patch
RUN bash setup.sh
CMD ["python", "inference.py", "/data"]

