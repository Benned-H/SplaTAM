# Use a reasonably recent version of CUDA (12.6.3) with a corresponding PyTorch version
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS splatam

### Install uv for Python dependency management ###
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install the uv installer, run it, and then remove it
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

### Install the corresponding version of PyTorch, plus all SplaTAM dependencies ###
# Need to use Python 3.10 to support faiss-gpu
ARG PYTHON_VERSION="3.10"

RUN uv venv --python ${PYTHON_VERSION}
RUN uv pip install torch torchvision torchaudio \
    tqdm \
    Pillow \
    opencv-python \
    imageio \
    matplotlib \
    kornia \
    natsort \
    pyyaml \
    wandb \
    lpips \
    open3d \
    torchmetrics \
    cyclonedds \
    pytorch-msssim \
    plyfile \
    faiss-gpu

### Install the diff-gaussian-rasterization-w-depth dependency from GitHub ###
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git ninja-build software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10-venv python3.10-dev libgl1-mesa-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX;8.9;9.0"
ENV CXXFLAGS="-std=c++17"
ENV TORCH_CUDA_CXX_FLAGS="-Wno-deprecated-declarations"

WORKDIR /src/diff-gaussian
RUN git init && \
    git remote add origin https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git && \
    git fetch --depth 1 origin cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110 && \
    git checkout FETCH_HEAD
RUN sed -i '1i#include <cstdint>\n#include <cstddef>' cuda_rasterizer/rasterizer_impl.h && \
    uv pip install --no-build-isolation .