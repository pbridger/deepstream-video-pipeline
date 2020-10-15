FROM nvcr.io/nvidia/deepstream:5.0.1-20.09-triton

RUN apt-get update && apt install --no-install-recommends -y \
    ca-certificates \
    python-gst-1.0 \
    wget \
    unzip

# allow GObject to find typelibs
ENV GI_TYPELIB_PATH /usr/lib/x86_64-linux-gnu/girepository-1.0/

# use conda to simplify some dependency managemeny
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda-10.2/compat:/opt/nvidia/deepstream/deepstream-5.0/lib:${LD_LIBRARY_PATH}:/opt/conda/lib

RUN conda install -y cmake

ENV CUDA_VER=10.2
ENV CUDNN_VER=8.0.4.30-1+cuda${CUDA_VER}
ENV TRT_VER=7.1.3-1+cuda${CUDA_VER}

RUN apt-get update
RUN apt-get install -y libcudnn8=${CUDNN_VER} libcudnn8-dev=${CUDNN_VER}
RUN apt-get install -y libnvinfer7=${TRT_VER} libnvonnxparsers7=${TRT_VER} libnvparsers7=${TRT_VER} libnvinfer-plugin7=${TRT_VER} libnvinfer-dev=${TRT_VER} libnvonnxparsers-dev=${TRT_VER} libnvparsers-dev=${TRT_VER} libnvinfer-plugin-dev=${TRT_VER} python3-libnvinfer=${TRT_VER} python3-libnvinfer-dev=${TRT_VER}

WORKDIR /build
RUN git clone https://github.com/NVIDIA/TensorRT.git
WORKDIR /build/TensorRT
RUN git submodule update --init --recursive
WORKDIR /build/TensorRT/build
RUN cmake .. -DTRT_OUT_DIR=$(pwd)/out -DCMAKE_INSTALL_PREFIX=/usr && \
    make -j$(nproc) && \
    make install

ENV TRT_VERSION=7.1.3.4

RUN conda install -y -c pytorch \
    cudatoolkit=${CUDA_VER} \
    pytorch \
    torchvision

RUN conda install -y -c conda-forge \
    pygobject \
    scikit-image \
    cython

RUN python -c "import torch; torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32')" 2>/dev/null | :
RUN python -c "import torch; torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp16')" 2>/dev/null | :


WORKDIR /opt
RUN wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip
RUN unzip libtorch*zip && rm libtorch*zip

WORKDIR /build
RUN git clone --depth 1 https://github.com/pytorch/vision.git --branch v0.7.0

# grab a bugfix to make NMS op available on CUDA
WORKDIR /build/vision
RUN wget -q -O - https://github.com/pytorch/vision/commit/1af4edc6507f03513ef23c7f8d6050b8b7a4721c.patch | git apply

WORKDIR /build/vision/build
RUN cmake .. -DWITH_CUDA=on -DCMAKE_PREFIX_PATH="/opt/libtorch"
RUN rm /usr/lib/gcc/x86_64-linux-gnu/7/../../../x86_64-linux-gnu/libcuda.so
RUN make -j$(nproc) && \
    make install

ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/opt/libtorch/lib

# Nvidia Apex for mixed-precision inference
RUN git clone https://github.com/NVIDIA/apex.git /build/apex
WORKDIR /build/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

RUN pip install --upgrade cython
RUN pip install --upgrade gil_load

# Gstreamer debug output location
env GST_DEBUG_DUMP_DOT_DIR=/app/logs

# RUN rm -rf /var/lib/apt/lists/* && \
#     conda clean -afy

WORKDIR /app
