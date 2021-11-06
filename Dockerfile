FROM tanmaniac/opencv3-cudagl

ENV http_proxy="http://www-cache.rd.bbc.co.uk:8080"
ENV https_proxy="http://www-cache.rd.bbc.co.uk:8080"

RUN mkdir /app
WORKDIR /app

# install prerequisites
RUN apt-get update \
 && apt-get install -y wget git curl nano \
 && apt-get install -y libsm6 libxext6 libxrender-dev

# install Cudnn
ENV CUDNN_VERSION 7.6.0.64
RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

# Create a Python 3.6 environment
ENV PATH=/miniconda/bin:$PATH

RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name unet python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=unet
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -c anaconda scikit-learn
RUN conda install -y scikit-image ipython tensorflow-gpu=1.14.0 keras=2.3.1
RUN conda install -c conda-forge opencv