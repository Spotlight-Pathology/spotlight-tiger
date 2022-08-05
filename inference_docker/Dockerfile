# Based on https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example/blob/main/Dockerfile

FROM nvidia/cuda:11.1-runtime-ubuntu20.04
# pull an image that supports cuda and cudnn

RUN : \
    && rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-get update \
    && apt-get -y install curl \
    && curl -L -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm cuda-keyring_1.0-1_all.deb \
    && :

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.8
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.8-venv \
    && apt-get install libpython3.8-dev -y \
    && apt-get clean \
    && :
    
# Add env to PATH
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install ASAP
RUN : \
    && apt-get update \
    && apt-get -y install curl \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2004.deb" \
    && dpkg --install ASAP-2.1-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.8/site-packages/asap.pth \
    && rm ASAP-2.1-Ubuntu2004.deb \
    && :

# # Install openslide
ARG DEBIAN_FRONTEND=noninteractive

RUN : \
    && apt-get update \
    && apt-get install openslide-tools -y \
    && apt-get install build-essential -y \
    && :

# # Install requirements
ARG USE_NNPACK=0
COPY ./requirements.txt /home/user/requirements.txt
RUN : \
    && pip install wheel==0.37.0 \
    && pip install --no-cache-dir -r /home/user/requirements.txt \
    && :

COPY ./ /home/user/pathology-tiger-algorithm/
RUN : \
    && pip install /home/user/pathology-tiger-algorithm \
    && :

# Make user
RUN groupadd -r user && useradd -r -g user user
RUN chown user /home/user/
RUN mkdir /output/
RUN chown user /output/
USER user
WORKDIR /home/user

# Cmd and entrypoint
CMD ["-mtigeralgorithmexample"]
ENTRYPOINT ["python"]

# Compute requirements
LABEL processor.cpus="4"
LABEL processor.cpu.capabilities="null"
LABEL processor.memory="30G"
LABEL processor.gpu_count="1"
LABEL processor.gpu.compute_capability="null"
LABEL processor.gpu.memory="15G"