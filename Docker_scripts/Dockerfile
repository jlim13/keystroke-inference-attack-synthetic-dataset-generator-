FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04

ARG local_xauth_
USER root
RUN echo "$local_xauth_"
RUN mkdir /app
WORKDIR /app

ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},display
RUN touch /root/.Xauthority
RUN apt-get update && apt-get install -y --no-install-recommends \
        mesa-utils xauth && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    vim \
    git \
    bzip2 \
    libx11-6 && \
    rm -rf /var/lib/apt/lists/*

CMD ["glxgears", "-info"]

# Install Miniconda

RUN curl -L -o ~/miniconda.sh  https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/root/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false
#
# Create a Python 3.6 environment
RUN /root/miniconda/bin/conda install conda-build \
 && /root/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /root/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/root/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# CUDA 10.0-specific steps
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch && \
    conda install torchvision && \
    conda clean -ya && \
    conda install -y h5py=2.8.0 && \
    conda clean -ya && \
    pip install h5py-cache==1.0 torchnet==0.0.4 && \
    conda install -y requests=2.19.1 && \
    conda clean -ya && \
    conda install -y ipython && \
    conda clean -ya && \
    sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module && \
    sudo rm -rf /var/lib/apt/lists/* && \
    conda install -y -c menpo opencv3=3.1.0 && \
    conda install -c menpo cyvlfeat && \
    pip install pygame scikit-learn scikit-image && \
    pip install future tb-nightly && \
    conda clean -ya

RUN apt-get update -y && \
 apt-get install software-properties-common -y && \
 add-apt-repository ppa:webupd8team/atom -y && \
 apt-get update -y && \
 apt-get install atom -y && \
 apm install remote-atom && \
 curl -o /usr/local/bin/rmate https://raw.githubusercontent.com/aurora/rmate/master/rmate && \
 sudo chmod +x /usr/local/bin/rmate && \
 mv /usr/local/bin/rmate /usr/local/bin/ratom && \
 apt-get install eog -y && \
 apt-get update -y && \
 pip install torchtext tqdm imageio configparser matplotlib tifffile pyquaternion panda3d==1.10.0 && \
 apt-get install -y x11-apps && \
 touch /root/.Xauthority

RUN touch /root/.vimrc && chmod 777 /root/.vimrc

RUN apt-get update -y && \
     apt install default-jre -y && \
     apt install openjdk-8-jdk -y
# Set the default command to python3
CMD ["python3"]



