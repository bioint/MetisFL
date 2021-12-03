# Date: Dec 2, 2021
# Description: (A) This docker file builds on top of projectmetis base image which should be build with the name projectmetisrc/projectmetis:rocky_linux_8
#              (B) The projectmetisrc/projectmetis:rocky_linux_8 can be built using Dockerfile in base path.
#              (C) This installs the CuDNN 8 and Cuda Toolkit 11.3 on the base image
# How to build: Pre-requisite is that the projectmetisrc/projectmetis:rocky_linux_8 image needs to be built and after that this image is build
# Required File: Two files cuda.repo-x86_64 and nvidia-ml.repo-x86_64 are copied the image in the repository folders and they are used to download
#                and install the required Cuda packages. The files and docker file inspiration is taken from
#                https://gitlab.com/nvidia/container-images/cuda/-/tree/master/dist



FROM projectmetisrc/projectmetis:rocky_linux_8  as base

ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.3 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450"
ENV NV_CUDA_CUDART_VERSION 11.3.58-1

COPY cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo
COPY nvidia-ml.repo-x86_64 /etc/yum.repos.d/nvidia-ml.repo

RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${NVARCH}/7fa2af80.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
    echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

ENV CUDA_VERSION 11.3.0

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN yum upgrade -y && yum install -y \
    cuda-cudart-11-3-${NV_CUDA_CUDART_VERSION} \
    cuda-compat-11-3 \
    && ln -s cuda-11.3 /usr/local/cuda \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# nvidia-docker 1.0
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64


# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# RUNTIME ENV

ENV NV_CUDA_LIB_VERSION 11.3.0-1
ENV NV_NVTX_VERSION 11.3.58-1
ENV NV_LIBNPP_VERSION 11.3.3.44-1
ENV NV_LIBNPP_PACKAGE libnpp-11-3-${NV_LIBNPP_VERSION}
ENV NV_LIBCUBLAS_VERSION 11.4.2.10064-1
ENV NV_LIBNCCL_PACKAGE_NAME libnccl
ENV NV_LIBNCCL_PACKAGE_VERSION 2.9.6-1
ENV NV_LIBNCCL_VERSION 2.9.6
ENV NCCL_VERSION 2.9.6
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}-${NV_LIBNCCL_PACKAGE_VERSION}+cuda11.3


RUN yum install -y \
    cuda-libraries-11-3-${NV_CUDA_LIB_VERSION} \
    cuda-nvtx-11-3-${NV_NVTX_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    libcublas-11-3-${NV_LIBCUBLAS_VERSION} \
    ${NV_LIBNCCL_PACKAGE} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# CUDA DNN Libraries

ENV NV_CUDNN_VERSION 8.2.0.53-1
ENV NV_CUDNN_PACKAGE libcudnn8-${NV_CUDNN_VERSION}.cuda11.3
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN yum install -y ${NV_CUDNN_PACKAGE} \
    && yum clean all \
    && rm -rf /var/cache/yum/*


