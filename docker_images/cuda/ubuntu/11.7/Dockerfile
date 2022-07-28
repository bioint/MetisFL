FROM projectmetis_ubuntu_22_04 as base

# On Centos/RHEL machines, after you build the image in order to make sure that the nvidia driver is properly
# configured, you need to run the following command
#   nvidia-docker run --rm --gpus all `cuda_image_name` nvidia-smi
#   e.g., nvidia-docker run --rm --gpus all projectmetis_ubuntu_22_04_cuda nvidia-smi
# Image content based on: https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.7.0/ubuntu2204/base/Dockerfile
CMD ["bash"]
ENV NVARCH=x86_64
ENV NVIDIA_REQUIRE_CUDA=cuda>=11.7
ENV NV_CUDA_CUDART_VERSION=11.7.60-1
ENV CUDA_VERSION=11.7.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates \
    && curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH}/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update  \
    && apt-get install -y --no-install-recommends cuda-cudart-11-7=${NV_CUDA_CUDART_VERSION} ${NV_CUDA_COMPAT_PACKAGE} \
    && ln -s cuda-11.7 /usr/local/cuda \
    && rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
