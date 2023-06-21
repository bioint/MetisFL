# Install Ubuntu image from docker hub.
FROM ubuntu:focal as ubuntu
ARG PYV="3.10"
# Assign simple root password.
RUN echo "root:root" | chpasswd

# Assign simple root password.
RUN echo "root:root" | chpasswd

# Set environmental variables.
ENV PROJECT_HOME=/metisfl
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
ENV PYTHONPATH=/workspaces/metisfl

WORKDIR /metisfl

# Simple update.
RUN apt-get -y update

# Install helper packages and bazelisk (i.e., bazel versioning resolution tool at runtime).
RUN apt-get -y install wget nano git
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64
RUN chmod +x ./bazelisk-linux-amd64
RUN mv ./bazelisk-linux-amd64 /usr/bin/bazelisk

# Install and setup gcc
RUN apt-get -y install build-essential gcc autoconf libomp-dev

# Install and setup python$PYV.
RUN apt -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update
RUN apt-get -y install -f python3.8-dev python3.8-distutils python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Clean up.
RUN apt-get clean

RUN apt-get -y install openssh-server
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
ENTRYPOINT service ssh start && bash
