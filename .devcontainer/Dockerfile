# Install Ubuntu image from docker hub.
FROM --platform=linux/amd64 python:3.10

# Set environmental variables.
ENV PROJECT_HOME=/metisfl
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
ENV PYTHONPATH=/workspaces/metisfl

WORKDIR /metisfl

# Simple update.
RUN apt-get -y update

# Install helper packages and bazelisk (i.e., bazel versioning resolution tool at runtime).
RUN apt-get -y install wget git nano 
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64
RUN chmod +x ./bazelisk-linux-amd64
RUN mv ./bazelisk-linux-amd64 /usr/bin/bazelisk

# Install and setup gcc
RUN apt-get -y install build-essential gcc autoconf libomp-dev

# Clean up.
RUN apt-get clean
