&nbsp;
<div align="center">

# MetisFL: The Open Federated Learning Framework for Scalable, Efficient, and Secure Federated Learning Workflows

MetisFL - The First Open Federated Learning Framework implemented in C++ and Python3.

[![BSD-3 License](https://badgen.net/badge/License/BSD-3-Clause/green?icon=github)](https://github.com/NevronAI/MetisFL/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/metisfl/)
<!-- [![Documentation](https://badgen.net/badge/Read/Documentation/orange?icon=buymeacoffee)](https://docs.metisfl.ai)
[![Blog](https://badgen.net/badge/Mentions/Blog?icon=awesome)](https://blog.metisfl.ai)
[![Slack Community](https://img.shields.io/badge/JoinSlack-@metisfl-brightgreen.svg?logo=slack)](https://join.slack.com/t/metisfl/shared_invite/zt-233d3rg4x-9HNnRloTkyEh8_XPch9mfQ) -->
[![Citation](https://img.shields.io/badge/cite-citation-brightgreen)](https://arxiv.org/pdf/2205.05249.pdf)

</div>
&nbsp;

## Project Installation Steps - Docker
Due to some library inconsistencies that appeared across operating systems (e.g., Centos vs MacOS) we concluded that we
should build a docker image and run the entire project within a container. The Dockerfile(s) contain all required steps.

System prerequisites:

1. python
2. docker

To compile and run the project through docker, navigate to the parent directory of the project and then:

1. Run `chmod +x ./configure.sh && ./configure.sh` to configure metis fl project.
   
   Note: we run the above command before building the docker image because to configure all project dependencies.

2. Build docker image for the entire project.
   - Ubuntu image (stable, preferable): `docker build -t projectmetis_ubuntu_22_04 -f DockerfileUbuntu .`
   - Ubuntu Dev image (development purposes): `docker build -t projectmetis_dev -f DockerfileDev .` 
   - RockyLinux image (not stable): `docker build -t projectmetis_rockylinux_8 -f DockerfileRockyLinux .`
   Approximate size for any of the following images (using docker): ~9GB (without CUDA), ~12GB (with CUDA)
   
4. Build docker CUDA image (only applicable to Ubuntu and RockyLinux images).
   - Ubuntu + CUDA `cd docker_images/cuda/ubuntu/11.7 && docker build -t projectmetis_ubuntu_22_04_cuda -f Dockerfile .`
   - RockyLinux + CUDA `cd docker_images/cuda/rockylinux/11.3 && docker build -t projectmetis_rockylinux_8_cuda -f Dockerfile .`
   - Verify docker cuda driver installation as: `nvidia-docker run --rm --gpus all projectmetis_ubuntu_22_04_cuda nvidia-smi`

## Standalone (Docker-Free) Prerequisites
- Install googletest (MacOS as `brew install googletest`)
- Install protobuf (MacOS as `brew install protobuf`)
- Run ./configure script 

## Bazel CLion comments 
If project files are not identifiable then you need to sync Bazel. To do so:

1. select the Bazel tab above
2. select the Sync subtab
3. and then Sync Project with BUILD Files

## Trello UI
https://trello.com/b/bYLUYqGK/metis-v01

