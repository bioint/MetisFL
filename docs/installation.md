## Project Installation Steps - Docker
Due to some library inconsistencies that appeared across operating systems (e.g., Centos vs MacOS) we concluded that we
should build a docker image and run the entire project within a container. The Dockerfile(s) contain all required steps.

System prerequisites:

1. python
2. docker

To compile and run the project through docker, navigate to the parent directory of the project and then:

1. Run `chmod +x ./configure.sh && ./configure.sh` to configure metis fl project.

   Note: we run the above command before building the docker image because to configure all project dependencies.

2. Build docker image for the entire project. If the server hosting the docker container has GPUs, then we need to also enable the CUDA GPU environment. To do this, we need to also pass as argument the following during build: `--build-arg ENV_CONDA_CUDA_ENABLED=0`
    - Ubuntu Dev image (development purposes): `docker build -t projectmetis_dev -f DockerfileDev .`
    - Ubuntu Dev image + CONDA CUDA (development purposes): `docker build -t projectmetis_dev --build-arg ENV_CONDA_CUDA_ENABLED=1 -f DockerfileDev .`
    - Ubuntu image (stable, preferable): `docker build -t projectmetis_ubuntu_22_04 -f DockerfileUbuntu .`
    - Ubuntu image + CONDA CUDA (stable, preferable): `docker build -t projectmetis_ubuntu_22_04 --build-arg ENV_CONDA_CUDA_ENABLED=1 -f DockerfileUbuntu .`
    - RockyLinux image (not stable): `docker build -t projectmetis_rockylinux_8 -f DockerfileRockyLinux .`
      Approximate size for any of the following images (using docker): ~9GB (without CUDA), ~12GB (with CUDA)

3. Build docker CUDA image (only applicable to Ubuntu and RockyLinux images). Careful in the image name used by the FROM clause in the CUDA Dockerfile.
    - Ubuntu + CUDA `cd docker_images/cuda/ubuntu/11.7 && docker build -t projectmetis_ubuntu_22_04_cuda -f Dockerfile .`
    - RockyLinux + CUDA `cd docker_images/cuda/rockylinux/11.3 && docker build -t projectmetis_rockylinux_8_cuda -f Dockerfile .`
    - Verify docker cuda driver installation as: `nvidia-docker run --rm --gpus all projectmetis_ubuntu_22_04_cuda nvidia-smi`

## Installing Redis As A Backend Model Store.
- In the host server need to execute `docker run -d --name redis-svr -p 6379:6379 redis:latest` to allow Controller to
  connect to Redis Backend.

## Standalone (Docker-Free) Prerequisites
- Install googletest (MacOS as `brew install googletest`)
- Install protobuf (MacOS as `brew install protobuf`)
- Run ./configure script

## Bazel CLion comments
If project files are not identifiable then you need to sync Bazel. To do so:

1. select the Bazel tab above
2. select the Sync subtab
3. and then Sync Project with BUILD Files
