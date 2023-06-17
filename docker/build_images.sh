#/bin/bash

docker build - < DockerfileUbuntuPY38 -t nevron/builder_ubuntu_x86_64_py38
docker build - < DockerfileUbuntuPY39 -t nevron/builder_ubuntu_x86_64_py39
docker build - < DockerfileUbuntuPY310 -t nevron/builder_ubuntu_x86_64_py310