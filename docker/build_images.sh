#/bin/bash

docker build - < DockerfileUbuntuPY38 -t nevron/ubuntu_focal_x86_64_py38
# docker build - < DockerfileUbuntuPY39 -t nevron/ubuntu_focal_x86_64_py39
# docker build - < DockerfileUbuntuPY310 -t nevron/ubuntu_focal_x86_64_py310

# docker tag nevron/ubuntu_focal_x86_64_py38 us-west2-docker.pkg.dev/nevron-385600/builders/ubuntu_focal_x86_64_py38
# docker tag nevron/ubuntu_focal_x86_64_py39 us-west2-docker.pkg.dev/nevron-385600/builders/ubuntu_focal_x86_64_py39
# docker tag nevron/ubuntu_focal_x86_64_py310 us-west2-docker.pkg.dev/nevron-385600/builders/ubuntu_focal_x86_64_py310

# docker push us-west2-docker.pkg.dev/nevron-385600/builders/ubuntu_focal_x86_64_py38
# docker push us-west2-docker.pkg.dev/nevron-385600/builders/ubuntu_focal_x86_64_py39
# docker push us-west2-docker.pkg.dev/nevron-385600/builders/ubuntu_focal_x86_64_py310
