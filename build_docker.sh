#!/bin/bash

CID=$(docker run -dit -v .:/metisfl nevron/builder_ubuntu_x86_64_py39)

# Run build
docker exec -it $CID /bin/bash -c build.sh

# Stop and remove container
docker stop $CID
docker rm $CID

