#!/bin/bash

docker pull nevron/builder_ubuntu_x86_64_py39
CID=$(docker run -dit -v .:/metisfl nevron/builder_ubuntu_x86_64_py39)

# Run build
docker exec -it $CID /bin/bash -c build.sh

# Copy wheel to parent root
WHEEL_NAME=$(docker exec -it c05d90e844ae /bin/bash -c "ls | grep .whl")
docker cp $CID:/metisfl/$WHEEL_NAME .

# Stop and remove container
docker stop $CID
docker rm $CID

