#!/bin/bash
IMAGE_NAME="${1:-nevron/ubuntu_focal_x86_64_py38}"

 IMG_LS=$(docker images | awk '$1 ~ /"$IMAGE_NAME"/ { print $1 }')
nevron/ubuntu_focal_x86_64_py38
if [[ $IMG_LS == "" ]]; then
  echo "Image $IMAGE_NAME not found. Pulling.."
  docker pull $IMAGE_NAME
fi

CID=$(docker run -dit -v .:/metisfl nevron/ubuntu_focal_x86_64_py38)

echo "Waiting for container to start..."
until [ "`docker inspect -f {{.State.Running}} $CID`"=="true" ]; do
    sleep 0.1;
done;
echo "Container started. Building"

# Run build
docker exec -it $CID /bin/bash -c /metisfl/build.sh

# Stop container
docker stop $CID
#docker rm $CID

