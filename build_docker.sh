#!/bin/bash
IMAGE_NAME=$1 || "ubuntu_focal_x86_64_py38"

IMG_LS=$(docker image list | grep $IMAGE_NAME) 
if [[ $IMG_LS == "" ]]; then
  echo "Image $IMAGE_NAME not found. Pulling.."
  docker pull $IMAGE_NAME
fi

CID=$(docker run -dit -v .:/metisfl nevron/$IMAGE_NAME)

echo "Waiting for container to start..."
until [ "`docker inspect -f {{.State.Running}} $CID`"=="true" ]; do
    sleep 0.1;
done;
echo "Container started. Building"

# Run build
docker exec -it $CID /bin/bash -c /metisfl/build.sh

# Stop and remove container
docker stop $CID
docker rm $CID

