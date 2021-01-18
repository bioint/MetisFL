#! /bin/bash -l

PROJECT_HOME=/lfs1/stripeli/condaprojects/projectmetis/projectmetis
cd $PROJECT_HOME

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda/extras/CUPTI/lib64/
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:/opt/anaconda3/condabin:/opt/anaconda3/bin/:/opt/anaconda2/bin/:/opt/cmake-3.7/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/nas/home/stripeli/.local/bin:/nas/home/stripeli/bin:/lfs1/prisms/golang/go/bin
export PYTHONPATH=.

# This is the main script for testing ResNet Model
WITH_CACHING_LAYER_ARR=("True")
RESNET_NUMBER_OF_LAYERS_ARR=(200)
NUMBER_OF_CLIENTS_ARR=(750 1000)
NUMBER_OF_UPDATE_REQUESTS=11
for index_1 in ${!WITH_CACHING_LAYER_ARR[@]}; do
    for index_2 in ${!RESNET_NUMBER_OF_LAYERS_ARR[@]}; do
        for index_3 in ${!NUMBER_OF_CLIENTS_ARR[@]}; do
            export WITH_CACHING_LAYER=${WITH_CACHING_LAYER_ARR[$index_1]}
            export RESNET_NUMBER_OF_LAYERS=${RESNET_NUMBER_OF_LAYERS_ARR[$index_2]}
            export NUMBER_OF_CLIENTS=${NUMBER_OF_CLIENTS_ARR[$index_3]}
            export NUMBER_OF_UPDATE_REQUESTS=${NUMBER_OF_UPDATE_REQUESTS}
            /lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/grpc_controller_caching_layer/fed_grpc_controller_caching_layer_resnet_cifar_test.py
        done
    done
done