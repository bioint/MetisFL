#!/bin/bash

BAZEL_CMD=bazelisk
CONDA_CMD=conda
BAZEL_BIN_DIR=bazel-bin
OUTPUT_DIR=.build
CONDA_ENV_BASE_PATH=/opt/conda/envs
PYTHON_VESIONS=("3.10")

generate_wheel(){
    PY_VERSION=$1
    ENV_NAME="py${PY_VERSION//"."/""}"
    ENV_PATH=$CONDA_ENV_BASE_PATH/$ENV_NAME
    if [ ! -d $ENV_PATH ]; then
        echo "Conda environment for Python $PY_VERSION not found. Exiting."
        exit
    fi
    
    if [ ! -d $OUTPUT_DIR ]; then
        mkdir $OUTPUT_DIR
    fi
    
    # FIXME: (dstripelis, pkyriakis). MAJOR ISSUE with python version.
    # The pybind_bazel extension is used to configure the python env used for compilation.
    # Look here  https://github.com/pybind/pybind11_bazel/blob/master/python_configure.bzl
    # The controller is build using bazel and the python env can be set using PYTHON_BIN_PATH and PYTHON_LIB_PATH
    # However, the FHE is build using cmake and it's not clear which python version is used and how to change it
    
    PYTHON_BIN_PATH=$ENV_PATH/bin/python
    PYTHON_LIB_PATH=$ENV_PATH/lib
    # $BAZEL_CMD clean --expunge
    $BAZEL_CMD build //:metisfl-wheel \
    --incompatible_strict_action_env=true \
    --action_env=BAZEL_CXXOPTS=-std=c++17 \
    --action_env=PYTHON_EXECUTABLE=$PYTHON_BIN_PATH \
    --action_env=PYTHON_BIN_PATH=$PYTHON_BIN_PATH \
    --action_env=PYTHON_LIB_PATH=$PYTHON_LIB_PATH \
    --define python=$PY_VERSION``
    cp -f $BAZEL_BIN_DIR/*.whl $OUTPUT_DIR``
    
}

build_dev() {
    BAZEL_BIN_DIR="$(pwd)/bazel-bin"
    BAZEL_PROTO_DIR=$BAZEL_BIN_DIR/metisfl/proto/python_grpc_srcs/metisfl/proto
    
    export PYTHONPATH=$(pwd)
    
    $BAZEL_CMD build //metisfl/controller:controller.so
    $BAZEL_CMD build //metisfl/encryption:fhe.so
    $BAZEL_CMD build //metisfl/proto:metisfl_cc_proto_lib
    
    cp -f $BAZEL_BIN_DIR/metisfl/controller/controller.so metisfl/controller/controller.so
    cp -f "$BAZEL_BIN_DIR/metisfl/encryption/fhe.so" metisfl/encryption/fhe.so
    cp -f $BAZEL_PROTO_DIR/*.py metisfl/proto
}

build_prod() {
    for version in "${PYTHON_VESIONS[@]}"
    do
        generate_wheel $version
    done
    
}

if [ "$1" == "dev" ]; then
    build_dev
    elif [ "$1" == "prod" ]; then
    build_prod
else
    echo "Must specify target; "dev" or "prod""
fi