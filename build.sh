#!/bin/bash

BAZEL_CMD=bazel
PYTHON_DEV_VERSION=3.8

build_dev() {
    BAZEL_BIN_DIR="$(pwd)/bazel-bin"
    BAZEL_PROTO_DIR=$BAZEL_BIN_DIR/metisfl/proto/metisfl_py_proto_lib_pb/metisfl/proto
    # export PYTHONPATH=$(pwd)
    # export LD_LIBRARY_PATH=$BAZEL_BIN_DIR/metisfl/encryption/core/palisade_cmake/lib:$LD_LIBRARY_PATH
    
    $BAZEL_CMD build //metisfl/controller:controller.so
    $BAZEL_CMD build //metisfl/encryption:fhe.so
    $BAZEL_CMD build //metisfl/proto:metisfl_py_proto_lib
    
    cp -f $BAZEL_BIN_DIR/metisfl/controller/controller.so metisfl/controller/controller.so
    cp -f "$BAZEL_BIN_DIR/metisfl/encryption/fhe.so" metisfl/encryption/fhe.so
    cp -f $BAZEL_PROTO_DIR/*.py metisfl/proto
}

build_prod() {
    $BAZEL_CMD build //:metisfl-wheel
    cp -f bazel-bin/*.whl .
}

if [ "$1" == "dev" ]; then
    build_dev
    elif [ "$1" == "prod" ]; then
    build_prod
else
    echo "Must specify target; "dev" or "prod""
fi
