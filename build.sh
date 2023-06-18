#!/bin/bash

BAZEL_CMD=bazelisk
BAZEL_BIN_DIR="$(pwd)/bazel-bin"
BAZEL_PROTO_DIR=$BAZEL_BIN_DIR/metisfl/proto/py_grpc_src/metisfl/proto

PY_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ $PY_VERSION != "3.8" ]] && [[ $PY_VERSION != "3.9" ]] && [[ $PY_VERSION != "3.10" ]]; then 
    echo "Requires Python >= 3.8 and <=3.10. Found Python $PY_VERSION"
    exit 
fi

$BAZEL_CMD build //metisfl/controller:controller.so
$BAZEL_CMD build //metisfl/encryption:fhe.so
$BAZEL_CMD build //metisfl/proto:py_grpc_src

cp -f $BAZEL_BIN_DIR/metisfl/controller/controller.so metisfl/controller/controller.so
cp -f $BAZEL_BIN_DIR/metisfl/encryption/fhe.so metisfl/encryption/fhe.so
cp -f $BAZEL_PROTO_DIR/*.py metisfl/proto

mkdir metisfl/examples
mkdir metisfl/resources
cp -r examples/* metisfl/examples/
cp -r resources/* metisfl/resources/

$BAZEL_CMD build //:metisfl-wheel --define python=$PY_VERSION
[[ -d build ]] || mkdir build
cp -f $BAZEL_BIN_DIR/*.whl build

rm -fr metisfl/examples
rm -fr metisfl/resources
