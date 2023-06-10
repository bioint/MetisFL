#!/bin/bash          

BAZEL_CMD=bazel-4.2.1
VENV_LIB_DIR="$(pwd)/.venv/lib/python3.8/site-packages"
BAZEL_BIN_DIR="$(pwd)/bazel-bin"
BAZEL_PROTO_DIR=$BAZEL_BIN_DIR/metisfl/proto/metisfl_py_proto_lib_pb/metisfl/proto
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$BAZEL_BIN_DIR/metisfl/encryption/palisade/palisade_cmake/lib:$LD_LIBRARY_PATH

if [ ! -d $VENV_LIB_DIR ]; then
  echo "Python virtual env not found. Exiting."
  exit
fi

$BAZEL_CMD build //metisfl/pybind/controller:controller.so
$BAZEL_CMD build //metisfl/pybind/fhe:fhe.so
$BAZEL_CMD build //metisfl/proto:metisfl_py_proto_lib

cp -f "$BAZEL_BIN_DIR/metisfl/pybind/controller/controller.so" "$VENV_LIB_DIR/controller.so"
cp -f "$BAZEL_BIN_DIR/metisfl/pybind/fhe/fhe.so" "$VENV_LIB_DIR/fhe.so"

cp -f $BAZEL_BIN_DIR/metisfl/pybind/controller/controller.so metisfl/pybind/controller/controller.so
cp -f "$BAZEL_BIN_DIR/metisfl/pybind/fhe/fhe.so" metisfl/pybind/fhe/fhe.so

cp -f $BAZEL_PROTO_DIR/*.py metisfl/proto

