#!/bin/bash          

BAZEL_CMD=bazel-4.2.1
VENV_LIB_DIR="$(pwd)/.venv/lib/python3.8/site-packages"
BAZEL_BIN_DIR="$(pwd)/bazel-bin"
BAZEL_PROTO_DIR=$BAZEL_BIN_DIR/src/proto/metisfl_py_proto_lib_pb/src/proto

if [ ! -d $VENV_LIB_DIR ]; then
  echo "Python virtual env not found. Exiting."
fi

$BAZEL_CMD build //src/pybind/controller:controller.so
$BAZEL_CMD build //src/pybind/fhe:fhe.so
$BAZEL_CMD build //src/proto:metisfl_py_proto_lib

cp -f "$BAZEL_BIN_DIR/src/pybind/controller/controller.so" "$VENV_LIB_DIR/controller.so"
cp -f "$BAZEL_BIN_DIR/src/pybind/fhe/fhe.so" "$VENV_LIB_DIR/fhe.so"
cp -f $BAZEL_PROTO_DIR/*2.py src/proto
