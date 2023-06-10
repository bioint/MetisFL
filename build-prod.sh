#!/bin/bash          

BAZEL_CMD=bazel-4.2.1
$BAZEL_CMD build //:metisfl-wheel
cp -f bazel-bin/*.whl .