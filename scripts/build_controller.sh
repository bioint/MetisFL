#!/bin/bash
bazel query //... | grep -i -e "//src/cc/controller" | xargs bazel build