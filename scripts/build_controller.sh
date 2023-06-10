#!/bin/bash
bazel query //... | grep -i -e "//metisfl/controller" | xargs bazel build