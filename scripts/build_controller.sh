#!/bin/bash
bazel query //... | grep -i -e "//projectmetis/controller" | xargs bazel build