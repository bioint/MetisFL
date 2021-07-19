# Toolchain setup.
# Demonstrate that we can use the same python interpreter for the toolchain and
# executing pip in pip install (see WORKSPACE).
load("@rules_python//python:defs.bzl", "py_runtime_pair")

py_runtime(
    name = "python3_runtime",
    files = ["@metis_python//:files"],
    interpreter = "@metis_python//:bin/python",
    python_version = "PY3",
    visibility = ["//visibility:public"],
)

py_runtime_pair(
    name = "my_py_runtime_pair",
    py2_runtime = None,
    py3_runtime = ":python3_runtime",
)

toolchain(
    name = "my_py_toolchain",
    toolchain = ":my_py_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)
# End of toolchain setup.
