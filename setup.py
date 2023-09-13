
""" Builds MetisFL wheel package. """

import os
import shutil
import site
import sys

from setuptools import setup, find_packages


os.environ["PYTHON_BIN_PATH"] = sys.executable
os.environ["PYTHON_LIB_PATH"] = site.getsitepackages()[0]

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
BAZEL_CMD = "bazelisk"
CONTROLER_SO_TARGET = "//metisfl/controller/core:controller.so"
CONTROLER_SO_SRC = "bazel-bin/metisfl/controller/core/controller.so"
CONTROLER_SO_DST = "metisfl/controller/controller.so"
FHE_SO_TARGET = "//metisfl/encryption/palisade:fhe.so"
FHE_SO_SRC = "bazel-bin/metisfl/encryption/palisade/fhe.so"
FHE_SO_DST = "metisfl/encryption/fhe.so"
PY_VERSIONS = ["3.8", "3.9", "3.10"]


def get_python_version():
    """Returns the current Python version."""
    return ".".join(map(str, [sys.version_info.major, sys.version_info.minor]))


def copy_helper(src_path, dst):
    """Copies a file to the given destination. If the destination is a directory, the file is copied into it."""
    if os.path.isdir(dst):
        fname = os.path.basename(src_path)
        dst = os.path.join(dst, fname)

    if os.path.isfile(dst):
        os.remove(dst)

    shutil.copy(src_path, dst)


def check_env():
    """Checks python version and PYTHON_LIB_PATH environment variable."""
    py_version = get_python_version()
    if py_version not in PY_VERSIONS:
        raise ValueError(
            "Python version {} is not supported. Supported versions are: {}".format(
                py_version, ", ".join(PY_VERSIONS)
            )
        )
    lib_path = os.environ["PYTHON_LIB_PATH"]
    if not os.path.isdir(lib_path):
        raise ValueError("PYTHON_LIB_PATH {} does not exist".format(lib_path))


def run_build():
    """Builds MetisFL C++ dependancies for the given Python version."""

    # Check environment
    check_env()

    # Build controller and encryption .so
    os.system("{} build {}".format(BAZEL_CMD, CONTROLER_SO_TARGET))
    os.system("{} build {}".format(BAZEL_CMD, FHE_SO_TARGET))

    # Compile proto files
    os.system(f"{sys.executable} {ROOT_DIR}/metisfl/proto/protoc.py")

    # Copy .so files
    copy_helper(CONTROLER_SO_SRC, CONTROLER_SO_DST)
    copy_helper(FHE_SO_SRC, FHE_SO_DST)


# Run build
run_build()

# TODO: add py version and arch to wheel name
setup(
    name="metisfl",
    version="0.1.0",
    description="MetisFL: The developer-friendly federated learning framework",
    author="MetisFL Team",
    author_email="hello@nevron.ai",
    url="https://github.com/nevronai/metisfl",
    classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: The Clear BSD License",
            "Operating System :: UNIX",
            "Topic :: Software Development :: Testing",
            "Topic :: Software Development :: Libraries",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: C++"
    ],
    packages=find_packages(),
    include_package_data=True,
    data_files=[
        ("metisfl/controller", ["metisfl/controller/controller.so"]),
        ("metisfl/encryption", ["metisfl/encryption/fhe.so"]),
    ],
    install_requires=[
        "Pebble>=5.0.3",
        "PyYAML>=6.0",
        "pandas>=1.3.2",
        "protobuf>=4.23.4",
        "termcolor>=2.3.0",
        "pyfiglet>=0.8.post1",
        "loguru>=0.7.1",
    ],
)
