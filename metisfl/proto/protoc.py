"""This module contains functions related to proto compilation."""

import glob
from os import path

import grpc_tools
from grpc_tools import protoc

GRPC_PATH = grpc_tools.__path__[0]

DIR_PATH = path.dirname(path.realpath(__file__))
PATH = path.normpath(path.join(DIR_PATH, "..", ".."))
PROTO_FILES = glob.glob(f"{DIR_PATH}/*.proto")


def compile() -> None:
    """Compile all protos in the `metisfl/proto` directory."""

    command = [
        "grpc_tools.protoc",
        f"--proto_path={GRPC_PATH}/_proto",
        f"--proto_path={PATH}",
        f"--python_out={PATH}",
        f"--mypy_out={PATH}",
        f"--mypy_grpc_out={PATH}",
        f"--grpc_python_out={PATH}",
    ] + PROTO_FILES

    exit_code = protoc.main(command)

    if exit_code != 0:
        raise Exception(f"Error: {command} failed")


if __name__ == "__main__":
    compile()
