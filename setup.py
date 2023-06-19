import argparse
import sys
import os
import shutil
import glob
import docker

BAZEL_CMD = "bazelisk"
BUILD_DIR = "build"
CONTROLER_SO_DST = "metisfl/controller/controller.so"
CONTROLER_SO_SRC = "bazel-bin/metisfl/controller/controller.so"
CONTROLER_SO_TARGET = "//metisfl/controller:controller.so"
DOCKER_REPO = "us-west2-docker.pkg.dev/nevron-385600/builders/ubuntu_focal_x86_64_py38"
FHE_SO_DST = "metisfl/encryption/fhe.so"
FHE_SO_SRC = "bazel-bin/metisfl/encryption/fhe.so"
FHE_SO_TARGET = "//metisfl/encryption:fhe.so"
PROT_DST_DIR = "metisfl/proto"
PROTO_GRPC_TARGET = "//metisfl/proto:py_grpc_src"
PROTO_SRC_DIR = "bazel-bin/metisfl/proto/py_grpc_src/metisfl/proto/"
PY_VERSIONS = ["3.8", "3.9", "3.10"]


def run_build(python_verion):
    # Build targets
    os.system("{} build {}".format(BAZEL_CMD, CONTROLER_SO_TARGET))
    os.system("{} build {}".format(BAZEL_CMD, FHE_SO_TARGET))
    os.system("{} build {}".format(BAZEL_CMD, PROTO_GRPC_TARGET))

    # Copy .so
    copy_helper(CONTROLER_SO_SRC, CONTROLER_SO_DST)
    copy_helper(FHE_SO_SRC, FHE_SO_DST)

    # Copy proto/grpc classes
    for file in glob.glob("{}*.py".format(PROTO_SRC_DIR)):
        copy_helper(file, PROT_DST_DIR)

    # Build wheel
    os.system(
        "{bazel} build //:metisfl-wheel --define python={python}".format(
            bazel=BAZEL_CMD, python=python_verion
        )
    )

    # Copy wheel
    os.makedirs(BUILD_DIR, exist_ok=True)
    for file in glob.glob("bazel-bin/*.whl"):
        copy_helper(file, BUILD_DIR)


def copy_helper(src_file, dst):
    if os.path.isdir(dst):
        fname = os.path.basename(src_file)
        dst = os.path.join(dst, fname)

    if os.path.isfile(dst):
        os.remove(dst)
    shutil.copy(src_file, dst)


def build_for_host():
    py_version = ".".join(map(str, sys.version_info[:2]))
    if py_version not in PY_VERSIONS:
        print(
            "Detected Python {} in environment. Need {}".format(py_version, PY_VERSIONS)
        )
        exit(1)
    run_build(py_version)


def spawn_container(image_name):
    client = docker.from_env()
    image_url = os.path.join(DOCKER_REPO, image_name)
    client.images.pull(image_url)
    cwd = os.getcwd()
    container = client.containers.run(
        image=image_name, detach=True, volumes=[cwd, "/metisfl"]
    )
    container.exec_run('/bin/bash -c "python setup.py"')
    container.kill()
    container.remove()


def create_image(platform, python):
    image_name = get_image(platform, python)
    client = docker.from_env()
    image = client.images.build(
        path="docker",
        buildargs={"python": python},
        quiet=False,
        rm=True
    )
    image[0].tag(image_name)
    return image

def get_image(platform, python):
    image_name = get_image_name(platform, python)
    try:
        return docker.image.get(image_name)
    except:
        return None

def get_image_name(platform, python):
    return "{}_py{}".format(platform, python.replace(".", ""))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-P",
        "--platform",
        type=str,
        choices=["host", "ubuntu_x86_64"],
        default="host",
        help="Platform to build for.",
    )
    parser.add_argument(
        "-p",
        "--python",
        type=str,
        choices=["3.8", "3.9", "3.10"],
        default="3.10",
        help="Python version. If platform is 'host', will use host os python",
    )
    args = parser.parse_args()

    if args.platform == "host":
        build_for_host()
    else:
        create_image(args.platform, args.python)
