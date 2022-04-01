workspace(name = "projectmetis")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
# Imports grpc.
http_archive(
    name = "rules_proto_grpc",
    sha256 = "7954abbb6898830cd10ac9714fbcacf092299fda00ed2baf781172f545120419",
    strip_prefix = "rules_proto_grpc-3.1.1",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/3.1.1.tar.gz"],
)

load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_toolchains", "rules_proto_grpc_repos")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

load("@rules_proto_grpc//cpp:repositories.bzl", rules_proto_grpc_cpp_repos = "cpp_repos")
rules_proto_grpc_cpp_repos()

load("@rules_proto_grpc//python:repositories.bzl", rules_proto_grpc_py_repos = "python_repos")
rules_proto_grpc_py_repos()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

# Imports abseil.
http_archive(
    name = "absl",
    sha256 = "59b862f50e710277f8ede96f083a5bb8d7c9595376146838b9580be90374ee1f",
    strip_prefix = "abseil-cpp-20210324.2",
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20210324.2.tar.gz"],
)

# Imports googletest.
http_archive(
    name = "gtest",
    sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
    strip_prefix = "googletest-release-1.10.0",
    urls = ["https://github.com/google/googletest/archive/release-1.10.0.tar.gz"],
)

# Imports PyBind11 Bazel plugin.
http_archive(
  name = "pybind11_bazel",
  sha256 = "3dc6435bd41c058453efe102995ef084d0a86b0176fd6a67a6b7100a2e9a940e",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/992381ced716ae12122360b0fbadbc3dda436dbf.zip"],
  strip_prefix = "pybind11_bazel-992381ced716ae12122360b0fbadbc3dda436dbf",
)

# Imports PyBind11 library.
http_archive(
  name = "pybind11",
  sha256 = "9ca7770fc5453b10b00a4a2f99754d7a29af8952330be5f5602e7c2635fa3e79",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.8.0",
  urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.8.0.tar.gz"],
)

# Imports Python rules.
http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.5.0/rules_python-0.5.0.tar.gz",
    sha256 = "cd6730ed53a002c56ce4e2f396ba3b3be262fd7cb68339f0377a45e8227fe332",
)

# Imports Conda rules.
http_archive(
    name = "rules_conda",
    sha256 = "9793f86162ec5cfb32a1f1f13f5bf776e2c06b243c4f1ee314b9ec870144220d",
    url = "https://github.com/spietras/rules_conda/releases/download/0.1.0/rules_conda-0.1.0.zip"
)

load("@rules_conda//:defs.bzl", "conda_create", "load_conda", "register_toolchain")

load_conda(
    conda_version = "4.10.3",  # version of conda to download, default is 4.10.3
    installer = "miniconda",  # which conda installer to download, either miniconda or miniforge, default is miniconda
)

conda_create(
    name = "py3_env",
    environment = "//python:conda_env.yaml",
    quiet = False,
)

register_toolchain(py3_env = "py3_env")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
# Imports Palisades library.
new_git_repository(
    name = "palisade_git",
    commit = "fded3711c5855e968467e2e73ccf2fcd7948dc7d",
    shallow_since = "1622060746 -0400",
    remote = "https://gitlab.com/palisade/palisade-development.git",
    verbose = True,
    recursive_init_submodules = True,
    build_file_content = """
filegroup(
    name = "all",
    srcs = glob(["**/*"]),
    visibility = ["//visibility:public"],
)
""",
)

# Imports SHELFI-FHE library.
new_git_repository(
    name = "shelfi_fhe_git",
    #branch = "main",
    commit = "5cba05be07320b5a5a828a40bd148fcd59720c19",
    shallow_since = "1635215298 -0700",
    remote = "https://github.com/tanmayghai18/he-encryption-shelfi.git",
    verbose = True,
    build_file_content = """
filegroup(
    name = "all",
    srcs = glob(["**/*"]),
    visibility = ["//visibility:public"],
)
""",
)

# Imports foreign rules repository.
http_archive(
    name = "rules_foreign_cc",
    sha256 = "33a5690733c5cc2ede39cb62ebf89e751f2448e27f20c8b2fbbc7d136b166804",
    strip_prefix = "rules_foreign_cc-0.5.1",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.5.1.tar.gz"
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

# This sets up some common toolchains for building targets. For more details, please see
# https://bazelbuild.github.io/rules_foreign_cc/0.4.0/flatten.html#rules_foreign_cc_dependencies
rules_foreign_cc_dependencies()

# Add boost library rule for future development.
#http_archive(
#    name = "boost",
#    build_file_content = """
#filegroup(
#    name = "all",
#    srcs = glob(["**"]),
#    visibility = ["//visibility:public"])
#""",
#    strip_prefix = "boost_1_78_0",
#    sha256 = "94ced8b72956591c4775ae2207a9763d3600b30d9d7446562c552f0a14a63be7",
#    urls = ["https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz"],
#)
