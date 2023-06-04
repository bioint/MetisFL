workspace(name = "metisfl")

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
  sha256 = "3700ef34597cda8e9fa4a43a6f177bec87d6d07fb400f3dfdf636a71332282e1",
  urls = ["https://github.com/raschild/pybind11_bazel/archive/refs/heads/main.zip"],
  strip_prefix = "pybind11_bazel-main",
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
    remote = "https://gitlab.com/palisade/palisade-release.git",
    # tag: v1.11.7
    commit = "19e84ae53ecdd1ca3360c4cdbb6df2eb31f21195",
    strip_prefix = "",
    init_submodules = True,
    shallow_since = "1651346647 +0000",
    verbose = True,
    recursive_init_submodules = True,
    build_file_content = """
filegroup(
    name = "palisade_srcs",
    srcs = glob(["**/*"]),
    visibility = ["//visibility:public"],
)
""",
)


# Imports SHELFI-FHE library.
new_git_repository(
    name = "shelfi_fhe_git",
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

http_archive(
    name = "rules_foreign_cc",
    sha256 = "6041f1374ff32ba711564374ad8e007aef77f71561a7ce784123b9b4b88614fc",
    strip_prefix = "rules_foreign_cc-0.8.0",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.8.0.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
# This sets up some common toolchains for building targets. For more details, please see
# https://bazelbuild.github.io/rules_foreign_cc/0.8.0/flatten.html#rules_foreign_cc_dependencies
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

# Use Google Logging (glog) to implement application-level logging at the controller.
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    url = "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz",
)

http_archive(
    name = "com_github_google_glog",
    sha256 = "122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022",
    strip_prefix = "glog-0.6.0",
    url = "https://github.com/google/glog/archive/v0.6.0.zip",
)

# Imports Cryptopp library.
new_git_repository(
    name = "cryptopp",
    commit = "59895be61b03f973416724cfe0274693ce73b6a1",
    shallow_since = "1649218442 -0400",
    remote = "https://github.com/weidai11/cryptopp.git",
    verbose = True,
    build_file_content = """
cc_library(
    name = "cryptopp",
    visibility = ['//visibility:public'],
    hdrs = glob([
        '*.h',
        '*.cpp',
      ]),
    srcs = [
      '3way.cpp',
      'adler32.cpp',
      'algebra.cpp',
      'algparam.cpp',
      'arc4.cpp',
      'asn.cpp',
      'authenc.cpp',
      'base32.cpp',
      'base64.cpp',
      'basecode.cpp',
      'bfinit.cpp',
      'blowfish.cpp',
      'blumshub.cpp',
      'camellia.cpp',
      'cast.cpp',
      'casts.cpp',
      'cbcmac.cpp',
      'ccm.cpp',
      'channels.cpp',
      'cmac.cpp',
      'cpu.cpp',
      'crc.cpp',
      'cryptlib.cpp',
      'default.cpp',
      'des.cpp',
      'dessp.cpp',
      'dh2.cpp',
      'dh.cpp',
      'dll.cpp',
      'dsa.cpp',
      'eax.cpp',
      'ec2n.cpp',
      'eccrypto.cpp',
      'ecp.cpp',
      'elgamal.cpp',
      'emsa2.cpp',
      'eprecomp.cpp',
      'esign.cpp',
      'files.cpp',
      'filters.cpp',
      'fips140.cpp',
      'fipstest.cpp',
      'gcm.cpp',
      'gf2_32.cpp',
      'gf256.cpp',
      'gf2n.cpp',
      'gfpcrypt.cpp',
      'gost.cpp',
      'gzip.cpp',
      'hex.cpp',
      'hmac.cpp',
      'hrtimer.cpp',
      'ida.cpp',
      'idea.cpp',
      'integer.cpp',
      'iterhash.cpp',
      'luc.cpp',
      'mars.cpp',
      'marss.cpp',
      'md2.cpp',
      'md4.cpp',
      'md5.cpp',
      'misc.cpp',
      'modes.cpp',
      'mqueue.cpp',
      'mqv.cpp',
      'nbtheory.cpp',
      'oaep.cpp',
      'osrng.cpp',
      'panama.cpp',
      'pch.cpp',
      'pkcspad.cpp',
      'polynomi.cpp',
      'pssr.cpp',
      'pubkey.cpp',
      'queue.cpp',
      'rabin.cpp',
      'randpool.cpp',
      'rc2.cpp',
      'rc5.cpp',
      'rc6.cpp',
      'rdtables.cpp',
      'rijndael.cpp',
      'ripemd.cpp',
      'rng.cpp',
      'rsa.cpp',
      'rw.cpp',
      'safer.cpp',
      'salsa.cpp',
      'seal.cpp',
      'seed.cpp',
      'serpent.cpp',
      'sha3.cpp',
      'shacal2.cpp',
      'sha.cpp',
      'sharkbox.cpp',
      'shark.cpp',
      'simple.cpp',
      'skipjack.cpp',
      'sosemanuk.cpp',
      'square.cpp',
      'squaretb.cpp',
      'strciphr.cpp',
      'tea.cpp',
      'tftables.cpp',
      'tiger.cpp',
      'tigertab.cpp',
      'ttmac.cpp',
      'twofish.cpp',
      'vmac.cpp',
      'wake.cpp',
      'whrlpool.cpp',
      'xtr.cpp',
      'xtrcrypt.cpp',
      'zdeflate.cpp',
      'zinflate.cpp',
      'zlib.cpp'
    ],
    linkopts = [
      '-lm',
    ]
)
""",
)

http_archive(
    name = "hiredis_git",
    sha256 = "e0ab696e2f07deb4252dda45b703d09854e53b9703c7d52182ce5a22616c3819",
    strip_prefix = "hiredis-1.0.2",
    url = "https://github.com/redis/hiredis/archive/refs/tags/v1.0.2.tar.gz",
    build_file_content =
"""
# The build file's content is derived from
# https://github.com/ray-project/ray/blob/master/bazel/BUILD.hiredis
# This library is for internal hiredis use, because hiredis assumes a
# different include prefix for itself than external libraries do.
cc_library(
    name = "_hiredis",
    hdrs = [
        "dict.c",
    ],
)

cc_library(
    name = "hiredis",
    srcs = glob(
        [
            "*.c",
            "*.h",
        ],
        exclude =
        [
            "ssl.c",
            "test.c",
        ],
    ),
    hdrs = glob([
        "*.h",
        "adapters/*.h",
    ]),
    include_prefix = "hiredis",
    deps = [
        ":_hiredis",
    ],
    visibility = ["//visibility:public"],
)
""",
)
