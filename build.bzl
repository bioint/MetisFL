load("@rules_python//python:packaging.bzl", "py_wheel")

METISFL_VERSION = "1.0.0"

def _wheel_build_wrapper(ctx):
    tag = ctx.configuration.default_shell_env["PYTHON_TAG"]
    py_wheel(
        name = "metisfl-wheels",
        classifiers = [
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: UNIX",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: MacOS :: MacOS X",
            "Topic :: Software Development :: Testing",
            "Topic :: Software Development :: Libraries",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: C++"
        ],
        description_file = "README.md",
        distribution = "metisfl",
        homepage = "https://www.nevron.ai",
        license = "Apache 2.0",
        python_requires = ">=3.8",
        python_tag = tag,
        requires = [
            "cloudpickle",
            "fabric",
            "future",
            "grpcio",
            "matplotlib",
            "nibabel",
            "numpy",
            "pandas",
            "Pebble",
            "protobuf",
            "psycopg-binary",
            "pycparser",
            "pyfiglet",
            "pyparsing",
            "torch",
            "PyYAML",
            "requests",
            "scikit-learn",
            "scipy",
            "simplejson",
            "six",
            "SQLAlchemy",
            "tensorboard",
            "tensorflow",
            "tensorflow-addons",
            "termcolor",
            "threadpoolctl"
        ],
        version = METISFL_VERSION,
        visibility = ["//visibility:public"],
        deps = [
            ":metisfl-pkg",
        ]
    )

metis_build_wheel = rule(
    implementation=_wheel_build_wrapper,
)
