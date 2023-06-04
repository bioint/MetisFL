import io
import os
import platform

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("python_example",
        ["pybind/controller/controller_pybind.cc"],
        ),
]
requirements = [
]

requirements_extra_tf = [
    "tensorflow",
    "tensorflow_datasets",
    "tensorflow_federated",
]

setup(
    name="metisfl",
    version="0.0.1",
    author="Nevron.AI",
    author_email="hello@nevron.ai",
    description="A developer-friendly and enterprise-ready federated learning platform.", 
    long_description=io.open(os.path.join("README.md"), "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NevronAI",
    keywords=[
        "federated learning",
        "privacy-presercing machine learning",
        "natural language processing",
        "computer vision",
    ],
    classifiers=[
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    packages=find_packages(),
    include_package_data=True,
    data_files=[
        # NOTE: figure out
    ],
    install_requires=requirements,
    extras_require={
        "gRPC": "grpcio",
        "tensorflow": requirements_extra_tf,
    },
    package_data={"": ["py.typed"]},
    license="Apache 2.0",
    
    # NOTE: figure out
    entry_points={
        "console_scripts": [
            
        ]
    }
)