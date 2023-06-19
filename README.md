# MetisFL: The blazing-fast and developer-friendly federated learning framework.

<div align="center">
 <img 
    style="padding: 40px 0px"
    src="docs/img/logos/logo_png_01.png" width="400px">
</div>


MetisFL is a federated learning framework that allows developers to federate their machine learning workflows and train their models across distributed datasets without having to collect the data in a centralized location. Currently, the project is transitioning from a private, experimental version to a public, beta phase. We are actively encouraging developers, researchers and data scientists to experiment with the framework and contribute to the codebase. 


Homepage: https://nevron.ai/ \
Github: https://github.com/NevronAI \
Docs: https://docs.nevron.ai/ \
Slack: https://nevronai.slack.com 

MetisFL sprung up from the Information and Science Institute (ISI) in the University of Southern California (USC). It is backed by several years of Ph.D. and Post-Doctoral research and several publications in top-tier machine learning and system conferences. It is being developed with the following guiding principles in mind:

* **Scalability**: MetisFL is the only federated learning framework with the core controller infrastructure developed solely on C++. This allows for the system to scale and support up to 100K+ learners! 

* **Speed**: The core operations at the controller as well as the controller-learner communication overhead has been optimized for efficiency. This allows MetisFL to achieve improvements of up to 1000x on the federation round time compared to other federated learning frameworks. 

* **Efficiency and Flexibility**: MetisFL supports synchronous, semi-synchronous and asynchronous protocols. The different choices make our framework flexible enough to adapt to the needs of each use-case. Additionally, the support of fully asynchronous protocol makes MetisFL a highly efficient solution for use-cases with high heterogeneity on the compute/communication capabilities of the learners.

* **Strong Security**: MetisFL supports secure aggregations with fully homomorphic encryption using the [Palisade](https://gitlab.com/palisade/palisade-release) C++ cryptographic library.  This ensure that the weights of the produced models remain private and secure in transit. 


# Development Environment 

Developers interested in making core contributions should setup up their environment accordingly. There are currently 3 options for setting up your development environment: 

## Codespaces Container

## Local Container

## Host Machine 

If you want to develop on your host machine, you need to ensure that it satisfies the requirement and that all needed packages are installed. Currently, the development environment mentioned bellow has been tested on Ubuntu OS and for the x86_64 architecture. It should, however, work for different Linux-like OS on the same architecture. Support for different architectures is under development. The requirements for compiling and testing the code on your local machine are: 

* Bazel 
* Python 3.8 - 3.10
* Python header and distutils
* build-essential, autoconf and libomp-dev

The recommended way to install Bazel is to use the [Bazelisk](https://github.com/bazelbuild/bazelisk) launcher and place the executable somewhere in your PATH, i.e., `/usr/bin/bazel` or `/usr/bin/bazelisk`. Please make make sure that the name of the Bazelisk executable matches the BAZEL_CMD variable in `setup.py`. By default, the setup script will search for `bazelisk`. Bazelisk will automatically pick up the version from the `.bezelversion` file and then download and execute the corresponding Bazel executable. 

The Python headers and distutils are needed so that the C++ controller and encryption code can be compiled as Python modules. On Ubuntu, they can be install with the following command: 

```Bash
apt-get -y install python3.10-dev python3.10-distutils
```

Finally, the remaining requirements contain the compiler, autoconf and libomp (which is used for parallelization). Please make sure that they are available in your system by running:

```Bash
apt-get -y install build-essential autoconf libomp-dev
```

# Build Project

Clone the MetisFL Github repository:

```Bash
git clone https://github.com/NevronAI/metisfl.git
```
   
# Project Structure Overview
The project uses a unified codebase for both the Python and C++ code. The C++ modules, i.e., `controller` and `encryption`, are shipped with simple python binding to expose their functionality to python code. 

    .
    ├── docker                # Docker files for packaging MetisFL in a docker container       
    ├── examples              # Examples and use-cases for MetisF
    ├── metisfl               # Main source code directory
        ├── controller        # C++ implementation of the Controller/Aggregator
        ├── driver            # Python library for the MetisFL Driver
        ├── encryption        # C++ Palisade library
        ├── learner           # MetisFL Python Learner library 
        ├── models            # MetisFL Python Learner library         
        ├── proto             # Protobuf definitions for controller/learner/driver communication
        ├── resources         # FHE/SSL related resource files and keys
        ├── proto             # Protobuf definitions for controller/learner/driver communication
        ├── utils             # Utilities classes and functions      
    ├── test                  # Testing folder (under construction)
    ├── LICENSE               # License file
    ├── BUILD.bazel           # Bazel build file; contains main target definitions
    ├── setup.py              # Setup script; compiles and produces a Python Wheel
    ├── WORKSPACE             # Bazel workspace; contains external dependencies

## Controller 
The controller (or aggregator) is responsible for collecting the (encrypted) model weights from the learners, aggregating them, encrypting them and distributing the encrypted weights back to the learners for another federation round. To ensure fast performance and short federation rounds, we have implemented the MetisFL Controller in C++ and we have exposed its main functionality (start, wait, stop) via python bindings. The main build target of the `controller` folder is a Pybind extension defined in `metisfl/controller/BUILD.bazel` which produces the `controller.so` shared object. The shared object and the python files in the folder are packaged in a `controller` python module. The `__main__.py` folder makes this module runnable, i.e., the controller instance can be started by running 

```Python
python -m metisfl.controller
``` 

## Driver
The driver is a python library that starts the controller and learners and initiates the training. 

## Encryption 

## Learner 

## Proto 
