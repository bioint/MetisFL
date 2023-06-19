# MetisFL: The blazing-fast and developer-friendly federated learning framework.

<div align="center">
 <img 
    style="padding: 20px 0px"
    src="docs/img/logos/logo_png_01.png" width="400px">
</div>


MetisFL is a federated learning framework that allows developers to easily federate their machine learning workflows and train their models across distributed datasets without having to collect the data in a centralized location. 

Homepage: https://nevron.ai/ \
Github: https://github.com/NevronAI \
Docs: https://docs.nevron.ai/ \
Slack: https://nevronai.slack.com 

This open-source project sprung up from the Information and Science Institute (ISI) in the University of Southern California (USC). It is backed by several years of Ph.D. and Post-Doctoral research and several publications in top-tier machine learning and system conferences. MetisFL is being developed with the following guiding principles in mind:

* **Scalability**: MetisFL is the only federated learning framework with the core controller infrastructure developed solely on C++. This allows for the system to scale and support up to 100K+ learners! 

* **Speed**: The core operations at the controller as well as the controller-learner communication overhead has been optimized for efficiency. This allows MetisFL to achieve improvements of up to 1000x on the federation round time compared to other federated learning frameworks. 

* **Efficiency and Flexibility**: MetisFL supports synchronous, semi-synchronous and asynchronous protocols. The different choices make our framework flexible enough to adapt to the needs of each use-case. Additionally, the support of fully asynchronous protocol makes MetisFL a highly efficient solution for use-cases with high heterogeneity on the compute/communication capabilities of the learners.

Currently, MetisFL is transitioning from a private, experimental version to a public, beta testing phase. We are actively encouraging developers, researchers and data scientists to experiment with the framework and contribute to the codebase. 
   
# Project Structure Overview
The project uses a unified codebase for both the Python and C++ code. 

    .
    ├── docker                # Docker files for packaging MetisFL in a docker container       
    ├── examples              # Examples and use-cases for MetisF
    ├── metisfl               # Main source code directory
        ├── controller        # C++ implementation of the Controller/Aggregator
        ├── driver            # Python library for the MetisFL Driver
        ├── encryption        # C++ Palisade helper files
        ├── learner           # MetisFL Python Learner library 
        ├── models            # MetisFL Python Learner library         
        ├── proto             # Protobuf definitions for controller/learner/driver communication
        ├── resources         # FHE/SSL
        ├── proto             # Protobuf definitions for controller/learner/driver communication
    ├── resources             # Resource files (public keys, certs etc)
    ├── test                  # Testing folder
    ├── build.sh              # Build script 
    ├── BUILD.bazel           # Bazel build file; contains main target definitions
    ├── WORKSPACE             # Bazel workspace; contains external dependancies

## Controller 
The controller (or aggregator) is responsible for collecting the (encrypted) model weights from the learners, aggregating them, encrypting them and distributing the encrypted weights back to the learners for another federation round. To ensure fast performance and short federation rounds, we have implemented the MetisFL Controller in C++ and we have exposed its main functionality (start, wait, stop) via python bindings. The main build target of the `controller` folder is a Pybind extension defined in `controller/BUILD.bazel` which produces the `controller.so` shared object. The shared object and the python files in the folder are packaged in a `controller` python module. The `__main__.py` folder makes this module runnable, i.e., the controller instance can be started by running 

```Python
python -m metisfl.controller
``` 

and supplying the arguments listed in `__main__.py` file.

## Driver
The driver is a python library that starts the controller and learners and initiates the training. 

## Encryption 

## Learner 

## Proto 
