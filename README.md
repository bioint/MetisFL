# MetisFL - The developer-friendly federated learning framework

MetisFL is a federated learning framework that allows developers to easily federate their machine learning workflows and train their models across distributed datasets without having to collect the data in a centralized location. 

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
        ├── proto             # Protobuf definitions for controller/learner/driver communication
        ├── pybind            # Controller and Palisade python bindings 
    ├── resources             # Resource files (public keys, certs etc)
    ├── test                  # Testing folder
    ├── build.sh              # Build script 
    ├── BUILD.bazel           # Bazel build file; contains main target definitions
    ├── WORKSPACE             # Bazel workspace; contains external dependancies

## Controller 
The controller (or aggregator) is responsible for collecting the (encrypted) model weights from the learners, aggregating them, encrypting them and distributing the encrypted weights back to the learners for another federation round. 

## Driver
The driver is a python library that starts the controller and learners and initiates the training. 

## Encryption 

## Learner 

## Proto 

## Pybind
