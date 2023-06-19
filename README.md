# MetisFL: The blazing-fast and developer-friendly federated learning framework.

<div align="center">
 <img src="docs/img/logos/logo_png_01.png" width="600px">
</div>


MetisFL is a federated learning framework that allows developers to easily federate their machine learning workflows and train their models across distributed datasets without having to collect the data in a centralized location. 

Homepage: https://nevron.ai/ \
Github: https://github.com/NevronAI \
Docs: https://docs.nevron.ai/ \
Slack: https://nevronai.slack.com \


### Federated Training

This is a high-level overview of how federated training is performed and when and how the synchronization points between
the learners and the controller are created.

**1. Controller Training Assignment**\
The controller creates and assigns the task that each learner needs to run by sending to every learner (the state of)
the community model and any other learning parameters, such as the number of local updates, the learning hyperparameters
and any other necessary training information. Subsequently, the controller pings each learner though the respective gRPC
endpoint (see `RunTask` gRPC endpoint in `learner.proto`). 

**2. Learner Local Training** \
Upon receiving the training task, the learner starts training locally on its local dataset and once it completes its
training, it sends its local model along with any associated training metadata to the controller. At this point, the 
learner pings the controller and sends a local training completion request (see `MarkTaskCompleted` gRPC endpoint in `controller.proto`).

**3. Synchronization Points (a.k.a. Federation Round)** \
The controller receives the local models and if a quorum exists (e.g., received local models from all learners), then it
computes the new community model using the local models and their associated scaling factors
(e.g., number of training examples) and creates and reassigns the new training task to each learner. At this point a new
global training iteration begins. 
   


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
