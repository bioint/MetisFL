# MetisFL: The blazing-fast and developer-friendly federated learning framework.

&nbsp;
<div align="center">
 <img 
    src="docs/img/logos/logo_icon_03.png" width="200px">
</div>
&nbsp;

<p align="center">
    <a href="https://nevron.ai/"><img src="https://img.shields.io/badge/View%20Homepage-Homepage-green"></a>
  <a href="https://docs.nevron.ai/"><img src="https://img.shields.io/badge/View%20Documentation-Docs-yellow"></a>
  <a href="https://join.slack.com/t/nevronai/shared_invite/zt-1xyc308jq-FeNBbuF1UBBwhp4H~zHCww"><img src="https://img.shields.io/badge/Join%20our%20community-Slack-blue"></a>
</p>


MetisFL is a federated learning framework that allows developers to federate their machine learning workflows and train their models across distributed datasets without having to collect the data in a centralized location. Currently, the project is transitioning from a private, experimental version to a public, beta phase. We are actively encouraging developers, researchers and data scientists to experiment with the framework and contribute to the codebase.


Homepage: https://nevron.ai/ \
Github: https://github.com/NevronAI \
Docs: https://docs.nevron.ai/ \
Slack: [https://nevronai.slack.com](https://join.slack.com/t/nevronai/shared_invite/zt-1xyc308jq-FeNBbuF1UBBwhp4H~zHCww)


MetisFL sprung up from the Information and Science Institute (ISI) in the University of Southern California (USC). It is backed by several years of Ph.D. and Post-Doctoral research and several publications in top-tier machine learning and system conferences. It is being developed with the following guiding principles in mind:

* **Scalability**: MetisFL is the only federated learning framework with the core controller infrastructure developed solely on C++. This allows for the system to scale and support up to 100K+ learners!

* **Speed**: The core operations at the controller as well as the controller-learner communication overhead has been optimized for efficiency. This allows MetisFL to achieve improvements of up to 1000x on the federation round time compared to other federated learning frameworks.

* **Efficiency and Flexibility**: MetisFL supports synchronous, semi-synchronous and asynchronous protocols. The different choices make our framework flexible enough to adapt to the needs of each use-case. Additionally, the support of fully asynchronous protocol makes MetisFL a highly efficient solution for use-cases with high heterogeneity on the compute/communication capabilities of the learners.

* **Strong Security**: MetisFL supports secure aggregations with fully homomorphic encryption using the [Palisade](https://gitlab.com/palisade/palisade-release) C++ cryptographic library.  This ensures that the weights of the produced models remain private and secure in transit.

# Quickstart (coming soon)
As an introductory example to quickly demonstrate the MetisFL framework in practice, we will run the `Hello World` example of deep learning. To get started, first ensure that your system meets the requirements:

* Python 3.8 to 3.10
* A x86_64 Linux distro (tested on Ubuntu Focal)

Install the `metisfl` Python package by running:

```Bash
pip install metisfl
```

and then clone the repository on you local machine:

```Bash
git clone https://github.com/NevronAI/metisfl.git
```
Navigate to the `Tensorflow Quickstart` example under `examples/tensorflow-quickstart` and open up 5 terminals. In the first terminal, run the controller:

```Bash
python controller.py
```

This is going to startup the controller at port 50001 in the local machine. In the next 3 terminals run the learners:

```Bash
python learner.py -l ID
```

where `ID` is the unique identifier of the learner (1, 2, 3). This will startup 3 learners servers at ports 50002, 50003 and 50004 and the learners will connect to the controller and wait for the training task. Finally, in the last terminal run the driver:

```Bash
python driver.py
```
The driver will initiate the federated training, monitor the controller and learners and terminate the experiment once 5 federated rounds have been completed. Congratulations! You are now running your first federated learning experiment using MetisFL!


# Project Structure Overview
The project uses a unified codebase for both the Python and C++ code. The C++ modules, i.e., `controller` and `encryption`, have simple python bindings to expose their functionality to python code. The python bindings of the encryption module are directly used by the Learner for encrypting the weight before sending them to the controller. 

    .
    ├── examples              # Examples and use-cases for MetisFL
    ├── metisfl               # Main source code folder
        ├── common            # Common utilities and helper functions    
        ├── controller        # C++ implementation of the Controller/Aggregator
        ├── driver            # Python library for the MetisFL Driver
        ├── encryption        # C++ Palisade library
        ├── learner           # Python Learner library 
        ├── proto             # Protobuf definitions for communication
        ├── resources         # FHE/SSL related resource files and keys
    ├── test                  # Testing folder (under construction)
    ├── LICENSE               # License file
    ├── BUILD.bazel           # Bazel build file; contains main target definitions
    ├── setup.py              # Setup script; compiles and produces a Python Wheel
    ├── WORKSPACE             # Bazel workspace; contains external dependencies

# Architecture Overview
The architecture of MetisFL consists of three main components: the **Federation Controller**, the **Federation Learner** and the **Federation Driver**.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="./docs/img/light/MetisFL-Components-Overview-02.png" width="700px">
  <img alt="MetisFL Components Overview" src="./docs/img/dark/MetisFL-Components-Overview-02.png" width="700px">
</picture>
</div>

## Federation Controller
The Federation Controller acts as the federation cluster manager, and it is responsible for selecting and delegating training and evaluating tasks to the federation learners (cluster nodes) and storing and aggregating the learners’ local models (w/ or w/out encryption).

## Federation Learner
The Federation Learner(s) are the cluster node responsible for training and evaluating the federation model assigned to the by the Controller on the local, private dataset.

## Federation Driver
The Federation Driver parses the federated learning workflow defined by the system user and creates the Metis Context. The Metis Context is responsible for initializing and monitoring the federation cluster, initializing the original federation model state, defining the data loading recipe for each learner, and generating the security keys where needed (e.g., SSL certificates, and FHE key pair).


