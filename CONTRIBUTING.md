# Development Guide

Thank you for you interest in contributing to MetisFL. We welcome all contributions from bug fixes to new features and documentation. To get started, please read the following guide.

# Development Environment
First, you need to setup you development environment. Currently, the setup mentioned below has been tested on Ubuntu OS and for the x86_64 architecture. It should, however, work for different Linux-like OS on the same architecture. Support for different architectures is under development. The requirements for compiling and testing the code on your local machine are:

* Bazel 4.2.1
* Python 3.8 - 3.10
* Python header and distutils
* build-essential, autoconf and libomp-dev

The recommended way to install Bazel is to use the [Bazelisk](https://github.com/bazelbuild/bazelisk) launcher and place the executable somewhere in your PATH, i.e., `/usr/bin/bazel` or `/usr/bin/bazelisk`. Please make sure that the name of the Bazelisk executable matches the BAZEL_CMD variable in `setup.py`. By default, the setup script will search for `bazelisk`. Bazelisk will automatically pick up the version from the `.bezelversion`file and then download and execute the corresponding Bazel executable.

The Python headers and distutils are needed so that the C++ controller and encryption code can be compiled as Python modules. On Ubuntu, they can be installed with the following command:

```Bash
apt-get -y install python3.10-dev python3.10-distutils
```

Finally, the remaining requirements contain the compiler, autoconf and libomp (which is used for parallelization). Please make sure that they are available in your system by running:

```Bash
apt-get -y install build-essential autoconf libomp-dev
```

# Build Project
The main command to build the project and the Python Wheel is:

```Bash 
python setup.py 
```

The build target of this script is a Python Wheel which will be placed in the `build` folder. In the process of producing that build, several other targets will be built such as the controller binaries `metisfl/controller/controller.so`, the Palisade/encryption binaries `metisfl/encryption/fhe.so` and the Protobuf/gRPC Python classes in `metisfl/proto` directory. Please note that Pybind will use the python headers of the currently active Python version. 


# Fork and Develop
Once you have setup your development environment, you can start contributing to MetisFL. The first step is to fork the repository you want to contribute to. This will create a copy of the repository in your GitHub account. You can then clone the forked repository to your local machine and start making changes. Once you are done with your changes, you can push them to your forked repository and create a pull request to merge them into the main repository. The following steps will guide you through this process:

1. Fork the repository by clicking the "Fork" button on the project page.

2. Clone the repository to your local machine and enter the newly created repo using the following commands:

```
git clone https://github.com/YOUR-GITHUB-USERNAME/metisfl.git
cd metisfl
```
3. Add metisfl original repository as upstream, to easily sync with the latest changes.

```
git remote add upstream https://github.com/NevronAI/metisfl.git
```

4. Create a new branch for your changes using the following command:

```
git checkout -b "branch-name"
```
5. Make your changes to the code or documentation.

6. Add the changes to the staging area using the following command:
```
git add . 
```

7. Commit the changes with a meaningful commit message using the following command:
```
git commit -m "your commit message"
```
8. Push the changes to your forked repository using the following command:
```
git push origin branch-name
```
9. Go to the GitHub website and navigate to your forked repository.

10. Click the "New pull request" button.

11. Select the branch you just pushed to and the branch you want to merge into on the original repository.

12. Add a description of your changes and click the "Create pull request" button.

13. Wait for the project maintainer to review your changes and provide feedback.

14. Make any necessary changes based on feedback and repeat steps 5-12 until your changes are accepted and merged into the main project.

15. Once your changes are merged, you can update your forked repository and local copy of the repository with the following commands:

```
git fetch upstream
git checkout main
git merge upstream/main
```
Finally, delete the branch you created with the following command:
```
git branch -d branch-name
```
That's it you made it ⭐⭐

