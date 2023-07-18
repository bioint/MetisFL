#! /usr/bin/bash

export PYTHONPATH=`python -c "import os; print(os.getcwd())"`
export PYTHON_BIN_PATH=`python -c 'import sys; print(sys.executable)'`
export PYTHON_LIB_PATH=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'`
python setup.py