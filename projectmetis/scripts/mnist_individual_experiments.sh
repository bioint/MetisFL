#! /bin/bash -l

# If something goes wrong and we need to kill the process then run:
# kill -9 $(ps aux | grep '[s]trip' | grep '[g]rid' | awk '{print $2}')

#export GRPC_TRACE=all
#export GRPC_VERBOSITY=DEBUG

PROJECT_HOME=/lfs1/stripeli/condaprojects/projectmetis/projectmetis
cd $PROJECT_HOME

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64/
export CUDA_HOME=/usr/local/cuda-9.0
export PATH=/usr/local/cuda-9.0/bin:/opt/anaconda3/bin/:/opt/anaconda2/bin/:/opt/cmake-3.7/bin:/usr/local/cuda/bin:/opt/anaconda3/bin/:/opt/anaconda2/bin/:/opt/cmake-3.7/bin:/usr/local/cuda/bin:/opt/anaconda3/bin/:/opt/anaconda2/bin/:/opt/cmake-3.7/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/nas/home/stripeli/.local/bin:/nas/home/stripeli/bin:/nas/home/stripeli/.local/bin:/nas/home/stripeli/bin:/nas/home/stripeli/.local/bin:/nas/home/stripeli/bin
export PYTHONPATH=.

# Centralized execution
# This will execute for 500 epochs
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous True --federation_rounds 1 --initial_learning_rate 0.0001 --sgd_momentum 0.0 --batchsize 50 --update_frequency 500 --execution_time 10000


/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --learning_rate 0.0015 --momentum 0.0 --batchsize 50 --update_frequency 4 --execution_time 30

# Synchronous Execution
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous True --federation_rounds 30 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 16 --execution_time 10000
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous True --federation_rounds 60 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 8 --execution_time 10000
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous True --federation_rounds 120 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 4 --execution_time 10000
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous True --federation_rounds 240 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 2 --execution_time 10000
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous True --federation_rounds 400 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 1 --execution_time 10000

# Asynchronous Execution with Static UF
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 1 --execution_time 60
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 2 --execution_time 90
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 4 --execution_time 90
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 8 --execution_time 90
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 16 --execution_time 90

# Asynchronous Execution with Static UF for Heterogeneous
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 8 --execution_time 60
#export CLASSES_PER_PARTITION=1
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 4 --execution_time 60
#export CLASSES_PER_PARTITION=2
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 4 --execution_time 60
#export CLASSES_PER_PARTITION=3
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 4 --execution_time 60
#export CLASSES_PER_PARTITION=4
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 4 --execution_time 60
#export CLASSES_PER_PARTITION=5
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 4 --execution_time 60
#export CLASSES_PER_PARTITION=10
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 4 --execution_time 60


# Asynchronous Executions with Adaptive UF
#export MINVLOSS=10
#export MAXVLOSS=10
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
#export MINVLOSS=5
#export MAXVLOSS=10
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
#export MINVLOSS=5
#export MAXVLOSS=5
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
#export MINVLOSS=3
#export MAXVLOSS=10
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
#export MINVLOSS=3
#export MAXVLOSS=5
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 15
#export MINVLOSS=3
#export MAXVLOSS=3
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
#export MINVLOSS=1
#export MAXVLOSS=10
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
#export MINVLOSS=1
#export MAXVLOSS=5
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
#export MINVLOSS=1
#export MAXVLOSS=3
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
#export MINVLOSS=1
#export MAXVLOSS=1
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
#export MINVLOSS=0.5
#export MAXVLOSS=10
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
#export MINVLOSS=0.5
#export MAXVLOSS=5
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
#export MINVLOSS=0.5
#export MAXVLOSS=3
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
#export MINVLOSS=0.5
#export MAXVLOSS=1
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/mnist/mnist_federated_main_sys_args.py --synchronous False --federation_rounds 1 --initial_learning_rate 0.0015 --sgd_momentum 0.0 --batchsize 50 --update_frequency 0 --execution_time 60
