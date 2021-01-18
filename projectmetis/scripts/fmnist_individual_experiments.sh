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
# This will execute for 300 epochs
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/fmnist/fmnist_federated_main_sys_args.py --synchronous True --federation_rounds 1 --learning_rate 0.0001 --momentum 0.0 --batchsize 50 --update_frequency 300 --execution_time 10000
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/fmnist/fmnist_federated_main_sys_args.py --synchronous True --federation_rounds 1 --learning_rate 0.0001 --momentum 0.0 --batchsize 100 --update_frequency 300 --execution_time 10000
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/fmnist/fmnist_federated_main_sys_args.py --synchronous True --federation_rounds 1 --learning_rate 0.0015 --momentum 0.0 --batchsize 50 --update_frequency 300 --execution_time 10000
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/fmnist/fmnist_federated_main_sys_args.py --synchronous True --federation_rounds 1 --learning_rate 0.0015 --momentum 0.0 --batchsize 100 --update_frequency 300 --execution_time 10000

export GLOBAL_EPOCH_ANNEALING=1000
export WITH_LR_FEDERATED_ANNEALING=0
export WITH_MOMENTUM_FEDERATED_ANNEALING=0
export COMMUNITY_FUNCTION=""

#export BALANCED_PARTITIONING="False"
#export CLASSES_PER_PARTITION=3
#export VALIDATION_PROPORTION=0.05
#export FAST_LEARNER_VC_TOMBSTONES=4
#export SLOW_LEARNER_VC_TOMBSTONES=0
#export FAST_LEARNER_VC_LOSS_PCT_THRESHOLD=1
#export SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD=3
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous False --federation_rounds 1 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 0 --execution_time 244
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous False --federation_rounds 1 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 0 --execution_time 245


#export BALANCED_PARTITIONING="False"
#export CLASSES_PER_PARTITION=5
#export VALIDATION_PROPORTION=0.05
#export FAST_LEARNER_VC_TOMBSTONES=4
#export SLOW_LEARNER_VC_TOMBSTONES=2
#export FAST_LEARNER_VC_LOSS_PCT_THRESHOLD=1
#export SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD=1
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous False --federation_rounds 1 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 0 --execution_time 241
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous False --federation_rounds 1 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 0 --execution_time 242
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous False --federation_rounds 1 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 0 --execution_time 243
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous False --federation_rounds 1 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 0 --execution_time 244
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous False --federation_rounds 1 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 0 --execution_time 245

export BALANCED_PARTITIONING="False"
export CLASSES_PER_PARTITION=3
export VALIDATION_PROPORTION=0.0
export COMMUNITY_FUNCTION=""
/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous True --federation_rounds 120 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 4 --execution_time 100005
/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous True --federation_rounds 120 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 4 --execution_time 100010
/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous True --federation_rounds 120 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 4 --execution_time 100015
/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous True --federation_rounds 120 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 4 --execution_time 100020

export BALANCED_PARTITIONING="False"
export CLASSES_PER_PARTITION=3
export VALIDATION_PROPORTION=0.05
export COMMUNITY_FUNCTION="FedValidationScore"
/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous True --federation_rounds 120 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 4 --execution_time 100006
/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous True --federation_rounds 120 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 4 --execution_time 100011
/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous True --federation_rounds 120 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 4 --execution_time 100016
/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/fmnist/fmnist_cnn2_federated_main_sys_args.py --synchronous True --federation_rounds 120 --learning_rate 0.01 --momentum 0.9 --batchsize 100 --update_frequency 4 --execution_time 100021
