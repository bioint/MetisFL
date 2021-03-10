#! /bin/bash -l

# If something goes wrong and we need to kill the process then run:
# kill -9 $(ps aux | grep '[s]trip' | grep '[g]rid' | awk '{print $2}')

#export GRPC_TRACE=all
#export GRPC_VERBOSITY=DEBUG

PROJECT_HOME=/lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis
cd $PROJECT_HOME

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:/opt/anaconda3/condabin:/opt/anaconda3/bin/:/opt/anaconda2/bin/:/opt/cmake-3.7/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/nas/home/stripeli/.local/bin:/nas/home/stripeli/bin:/lfs1/prisms/golang/go/bin
export PYTHONPATH=.

export SYNCHRONOUS_CLUSTER_CONFIGURATION_FILES_DIRECTORY="${PROJECT_HOME}"/resources/config/experiments_configs/cifar100/synchronous
export ASYNCHRONOUS_CLUSTER_CONFIGURATION_FILES_DIRECTORY="${PROJECT_HOME}"/resources/config/experiments_configs/cifar100/asynchronous


function run_synchronous_fedavg_heterogeneous_clusters_experiments {

    # Synchronous FedAvg with x5 Fast and x5 Slow Learners
    for file in "${SYNCHRONOUS_CLUSTER_CONFIGURATION_FILES_DIRECTORY}"/FedAvg/*cifar100.resnet.federation.10Learners.5Fast_atBDNF_5Slow_atLEARN.SyncFedAvg*; do
        if [[ "${file}" == *"SyncFedAvg.run1.yaml"* ]]; then
            echo "Running" "${file}"
            /lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/cifar/resnet/cifar100_resnet_federated_main_sys_args.py --learning_rate 0.1 --momentum 0.9 --federation_environment_filepath "${file}" &
            sleep 1
        fi
        pids[${i}]=$!
    done

    # Wait for all pids to finish
    for pid in ${pids[*]}; do
        while s=`ps -p $pid -o s=` && [[ "$s" && "$s" != 'Z' ]]; do
            sleep 1
        done
    done
}

function run_asynchronous_fedavg_heterogeneous_clusters_experiments {

    # Synchronous FedAvg with x5 Fast and x5 Slow Learners
    for file in "${ASYNCHRONOUS_CLUSTER_CONFIGURATION_FILES_DIRECTORY}"/FedAvg/*cifar100.resnet.federation.10Learners.5Fast_atBDNF_5Slow_atLEARN.AsyncFedAvg*; do
        if [[ "${file}" == *"AsyncFedAvg.run1.yaml"* ]]; then
            echo "Running" "${file}"
            /lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/cifar/resnet/cifar100_resnet_federated_main_sys_args.py --learning_rate 0.1 --momentum 0.9 --federation_environment_filepath "${file}" &
            sleep 1
        fi
        pids[${i}]=$!
    done

    # Wait for all pids to finish
    for pid in ${pids[*]}; do
        while s=`ps -p $pid -o s=` && [[ "$s" && "$s" != 'Z' ]]; do
            sleep 1
        done
    done
}

function run_synchronous_dvw_heterogeneous_clusters_experiments {

    # Synchronous DVW with x5 Fast and x5 Slow Learners
    for file in "${SYNCHRONOUS_CLUSTER_CONFIGURATION_FILES_DIRECTORY}"/DVW/*cifar100.resnet.federation.10Learners.5Fast_atBDNF_5Slow_atLEARN.AsyncFedAvg*; do
        if [[ "${file}" == *"SyncDVWMicroF1.run1.yaml"* ]]; then
            echo "Running" "${file}"
            /lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/cifar/resnet/cifar100_resnet_federated_main_sys_args.py --learning_rate 0.1 --momentum 0.9 --federation_environment_filepath "${file}" &
            sleep 1
        fi
        pids[${i}]=$!
    done

    # Wait for all pids to finish
    for pid in ${pids[*]}; do
        while s=`ps -p $pid -o s=` && [[ "$s" && "$s" != 'Z' ]]; do
            sleep 1
        done
    done
}

function run_asynchronous_dvw_heterogeneous_clusters_experiments {

    # Synchronous DVW with x5 Fast and x5 Slow Learners
    for file in "${ASYNCHRONOUS_CLUSTER_CONFIGURATION_FILES_DIRECTORY}"/DVW/*cifar100.resnet.federation.10Learners.5Fast_atBDNF_5Slow_atLEARN.AsyncDVWMicroF1*; do
        if [[ "${file}" == *"AsyncDVWMicroF1.run1.yaml"* ]]; then
            echo "Running" "${file}"
            /lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/cifar/resnet/cifar100_resnet_federated_main_sys_args.py --learning_rate 0.1 --momentum 0.9 --federation_environment_filepath "${file}" &
            sleep 1
        fi
        pids[${i}]=$!
    done

    # Wait for all pids to finish
    for pid in ${pids[*]}; do
        while s=`ps -p $pid -o s=` && [[ "$s" && "$s" != 'Z' ]]; do
            sleep 1
        done
    done
}


#############################################
### UNIFORM & IID ####
#############################################
export CLASSES_PER_PARTITION=100
export SKEWNESS_FACTOR=0.0
export BALANCED_RANDOM_PARTITIONING="True"
export BALANCED_CLASS_PARTITIONING="False"
export STRICTLY_UNBALANCED="False"

#export FAST_LEARNERS_VC_TOMBSTONES=2
#export SLOW_LEARNERS_VC_TOMBSTONES=2
#export FAST_LEARNERS_VC_LOSS_PCT_THRESHOLD=0
#export SLOW_LEARNERS_VC_LOSS_PCT_THRESHOLD=1

#export SEMI_SYNCHRONOUS_EXECUTION="False"
#run_synchronous_fedavg_heterogeneous_clusters_experiments
#
#export SEMI_SYNCHRONOUS_EXECUTION="True"
#export SEMI_SYNCHRONOUS_K_VALUE=2
#export GPU_TIME_PER_BATCH_MS=60
#export CPU_TIME_PER_BATCH_MS=2000
#
#run_synchronous_fedavg_heterogeneous_clusters_experiments


#############################################
### UNIFORM & Non-IID(50) ####
#############################################
export CLASSES_PER_PARTITION=50
export SKEWNESS_FACTOR=0.0
export BALANCED_RANDOM_PARTITIONING="False"
export BALANCED_CLASS_PARTITIONING="True"
export STRICTLY_UNBALANCED="False"

#export FAST_LEARNERS_VC_TOMBSTONES=4
#export SLOW_LEARNERS_VC_TOMBSTONES=2
#export FAST_LEARNERS_VC_LOSS_PCT_THRESHOLD=3
#export SLOW_LEARNERS_VC_LOSS_PCT_THRESHOLD=3

#export SEMI_SYNCHRONOUS_EXECUTION="False"
#run_synchronous_fedavg_heterogeneous_clusters_experiments
#
#export SEMI_SYNCHRONOUS_EXECUTION="True"
#export SEMI_SYNCHRONOUS_K_VALUE=2
#export GPU_TIME_PER_BATCH_MS=60
#export CPU_TIME_PER_BATCH_MS=2000
#
#run_synchronous_fedavg_heterogeneous_clusters_experiments


#############################################
### SKEWED & IID ####
#############################################
export CLASSES_PER_PARTITION=100
export SKEWNESS_FACTOR=1.2
export BALANCED_RANDOM_PARTITIONING="False"
export BALANCED_CLASS_PARTITIONING="False"
export UNBALANCED_CLASS_PARTITIONING="True"
export STRICTLY_UNBALANCED="False"

#export FAST_LEARNERS_VC_TOMBSTONES=4
#export SLOW_LEARNERS_VC_TOMBSTONES=2
#export FAST_LEARNERS_VC_LOSS_PCT_THRESHOLD=0
#export SLOW_LEARNERS_VC_LOSS_PCT_THRESHOLD=1

#export SEMI_SYNCHRONOUS_EXECUTION="False"
#run_synchronous_fedavg_heterogeneous_clusters_experiments


#############################################
### SKEWED & Non-IID(50) ####
#############################################
export CLASSES_PER_PARTITION=50
export SKEWNESS_FACTOR=1.3
export BALANCED_RANDOM_PARTITIONING="False"
export BALANCED_CLASS_PARTITIONING="False"
export UNBALANCED_CLASS_PARTITIONING="True"
export STRICTLY_UNBALANCED="False"

#export FAST_LEARNERS_VC_TOMBSTONES=6
#export SLOW_LEARNERS_VC_TOMBSTONES=1
#export FAST_LEARNERS_VC_LOSS_PCT_THRESHOLD=3
#export SLOW_LEARNERS_VC_LOSS_PCT_THRESHOLD=3

#export SEMI_SYNCHRONOUS_EXECUTION="False"
#run_synchronous_fedavg_heterogeneous_clusters_experiments


#############################################
### POWER LAW & IID ####
#############################################
export CLASSES_PER_PARTITION=100
export SKEWNESS_FACTOR=1.5
export BALANCED_RANDOM_PARTITIONING="False"
export BALANCED_CLASS_PARTITIONING="False"
export UNBALANCED_CLASS_PARTITIONING="True"
export STRICTLY_UNBALANCED="True"

#export FAST_LEARNERS_VC_TOMBSTONES=4
#export SLOW_LEARNERS_VC_TOMBSTONES=2
#export FAST_LEARNERS_VC_LOSS_PCT_THRESHOLD=0
#export SLOW_LEARNERS_VC_LOSS_PCT_THRESHOLD=1

export SEMI_SYNCHRONOUS_EXECUTION="True"
export SEMI_SYNCHRONOUS_K_VALUE=0.5
export GPU_TIME_PER_BATCH_MS=60
export CPU_TIME_PER_BATCH_MS=2000

run_synchronous_fedavg_heterogeneous_clusters_experiments


#############################################
### POWER LAW & Non-IID(50) ####
#############################################
export CLASSES_PER_PARTITION=50
export SKEWNESS_FACTOR=1.5
export BALANCED_RANDOM_PARTITIONING="False"
export BALANCED_CLASS_PARTITIONING="False"
export UNBALANCED_CLASS_PARTITIONING="True"
export STRICTLY_UNBALANCED="True"

#export FAST_LEARNERS_VC_TOMBSTONES=4
#export SLOW_LEARNERS_VC_TOMBSTONES=2
#export FAST_LEARNERS_VC_LOSS_PCT_THRESHOLD=1
#export SLOW_LEARNERS_VC_LOSS_PCT_THRESHOLD=1

#export SEMI_SYNCHRONOUS_EXECUTION="True"
#export SEMI_SYNCHRONOUS_K_VALUE=0.5
#export GPU_TIME_PER_BATCH_MS=60
#export CPU_TIME_PER_BATCH_MS=2000
#
#run_synchronous_fedavg_heterogeneous_clusters_experiments


#############################################
### SKEWED & IID ####
#############################################
export CLASSES_PER_PARTITION=100
export SKEWNESS_FACTOR=1.2
export BALANCED_RANDOM_PARTITIONING="False"
export BALANCED_CLASS_PARTITIONING="False"
export UNBALANCED_CLASS_PARTITIONING="True"
export STRICTLY_UNBALANCED="False"

#export FAST_LEARNERS_VC_TOMBSTONES=4
#export SLOW_LEARNERS_VC_TOMBSTONES=2
#export FAST_LEARNERS_VC_LOSS_PCT_THRESHOLD=0
#export SLOW_LEARNERS_VC_LOSS_PCT_THRESHOLD=1

export SEMI_SYNCHRONOUS_EXECUTION="True"
export SEMI_SYNCHRONOUS_K_VALUE=0.5
export GPU_TIME_PER_BATCH_MS=60
export CPU_TIME_PER_BATCH_MS=2000

run_synchronous_fedavg_heterogeneous_clusters_experiments