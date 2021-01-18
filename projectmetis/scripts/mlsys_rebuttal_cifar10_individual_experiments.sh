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

export SYNCHRONOUS_CLUSTER_CONFIGURATION_FILES_DIRECTORY="${PROJECT_HOME}"/resources/config/experiments_configs/cifar10/synchronous
export ASYNCHRONOUS_CLUSTER_CONFIGURATION_FILES_DIRECTORY="${PROJECT_HOME}"/resources/config/experiments_configs/cifar10/asynchronous


function run_synchronous_fedavg_homogeneous_clusters_experiments {

    # Synchronous FedAvg with x10 Fast Learners
    for file in "${SYNCHRONOUS_CLUSTER_CONFIGURATION_FILES_DIRECTORY}"/FedAvg/*cifar10.cnn2.federation.10Learners.10Fast_atBDNF.SyncFedAvg*; do
        if [[ "${file}" == *"SyncFedAvg.run1.yaml"* ]]; then
            echo "Running" "${file}"
            /lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/cifar/cnn2/cifar10_cnn2_federated_main_sys_args.py --learning_rate 0.05 --momentum 0.75 --federation_environment_filepath "${file}" &
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

function run_synchronous_fedavg_heterogeneous_clusters_experiments {

    # Synchronous FedAvg with x5 Fast and x5 Slow Learners
#    for file in "${SYNCHRONOUS_CLUSTER_CONFIGURATION_FILES_DIRECTORY}"/FedAvg/*cifar10.cnn2.federation.10Learners.5Fast_atBDNF_5Slow_atLEARN.SyncFedAvg*; do
    for file in "${SYNCHRONOUS_CLUSTER_CONFIGURATION_FILES_DIRECTORY}"/FedAvg/*cifar10.cnn2.federation.10Learners.5Fast_atBDNF_5Slow_atBDNF.SyncFedAvg*; do
        if [[ "${file}" == *"SyncFedAvg.run1.yaml"* ]]; then
            echo "Running" "${file}"

            /lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/cifar/cnn2/mlsys_rebuttal_cifar10_cnn2_federated_main_sys_args.py --learning_rate 0.05 --momentum 0.75 --federation_environment_filepath "${file}" &
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
### REVIEWER_2 ####
##############################################

#export SEMI_SYNCHRONOUS_EXECUTION="False"
#run_synchronous_fedavg_heterogeneous_clusters_experiments


export SEMI_SYNCHRONOUS_EXECUTION="True"
export GPU_TIME_PER_BATCH_MS=30
export CPU_TIME_PER_BATCH_MS=300

#export SEMI_SYNCHRONOUS_K_VALUE=2
#run_synchronous_fedavg_heterogeneous_clusters_experiments
#
#export SEMI_SYNCHRONOUS_K_VALUE=2.5
#run_synchronous_fedavg_heterogeneous_clusters_experiments
#
#export SEMI_SYNCHRONOUS_K_VALUE=0.5
#run_synchronous_fedavg_heterogeneous_clusters_experiments

export SEMI_SYNCHRONOUS_K_VALUE=0.25
run_synchronous_fedavg_heterogeneous_clusters_experiments


#############################################
### REVIEWER_3 ####
##############################################

#############################################
### SKEWED & Non-IID(5) ####
#############################################
export CLASSES_PER_PARTITION=5
export SKEWNESS_FACTOR=1.5
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

export SEMI_SYNCHRONOUS_EXECUTION="True"
export SEMI_SYNCHRONOUS_K_VALUE=2
export GPU_TIME_PER_BATCH_MS=30
export CPU_TIME_PER_BATCH_MS=300
#run_synchronous_fedavg_heterogeneous_clusters_experiments