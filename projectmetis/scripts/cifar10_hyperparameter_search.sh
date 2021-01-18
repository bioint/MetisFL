#! /bin/bash -l

PROJECT_HOME=/lfs1/stripeli/condaprojects/projectmetis/projectmetis
cd $PROJECT_HOME

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda/extras/CUPTI/lib64/
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:/opt/anaconda3/condabin:/opt/anaconda3/bin/:/opt/anaconda2/bin/:/opt/cmake-3.7/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/nas/home/stripeli/.local/bin:/nas/home/stripeli/bin:/lfs1/prisms/golang/go/bin
export PYTHONPATH=.

export CLASSES_PER_PARTITION=3
export VALIDATION_PROPORTION=0.05
export WITH_LR_FEDERATED_ANNEALING=0
export WITH_MOMENTUM_FEDERATED_ANNEALING=0

FAST_LEARNER_VC_TOMBSTONES_ARR=(5 6 7)
SLOW_LEARNER_VC_TOMBSTONES_ARR=(2 3)
FAST_LEARNER_VC_LOSS_PCT_THRESHOLD_ARR=(1 3)
SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD_ARR=(1 3)
k=0
for index_1 in ${!FAST_LEARNER_VC_TOMBSTONES_ARR[@]}; do
    for index_2 in ${!SLOW_LEARNER_VC_TOMBSTONES_ARR[@]}; do
        for index_3 in ${!FAST_LEARNER_VC_LOSS_PCT_THRESHOLD_ARR[@]}; do
            for index_4 in ${!SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD_ARR[@]}; do
                export FAST_LEARNER_VC_TOMBSTONES=${FAST_LEARNER_VC_TOMBSTONES_ARR[$index_1]}
                export SLOW_LEARNER_VC_TOMBSTONES=${SLOW_LEARNER_VC_TOMBSTONES_ARR[$index_2]}
                export FAST_LEARNER_VC_LOSS_PCT_THRESHOLD=${FAST_LEARNER_VC_LOSS_PCT_THRESHOLD_ARR[$index_3]}
                export SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD=${SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD_ARR[$index_4]}
#                if [[ ${SLOW_LEARNER_VC_TOMBSTONES} -le ${FAST_LEARNER_VC_TOMBSTONES} && ${SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD} -ge ${FAST_LEARNER_VC_LOSS_PCT_THRESHOLD} ]]; then
                 echo "Hyperparameter Combination:" ${FAST_LEARNER_VC_TOMBSTONES}, ${SLOW_LEARNER_VC_TOMBSTONES}, ${FAST_LEARNER_VC_LOSS_PCT_THRESHOLD}, ${SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD}
                 k=$(($k+1))
#                 /lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/cifar/cifar10_federated_main_sys_args.py --synchronous False --federation_rounds 1 --learning_rate 0.05 --momentum 0.75 --batchsize 100 --update_frequency 0 --execution_time 35
#                fi
            done
        done
    done
done
echo $k