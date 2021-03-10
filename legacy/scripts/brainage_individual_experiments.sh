#! /bin/bash -l

# Run the current script as:
#   nohup ./brainage_individual_experiments.sh < /dev/null > nohup_brainage_individual_experiments.out &

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


###### SYNCHRONOUS EXECUTION ######

# ---- FedAvg ---
# Synchronous FedAvg Eval 4xLearners (Uniform & IID)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.4FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.test_adni.cnn5.federation.4FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.test_ukbb.cnn5.federation.4FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.adni.test_adni.cnn5.federation.4FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.adni.test_ukbb.cnn5.federation.4FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.run1.yaml

# Synchronous FedAvg Eval 8xLearners (Uniform & IID)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.run1.yaml

# Synchronous FedAvg Eval 8xLearners (Uniform & IID - ADNI & UKBB)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb_with_adni.test_jointly.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb_with_adni.test_adni.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb_with_adni.test_ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.run1.yaml

# Synchronous FedAvg Eval 8xLearners (Uniform & IID) -- Small Consortia: 200 examples per learner
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.smallconsortia.run1.yaml

# Synchronous FedAvg Eval 8xLearners (Uniform & Non-IID)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_noniid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_noniid.run1.yaml

# Synchronous FedAvg Eval 8xLearners (Uniform & Non-IID) -- Small Consortia: 200 examples per learner
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_noniid.smallconsortia.run1.yaml

# Synchronous FedAvg Eval 8xLearners (Skewed & Non-IID Skewness Factor: 1.25)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.skewed_125_datasize_noniid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.skewed_125_datasize_noniid.run1.yaml

# Synchronous FedAvg Eval 8xLearners (Skewed & Non-IID Skewness Factor: 1.35)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.skewed_135_datasize_noniid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.skewed_135_datasize_noniid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.skewed_135_datasize_noniid.run1.yaml

# Synchronous FedAvg Eval 8xLearners (x7clean, x1corrupted)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.7clean.1corrupted.run1.yaml

# Synchronous FedAvg Eval 8xLearners (x6clean, x2corrupted)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.6clean.2corrupted.run1.yaml

# Synchronous FedAvg Eval 8xLearners (x4clean, x4corrupted)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.4clean.4corrupted.run1.yaml

# Synchronous FedAvg Eval 8xLearners (x4clean, x2blurred, x2noisy)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.4clean.2blurred.2noisy.run1.yaml


# ---- DVW ---
# Synchronous DVW Eval 4xLearners (Uniform & IID)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/DVW/brainage.ukbb.cnn5.federation.4FastLearners_atBDNF.SyncDVW.uniform_datasize_iid.run1.yaml

# Synchronous DVW Eval 8xLearners (Uniform & IID)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/DVW/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncDVW.uniform_datasize_iid.run1.yaml

# Synchronous DVW Eval 8xLearners (Uniform & IID - ADNI & UKBB)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/DVW/brainage.ukbb_with_adni.test_jointly.cnn5.federation.8FastLearners_atBDNF.SyncDVW.uniform_datasize_iid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/DVW/brainage.ukbb_with_adni.test_adni.cnn5.federation.8FastLearners_atBDNF.SyncDVW.uniform_datasize_iid.run1.yaml

# Synchronous DVW Eval 8xLearners (Uniform & IID) -- Small Consortia: 200 examples per learner
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/DVW/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncDVW.uniform_datasize_iid.smallconsortia.run1.yaml

# Synchronous DVW Eval 8xLearners (Uniform & Non-IID)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/DVW/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncDVW.uniform_datasize_noniid.run1.yaml

# Synchronous DVW Eval 8xLearners (Uniform & Non-IID) -- Small Consortia: 200 examples per learner
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/DVW/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncDVW.uniform_datasize_noniid.smallconsortia.run1.yaml

# Synchronous DVW Eval 8xLearners (Skewed & Non-IID)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/DVW/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncDVW.skewed_datasize_noniid.run1.yaml

# Synchronous DVW Eval 8xLearners (x7clean, x1corrupted)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/DVW/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncDVW.uniform_datasize_iid.7clean.1corrupted.run1.yaml

# Synchronous DVW Eval 8xLearners (x6clean, x2corrupted)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/DVW/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncDVW.uniform_datasize_iid.6clean.2corrupted.run1.yaml

# Synchronous DVW Eval 8xLearners (x4clean, x4corrupted)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/DVW/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncDVW.uniform_datasize_iid.4clean.4corrupted.run1.yaml

# Synchronous DVW Eval 8xLearners (x4clean, x2blurred, x2noisy)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/DVW/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncDVW.uniform_datasize_iid.4clean.2blurred.2noisy.run1.yaml


###### SEMI-SYNCHRONOUS EXECUTION ######

# ---- FedAvg ---
# Semi-Synchronous FedAvg Eval 8xLearners (Skewed & Non-IID)
#export SEMI_SYNC_NUMBER_OF_BATCHES=3842

#export SEMI_SYNCHRONOUS_EXECUTION="True"
#export SEMI_SYNCHRONOUS_K_VALUE=2
#export GPU_TIME_PER_BATCH_MS=120
#export CPU_TIME_PER_BATCH_MS=1
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/semisynchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SemiSyncFedAvg.skewed_datasize_noniid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/semisynchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SemiSyncFedAvg.skewed_datasize_noniid.run1.yaml

#export SEMI_SYNCHRONOUS_EXECUTION="True"
#export SEMI_SYNCHRONOUS_K_VALUE=3
#export GPU_TIME_PER_BATCH_MS=120
#export CPU_TIME_PER_BATCH_MS=1
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/semisynchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SemiSyncFedAvg.skewed_datasize_noniid.run1.yaml

#export SEMI_SYNCHRONOUS_EXECUTION="True"
#export SEMI_SYNCHRONOUS_K_VALUE=4
#export GPU_TIME_PER_BATCH_MS=120
#export CPU_TIME_PER_BATCH_MS=1

# Synchronous FedAvg Eval 8xLearners (Uniform & IID)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb_with_adni.test_adni.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.uniform_datasize_iid.run1.yaml


# Synchronous FedAvg Eval 8xLearners (Skewed & Non-IID Skewness Factor: 1.25)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/semisynchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SemiSyncFedAvg.skewed_125_datasize_noniid.run1.yaml

# Synchronous FedAvg Eval 8xLearners (Skewed & Non-IID Skewness Factor: 1.35)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.skewed_135_datasize_noniid.run1.yaml
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/synchronous/FedAvg/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SyncFedAvg.skewed_135_datasize_noniid.run1.yaml


# ---- DVW ---
# Semi-Synchronous DVW Eval 8xLearners (Skewed & Non-IID)
#export SEMI_SYNC_NUMBER_OF_BATCHES=3842
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/semisynchronous/DVW/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.SemiSyncDVW.skewed_datasize_noniid.run1.yaml


###### ASYNCHRONOUS EXECUTION ######

# Asynchronous DVW Eval 8xLearners (Skewed & Non-IID)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/asynchronous/DVW/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.AsyncDVW.skewed_datasize_noniid.run1.yaml

# Asynchronous FedAnnealing 8xLearners (Skewed & Non-IID)
#/lfs1/stripeli/metiscondaenv/bin/python3 $PROJECT_HOME/experiments/run_fedmodels/neuroimaging/brainage_cnn/brainage_cnn5_federated_main.py --federation_environment_filepath /lfs1/stripeli/condaprojects/federatedneuroimaging/projectmetis/resources/config/experiments_configs/brainage/policies/asynchronous/FedAnnealing/brainage.ukbb.cnn5.federation.8FastLearners_atBDNF.FedAnnealing.skewed_datasize_noniid.run1.yaml


