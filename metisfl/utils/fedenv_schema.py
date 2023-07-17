import os

from schema import Schema, And, Use, Optional, Or
from typing import List


METRICS = ["accuracy", "loss"]
COMMUNICATION_PROTOCOLS = ["Synchronous", "Asynchronous", "SemiSynchronous"]
MODEL_STORES = ["InMemory", "Redis"]
EVICTION_POLICIES = ["LineageLengthEviction", "NoEviction"]
HE_SCHEMES = ["CKKS"]
AGGREGATION_RULES = ["FedAvg", "FedRec", "FedStride", "PWA"]
SCALING_FACTORS = ["NumTrainingExamples", "NUM_COMPLETED_BATCHES",
                   "NUM_PARTICIPANTS", "NUM_TRAINING_EXAMPLES"]
OPTIMIZERS = ["SGD", "Adam", "Adagrad", "Adadelta", "RMSprop"]


def _existing_file(s):
    if not os.path.exists(s):
        raise FileNotFoundError(f"{s} does not exist.")
    return s


remote_host_schema = Schema({
    "ProjectHome": str,
    "Hostname": str,
    Optional("Port"): And(Use(int), lambda n: n > 0),
    "Username": str,
    Or("Password", "KeyFilename", only_one=True): str,
    Optional("Passphrase"): str,
    "OnLoginCommand": str,
    "GRPCServicerHostname": str,
    "GRPCServicerPort": And(Use(int), lambda n: n > 0),
    # These two must exist together
    Optional("SSLPrivateKey"): And(_existing_file, str),
    Optional("SSLPublicCertificate"): And(_existing_file, str),
    Optional("CudaDevices"): List[int]
})

env_schema = Schema({
    # required only on sync protocol
    "FederationRounds": And(Use(int), lambda n: n > 0),
    # required only on async protocol
    "ExecutionCutoffTimeMins": And(Use(int), lambda n: n > 0),
    Optional("EvaluationMetric"): And(str, lambda s: s in METRICS),
    Optional("EvaluationMetricCutoffScore"): And(Use(float), lambda n: n > 0),
    "CommunicationProtocol": And(str, lambda s: s in COMMUNICATION_PROTOCOLS),
    Optional("EnableSSL"): bool,
    # this and hostname/port are required ONLY if REDIS
    Optional("ModelStore"): And(str, lambda s: s in MODEL_STORES),
    Optional("ModelStoreHostname"): str,
    Optional("ModelStorePort"): And(Use(int), lambda n: n > 0),
    # this and lineage length are required TOGETHER
    "EvictionPolicy": And(str, lambda s: s in EVICTION_POLICIES),
    Optional("LineageLength"): And(Use(int), lambda n: n > 0),

    Optional("HEScheme"): And(str, lambda s: s in HE_SCHEMES),
    Optional("HEBatchSize"): And(Use(int), lambda n: n > 0),
    Optional("HEScalingBits"): And(Use(int), lambda n: n > 0),

    # FedRec only if async; PWA only if FHE
    "AggregationRule": And(str, lambda s: s in AGGREGATION_RULES),
    # required; forbitten in fedrec
    "ScalingFactor": And(Use(str), lambda s: s in SCALING_FACTORS),
    # required only if FedStride; forbitten o/w
    Optional("StrideLength"): And(Use(int), lambda n: n > 0),
    "BatchSize": And(Use(int), lambda n: n > 0),
    "LocalEpochs": And(Use(int), lambda n: n > 0),
    "Controller": remote_host_schema,
    "Learners": And(list, lambda l: len(l) > 0, error="Learners must be a non-empty list."),
})
