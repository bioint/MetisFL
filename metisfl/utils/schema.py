from schema import Schema, And, Use, Optional, Or

from metisfl.proto import model_pb2

OPTIMIZER_PB_MAP = {
    "VanillaSGD": model_pb2.VanillaSGD,
    "MomentumSGD": model_pb2.MomentumSGD,
    "FedProx": model_pb2.FedProx,
    "Adam": model_pb2.Adam,
    "AdamWeightDecay": model_pb2.AdamWeightDecay,
}

METRICS = ["accuracy", "mse"]
COMMUNICATION_PROTOCOLS = ["Synchronous", "Asynchronous", "SemiSynchronous"]
MODEL_STORES = ["InMemory", "Redis"]
EVICTION_POLICIES = ["LineageLengthEviction", "NoEviction"]
HE_SCHEMES = ["CKKS"]
AGGREGATION_RULES = ["FedAvg", "FedRec", "FedStride", "PWA"]
SCALING_FACTORS = ["NumTrainingExamples", "NUM_COMPLETED_BATCHES",
                   "NUM_PARTICIPANTS", "NUM_TRAINING_EXAMPLES"]

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
    "SSLPrivateKey": str, 
    "SSLPublicCertificate": str,
    Optional("CudaDevices"): list[int]
})

env_schema = Schema({
    "FederationRounds": And(Use(int), lambda n: n > 0), # required only on sync protocol 
    "ExecutionCutoffTimeMins": And(Use(int), lambda n: n > 0), # required only on async protocol
    Optional("EvaluationMetric"): And(str, lambda s: s in METRICS),
    Optional("EvaluationMetricCutoffScore"): And(Use(float), lambda n: n > 0),
    "CommunicationProtocol": And(str, lambda s: s in COMMUNICATION_PROTOCOLS),
    Optional("EnableSSL"): bool,
    Optional("ModelStore"): And(str, lambda s: s in MODEL_STORES), # this and hostname/port are required ONLY if REDIS
    Optional("ModelStoreHostname"): str,
    Optional("ModelStorePort"): And(Use(int), lambda n: n > 0),
    "EvictionPolicy": And(str, lambda s: s in EVICTION_POLICIES), # this and lineage length are required TOGETHER
    Optional("LineageLength"): And(Use(int), lambda n: n > 0),
    
    Optional("HEScheme"): And(str, lambda s: s in HE_SCHEMES),
    Optional("BatchSize"): And(Use(int), lambda n: n > 0),
    Optional("ScalingBits"): And(Use(int), lambda n: n > 0),
    
    "AggregationRule": And(str, lambda s: s in AGGREGATION_RULES), # FedRec only if async; PWA only if FHE
    "ScalingFactor": And(Use(str), lambda s: s in SCALING_FACTORS), # required; forbitten in fedrec
    "StrideLength": And(Use(int), lambda n: n > 0), # required only if FedStride; forbitten o/w 
    Optional("ParticipationRatio"): And(Use(float), lambda n: n > 0 and n <= 1),  
    "BatchSize": And(Use(int), lambda n: n > 0),
    "LocalEpochs": And(Use(int), lambda n: n > 0),
    "Optimizer": And(str, lambda s: s in OPTIMIZER_PB_MAP.keys()),
    "OptimizerParams": dict,
    "Controller": remote_host_schema,
    "Learners": [remote_host_schema],
})
