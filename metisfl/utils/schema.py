from schema import Schema, And, Use, Optional, SchemaError, Or

METRICS = ["accuracy", "mse"]
COMMUNICATION_PROTOCOLS = ["Synchronous", "Asynchronous", "SemiSynchronous"]
MODEL_STORES = ["InMemory", "Redis"]
EVICTION_POLICIES = ["LineageLengthEviction", "NoEviction"]
HE_SCHEMES = ["CKKS"]
AGGREGATION_RULES = ["FedAvg", "FedStride", "FedStride", "PWA"]
SCALING_FACTORS = ["NumTrainingExamples", "NUM_COMPLETED_BATCHES",
                   "NUM_PARTICIPANTS", "NUM_TRAINING_EXAMPLES"]
OPTMIZERS = ["VanillaSGD", "MomentumSGD", "Adam", "RMSProp"]

remote_host_schema = Schema({
    Optional("LearnerID"): str,
    "ProjectHome": str,
    "Hostname": str,
    Optional("Port"): And(Use(int), lambda n: n > 0),
    "Username": str,
    Or("Password", "KeyFilename", only_one=True): str,
    "OnLoginCommand": str,
    "GRPCServicerHostname": str,
    "GRPCServicerPort": And(Use(int), lambda n: n > 0),
    "SSLPrivateKey": str,
    "SSLPublicCertificate": str,
    Optional("CudaDevices"): list[int]
})

env_schema = Schema({
    "FederationRounds": And(Use(int), lambda n: n > 0),
    "ExecutionCutoffTimeMins": And(Use(int), lambda n: n > 0),
    "EvaluationMetric": And(str, lambda s: s in METRICS),
    "EvaluationMetricCutoffScore": And(Use(float), lambda n: n > 0),
    "CommunicationProtocol": And(str, lambda s: s in COMMUNICATION_PROTOCOLS),
    "EnableSSL": bool,
    "ModelStore": And(str, lambda s: s in MODEL_STORES),
    Optional("ModelStoreHostname"): str,
    Optional("ModelStorePort"): And(Use(int), lambda n: n > 0),
    "EvictionPolicy": And(str, lambda s: s in EVICTION_POLICIES),
    Optional("LineageLength"): And(Use(int), lambda n: n > 0),
    Optional("HEScheme"): And(str, lambda s: s in HE_SCHEMES),
    Optional("BatchSize"): And(Use(int), lambda n: n > 0),
    Optional("ScalingBits"): And(Use(int), lambda n: n > 0),
    "AggregationRule": And(str, lambda s: s in AGGREGATION_RULES),
    "ScalingFactor": And(Use(str), lambda s: s in SCALING_FACTORS),
    "StrideLength": And(Use(int), lambda n: n > 0),
    "ParticipationRatio": And(Use(float), lambda n: n > 0),
    "BatchSize": And(Use(int), lambda n: n > 0),
    "LocalEpochs": And(Use(int), lambda n: n > 0),
    Optional("ValidationPercentage"): float,
    "Optimizer": And(str, lambda s: s in OPTMIZERS),
    "OptimizerParams": dict,
    "Controller": remote_host_schema,
    "Learners": [remote_host_schema],
})

class SSLSchema(Schema):
    def __init__(self):
        super().__init__({
            "SSLPrivateKey": And(str, lambda s: os.path.exists(s)),
            "SSLPublicCertificate": And(str, lambda s: os.path.exists(s)),
        })

    def __call__(self, d):
        try:
            return super().__call__(d)
        except SchemaError as e:
            raise SchemaError("SSL private key or public certificate not found") from e
    