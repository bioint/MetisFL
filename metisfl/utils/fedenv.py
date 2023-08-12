import yaml

from dataclasses import dataclass
from typing import List, Optional
from metisfl.utils.formatting import DataTypeFormatter

METRICS = ["accuracy"]
COMMUNICATION_PROTOCOLS = ["Synchronous", "Asynchronous", "SemiSynchronous"]
MODEL_STORES = ["InMemory", "Redis"]
EVICTION_POLICIES = ["LineageLengthEviction",
                     "NoEviction", "LineageLengthEviction"]
HE_SCHEMES = ["CKKS"]
AGGREGATION_RULES = ["FedAvg", "FedRec", "FedStride", "SecAgg"]
SCALING_FACTORS = ["NumTrainingExamples",
                   "NumCompletedBatches", "NumParticipants"]


def file_exists(path: str) -> bool:
    try:
        open(path, 'r')
        return True
    except FileNotFoundError:
        return False


@dataclass
class TerminationSingals(object):
    """Termination signals for the federated training. Controls when the training should stop."""
    
    federation_rounds: int
    federation_rounds_cutoff: int
    execution_cutoff_time_mins: int
    evaluation_metric: str
    evaluation_metric_cutoff_score: float

    @classmethod
    def from_yaml(cls, yaml_dict: dict) -> 'TerminationSingals':
        yaml_dict = DataTypeFormatter.camel_to_snake_dict_keys(yaml_dict)
        return cls(**yaml_dict)

    def __post_init__(self) -> None:
        if self.evaluation_metric not in METRICS:
            raise ValueError(
                f"Invalid evaluation metric: {self.evaluation_metric}")


@dataclass
class ModelStoreConfig(object):
    """Configuration for the model store. Controls where the model is stored and how it is evicted."""
    
    model_store: str
    eviction_policy: str
    model_store_hostname: Optional[str] = None
    model_store_port: Optional[int] = None
    lineage_length: Optional[int] = None

    @classmethod
    def from_yaml(cls, yaml_dict: dict) -> 'ModelStoreConfig':
        yaml_dict = DataTypeFormatter.camel_to_snake_dict_keys(yaml_dict)
        return cls(**yaml_dict)

    def __post_init__(self):
        if self.model_store not in MODEL_STORES:
            raise ValueError(f"Invalid model store: {self.model_store}")
        if self.eviction_policy not in EVICTION_POLICIES:
            raise ValueError(
                f"Invalid eviction policy: {self.eviction_policy}")
        if self.model_store == "Redis":
            if self.model_store_hostname is None:
                raise ValueError(
                    "Redis model store requires a hostname to be specified")
            if self.model_store_port is None:
                raise ValueError(
                    "Redis model store requires a port to be specified")
        if self.eviction_policy == "LineageLengthEviction":
            if self.lineage_length is None:
                raise ValueError(
                    "LineageLengthEviction requires a lineage length to be specified")


@dataclass
class GlobalTrainConfig(object):
    """Configuration for the federated training. Controls how the training is performed."""
    
    aggregation_rule: str
    communication_protocol: str
    scaling_factor: int
    participation_ratio: float
    stride_length: Optional[int] = None
    encryption_scheme: Optional[str] = None
    he_batch_size: Optional[int] = None
    he_scaling_factor_bits: Optional[int] = None
    semi_sync_lambda: Optional[float] = None
    semi_sync_recompute_num_updates: Optional[int] = None


    @classmethod
    def from_yaml(cls, yaml_dict: dict) -> 'GlobalTrainConfig':
        yaml_dict = DataTypeFormatter.camel_to_snake_dict_keys(yaml_dict)
        return cls(**yaml_dict)

    def __post_init__(self):
        if self.aggregation_rule not in AGGREGATION_RULES:
            raise ValueError(
                f"Invalid aggregation rule: {self.aggregation_rule}")
        if self.communication_protocol not in COMMUNICATION_PROTOCOLS:
            raise ValueError(
                f"Invalid communication protocol: {self.protocol}")
        if self.scaling_factor not in SCALING_FACTORS:
            raise ValueError(f"Invalid scaling factor: {self.scaling_factor}")
        if self.encryption_scheme not in HE_SCHEMES:
            raise ValueError(
                f"Invalid encryption scheme: {self.encryption_scheme}")


@dataclass
class LocalTrainConfig(object):
    """Configuration for the local training. Controls how the local training is performed."""
    
    batch_size: int
    local_epochs: int

    @classmethod
    def from_yaml(cls, yaml_dict):
        yaml_dict = DataTypeFormatter.camel_to_snake_dict_keys(yaml_dict)
        return cls(**yaml_dict)


@dataclass
class ServerParams(object):
    """Server configuration parameters."""
    
    hostname: str
    port: int
    root_certificate: Optional[str] = None
    server_certificate: Optional[str] = None
    private_key: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_dict: dict) -> 'ServerParams':
        yaml_dict = DataTypeFormatter.camel_to_snake_dict_keys(yaml_dict)
        return cls(**yaml_dict)

    def __post_init__(self):
        if self.root_certificate is not None and not file_exists(self.root_certificate):
            raise ValueError(
                f"Root certificate file {self.root_certificate} does not exist")
        if self.server_certificate is not None and not file_exists(self.server_certificate):
            raise ValueError(
                f"Server certificate file {self.server_certificate} does not exist")
        if self.private_key is not None and not file_exists(self.private_key):
            raise ValueError(
                f"Private key file {self.private_key} does not exist")


@dataclass
class ClientParams(object):
    hostname: str
    port: int
    root_certificate: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_dict: dict) -> 'ClientParams':
        yaml_dict = DataTypeFormatter.camel_to_snake_dict_keys(yaml_dict)
        return cls(**yaml_dict)

    def __post_init__(self):
        if self.root_certificate is not None and not file_exists(self.root_certificate):
            raise ValueError(
                f"Root certificate file {self.root_certificate} does not exist")


KEY_TO_CLASS = {
    "global_train_config": GlobalTrainConfig,
    "termination_signals": TerminationSingals,
    "model_store_config": ModelStoreConfig,
    "local_train_config": LocalTrainConfig,
}


@dataclass
class FederationEnvironment(object):
    """MetisFL federated training environment configuration."""
    
    global_train_config: GlobalTrainConfig
    termination_signals: TerminationSingals
    model_store_config: ModelStoreConfig
    local_train_config: LocalTrainConfig

    controller: ServerParams
    learners: List[ServerParams]

    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'FederationEnvironment':
        
        yaml_dict = yaml.safe_load(open(yaml_file, 'r'))
        
        yaml_dict = DataTypeFormatter.camel_to_snake_dict_keys(yaml_dict)

        for key, value in yaml_dict.items():
            if key in KEY_TO_CLASS:
                yaml_dict[key] = KEY_TO_CLASS[key].from_yaml(value)

        yaml_dict['controller'] = ServerParams.from_yaml(
            yaml_dict['controller'])
        
        yaml_dict['learners'] = [ServerParams.from_yaml(
            learner) for learner in yaml_dict['learners']]
        
        return cls(**yaml_dict)
