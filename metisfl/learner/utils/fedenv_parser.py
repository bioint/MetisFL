import yaml


class Docker(object):

    def __init__(self, docker_image):
        self.docker_image = docker_image


class TerminationSignals(object):

    def __init__(self, termination_signals_map):
        self.federation_rounds = termination_signals_map.get("FederationRounds", 100)
        self.execution_time_cutoff_mins = termination_signals_map.get("ExecutionCutoffTimeMins", 1e6)
        if not self.execution_time_cutoff_mins:
            self.execution_time_cutoff_mins = 1e6
        self.metric_cutoff_score = termination_signals_map.get("MetricCutoffScore", 1)


class HomomorphicEncryption(object):

    def __init__(self, homomorphic_encryption_map):
        self.scheme = homomorphic_encryption_map.get("Scheme")
        if self.scheme.upper() == "CKKS":
            self.batch_size = homomorphic_encryption_map.get("BatchSize")
            self.scaling_bits = homomorphic_encryption_map.get("ScalingBits")


class CommunicationProtocol(object):

    def __init__(self, communication_protocol):
        self.name = communication_protocol.get("Name")
        self.is_asynchronous = self.name.upper() == "ASYNCHRONOUS"
        self.is_synchronous = self.name.upper() == "SYNCHRONOUS"
        self.is_semi_synchronous = self.name.upper() == "SEMI_SYNCHRONOUS"
        self.specifications = communication_protocol.get("Specifications", None)
        self.semi_synchronous_lambda, self.semi_sync_recompute_num_updates = None, None
        if self.specifications and self.is_semi_synchronous:
            self.semi_synchronous_lambda = self.specifications.get("SemiSynchronousLambda", None)
            self.semi_sync_recompute_num_updates = self.specifications.get("SemiSynchronousRecomputeSteps", None)


class FHEScheme(object):

    def __init__(self, fhe_scheme_map):
        self.scheme_name = fhe_scheme_map.get("Name")
        self.batch_size = fhe_scheme_map.get("BatchSize")
        self.scaling_bits = fhe_scheme_map.get("ScalingBits")


class AggregationRule(object):

    def __init__(self, aggregation_rule_map):
        self.aggregation_rule_name = aggregation_rule_map.get("Name", None)
        self.aggregation_rule_specifications = aggregation_rule_map.get("RuleSpecifications", {})
        self.aggregation_rule_scaling_factor = \
            self.aggregation_rule_specifications.get("ScalingFactor", None)
        self.aggregation_rule_stride_length = \
            self.aggregation_rule_specifications.get("StrideLength", -1)

    def __str__(self):
        return """ RuleName: {}, RuleScalingFactor: {}, RuleStrideLength: {} """.format(
            self.aggregation_rule_name,
            self.aggregation_rule_scaling_factor,
            self.aggregation_rule_stride_length)


class GlobalModelConfig(object):

    def __init__(self, global_model_map):
        self.aggregation_rule = AggregationRule(global_model_map.get("AggregationRule", None))
        self.participation_ratio = global_model_map.get("ParticipationRatio", 1)


class LocalModelConfig(object):

    def __init__(self, local_model_map):
        self.batch_size = local_model_map.get("BatchSize", 100)
        self.local_epochs = local_model_map.get("LocalEpochs", 5)
        self.validation_percentage = local_model_map.get("ValidationPercentage", 0)
        self.optimizer_config = OptimizerConfig(local_model_map.get("OptimizerConfig"))


class ModelStoreConfig(object):

    def __init__(self, model_store_map):
        if not model_store_map:
            self.name = "InMemory"
            self.eviction_policy = "LineageLengthEviction"
            self.eviction_lineage_length = 1
            self.connection_configs = ConnectionConfigsBase({})
        else:
            self.name = model_store_map.get("Name", None)
            self.eviction_policy = model_store_map.get("EvictionPolicy")
            self.eviction_lineage_length = model_store_map.get("LineageLength", 1)
            self.connection_configs = ConnectionConfigsBase(model_store_map.get("ConnectionConfigs", {}))


class OptimizerConfig(object):

    def __init__(self, optimizer_map):
        self.optimizer_name = optimizer_map.get("OptimizerName")
        self.learning_rate = optimizer_map.get("LearningRate")
        self.optimizer_pb_kwargs = self.create_optimizer_pb_kwargs(optimizer_map)

    def create_optimizer_pb_kwargs(self, optimizer_map):
        optimizer_pb_kwargs = dict()
        optimizer_pb_kwargs["learning_rate"] = self.learning_rate
        # For the optimizer name we use the name from the proto messages.
        if self.optimizer_name.upper() == "VANILLASGD":
            optimizer_pb_kwargs["name"] = "VanillaSGD"
            if "L1Reg" in optimizer_map:
                optimizer_pb_kwargs["l1_reg"] = optimizer_map['L1Reg']
            if "L2Reg" in optimizer_map:
                optimizer_pb_kwargs["l2_reg"] = optimizer_map['L2Reg']
        elif self.optimizer_name.upper() == "MOMENTUMSGD":
            optimizer_pb_kwargs["name"] = "MomentumSGD"
            if "MomentumFactor" in optimizer_map:
                optimizer_pb_kwargs["momentum_factor"] = optimizer_map['MomentumFactor']
        elif self.optimizer_name.upper() == "FEDPROX":
            optimizer_pb_kwargs["name"] = "FedProx"
            if "ProximalTerm" in optimizer_map:
                optimizer_pb_kwargs["proximal_term"] = optimizer_map['ProximalTerm']
        elif self.optimizer_name.upper() == "ADAM":
            optimizer_pb_kwargs["name"] = "Adam"
            if "Beta1" in optimizer_map:
                optimizer_pb_kwargs["beta_1"] = optimizer_map['Beta1']
            if "Beta2" in optimizer_map:
                optimizer_pb_kwargs["beta_2"] = optimizer_map['Beta2']
            if "Epsilon" in optimizer_map:
                optimizer_pb_kwargs["epsilon"] = optimizer_map['Epsilon']
        elif self.optimizer_name.upper() == "ADAMW":
            optimizer_pb_kwargs["name"] = "AdamWeightDecay"
            optimizer_pb_kwargs["weight_decay"] = 1e-4
            if "WeightDecay" in optimizer_map:
                optimizer_pb_kwargs["weight_decay"] = optimizer_map['WeightDecay']
        else:
            raise RuntimeError("Not a supported optimizer.")
        return optimizer_pb_kwargs


class Controller(object):

    def __init__(self, controller_map):
        self.project_home = controller_map.get("ProjectHome", "")
        assert self.project_home, "Need to define ProjectHome for controller."
        self.connection_configs = ConnectionConfigs(controller_map.get("ConnectionConfigs"))
        self.grpc_servicer = GRPCServicer(controller_map.get("GRPCServicer"))


class Learners(object):

    def __init__(self, learners_map):
        self.learners = []
        for learner_def in learners_map:
            self.learners.append(Learner(learner_def))

    def __iter__(self):
        self.itr_index = 0
        return self

    def __next__(self):
        end = len(self.learners)
        if self.itr_index >= end:
            raise StopIteration
        current = self.learners[self.itr_index]
        self.itr_index += 1
        return current


class Learner(object):

    def __init__(self, learner_def_map):
        self.learner_id = learner_def_map.get("LearnerID")
        self.project_home = learner_def_map.get("ProjectHome", "")
        assert self.project_home, "Need to define ProjectHome for learner: {}.".format(self.learner_id)
        self.connection_configs = ConnectionConfigs(learner_def_map.get("ConnectionConfigs"))
        self.grpc_servicer = GRPCServicer(learner_def_map.get('GRPCServicer'))
        self.cuda_devices = learner_def_map.get('CudaDevices', [])
        self.dataset_configs = DatasetConfigs(learner_def_map.get("DatasetConfigs"))

    def __str__(self):
        return """ LearnerID: {}, ProjectHome: {}, ConnectionConfigs: {}, GRPCServicer:{}, CUDA_DEVICES: {}," \
               "DatasetConfigs: {}""".format(
            self.learner_id,
            self.project_home,
            self.connection_configs,
            self.grpc_servicer,
            self.cuda_devices,
            self.dataset_configs)


class ConnectionConfigsBase(object):

    def __init__(self, connection_configs_map):
        self.hostname = connection_configs_map.get("Hostname", None)
        self.port = connection_configs_map.get("Port", None)


class ConnectionConfigs(ConnectionConfigsBase):

    def __init__(self, connection_configs_map):
        super().__init__(connection_configs_map)
        self.username = connection_configs_map.get("Username", "")
        # Username is necessary to establish connection.
        assert self.username, "Need to define username."
        # Password might not be necessary, e.g., if key filename is provided.
        self.password = connection_configs_map.get("Password", "")

        self.key_filename = connection_configs_map.get("KeyFilename", "")
        self.passphrase = connection_configs_map.get("Passphrase", "")
        self.on_login = connection_configs_map.get("OnLogin", "clear")

    def get_netmiko_connection_config(self):
        conn_config = {
            "device_type": "linux",
            "host": self.hostname,
            "username": self.username,
            "password": self.password
        }
        return conn_config

    def get_fabric_connection_config(self):
        # Look for parameters values here:
        # https://docs.paramiko.org/en/latest/api/client.html#paramiko.client.SSHClient.connect
        # 'allow_agent' show be disabled if working with username/password.
        connect_kwargs = {
            "password": self.password,
            "allow_agent": False if self.password else True,
            "look_for_keys": True if self.key_filename else False,
        }
        if self.key_filename:
            connect_kwargs["key_filename"] = self.key_filename
            connect_kwargs["passphrase"] = self.passphrase

        conn_config = {
            "host": self.hostname,
            "port": self.port,
            "user": self.username,
            "connect_kwargs": connect_kwargs
        }
        return conn_config


class DatasetConfigs(object):

    def __init__(self, dataset_configs_map):
        self.train_dataset_path = dataset_configs_map.get("TrainDatasetPath")
        self.validation_dataset_path = dataset_configs_map.get("ValidationDatasetPath", "")
        self.test_dataset_path = dataset_configs_map.get("TestDatasetPath", "")


class GRPCServicer(object):

    def __init__(self, grpc_servicer_map):
        self.port = grpc_servicer_map.get("Port")


class FederationEnvironment(object):

    def __init__(self, federation_environment_config_fp):
        # Read YAML Configs.
        fstream = open(federation_environment_config_fp).read()
        self.loaded_stream = yaml.load(fstream, Loader=yaml.SafeLoader)

        federation_environment = self.loaded_stream.get("FederationEnvironment")
        # TODO(dstripelis) Needs correction to DockerImage
        self.docker = Docker(federation_environment.get("Docker")) 
        self.termination_signals = TerminationSignals(federation_environment.get("TerminationSignals"))
        self.evaluation_metric = federation_environment.get("EvaluationMetric")
        self.communication_protocol = CommunicationProtocol(federation_environment.get("CommunicationProtocol"))
        self.global_model_config = GlobalModelConfig(federation_environment.get("GlobalModelConfig"))
        self.local_model_config = LocalModelConfig(federation_environment.get("LocalModelConfig"))
        # The model store config is not mandatory, hence the None value if the
        # store configuration is not defined in the environment's yaml file.
        self.model_store_config = ModelStoreConfig(federation_environment.get("ModelStoreConfig", None))
        self.controller = Controller(federation_environment.get("Controller"))
        self.learners = Learners(federation_environment.get("Learners"))

        self.homomorphic_encryption = None
        if "HomomorphicEncryption" in federation_environment:
            self.homomorphic_encryption = HomomorphicEncryption(federation_environment.get("HomomorphicEncryption"))
            # To use homomorphic encryption (fully, partial, somewhat) the user needs to define
            # specific aggregation functions. In particular:
            #   - Case 1: Fully Homomorphic Encryption with Private Weighted Aggregation - (FHE, PWA)
            # TODO Expand when we get support additional encryption schemes and aggregation functions.
            assert self.global_model_config.aggregation_rule.aggregation_rule_name == "PWA", \
                "Since you have enabled Homomorphic Encryption, you need to use the PWA aggregation function."

