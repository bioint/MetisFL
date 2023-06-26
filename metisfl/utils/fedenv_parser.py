import yaml
from typing import List

from metisfl.encryption import fhe
from metisfl.models.model_wrapper import ModelWeightsDescriptor
from metisfl.proto import model_pb2
from metisfl.utils.proto_messages_factory import ModelProtoMessages
import metisfl.utils.proto_messages_factory as proto_messages_factory

# FIXME: this can go in the yaml file.
CRYPTO_RESOURCES_DIR = "resources/fhe/cryptoparams/"


class TerminationSignals(object):

    def __init__(self, termination_signals_map):
        self.federation_rounds = termination_signals_map.get("FederationRounds", 100)
        self.execution_time_cutoff_mins = termination_signals_map.get("ExecutionCutoffTimeMins", 1e6)
        if not self.execution_time_cutoff_mins:
            self.execution_time_cutoff_mins = 1e6
        self.metric_cutoff_score = termination_signals_map.get("MetricCutoffScore", 1)


class HomomorphicEncryption(object):

    def __init__(self, homomorphic_encryption_map):
        self.scheme = homomorphic_encryption_map.get("Scheme", None)
        if self.scheme and self.scheme.upper() == "CKKS":
            self.batch_size = homomorphic_encryption_map.get("BatchSize")
            self.scaling_bits = homomorphic_encryption_map.get("ScalingBits")
            self._he_scheme = fhe.CKKS(self.batch_size, self.scaling_bits, CRYPTO_RESOURCES_DIR)
            self._he_scheme.load_crypto_params()
            fhe_scheme_pb = proto_messages_factory.MetisProtoMessages.construct_fhe_scheme_pb(
                    batch_size=self.batch_size, scaling_bits=self.scaling_bits)
            self._he_scheme_pb = proto_messages_factory.MetisProtoMessages.construct_he_scheme_pb(
                    enabled=True, name="CKKS", fhe_scheme_pb=fhe_scheme_pb)
        else:
            empty_scheme_pb = proto_messages_factory.MetisProtoMessages.construct_empty_he_scheme_pb()
            self._he_scheme_pb = proto_messages_factory.MetisProtoMessages.construct_he_scheme_pb(
                enabled=False, empty_scheme_pb=empty_scheme_pb)
            
    @staticmethod
    def from_proto(he_scheme_pb):
        assert isinstance(he_scheme_pb, model_pb2.HEScheme), "Not a valid HE scheme protobuf."
        if he_scheme_pb.HasField("fhe_scheme"):
            he_map = {
                "Scheme": "CKKS",
                "BatchSize": he_scheme_pb.fhe_scheme.batch_size,
                "ScalingBits": he_scheme_pb.fhe_scheme.scaling_bits
            }
            return HomomorphicEncryption(he_map)
        else:
            return HomomorphicEncryption({})
        

    def decrypt_pb_weights(self, variables: List[model_pb2.Model.Variable]):
        assert all([isinstance(var, model_pb2.Model.Variable) for var in variables])
        var_names, var_trainables, var_nps = list(), list(), list()
        for var in variables:
            # Variable specifications.
            var_name = var.name
            var_trainable = var.trainable

            if var.HasField("ciphertext_tensor"):
                assert self._he_scheme is not None, "Need encryption scheme to decrypt tensor."
                # For a ciphertext tensor, first we need to decrypt it, and then load it
                # into a numpy array with the data type specified in the tensor specifications.
                tensor_spec = var.ciphertext_tensor.tensor_spec
                tensor_length = tensor_spec.length
                decoded_value = self._he_scheme.decrypt(tensor_spec.value, tensor_length, 1)
                # Since the tensor is decoded we just need to recreate the numpy array
                # to its original data type and shape.
                np_array = \
                    ModelProtoMessages.TensorSpecProto.proto_tensor_spec_with_list_values_to_numpy_array(
                        tensor_spec, decoded_value)
            elif var.HasField('plaintext_tensor'):
                tensor_spec = var.plaintext_tensor.tensor_spec
                # If the tensor is a plaintext tensor, then we need to read the byte buffer
                # and load the tensor as a numpy array casting it to the specified data type.
                np_array = ModelProtoMessages.TensorSpecProto.proto_tensor_spec_to_numpy_array(tensor_spec)
            else:
                raise RuntimeError("Not a supported tensor type.")

            # Append variable specifications to model's variable list.
            var_names.append(var_name)
            var_trainables.append(var_trainable)
            var_nps.append(np_array)

        # @stripeli what is var_nps? is this the same ase ModelWeightsDescriptor?
        return var_names, var_trainables, var_nps

    def construct_model_pb_from_np(self, weight_descriptor: ModelWeightsDescriptor) -> model_pb2.Model:
        weights_names = weight_descriptor.weights_names
        weights_trainable = weight_descriptor.weights_trainable
        weights_values = weight_descriptor.weights_values
        if not weights_names:
            # Populating weights names with surrogate keys.
            weights_names = ["arr_{}".format(widx) for widx in range(len(weights_values))]
        if weights_trainable:
            # Since weights have not specified as trainable or not, we default all weights to trainable.
            weights_trainable = [True for _ in range(len(weights_values))]

        variables_pb = []
        for w_n, w_t, w_v in zip(weights_names, weights_trainable, weights_values):
            ciphertext = None
            if self._he_scheme is not None:
                ciphertext = self._he_scheme.encrypt(w_v.flatten(), 1)
            # If we have a ciphertext we prioritize it over the plaintext.
            tensor_pb = ModelProtoMessages.construct_tensor_pb(nparray=w_v,
                                                               ciphertext=ciphertext)
            model_var = ModelProtoMessages.construct_model_variable_pb(name=w_n,
                                                                       trainable=w_t,
                                                                       tensor_pb=tensor_pb)
            variables_pb.append(model_var)
        return model_pb2.Model(variables=variables_pb)
    
    def to_proto(self):
        return self._he_scheme_pb

class CommunicationProtocol(object):

    def __init__(self, communication_protocol):
        # If the user just provides the `EnableSSL` field then we enable TLS/SSL.
        self.enable_ssl = communication_protocol.get("EnableSSL", False)
        if self.enable_ssl:
            self.enable_ssl = True
        self.name = communication_protocol.get("Name")
        self.is_asynchronous = self.name.upper() == "ASYNCHRONOUS"
        self.is_synchronous = self.name.upper() == "SYNCHRONOUS"
        self.is_semi_synchronous = self.name.upper() == "SEMI_SYNCHRONOUS"
        self.specifications = communication_protocol.get("Specifications", None)
        self.semi_synchronous_lambda, self.semi_sync_recompute_num_updates = None, None
        if self.specifications and self.is_semi_synchronous:
            self.semi_synchronous_lambda = self.specifications.get("SemiSynchronousLambda", None)
            self.semi_sync_recompute_num_updates = self.specifications.get("SemiSynchronousRecomputeSteps", None)
            
    def to_proto(self):
        return proto_messages_factory.MetisProtoMessages.construct_communication_specs_pb(
            protocol=self.name,
            semi_sync_lambda=self.semi_synchronous_lambda,
            semi_sync_recompute_num_updates=self.semi_sync_recompute_num_updates)
                


class FHEScheme(object):

    def __init__(self, fhe_scheme_map):
        self.scheme_name = fhe_scheme_map.get("Name")
        self.batch_size = fhe_scheme_map.get("BatchSize")
        self.scaling_bits = fhe_scheme_map.get("ScalingBits")


class AggregationRule(object):

    def __init__(self, aggregation_rule_map, homomorphic_encryption):
        self.aggregation_rule_name = aggregation_rule_map.get("Name", None)
        self.aggregation_rule_specifications = aggregation_rule_map.get("RuleSpecifications", {})
        self.aggregation_rule_scaling_factor = \
            self.aggregation_rule_specifications.get("ScalingFactor", None)
        self.aggregation_rule_stride_length = \
            self.aggregation_rule_specifications.get("StrideLength", -1)
        self.homomorphic_encryption = homomorphic_encryption

    def __str__(self):
        return """ RuleName: {}, RuleScalingFactor: {}, RuleStrideLength: {} """.format(
            self.aggregation_rule_name,
            self.aggregation_rule_scaling_factor,
            self.aggregation_rule_stride_length)
    
    def to_proto(self):
        return proto_messages_factory.MetisProtoMessages.construct_aggregation_rule_pb(
            rule_name=self.aggregation_rule_name,
            scaling_factor=self.aggregation_rule_scaling_factor,
            stride_length=self.aggregation_rule_stride_length,
            he_scheme_pb=self.homomorphic_encryption.to_proto())

class GlobalModelConfig(object):

    def __init__(self, global_model_map, homomorphic_encryption):
        self.aggregation_rule = AggregationRule(global_model_map.get("AggregationRule", None), 
                                                homomorphic_encryption)
        self.participation_ratio = global_model_map.get("ParticipationRatio", 1)
        
    def to_proto(self):
        return proto_messages_factory.MetisProtoMessages.construct_global_model_specs(
            aggregation_rule_pb=self.aggregation_rule.to_proto(),
            learners_participation_ratio=self.participation_ratio)

class LocalModelConfig(object):

    def __init__(self, local_model_map):
        self.batch_size = local_model_map.get("BatchSize", 100)
        self.local_epochs = local_model_map.get("LocalEpochs", 5)
        self.validation_percentage = local_model_map.get("ValidationPercentage", 0)
        self.optimizer_config = OptimizerConfig(local_model_map.get("OptimizerConfig"))
        
    def to_proto(self):
        return proto_messages_factory.MetisProtoMessages.construct_local_model_specs(
            batch_size=self.batch_size,
            epochs=self.local_epochs,
            optimizer_pb=self.optimizer_config.to_proto(),
            percent_validation=self.validation_percentage)

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
            
    def to_proto(self):
        return proto_messages_factory.MetisProtoMessages.construct_model_store_config_pb(
                            name=self.name,
                            eviction_policy=self.eviction_policy,
                            lineage_length=self.eviction_lineage_length,
                            store_hostname=self.connection_configs.hostname,
                            store_port=self.connection_configs.port)


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
    
    def to_proto(self):
        return proto_messages_factory.MetisProtoMessages.construct_optimizer_pb(**self.optimizer_pb_kwargs)


class RemoteHost(object):
    def __init__(self, config_map):
        self.connection_configs = ConnectionConfigs(config_map.get("ConnectionConfigs"))
        self.grpc_servicer = GRPCServicer(config_map.get("GRPCServicer"))
        self.ssl_configs = None
        if "SSLConfigs" in config_map:
            self.ssl_configs = SSLConfigs(config_map.get("SSLConfigs"))


class Controller(RemoteHost):

    def __init__(self, controller_map):
        super().__init__(controller_map)
        self.project_home = controller_map.get("ProjectHome", "")
        assert self.project_home, "Need to define ProjectHome for controller."


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


class Learner(RemoteHost):

    def __init__(self, learner_def_map):
        super().__init__(learner_def_map)
        self.learner_id = learner_def_map.get("LearnerID")
        self.project_home = learner_def_map.get("ProjectHome", "")
        assert self.project_home, "Need to define ProjectHome for learner: {}.".format(self.learner_id)
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


class SSLConfigs(object):
    def __init__(self, ssl_config_map):
        self.public_certificate_filepath = ssl_config_map.get("PublicCertificate", None)
        self.private_key_filepath = ssl_config_map.get("PrivateKey", None)


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
        self.hostname = grpc_servicer_map.get("Hostname")
        self.port = grpc_servicer_map.get("Port")
        if not self.hostname and not self.port:
            raise RuntimeError("Malformed (hostname, port) combination. Both values need to be defined.")
        self.public_certificate_path = grpc_servicer_map.get("PublicCertificatePath", None)
        self.private_key_path = grpc_servicer_map.get("PrivateKeyPath", None)


class FederationEnvironment(object):

    def __init__(self, federation_environment_config_fp):
        # Read YAML Configs.
        fstream = open(federation_environment_config_fp).read()
        self.loaded_stream = yaml.load(fstream, Loader=yaml.SafeLoader)

        federation_environment = self.loaded_stream.get("FederationEnvironment")
        # TODO(dstripelis) Needs correction to DockerImage

        self.termination_signals = TerminationSignals(federation_environment.get("TerminationSignals"))
        self.evaluation_metric = federation_environment.get("EvaluationMetric")
        self.communication_protocol = CommunicationProtocol(federation_environment.get("CommunicationProtocol"))
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
        
        # @stripeli: added by panos. We need the HomomorphicEncryption object in the AggregationRule
        # in order to construct the protobuf. This is because the protobuf does not correctly replicate 
        # the structure of the yaml file. FIXME:
        self.global_model_config = GlobalModelConfig(
            federation_environment.get("GlobalModelConfig"), self.homomorphic_encryption)
        
    def get_controller(self):
        return self.loaded_stream["Controller"]


