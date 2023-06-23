import abc
import datetime
import queue
import os
import tarfile
import time
import shutil
from typing import Union
import cloudpickle

import multiprocessing as mp
from metisfl.driver.utils import create_server_entity
from metisfl.models.model_wrapper import MetisModel
import metisfl.utils.proto_messages_factory as proto_messages_factory
import metisfl.utils.fedenv_parser as fedenv_parser
 
from pebble import ProcessPool
from metisfl.utils.grpc_controller_client import GRPCControllerClient
from metisfl.utils.grpc_learner_client import GRPCLearnerClient
from metisfl.utils.metis_logger import MetisASCIIArt
from metisfl.utils.init_services_factory import MetisInitServicesCmdFactory
from metisfl.utils.ssl_configurator import SSLConfigurator
from metisfl.encryption import fhe


MODEL_SAVE_DIR = "model_definition"
TRAIN_RECEIPE_FILE = "model_train_dataset_ops.pkl"
VALIDATION_RECEIPE_FILE = "model_validation_dataset_ops.pkl"
TEST_RECEIPE_FILE = "model_test_dataset_ops.pkl"
CRYPTO_RESOURCES_DIR = "resources/fhe/cryptoparams/"

class DriverSessionBase(object):

    def __init__(self,
                 fed_env: Union[fedenv_parser.FederationEnvironment, object],
                 model: MetisModel,
                 train_dataset_recipe_fn: callable,
                 validation_dataset_recipe_fn: callable = None,
                 test_dataset_recipe_fn: callable = None,
                 working_dir="/tmp/metis/"):
        assert train_dataset_recipe_fn is not None, "Train dataset recipe function cannot be None."
    
        # Print welcome message.
        MetisASCIIArt.print()
        
        self.federation_environment = fed_env if isinstance(fed_env, fedenv_parser.FederationEnvironment) \
                                else fedenv_parser.FederationEnvironment(fed_env)
        self.num_learners = len(self.federation_environment.learners.learners)     

        self.prepare_working_dir(working_dir)
        self.save_dataset_receipe(train_dataset_recipe_fn, working_dir, TRAIN_RECEIPE_FILE)
        self.save_dataset_receipe(validation_dataset_recipe_fn, working_dir, VALIDATION_RECEIPE_FILE)
        self.save_dataset_receipe(test_dataset_recipe_fn, working_dir, TEST_RECEIPE_FILE)
        self.save_initial_model(model, working_dir)
        
        self.init_pool()
        self.setup_fh_scheme()
        self._driver_controller_grpc_client = self._create_driver_controller_grpc_client()
        self._driver_learner_grpc_clients = self._create_driver_learner_grpc_clients()

    def prepare_working_dir(self, working_dir):
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        os.makedirs(working_dir)

    def save_dataset_receipe(self, dataset_recipe_fn, working_dir, filename):
        if dataset_recipe_fn:
            train_dataset_pkl = os.path.join(working_dir, filename)
            cloudpickle.dump(obj=dataset_recipe_fn, file=open(train_dataset_pkl, "wb+"))
            self.train_dataset_recipe_fp = train_dataset_pkl
            
    def save_initial_model(self, model, working_dir):
        self._save_model_dir = os.path.join(working_dir, MODEL_SAVE_DIR)
        os.makedirs(self._save_model_dir)
        
        self._model_weights_descriptor = model.get_weights_descriptor()
        model.save(self._save_model_dir)
        
        self.model_definition_tar_fp = self._make_tarfile(
            output_filename=self._save_model_dir_name,
            source_dir=self._save_model_dir
        )
        
    def _make_tarfile(self, output_filename, source_dir):
        output_dir = os.path.abspath(os.path.join(source_dir, os.pardir))
        output_filepath = os.path.join(output_dir, "{}.tar.gz".format(output_filename))
        with tarfile.open(output_filepath, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        return output_filepath

    def init_pool(self):
        # Unix default is "fork", others: "spawn", "forkserver"
        # We use spawn so that the parent process starts a fresh Python interpreter process.
        mp_ctx = mp.get_context("spawn")
        self._executor = ProcessPool(max_workers=self.num_learners + 1, context=mp_ctx)
        self._executor_controller_tasks_q = queue.LifoQueue(maxsize=0)
        self._executor_learners_tasks_q = queue.LifoQueue(maxsize=0)

    def setup_fh_scheme(self):
        self._he_scheme, self._he_scheme_pb = None, None
        if self.federation_environment.homomorphic_encryption:
            if self.federation_environment.homomorphic_encryption.scheme.upper() == "CKKS":
                # Initialize encryption scheme to encode initial model.
                batch_size = self.federation_environment.homomorphic_encryption.batch_size
                scaling_bits = self.federation_environment.homomorphic_encryption.scaling_bits
                self._he_scheme = fhe.CKKS(batch_size, scaling_bits, CRYPTO_RESOURCES_DIR)
                self._he_scheme.load_crypto_params()

                # Construct serialized proto message.
                fhe_scheme_pb = proto_messages_factory.MetisProtoMessages.construct_fhe_scheme_pb(
                    batch_size=batch_size, scaling_bits=scaling_bits)
                self._he_scheme_pb = proto_messages_factory.MetisProtoMessages.construct_he_scheme_pb(
                    enabled=True, name="CKKS", fhe_scheme_pb=fhe_scheme_pb)
        else:
            empty_scheme_pb = proto_messages_factory.MetisProtoMessages.construct_empty_he_scheme_pb()
            self._he_scheme_pb = proto_messages_factory.MetisProtoMessages.construct_he_scheme_pb(
                enabled=False, empty_scheme_pb=empty_scheme_pb)

    def __getstate__(self):
        """
        Python needs to pickle the entire object, including its instance variables.
        Since one of these variables is the Pool object itself, the entire object cannot be pickled.
        We need to remove the Pool() variable from the object state in order to use the pool_task.
        The same holds for the gprc clients, which use a futures thread pool under the hood.
        See also: https://stackoverflow.com/questions/25382455
        """
        self_dict = self.__dict__.copy()
        del self_dict['_driver_controller_grpc_client']
        del self_dict['_driver_learner_grpc_clients']
        del self_dict['_executor']
        del self_dict['_executor_controller_tasks_q']
        del self_dict['_executor_learners_tasks_q']
        del self_dict['_he_scheme']
        return self_dict


    def _create_driver_controller_grpc_client(self):
        # Controller is a subtype of RemoteHost instance, hence we pass it as is.
        controller_server_entity_pb = create_server_entity(
            remote_host_instance=self.federation_environment.controller,
            connection_entity=True)
        grpc_controller_client = GRPCControllerClient(
            controller_server_entity_pb,
            max_workers=1)
        return grpc_controller_client

    def _create_driver_learner_grpc_clients(self):
        grpc_clients = {}
        for learner_instance in self.federation_environment.learners.learners:
            learner_server_entity_pb = create_server_entity(
                learner_instance,
                connection_entity=True)
            grpc_clients[learner_instance.learner_id] = \
                GRPCLearnerClient(learner_server_entity_pb, max_workers=1)
        return grpc_clients

    def _init_controller_cmd(self):
        # To create the controller grpc server entity, we need the hostname to which the server
        # will bind to and the port of the grpc servicer defined in the initial configuration file.
        # Controller is a subtype of RemoteHost instance, hence we pass it as is.
        controller_server_entity_pb = create_server_entity(
            remote_host_instance=self.federation_environment.controller,
            initialization_entity=True)

        communication_specs_pb = proto_messages_factory.MetisProtoMessages.construct_communication_specs_pb(
            protocol=self.federation_environment.communication_protocol.name,
            semi_sync_lambda=self.federation_environment.communication_protocol.semi_synchronous_lambda,
            semi_sync_recompute_num_updates=self.federation_environment.communication_protocol.semi_sync_recompute_num_updates)
        
        optimizer_pb_kwargs = self.federation_environment.local_model_config.optimizer_config.optimizer_pb_kwargs
        optimizer_pb = \
            proto_messages_factory.ModelProtoMessages.construct_optimizer_config_pb_from_kwargs(optimizer_pb_kwargs)
        model_hyperparameters_pb = proto_messages_factory.MetisProtoMessages \
            .construct_controller_modelhyperparams_pb(
                batch_size=self.federation_environment.local_model_config.batch_size,
                epochs=self.federation_environment.local_model_config.local_epochs,
                optimizer_pb=optimizer_pb,
                percent_validation=self.federation_environment.local_model_config.validation_percentage)
        aggregation_rule_pb = proto_messages_factory.MetisProtoMessages.construct_aggregation_rule_pb(
            rule_name=self.federation_environment.global_model_config.aggregation_rule.aggregation_rule_name,
            scaling_factor=self.federation_environment.global_model_config.aggregation_rule.aggregation_rule_scaling_factor,
            stride_length=self.federation_environment.global_model_config.aggregation_rule.aggregation_rule_stride_length,
            he_scheme_pb=self._he_scheme_pb)
        
        global_model_specs_pb = proto_messages_factory.MetisProtoMessages.construct_global_model_specs(
            aggregation_rule_pb=aggregation_rule_pb,
            learners_participation_ratio=self.federation_environment.global_model_config.participation_ratio)
        model_store_config_pb = proto_messages_factory.MetisProtoMessages.construct_model_store_config_pb(
            name=self.federation_environment.model_store_config.name,
            eviction_policy=self.federation_environment.model_store_config.eviction_policy,
            lineage_length=self.federation_environment.model_store_config.eviction_lineage_length,
            store_hostname=self.federation_environment.model_store_config.connection_configs.hostname,
            store_port=self.federation_environment.model_store_config.connection_configs.port)
        init_controller_cmd = MetisInitServicesCmdFactory().init_controller_target(
            controller_server_entity_pb_ser=controller_server_entity_pb.SerializeToString(),
            global_model_specs_pb_ser=global_model_specs_pb.SerializeToString(),
            communication_specs_pb_ser=communication_specs_pb.SerializeToString(),
            model_hyperparameters_pb_ser=model_hyperparameters_pb.SerializeToString(),
            model_store_config_pb_ser=model_store_config_pb.SerializeToString())
        return init_controller_cmd

    def _init_learner_cmd(self, learner_instance, controller_instance):
        remote_metis_path = "/tmp/metis/workdir_learner_{}".format(learner_instance.grpc_servicer.port)

        # The model and the training dataset file will never be empty if we reach this point.
        # Therefore, we do not need to check if they exist or not.
        # TODO: check this 
        remote_metis_model_path = os.path.join(remote_metis_path, self._save_model_dir_name)
        train_dataset_recipe_fp = os.path.join(remote_metis_path, self.train_dataset_recipe_fp.split("/")[-1])

        # However, the validation and the test files may be empty.
        validation_dataset_recipe_fp = ""
        if self.validation_dataset_recipe_fp:
            validation_dataset_recipe_fp = \
                os.path.join(remote_metis_path, self.validation_dataset_recipe_fp.split("/")[-1])

        test_dataset_recipe_fp = ""
        if self.test_dataset_recipe_fp:
            test_dataset_recipe_fp = \
                os.path.join(remote_metis_path, self.test_dataset_recipe_fp.split("/")[-1])

        # To create the controller grpc server entity, we need the hostname to which the server
        # will bind to and the port of the grpc servicer defined in the initial configuration file.
        learner_server_entity_pb = create_server_entity(
            remote_host_instance=learner_instance,
            initialization_entity=True)
        controller_server_entity_pb = create_server_entity(
            remote_host_instance=controller_instance,
            connection_entity=True)
        
        init_learner_cmd = MetisInitServicesCmdFactory().init_learner_target(
            learner_server_entity_pb_ser=learner_server_entity_pb.SerializeToString(),
            controller_server_entity_pb_ser=controller_server_entity_pb.SerializeToString(),
            he_scheme_pb_ser=self._he_scheme_pb.SerializeToString(),
            model_dir=remote_metis_model_path,
            train_dataset=learner_instance.dataset_configs.train_dataset_path,
            validation_dataset=learner_instance.dataset_configs.validation_dataset_path,
            test_dataset=learner_instance.dataset_configs.test_dataset_path,
            train_dataset_recipe=train_dataset_recipe_fp,
            validation_dataset_recipe=validation_dataset_recipe_fp,
            test_dataset_recipe=test_dataset_recipe_fp,
            neural_engine=self._model_weights_descriptor.nn_engine)
        return init_learner_cmd

    def _ship_model_to_controller(self):
        # @stripeli why unpacking the model weights here?
        # This makes the introduction of the model_weights_descriptor redundant.
        # Pass the model_weights_descriptor to the construct_model_pb_from_np method.
        # Readability!
        model_pb = proto_messages_factory.ModelProtoMessages.construct_model_pb_from_np(
            weights_values=self._model_weights_descriptor.weights_values,
            weights_names=self._model_weights_descriptor.weights_names,
            weights_trainable=self._model_weights_descriptor.weights_trainable,
            he_scheme=self._he_scheme)
        self._driver_controller_grpc_client.replace_community_model(
            num_contributors=self.num_learners,
            model_pb=model_pb)

    def _shutdown(self):
        # Collect all statistics related to learners before sending the shutdown signal.
        self._collect_local_statistics()
        # Send shutdown signal to all learners in a Round-Robin fashion.
        for learner_id, grpc_client in self._driver_learner_grpc_clients.items():
            # We send a single non-blocking shutdown request to every learner with a 30secs time-to-live.
            grpc_client.shutdown_learner(request_retries=1, request_timeout=30, block=False)
        # Blocking-call, wait for learners shutdown acknowledgment.
        for learner_id, grpc_client in self._driver_learner_grpc_clients.items():
            grpc_client.shutdown()

        # Collect all statistics related to the global execution before sending the shutdown signal.
        self._collect_global_statistics()

        # Similar to the learners, we also give a bit more time in between requests to
        # the controller since in needs to wrap pending tasks submitted by the learners.
        self._driver_controller_grpc_client.shutdown_controller(request_retries=2, request_timeout=30, block=True)
        self._driver_controller_grpc_client.shutdown()

        self._executor.close()
        self._executor.join()

    def initialize_federation(self):
        """
        This func will create N number of processes/workers to create the federation
        environment. One process for the controller and every other 
        It first initializes the federation controller and then each learner, with some
        lagging time till the federation controller is live so that every learner can
        connect to it.
        """
        # TODO If we need to test the pipeline we force a future return here, i.e., controller_future.result()
        # The following initialization futures are always running (status=running)
        # since we need to keep the connections open in order to retrieve logs
        # regarding the execution progress of the federation.
        controller_future = self._executor.schedule(function=self._init_controller)
        self._executor_controller_tasks_q.put(controller_future)
        if self._driver_controller_grpc_client.check_health_status(request_retries=10, request_timeout=30, block=True):
            self._ship_model_to_controller()
            for learner_instance in self.federation_environment.learners.learners:
                learner_future = self._executor.schedule(
                    function=self._init_learner,
                    args=(learner_instance,
                          self.federation_environment.controller))
                # TODO If we need to test the pipeline we can force a future return here, i.e., learner_future.result()
                self._executor_learners_tasks_q.put(learner_future)
                # TODO We perform a sleep because if the learners are co-located, e.g., localhost, then an exception
                #  is raised by the SSH client: """ Exception (client): Error reading SSH protocol banner """.
                time.sleep(0.1)

    @abc.abstractmethod
    def _init_controller(self):
        pass

    @abc.abstractmethod
    def _init_learner(self, learner_instance, controller_instance):
        pass