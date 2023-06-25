import abc
import datetime
import queue
import os
import tarfile
import time
import shutil
import cloudpickle
import inspect

import multiprocessing as mp
import metisfl.utils.proto_messages_factory as proto_messages_factory
import metisfl.utils.fedenv_parser as fedenv_parser
import tensorflow as tf
import torch

from fabric import Connection
from google.protobuf.json_format import MessageToDict
from pebble import ProcessPool
from metisfl.utils.grpc_controller_client import GRPCControllerClient
from metisfl.utils.grpc_learner_client import GRPCLearnerClient
from metisfl.utils.metis_logger import MetisASCIIArt, MetisLogger
from metisfl.utils.init_services_factory import MetisInitServicesCmdFactory
from metisfl.utils.ssl_configurator import SSLConfigurator
from metisfl.models.model_ops import ModelOps
from metisfl.encryption import fhe


class DriverSessionBase(object):

    def __init__(self,
                 fed_env,
                 model,
                 train_dataset_recipe_fn,
                 validation_dataset_recipe_fn=None,
                 test_dataset_recipe_fn=None,
                 working_dir="/tmp/metis/"):
        # Print welcome message.
        MetisASCIIArt.print()
        # If the provided federation environment is not a `FederationEnvironment` object then construct it.
        self.federation_environment = fed_env
        if not isinstance(fed_env, fedenv_parser.FederationEnvironment):
            self.federation_environment = fedenv_parser.FederationEnvironment(fed_env)
        self.num_participating_learners = len(self.federation_environment.learners.learners)

        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        os.makedirs(working_dir)

        self._save_model_dir_name = "model_definition"
        self._save_model_dir = os.path.join(working_dir, self._save_model_dir_name)
        os.makedirs(self._save_model_dir)

        # Extract weights description fom the given model.
        self._model_weights_descriptor = ModelOps(model).get_model_weights()
        if isinstance(model, tf.keras.Model):
            model.save(self._save_model_dir)
        elif isinstance(model, torch.nn.Module):
            model_weights_path = os.path.join(self._save_model_dir, "model_weights.pt")
            model_def_path = os.path.join(self._save_model_dir, "model_def.pkl")
            torch.save(model.state_dict(), model_weights_path)
            cloudpickle.register_pickle_by_value(inspect.getmodule(model))
            cloudpickle.dump(obj=model, file=open(model_def_path, "wb+"))
        else:
            raise RuntimeError("Not a supported model type.")

        self.model_definition_tar_fp = self._make_tarfile(
            output_filename=self._save_model_dir_name,
            source_dir=self._save_model_dir)

        if train_dataset_recipe_fn:
            train_dataset_pkl = os.path.join(working_dir, "model_train_dataset_ops.pkl")
            cloudpickle.dump(obj=train_dataset_recipe_fn, file=open(train_dataset_pkl, "wb+"))
            self.train_dataset_recipe_fp = train_dataset_pkl
        else:
            raise RuntimeError("Train dataset recipe cannot be empty.")

        if validation_dataset_recipe_fn:
            validation_dataset_pkl = os.path.join(working_dir, "model_validation_dataset_ops.pkl")
            cloudpickle.dump(obj=validation_dataset_recipe_fn, file=open(validation_dataset_pkl, "wb+"))
            self.validation_dataset_recipe_fp = validation_dataset_pkl
        else:
            self.validation_dataset_recipe_fp = None

        if test_dataset_recipe_fn:
            test_dataset_pkl = os.path.join(working_dir, "model_test_dataset_ops.pkl")
            cloudpickle.dump(obj=test_dataset_recipe_fn, file=open(test_dataset_pkl, "wb+"))
            self.test_dataset_recipe_fp = test_dataset_pkl
        else:
            self.test_dataset_recipe_fp = None

        # Unix default is "fork", others: "spawn", "forkserver"
        # We use spawn so that the parent process starts a fresh Python interpreter process.
        self._mp_ctx = mp.get_context("spawn")
        self._executor = ProcessPool(max_workers=self.num_participating_learners + 1, context=self._mp_ctx)
        self._executor_controller_tasks_q = queue.LifoQueue(maxsize=0)
        self._executor_learners_tasks_q = queue.LifoQueue(maxsize=0)

        self.enable_ssl = self.federation_environment.communication_protocol.enable_ssl

        self._driver_controller_grpc_client = self._create_driver_controller_grpc_client()
        self._driver_learner_grpc_clients = self._create_driver_learner_grpc_clients()
        # This field is populated at different stages of the entire federated training lifecycle.
        self._federation_statistics = dict()

        self._he_scheme, self._he_scheme_pb = None, None
        if self.federation_environment.homomorphic_encryption:
            if self.federation_environment.homomorphic_encryption.scheme.upper() == "CKKS":
                # Initialize encryption scheme to encode initial model.
                batch_size = self.federation_environment.homomorphic_encryption.batch_size
                scaling_bits = self.federation_environment.homomorphic_encryption.scaling_bits
                self._he_scheme = fhe.CKKS(batch_size, scaling_bits, "resources/fheparams/cryptoparams/")
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

    def _create_server_entity(self,
                              remote_host_instance: fedenv_parser.RemoteHost,
                              initialization_entity=False,
                              connection_entity=False):
        if initialization_entity is False and connection_entity is False:
            raise RuntimeError("One field of Initialization or connection entity needs to be provided.")

        # By default ssl is disabled.
        ssl_config_pb = None
        if self.enable_ssl:
            ssl_configurator = SSLConfigurator()
            if remote_host_instance.ssl_configs:
                # If the given instance has the public certificate and the private key defined
                # then we just wrap the ssl configuration around the files.
                wrap_as_stream = False
                public_cert = remote_host_instance.ssl_configs.public_certificate_filepath
                private_key = remote_host_instance.ssl_configs.private_key_filepath
            else:
                # If the given instance has no ssl configuration files defined, then we use
                # the default non-verified (self-signed) certificates, and we wrap them as streams.
                wrap_as_stream = True
                print("SSL enabled but remote host needs custom config files!", flush=True)
                public_cert, private_key = \
                    ssl_configurator.gen_default_certificates(as_stream=True)

            if connection_entity:
                # We only need to use the public certificate
                # to issue requests to the remote entity,
                # hence the private key is set to None.
                private_key = None

            if wrap_as_stream:
                ssl_config_bundle_pb = \
                    proto_messages_factory.MetisProtoMessages.construct_ssl_config_stream_pb(
                        public_certificate_stream=public_cert,
                        private_key_stream=private_key)
            else:
                ssl_config_bundle_pb = \
                    proto_messages_factory.MetisProtoMessages.construct_ssl_config_files_pb(
                        public_certificate_file=public_cert,
                        private_key_file=private_key)

            ssl_config_pb = \
                proto_messages_factory.MetisProtoMessages.construct_ssl_config_pb(
                    enable_ssl=True,
                    config_pb=ssl_config_bundle_pb)

        # The server entity encapsulates the GRPC servicer to which remote host will
        # spaw its grpc server and listen for incoming requests. It does not refer
        # to the connection configurations used to connect to the remote host.
        server_entity_pb = \
            proto_messages_factory.MetisProtoMessages.construct_server_entity_pb(
                hostname=remote_host_instance.grpc_servicer.hostname,
                port=remote_host_instance.grpc_servicer.port,
                ssl_config_pb=ssl_config_pb)
        return server_entity_pb

    def _create_driver_controller_grpc_client(self):
        # Controller is a subtype of RemoteHost instance, hence we pass it as is.
        controller_server_entity_pb = self._create_server_entity(
            remote_host_instance=self.federation_environment.controller,
            connection_entity=True)
        grpc_controller_client = GRPCControllerClient(
            controller_server_entity_pb,
            max_workers=1)
        return grpc_controller_client

    def _create_driver_learner_grpc_clients(self):
        grpc_clients = {}
        for learner_instance in self.federation_environment.learners.learners:
            learner_server_entity_pb = self._create_server_entity(
                learner_instance,
                connection_entity=True)
            grpc_clients[learner_instance.learner_id] = \
                GRPCLearnerClient(learner_server_entity_pb, max_workers=1)
        return grpc_clients

    def _init_controller_cmd(self):
        # To create the controller grpc server entity, we need the hostname to which the server
        # will bind to and the port of the grpc servicer defined in the initial configuration file.
        # Controller is a subtype of RemoteHost instance, hence we pass it as is.
        controller_server_entity_pb = self._create_server_entity(
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
        learner_server_entity_pb = self._create_server_entity(
            remote_host_instance=learner_instance,
            initialization_entity=True)
        controller_server_entity_pb = self._create_server_entity(
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

    def _make_tarfile(self, output_filename, source_dir):
        output_dir = os.path.abspath(os.path.join(source_dir, os.pardir))
        output_filepath = os.path.join(output_dir, "{}.tar.gz".format(output_filename))
        with tarfile.open(output_filepath, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        return output_filepath

    def _ship_model_to_controller(self):
        model_pb = proto_messages_factory.ModelProtoMessages.construct_model_pb_from_np(
            weights_values=self._model_weights_descriptor.weights_values,
            weights_names=self._model_weights_descriptor.weights_names,
            weights_trainable=self._model_weights_descriptor.weights_trainable,
            he_scheme=self._he_scheme)
        self._driver_controller_grpc_client.replace_community_model(
            num_contributors=self.num_participating_learners,
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
        # TODO(stripeli): Figure out a way to run in DEBUG mode by calling `controller_future.result()`.
        #  This command is useful if we need to test the pipeline
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
                    args=[learner_instance, self.federation_environment.controller])
                # TODO(stripeli): Figure out a way to run in DEBUG mode by calling `learner_future.result()`.
                #  This command is useful if we need to test the pipeline
                self._executor_learners_tasks_q.put(learner_future)
                # We perform a sleep because if the learners are co-located, e.g., localhost, then an exception
                # is raised by the SSH client: """ Exception (client): Error reading SSH protocol banner """.
                time.sleep(0.1)

    def _collect_local_statistics(self):
        learners_pb = self._driver_controller_grpc_client.get_participating_learners()
        learners_collection = learners_pb.learner
        learners_id = [learner.id for learner in learners_collection]
        learners_descriptors_dict = MessageToDict(learners_pb,
                                                  preserving_proto_field_name=True)
        learners_results = self._driver_controller_grpc_client \
            .get_local_task_lineage(-1, learners_id)
        learners_results_dict = MessageToDict(learners_results,
                                              preserving_proto_field_name=True)
        self._federation_statistics["learners_descriptor"] = learners_descriptors_dict
        self._federation_statistics["learners_models_results"] = learners_results_dict

    def _collect_global_statistics(self):
        runtime_metadata_pb = self._driver_controller_grpc_client \
            .get_runtime_metadata(num_backtracks=0)
        runtime_metadata_dict = MessageToDict(runtime_metadata_pb,
                                              preserving_proto_field_name=True)
        community_results = self._driver_controller_grpc_client \
            .get_community_model_evaluation_lineage(-1)
        community_results_dict = MessageToDict(community_results,
                                               preserving_proto_field_name=True)
        self._federation_statistics["federation_runtime_metadata"] = runtime_metadata_dict
        self._federation_statistics["community_model_results"] = community_results_dict

    def get_federation_statistics(self):
        return self._federation_statistics

    def monitor_federation(self, request_every_secs=10):
        federation_rounds_cutoff = self.federation_environment.termination_signals.federation_rounds
        communication_protocol = self.federation_environment.communication_protocol
        execution_time_cutoff_mins = self.federation_environment.termination_signals.execution_time_cutoff_mins
        metric_cutoff_score = self.federation_environment.termination_signals.metric_cutoff_score
        evaluation_metric = self.federation_environment.evaluation_metric

        def monitor_termination_signals():
            # measuring elapsed wall-clock time
            st = datetime.datetime.now()
            signal_not_reached = True
            while signal_not_reached:
                # ping controller for latest execution stats
                time.sleep(request_every_secs)

                metadata_pb = self._driver_controller_grpc_client \
                    .get_runtime_metadata(num_backtracks=0).metadata

                # First condition is to check if we reached the desired
                # number of federation rounds for synchronous execution.
                if communication_protocol.is_synchronous or communication_protocol.is_semi_synchronous:
                    if federation_rounds_cutoff and len(metadata_pb) > 0:
                        current_global_iteration = max([m.global_iteration for m in metadata_pb])
                        if current_global_iteration > federation_rounds_cutoff:
                            MetisLogger.info("Exceeded federation rounds cutoff point. Exiting ...")
                            signal_not_reached = False

                community_results = self._driver_controller_grpc_client \
                    .get_community_model_evaluation_lineage(-1)
                # Need to materialize the iterator in order to get all community results.
                community_results = [x for x in community_results.community_evaluation]

                # Second condition is to check if we reached the
                # desired evaluation score in the test set.
                for res in community_results:
                    test_set_scores = []
                    # Since we evaluate the community model across all learners,
                    # we need to measure the average performance across the test sets.
                    for learner_id, evaluations in res.evaluations.items():
                        if evaluation_metric in evaluations.test_evaluation.metric_values:
                            test_score = evaluations.test_evaluation.metric_values[evaluation_metric]
                            test_set_scores.append(float(test_score))
                    if test_set_scores:
                        mean_test_score = sum(test_set_scores) / len(test_set_scores)
                        if mean_test_score >= metric_cutoff_score:
                            MetisLogger.info("Exceeded evaluation metric cutoff score. Exiting ...")
                            signal_not_reached = False

                # Third condition is to check if we reached the
                # desired execution time cutoff point.
                et = datetime.datetime.now()
                diff_mins = (et - st).seconds / 60
                if diff_mins > execution_time_cutoff_mins:
                    MetisLogger.info("Exceeded execution time cutoff minutes. Exiting ...")
                    signal_not_reached = False

        monitor_termination_signals()
        return

    @abc.abstractmethod
    def _init_controller(self):
        pass

    @abc.abstractmethod
    def _init_learner(self, learner_instance, controller_instance):
        pass


class DriverSession(DriverSessionBase):

    def __init__(self,
                 fed_env,
                 model,
                 train_dataset_recipe_fn,
                 validation_dataset_recipe_fn=None,
                 test_dataset_recipe_fn=None):
        super(DriverSession, self).__init__(
            fed_env,
            model,
            train_dataset_recipe_fn,
            validation_dataset_recipe_fn,
            test_dataset_recipe_fn)

    def _init_controller(self):
        fabric_connection_config = self.federation_environment.controller \
            .connection_configs.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        # We do not use asynchronous or disown, since we want the remote subprocess to return standard (error) output.
        # see also, https://github.com/pyinvoke/invoke/blob/master/invoke/runners.py#L109
        remote_metis_path = "/tmp/metis/controller"
        # Delete existing directory if it exists, then recreate it.
        connection.run("rm -rf {}".format(remote_metis_path))
        connection.run("mkdir -p {}".format(remote_metis_path))
        remote_on_login = self.federation_environment.controller.connection_configs.on_login
        if len(remote_on_login) > 0 and remote_on_login[-1] == ";":
            remote_on_login = remote_on_login[:-1]

        init_cmd = "{} && cd {} && {}".format(
            remote_on_login,
            self.federation_environment.controller.project_home,
            self._init_controller_cmd())
        MetisLogger.info("Running init cmd to controller host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()
        return

    def _init_learner(self, learner_instance, controller_instance):
        fabric_connection_config = \
            learner_instance.connection_configs.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        # We do not use asynchronous or disown, since we want the remote subprocess to return standard (error) output.
        remote_metis_path = "/tmp/metis/workdir_learner_{}".format(learner_instance.grpc_servicer.port)
        # Delete existing directory if it exists, then recreate it.
        connection.run("rm -rf {}".format(remote_metis_path))
        connection.run("mkdir -p {}".format(remote_metis_path))
        # Place/Copy model definition and dataset recipe files from the driver to the remote host.
        # Model definition ship .gz file and decompress it.
        MetisLogger.info("Copying model definition and dataset recipe files at learner: {}"
                         .format(learner_instance.learner_id))

        if self.model_definition_tar_fp:
            connection.put(self.model_definition_tar_fp, remote_metis_path)

        if self.train_dataset_recipe_fp:
            connection.put(self.train_dataset_recipe_fp, remote_metis_path)

        if self.validation_dataset_recipe_fp:
            connection.put(self.validation_dataset_recipe_fp, remote_metis_path)

        if self.test_dataset_recipe_fp:
            connection.put(self.test_dataset_recipe_fp, remote_metis_path)

        # Fabric runs every command on a non-interactive mode and therefore the $PATH that might be set for a
        # running user might not be visible while running the command. A workaround is to always
        # source the respective bash_environment files.
        cuda_devices_str = ""
        # Exporting this environmental variable works for both Tensorflow/Keras and PyTorch.
        if learner_instance.cuda_devices is not None and len(learner_instance.cuda_devices) > 0:
            cuda_devices_str = "export CUDA_VISIBLE_DEVICES=\"{}\" " \
                .format(",".join([str(c) for c in learner_instance.cuda_devices]))
        remote_on_login = learner_instance.connection_configs.on_login
        if len(remote_on_login) > 0 and remote_on_login[-1] == ";":
            remote_on_login = remote_on_login[:-1]

        # Un-taring model definition zipped file.
        MetisLogger.info("Un-taring model definition files at learner: {}"
                         .format(learner_instance.learner_id))
        connection.run("cd {}; tar -xvzf {}".format(
            remote_metis_path,
            self.model_definition_tar_fp))

        init_cmd = "{} && {} && cd {} && {}".format(
            remote_on_login,
            cuda_devices_str,
            learner_instance.project_home,
            self._init_learner_cmd(learner_instance, controller_instance))
        MetisLogger.info("Running init cmd to learner host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()
        return

    def shutdown_federation(self):
        self._shutdown()
