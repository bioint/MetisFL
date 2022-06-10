import abc
import datetime
import queue
import os
import statistics
import tarfile
import time

import projectmetis.python.utils.proto_messages_factory as proto_messages_factory

from fabric import Connection
from google.protobuf.json_format import MessageToDict
from pebble import ThreadPool
from projectmetis.python.utils.grpc_controller_client import GRPCControllerClient
from projectmetis.python.utils.grpc_learner_client import GRPCLearnerClient
from projectmetis.python.logging.metis_logger import MetisLogger
from projectmetis.python.utils.bazel_services_factory import BazelMetisServicesCmdFactory
from projectmetis.python.utils.docker_services_factory import DockerMetisServicesCmdFactory
from projectmetis.python.utils.fedenv_parser import FederationEnvironment


class DriverSessionBase(object):

    def __init__(self, fed_env, nn_engine, model_definition_dir,
                 train_dataset_recipe_fp, validation_dataset_recipe_fp="", test_dataset_recipe_fp=""):
        # If the provided federation environment is not a `FederationEnvironment` object then construct it.
        self.federation_environment = \
            fed_env if isinstance(fed_env, FederationEnvironment) else FederationEnvironment(fed_env)
        self.num_participating_learners = len(self.federation_environment.learners.learners)
        self.nn_engine = nn_engine
        self.model_definition_tar_fp = self._make_tarfile(
            output_filename="model_definition",
            source_dir=model_definition_dir)
        self.train_dataset_recipe_fp = train_dataset_recipe_fp
        self.validation_dataset_recipe_fp = validation_dataset_recipe_fp
        self.test_dataset_recipe_fp = test_dataset_recipe_fp

        self._executor = ThreadPool()
        self._executor_controller_tasks_q = queue.LifoQueue(maxsize=0)
        self._executor_learners_tasks_q = queue.LifoQueue(maxsize=0)
        self._driver_controller_grpc_client = self._create_driver_controller_grpc_client()
        self._driver_learner_grpc_clients = self._create_driver_learner_grpc_clients()
        # This field is populated at different stages of the entire federated training lifecycle.
        self._federation_statistics = dict()

    def __getstate__(self):
        """
        Python needs to pickle the entire object, including its instance variables.
        Since one of these variables is the Pool object itself, the entire object cannot be pickled.
        We need to remove the Pool() variable from the object state in order to use the pool_task.
        The same holds for the gprc clients, which use a futures thread pool under the hood.
        See also: https://stackoverflow.com/questions/25382455
        """
        self_dict = self.__dict__.copy()
        del self_dict['_executor']
        del self_dict['_executor_controller_tasks_q']
        del self_dict['_executor_learners_tasks_q']
        del self_dict['_driver_controller_grpc_client']
        del self_dict['_driver_learner_grpc_clients']
        return self_dict

    def _create_driver_controller_grpc_client(self):
        controller_server_entity_pb = \
            proto_messages_factory.MetisProtoMessages.construct_server_entity_pb(
                hostname=self.federation_environment.controller.connection_configs.hostname,
                port=self.federation_environment.controller.grpc_servicer.port)
        return GRPCControllerClient(controller_server_entity_pb, max_workers=1)

    def _create_driver_learner_grpc_clients(self):
        grpc_clients = {}
        for learner_instance in self.federation_environment.learners.learners:
            learner_server_entity_pb = \
                proto_messages_factory.MetisProtoMessages.construct_server_entity_pb(
                    hostname=learner_instance.connection_configs.hostname,
                    port=learner_instance.grpc_servicer.port)
            grpc_clients[learner_instance.learner_id] = \
                GRPCLearnerClient(learner_server_entity_pb, max_workers=1)
        return grpc_clients

    def _init_controller_bazel_cmd(self):
        protocol_pb = proto_messages_factory.MetisProtoMessages.construct_communication_specs_pb(
            protocol=self.federation_environment.communication_protocol.name,
            semi_sync_lambda=self.federation_environment.communication_protocol.semi_synchronous_lambda,
            semi_sync_recompute_num_updates=self.federation_environment.communication_protocol.semi_sync_recompute_num_updates)
        protocol_pb_serialized = protocol_pb.SerializeToString()
        optimizer_pb = self.federation_environment.local_model_config.optimizer_config.optimizer_pb
        model_hyperparameters_pb = proto_messages_factory.MetisProtoMessages \
            .construct_controller_modelhyperparams_pb(
             batch_size=self.federation_environment.local_model_config.batch_size,
             epochs=self.federation_environment.local_model_config.local_epochs,
             optimizer_pb=optimizer_pb,
             percent_validation=self.federation_environment.local_model_config.validation_percentage)
        model_hyperparameters_pb_serialized = model_hyperparameters_pb.SerializeToString()
        bazel_init_controller_cmd = BazelMetisServicesCmdFactory().bazel_init_controller_target(
             hostname=self.federation_environment.controller.connection_configs.hostname,
             port=self.federation_environment.controller.grpc_servicer.port,
             aggregation_rule=self.federation_environment.global_model_config.aggregation_function,
             participation_ratio=self.federation_environment.global_model_config.participation_ratio,
             protocol_pb=protocol_pb_serialized,
             model_hyperparameters_pb=model_hyperparameters_pb_serialized)
        return bazel_init_controller_cmd

    def _init_learner_bazel_cmd(self, learner_instance, controller_instance):
        model_def_name = "model_definition"
        remote_metis_model_path = "/tmp/metis/model_learner_{}".format(learner_instance.grpc_servicer.port)
        remote_model_def_path = os.path.join(remote_metis_model_path, model_def_name)

        init_learner_cmd = BazelMetisServicesCmdFactory().bazel_init_learner_target(
            learner_hostname=learner_instance.connection_configs.hostname,
            learner_port=learner_instance.grpc_servicer.port,
            controller_hostname=controller_instance.connection_configs.hostname,
            controller_port=controller_instance.grpc_servicer.port,
            model_definition=remote_model_def_path,
            train_dataset=learner_instance.dataset_configs.train_dataset_path,
            validation_dataset=learner_instance.dataset_configs.validation_dataset_path,
            test_dataset=learner_instance.dataset_configs.test_dataset_path,
            train_dataset_recipe=os.path.join(remote_metis_model_path, self.train_dataset_recipe_fp.split("/")[-1]),
            validation_dataset_recipe=os.path.join(remote_metis_model_path,
                                                   self.validation_dataset_recipe_fp.split("/")[-1]),
            test_dataset_recipe=os.path.join(remote_metis_model_path, self.test_dataset_recipe_fp.split("/")[-1]),
            neural_engine=self.nn_engine)
        return init_learner_cmd

    def _make_tarfile(self, output_filename, source_dir):
        output_filename = "{}.tar.gz".format(output_filename)
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        return output_filename

    def _ship_model(self, model_weights):
        self._driver_controller_grpc_client.replace_community_model(
            num_contributors=self.num_participating_learners,
            model_weights=model_weights)

    def _shutdown(self):
        # Collect all statistics related to learners before sending the shutdown signal.
        self._collect_local_statistics()
        # Send shutdown signal to all learners in a Round-Robin fashion.
        for learner_id, grpc_client in self._driver_learner_grpc_clients.items():
            # We give a bit more time in between requests since some of the learners
            # may be processing pending tasks and may need to wrap up their ongoing workload.
            grpc_client.shutdown_learner(request_retries=2, request_timeout=30, block=False)
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

    def initialize_federation(self, model_weights):
        """
        This func will create N number of processes/workers to create the federation
        environment. One process for the controller and every other learner.

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
        if self._driver_controller_grpc_client.check_health_status(request_retries=10, request_timeout=3, block=True):
            self._ship_model(model_weights=model_weights)
            for learner_instance in self.federation_environment.learners.learners:
                learner_future = self._executor.schedule(
                    function=self._init_learner,
                    args=(learner_instance,
                          self.federation_environment.controller))
                # TODO If we need to test the pipeline we can force a future return here, i.e., learner_future.result()
                self._executor_learners_tasks_q.put(learner_future)

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
                    if len(metadata_pb) > 0:
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

    def __init__(self, fed_env, nn_engine, model_definition_dir,
                 train_dataset_recipe_fp, validation_dataset_recipe_fp="", test_dataset_recipe_fp=""):
        super(DriverSession, self).__init__(
            fed_env, nn_engine, model_definition_dir,
            train_dataset_recipe_fp, validation_dataset_recipe_fp, test_dataset_recipe_fp)

    def _init_controller(self):
        fabric_connection_config = self.federation_environment.controller \
            .connection_configs.get_fabric_connection_config()
        remote_on_login = self.federation_environment.controller.connection_configs.on_login
        if len(remote_on_login) > 0 and remote_on_login[-1] == ";":
            remote_on_login = remote_on_login[:-1]
        connection = Connection(**fabric_connection_config)
        # We do not use asynchronous or disown, since we want the remote subprocess to return standard (error) output.
        # see also, https://github.com/pyinvoke/invoke/blob/master/invoke/runners.py#L109
        init_cmd = "{} && cd {} && {}".format(
            remote_on_login,
            self.federation_environment.controller.project_home,
            self._init_controller_bazel_cmd())
        MetisLogger.info("Running init cmd to controller host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()
        return

    def _init_learner(self, learner_instance, controller_instance):
        fabric_connection_config = \
            learner_instance.connection_configs.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        # We do not use asynchronous or disown, since we want the remote subprocess to return standard (error) output.
        remote_metis_model_path = "/tmp/metis/model_learner_{}".format(learner_instance.grpc_servicer.port)
        # Delete existing directory if it exists, then recreate it.
        connection.run("rm -rf {}".format(remote_metis_model_path))
        connection.run("mkdir -p {}".format(remote_metis_model_path))
        # Place/Copy model definition and dataset recipe files from the driver to the remote host.
        # Model definition ship .gz file and decompress it.
        connection.put(self.model_definition_tar_fp, remote_metis_model_path)
        connection.put(self.train_dataset_recipe_fp, remote_metis_model_path)
        connection.put(self.validation_dataset_recipe_fp, remote_metis_model_path)
        connection.put(self.test_dataset_recipe_fp, remote_metis_model_path)

        # Fabric runs every command on a non-interactive mode and therefore the $PATH that might be set for a
        # running user might not be visible while running the command. A workaround is to always
        # source the respective bash_environment files.
        cuda_devices_str = ""
        if learner_instance.cuda_devices is not None and len(learner_instance.cuda_devices) > 0:
            cuda_devices_str = "export CUDA_VISIBLE_DEVICES=\"{}\" " \
                .format(",".join([str(c) for c in learner_instance.cuda_devices]))
        remote_on_login = learner_instance.connection_configs.on_login
        if len(remote_on_login) > 0 and remote_on_login[-1] == ";":
            remote_on_login = remote_on_login[:-1]
        # Untar model definition zipped file.
        connection.run("cd {}; tar -xvzf {}".format(
            remote_metis_model_path,
            self.model_definition_tar_fp))
        init_cmd = "{} && {} && cd {} && {}".format(
            remote_on_login,
            cuda_devices_str,
            learner_instance.project_home,
            self._init_learner_bazel_cmd(learner_instance, controller_instance))
        MetisLogger.info("Running init cmd to learner host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()
        return

    def shutdown_federation(self):
        self._shutdown()


class DriverSessionDocker(DriverSessionBase):

    def __init__(self, fed_env, nn_engine, model_definition_dir,
                 train_dataset_recipe_fp, validation_dataset_recipe_fp="", test_dataset_recipe_fp=""):
        super(DriverSessionDocker, self).__init__(
            fed_env, nn_engine, model_definition_dir,
            train_dataset_recipe_fp, validation_dataset_recipe_fp, test_dataset_recipe_fp)

        # When initializing DriverSession based on Docker, we need to make sure that a docker image is provided so
        # that it can be used to trigger the initialization of the controller and learner instances.
        assert self.federation_environment.docker.docker_image is not None

        self.controller_containers = dict()
        self.learner_containers = dict()

    def _init_controller(self):
        docker_cmd_factory = DockerMetisServicesCmdFactory()
        controller_id = "{}:{}".format(
            self.federation_environment.controller.connection_configs.hostname,
            self.federation_environment.controller.grpc_servicer.port)
        self.controller_containers[controller_id] = docker_cmd_factory.container_name
        docker_init_container = \
            docker_cmd_factory.init_container(
                port=self.federation_environment.controller.grpc_servicer.port,
                docker_image=self.federation_environment.docker.docker_image)
        init_container = docker_init_container + " /bin/bash -c \"{}; {}\"".format(
            "source /opt/rh/gcc-toolset-9/enable", self._init_controller_bazel_cmd())
        print(init_container, flush=True)
        fabric_connection_config = self.federation_environment.controller \
            .connection_configs.get_fabric_connection_config()
        remote_source_path = "source /etc/profile; source ~/.bash_profile; source ~/.bashrc "
        connection = Connection(**fabric_connection_config)
        connection.run("{};{}".format(remote_source_path, docker_init_container))
        connection.close()
        return

    def _init_learner(self, learner_instance, controller_instance):
        docker_cmd_factory = DockerMetisServicesCmdFactory()
        learner_id = "{}:{}".format(
            learner_instance.connection_configs.hostname,
            learner_instance.grpc_servicer.port)
        self.learner_containers[learner_id] = docker_cmd_factory.container_name
        docker_init_container = \
            docker_cmd_factory.init_container(
                port=learner_instance.grpc_servicer.port,
                docker_image=self.federation_environment.docker.docker_image,
                cuda_devices=learner_instance.cuda_devices)
        init_container = docker_init_container + " /bin/bash -c \"{}; {}\"".format(
            "source /opt/rh/gcc-toolset-9/enable", self._init_controller_bazel_cmd())
        print(docker_init_container, flush=True)
        fabric_connection_config = self.federation_environment.controller \
            .connection_configs.get_fabric_connection_config()
        remote_source_path = "source /etc/profile; source ~/.bash_profile; source ~/.bashrc "
        connection = Connection(**fabric_connection_config)
        connection.run("{};{}".format(remote_source_path, init_container))
        connection.close()
        return

    def shutdown_federation(self):
        self._shutdown()

        for learner_instance in self.federation_environment.learners.learners:
            learner_id = "{}:{}".format(
                learner_instance.connection_configs.hostname,
                learner_instance.grpc_servicer.port)
            docker_cmd_factory = DockerMetisServicesCmdFactory(self.learner_containers[learner_id])
            stop_container = docker_cmd_factory.stop_container()
            rm_container = docker_cmd_factory.rm_container()
            fabric_connection_config = self.federation_environment.controller \
                .connection_configs.get_fabric_connection_config()
            remote_source_path = "source /etc/profile; source ~/.bash_profile; source ~/.bashrc "
            connection = Connection(**fabric_connection_config)
            connection.run("{};{} && {} ".format(
                remote_source_path,
                stop_container,
                rm_container))
            connection.close()

        controller_id = "{}:{}".format(
            self.federation_environment.controller.connection_configs.hostname,
            self.federation_environment.controller.grpc_servicer.port)
        docker_cmd_factory = DockerMetisServicesCmdFactory(self.controller_containers[controller_id])
        stop_container = docker_cmd_factory.stop_container()
        rm_container = docker_cmd_factory.rm_container()
        fabric_connection_config = self.federation_environment.controller \
            .connection_configs.get_fabric_connection_config()
        remote_source_path = "source /etc/profile; source ~/.bash_profile; source ~/.bashrc "
        connection = Connection(**fabric_connection_config)
        connection.run("{};{} && {} ".format(
            remote_source_path,
            stop_container,
            rm_container))
        connection.close()
