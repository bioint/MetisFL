import queue
import os
import random
import string
import tarfile
import time

import projectmetis.python.utils.proto_messages_factory as proto_messages_factory

from fabric import Connection
from pebble import ProcessPool
from projectmetis.python.utils.fedenv_parser import FederationEnvironment
from projectmetis.python.driver.driver_learner_client import DriverLearnerClient
from projectmetis.python.driver.driver_controller_client import DriverControllerClient


class MetisDockerServicesCmdFactory(object):

    def __init__(self,
                 port,
                 host_tmp_volume="/tmp/metis",
                 container_tmp_volume="/tmp/metis",
                 host_crypto_params_volume="/Users/Dstrip/CLionProjects/projectmetis-rc/resources/shelfi_cryptoparams",
                 container_crypto_params_volume="/metis/cryptoparams",
                 projectmetis_docker_image="projectmetis_rockylinux_8:0.0.1",
                 container_name=None,
                 cuda_devices=None):

        self.container_name = container_name
        if self.container_name is None:
            self.container_name = ''.join(random.SystemRandom().choice(
                string.ascii_uppercase + string.digits) for _ in range(10))

        self.cuda_devices = cuda_devices
        if self.cuda_devices is None:
            self.cuda_devices = []

        self.docker_template_cmd = \
            "docker run " \
            "-p {port}:{port} " \
            "-v {host_tmp_volume}:{container_tmp_volume} " \
            "-v {host_crypto_params_volume}:{container_crypto_params_volume} " \
            "--name {container_name} " \
            "--gpus '{gpu_devices}' " \
            "{projectmetis_docker_image} ".format(
                port=port,
                host_tmp_volume=host_tmp_volume,
                container_tmp_volume=container_tmp_volume,
                host_crypto_params_volume=host_crypto_params_volume,
                container_name=self.container_name,
                container_crypto_params_volume=container_crypto_params_volume,
                gpu_devices=','.join(self.cuda_devices),
                projectmetis_docker_image=projectmetis_docker_image)

    def docker_bazel_init_controller(self, *args, **kwargs):
        return self.docker_template_cmd + " " + MetisBazelServicesCmdFactory.bazel_init_controller_target(**kwargs)

    def docker_bazel_init_learner(self, *args, **kwargs):
        return self.docker_template_cmd + " " + MetisBazelServicesCmdFactory.bazel_init_learner_target(**kwargs)


class MetisBazelServicesCmdFactory(object):

    @classmethod
    def bazel_init_controller_target(cls,
                                     output_user_root,
                                     hostname,
                                     port,
                                     aggregation_rule,
                                     participation_ratio,
                                     protocol,
                                     model_hyperparameters_pb):
        bazel_cmd = \
            "bazel " \
            "--output_user_root={output_user_root} " \
            "run -- //projectmetis/python/driver:initialize_controller " \
            "--controller_hostname=\"{hostname}\" " \
            "--controller_port={port} " \
            "--aggregation_rule=\"{aggregation_rule}\" " \
            "--learners_participation_ratio={participation_ratio} " \
            "--communication_protocol=\"{protocol}\" " \
            "--model_hyperparameters_protobuff=\"{model_hyperparameters_pb}\" ".format(
                output_user_root=output_user_root, hostname=hostname, port=port,
                aggregation_rule=aggregation_rule, participation_ratio=participation_ratio,
                protocol=protocol, model_hyperparameters_pb=model_hyperparameters_pb)
        return bazel_cmd

    @classmethod
    def bazel_init_learner_target(cls,
                                  output_user_root,
                                  learner_hostname,
                                  learner_port,
                                  controller_hostname,
                                  controller_port,
                                  model_definition,
                                  train_dataset,
                                  train_dataset_recipe,
                                  validation_dataset="",
                                  test_dataset="",
                                  validation_dataset_recipe="",
                                  test_dataset_recipe="",
                                  neural_engine="keras"):
        bazel_cmd = \
            "bazel " \
            "--output_user_root={output_user_root} " \
            "run -- //projectmetis/python/driver:initialize_learner " \
            "--neural_engine=\"{neural_engine}\" " \
            "--learner_hostname=\"{learner_hostname}\" " \
            "--learner_port={learner_port} " \
            "--controller_hostname=\"{controller_hostname}\" " \
            "--controller_port={controller_port} " \
            "--model_definition=\"{model_definition}\" " \
            "--train_dataset=\"{train_dataset}\" " \
            "--validation_dataset=\"{validation_dataset}\" " \
            "--test_dataset=\"{test_dataset}\" " \
            "--train_dataset_recipe=\"{train_dataset_recipe}\" " \
            "--validation_dataset_recipe=\"{validation_dataset_recipe}\" " \
            "--test_dataset_recipe=\"{test_dataset_recipe}\"".format(
                output_user_root=output_user_root,
                neural_engine=neural_engine,
                learner_hostname=learner_hostname,
                learner_port=learner_port,
                controller_hostname=controller_hostname,
                controller_port=controller_port,
                model_definition=model_definition,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                test_dataset=test_dataset,
                train_dataset_recipe=train_dataset_recipe,
                validation_dataset_recipe=validation_dataset_recipe,
                test_dataset_recipe=test_dataset_recipe)
        return bazel_cmd


class DriverSession(object):

    def __init__(self, federation_environment_fp, nn_engine, model_definition_dir,
                 train_dataset_recipe_fp, validation_dataset_recipe_fp="", test_dataset_recipe_fp=""):
        self.federation_environment = FederationEnvironment(federation_environment_fp)
        self.nn_engine = nn_engine
        self.model_definition_dir = model_definition_dir
        self.train_dataset_recipe_fp = train_dataset_recipe_fp
        self.validation_dataset_recipe_fp = validation_dataset_recipe_fp
        self.test_dataset_recipe_fp = test_dataset_recipe_fp

        # Total number of workers is based on the number of participating learners and one for the controller.
        self.max_workers = len(self.federation_environment.learners.learners) + 1
        self._executor = ProcessPool(max_workers=self.max_workers)
        # We use a Last-In-First out queue because when we initialize the federation, we start by firstly
        # initializing the controller and then every other learner. Similarly, when we need to shutdown
        # the federation, we start by shutting down first the learners and finally the controller.
        self._executor_tasks_q = queue.LifoQueue(maxsize=self.max_workers)

    def __getstate__(self):
        """
        Python needs to pickle the entire object, including its instance variables.
        Since one of these variables is the Pool object itself, the entire object cannot be pickled.
        We need to remove the Pool() variable from the object state in order to use the pool_task.
        See also: https://stackoverflow.com/questions/25382455
        """
        self_dict = self.__dict__.copy()
        del self_dict['_executor']
        del self_dict['_executor_tasks_q']
        return self_dict

    def _tarify_directory(self, input_directory, output_filename):
        output_filename = "{}.tar.gz".format(output_filename)
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(input_directory, arcname=os.path.basename(input_directory))
        return output_filename

    def _init_controller(self):
        # dev_cmd_factory = MetisDockerServicesCmdFactory(
        #     port=self.federation_environment.controller.grpc_servicer.port)
        dev_cmd_factory = MetisBazelServicesCmdFactory()
        # controller_container_name = dev_cmd_factory.container_name
        fabric_connection_config = self.federation_environment.controller \
            .connection_configs.get_fabric_connection_config()
        # TODO This must be removed when running with docker. Makes sense
        #  only when running localhost to change directory and run bazel.
        metis_home = "/Users/Dstrip/CLionProjects/projectmetis-rc/"
        remote_bazel_output_user_root = "/tmp/metis/bazel"
        remote_source_path = "source /etc/profile; source ~/.bash_profile; source ~/.bashrc"
        optimizer_pb = self.federation_environment.local_model_config.optimizer_config.optimizer_pb
        model_hyperparameters_pb = proto_messages_factory.MetisProtoMessages\
            .construct_controller_modelhyperparams_pb(
                batch_size=self.federation_environment.local_model_config.batch_size,
                epochs=self.federation_environment.local_model_config.local_epochs,
                optimizer_pb=optimizer_pb,
                percent_validation=self.federation_environment.local_model_config.validation_percentage)
        model_hyperparameters_pb_serialized = model_hyperparameters_pb.SerializeToString()
        init_controller_cmd = dev_cmd_factory.bazel_init_controller_target(
                output_user_root=remote_bazel_output_user_root,
                hostname=self.federation_environment.controller.connection_configs.hostname,
                port=self.federation_environment.controller.grpc_servicer.port,
                aggregation_rule=self.federation_environment.global_model_config.aggregation_function,
                participation_ratio=self.federation_environment.global_model_config.participation_ratio,
                protocol=self.federation_environment.communication_protocol,
                model_hyperparameters_pb=model_hyperparameters_pb_serialized)
        print(init_controller_cmd, flush=True)
        connection = Connection(**fabric_connection_config)
        # Problem with fabric is that every command runs on non-interactive mode and therefore the $PATH that might
        # be set for a particular user might not be visible while running the command. A workaround is to always
        # source the respective bash_environment files.
        connection.run("{}; cd {}; {}".format(
            remote_source_path, metis_home, init_controller_cmd))
        connection.close()

    def _init_learner(self, learner_instance, controller_instance):
        fabric_connection_config = learner_instance.connection_configs.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)

        # Problem with fabric is that every command runs on non-interactive mode and therefore the $PATH that might
        # be set for a particular user might not be visible while running the command. A workaround is to always
        # source the respective bash_environment files.
        # TODO This must be removed when running with docker. Makes sense
        #  only when running localhost to change directory and run bazel.
        metis_home = "/Users/Dstrip/CLionProjects/projectmetis-rc/"
        model_def_name = "model_definition"
        remote_bazel_output_user_root = "/tmp/metis/bazel"
        remote_metis_model_path = "/tmp/metis/model_learner_{}".format(learner_instance.grpc_servicer.port)
        remote_model_def_path = os.path.join(remote_metis_model_path, model_def_name)
        remote_source_path = "source /etc/profile; source ~/.bash_profile; source ~/.bashrc"

        connection.run("mkdir -p {}".format(remote_metis_model_path))
        # Place/Copy model definition and dataset recipe files from the driver to the remote host.
        # Model definition compress, ship, decompress.
        model_definition_tar_fp = self._tarify_directory(
            input_directory=self.model_definition_dir,
            output_filename=model_def_name)
        connection.put(model_definition_tar_fp, remote_metis_model_path)
        connection.run("cd {}; tar -xvf {}".format(remote_metis_model_path, model_definition_tar_fp))
        # Data recipes.
        connection.put(self.train_dataset_recipe_fp, remote_metis_model_path)
        connection.put(self.validation_dataset_recipe_fp, remote_metis_model_path)
        connection.put(self.test_dataset_recipe_fp, remote_metis_model_path)

        # dev_cmd_factory = MetisDockerServicesCmdFactory(port=learner_instance.grpc_servicer.port,
        #                                                 cuda_devices=learner_instance.cuda_devices)
        dev_cmd_factory = MetisBazelServicesCmdFactory()
        init_learner_cmd = dev_cmd_factory.bazel_init_learner_target(
            output_user_root=remote_bazel_output_user_root,
            learner_hostname=learner_instance.connection_configs.hostname,
            learner_port=learner_instance.grpc_servicer.port,
            controller_hostname=controller_instance.connection_configs.hostname,
            controller_port=controller_instance.grpc_servicer.port,
            model_definition=remote_model_def_path,
            train_dataset=learner_instance.dataset_configs.train_dataset_path,
            validation_dataset=learner_instance.dataset_configs.validation_dataset_path,
            test_dataset=learner_instance.dataset_configs.test_dataset_path,
            train_dataset_recipe=os.path.join(remote_metis_model_path, self.train_dataset_recipe_fp),
            validation_dataset_recipe=os.path.join(remote_metis_model_path, self.validation_dataset_recipe_fp),
            test_dataset_recipe=os.path.join(remote_metis_model_path, self.test_dataset_recipe_fp,),
            neural_engine=self.nn_engine)
        connection.run("{}; cd {}; {}".format(
            remote_source_path, metis_home, init_learner_cmd))
        connection.close()

    def initialize_federation(self):
        """
        This func will create N number of processes/workers to create the federation
        environment. One process for the controller and every other learner.

        It first initializes the federation controller and then each learner, with some
        lagging time till the federation controller is live so that every learner can
        connect to it.
        """
        controller_future = self._executor.schedule(function=self._init_controller)
        self._executor_tasks_q.put(controller_future)
        # TODO We need to add driver-controller ping, so that we know when the controller is up
        #  so to start registering the learners. Maybe by pinging the
        #  CheckHealthStatus grpc endpoint?
        for learner_instance in self.federation_environment.learners.learners:
            learner_future = self._executor.schedule(
                function=self._init_learner,
                args=(learner_instance,
                      self.federation_environment.controller))
            self._executor_tasks_q.put(learner_future)

    def shutdown_federation(self):
        # Shutdown learners, controller, docker containers
        for learner_instance in self.federation_environment.learners.learners:
            learner_server_entity_pb = \
                proto_messages_factory.MetisProtoMessages.construct_server_entity_pb(
                    hostname=learner_instance.connection_configs.hostname,
                    port=learner_instance.grpc_servicer.port)
            DriverLearnerClient(learner_server_entity_pb).shutdown_learner()
        # TODO sleep needs to be removed!
        time.sleep(10)
        controller_server_entity_pb = \
            proto_messages_factory.MetisProtoMessages.construct_server_entity_pb(
                hostname=self.federation_environment.controller.connection_configs.hostname,
                port=self.federation_environment.controller.grpc_servicer.port)
        DriverControllerClient(controller_server_entity_pb).shutdown_controller()

        while not self._executor_tasks_q.empty():
            # Blocking retrieval of pebble.ProcessFuture from queue.
            self._executor_tasks_q.get().result()
        self._executor.close()
        self._executor.join()

    def monitor_federation(self):
        federation_rounds = self.federation_environment.termination_signals.federation_rounds
        exec_time_cutoff = self.federation_environment.termination_signals.execution_cutoff_time_mins
        exec_score_cutoff = self.federation_environment.termination_signals.execution_cutoff_score

        # measuring elapsed wall-clock time
        ts = time.time()
        while True:
            # ping controller for latest execution stats
            time.sleep(1)
            es = time.time()
            diff_mins = (es - ts) / 60
            if diff_mins >= exec_time_cutoff:
                self.shutdown_federation()
                break
