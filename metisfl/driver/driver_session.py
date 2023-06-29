import multiprocessing as mp
import queue
import time
from typing import Callable, Union

from pebble import ProcessPool

from metisfl.models.model_wrapper import MetisModel
from metisfl.utils import fedenv_parser
from metisfl.utils.metis_logger import MetisASCIIArt, MetisLogger

from . import constants
from .driver_initializer import DriverInitializer
from .grpc_controller_client import GRPCControllerClient
from .grpc_learner_client import GRPCLearnerClient
from .monitor import FederationMonitor
from .utils import create_server_entity


class DriverSession(object):

    def __init__(self,
                 fed_env: Union[str, fedenv_parser.FederationEnvironment],
                 model: MetisModel,
                 working_dir: str,
                 train_dataset_recipe_fn: Callable,
                 validation_dataset_recipe_fn: Callable = None,
                 test_dataset_recipe_fn: Callable = None):
        # Print welcome message.
        MetisASCIIArt.print()
        self._federation_environment = fedenv_parser.FederationEnvironment(
            fed_env) if isinstance(fed_env, str) else fed_env
        self._homomorphic_encryption = self._federation_environment.homomorphic_encryption
        self._num_learners = len(
            self._federation_environment.learners.learners)
        self._model = model
        dataset_recipe_fns = {
            constants.TRAIN: train_dataset_recipe_fn,
            constants.VALIDATION: validation_dataset_recipe_fn,
            constants.TEST: test_dataset_recipe_fn
        }
        self._init_pool()
        self._controller_server_entity_pb = self._create_controller_server_entity()
        self._learner_server_entities_pb = self._create_learning_server_entities()
        self._driver_controller_grpc_client = self._create_driver_controller_grpc_client()
        self._driver_learner_grpc_clients = self._create_driver_learner_grpc_clients()

        self._driver_initilizer = DriverInitializer(
            dataset_recipe_fns=dataset_recipe_fns,
            fed_env=self._federation_environment,
            controller_server_entity_pb=self._controller_server_entity_pb,
            learner_server_entities_pb=self._learner_server_entities_pb,
            model=self._model,
            working_dir=working_dir)
        self._monitor = FederationMonitor(
            federation_environment=self._federation_environment,
            driver_controller_grpc_client=self._driver_controller_grpc_client)

    def _init_pool(self):
        # Unix default is "fork", others: "spawn", "forkserver"
        # We use spawn so that the parent process starts a fresh Python interpreter process.
        mp_ctx = mp.get_context("spawn")
        self._executor = ProcessPool(
            max_workers=self._num_learners + 1, context=mp_ctx)
        self._executor_controller_tasks_q = queue.LifoQueue(maxsize=0)
        self._executor_learners_tasks_q = queue.LifoQueue(maxsize=0)

    def _create_controller_server_entity(self):
        return create_server_entity(
            enable_ssl=self._federation_environment.communication_protocol.enable_ssl,
            remote_host_instance=self._federation_environment.controller,
            initialization_entity=True)

    def _create_learning_server_entities(self):
        learning_server_entities_pb = []
        for learner_instance in self._federation_environment.learners.learners:
            learning_server_entities_pb.append(create_server_entity(
                enable_ssl=self._federation_environment.communication_protocol.enable_ssl,
                remote_host_instance=learner_instance,
                initialization_entity=True))
        return learning_server_entities_pb

    def _create_driver_controller_grpc_client(self):
        grpc_controller_client = GRPCControllerClient(
            controller_server_entity=self._controller_server_entity_pb,
            max_workers=1)
        return grpc_controller_client

    def _create_driver_learner_grpc_clients(self):
        grpc_clients = {}
        for index in range(self._num_learners):
            learner_instance = self._federation_environment.learners.learners[index]
            learner_server_entity_pb = self._learner_server_entities_pb[index]
            grpc_clients[learner_instance.learner_id] = \
                GRPCLearnerClient(
                    learner_server_entity=learner_server_entity_pb, max_workers=1)
        return grpc_clients

    def _ship_model_to_controller(self):
        weights_descriptor = self._model.get_weights_descriptor()
        model_pb = self._homomorphic_encryption.construct_model_pb_from_np(
            weights_descriptor)
        self._driver_controller_grpc_client.replace_community_model(
            num_contributors=self._num_learners,
            model_pb=model_pb)

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
        controller_future = self._executor.schedule(
            function=self._driver_initilizer.init_controller)
        self._executor_controller_tasks_q.put(controller_future)
        
        # TODO(@stripeli): what happens in the else case?
        # need to wait abit before checking the health status of the controller
        # o/w the first attempt will fail
        if self._driver_controller_grpc_client.check_health_status(request_retries=10, request_timeout=30, block=True):
            self._ship_model_to_controller()
            for index in range(self._num_learners):
                MetisLogger.info("Initializing learner {}...".format(index))
                learner_future = self._executor.schedule(
                    function=self._driver_initilizer.init_learner,
                    args=(index,))  # NOTE: args must be a tuple
                # TODO If we need to test the pipeline we can force a future return here, i.e., learner_future.result()
                self._executor_learners_tasks_q.put(learner_future)
                # TODO We perform a sleep because if the learners are co-located, e.g., localhost, then an exception
                #  is raised by the SSH client: """ Exception (client): Error reading SSH protocol banner """.
                time.sleep(0.1)

    def monitor_federation(self):
        self._monitor.monitor_federation()

    def get_federation_statistics(self):
        return self._monitor.get_federation_statistics()

    def shutdown_federation(self):
        # Collect all statistics related to learners before sending the shutdown signal.
        self._monitor.collect_local_statistics()
        # Send shutdown signal to all learners in a Round-Robin fashion.
        for learner_id, grpc_client in self._driver_learner_grpc_clients.items():
            # We send a single non-blocking shutdown request to every learner with a 30secs time-to-live.
            grpc_client.shutdown_learner(
                request_retries=1, request_timeout=30, block=False)
        # Blocking-call, wait for learners shutdown acknowledgment.
        for learner_id, grpc_client in self._driver_learner_grpc_clients.items():
            grpc_client.shutdown()

        # Collect all statistics related to the global execution before sending the shutdown signal.
        self._monitor.collect_global_statistics()

        # Similar to the learners, we also give a bit more time in between requests to
        # the controller since in needs to wrap pending tasks submitted by the learners.
        self._driver_controller_grpc_client.shutdown_controller(
            request_retries=2, request_timeout=30, block=True)
        self._driver_controller_grpc_client.shutdown()

        self._executor.close()
        self._executor.join()
