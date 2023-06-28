import queue
import time
import multiprocessing as mp
from metisfl.driver.monitor import FederationMonitor

from pebble import ProcessPool
from typing import Callable

import metisfl.utils.fedenv_parser as fedenv_parser
from metisfl.driver.driver_initializer import DriverInitializer
from metisfl.driver.utils import create_server_entity
from metisfl.grpc.grpc_controller_client import GRPCControllerClient
from metisfl.grpc.grpc_learner_client import GRPCLearnerClient
from metisfl.models.model_wrapper import MetisModel
from metisfl.utils.metis_logger import MetisASCIIArt

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
TASK_KEYS = [TRAIN, VALIDATION, TEST]


class DriverSessionBase(object):

    def __init__(self, 
                 fed_env_fp: str,
                 model: MetisModel,
                 working_dir: str,
                 train_dataset_recipe_fn: Callable,
                 validation_dataset_recipe_fn: Callable = None,
                 test_dataset_recipe_fn: Callable = None):    
        # Print welcome message.
        MetisASCIIArt.print()
        self.federation_environment = fedenv_parser.FederationEnvironment(fed_env_fp)
        self.homomorphic_encryption = self.federation_environment.homomorphic_encryption
        self.num_learners = len(self.federation_environment.learners.learners)     
        self.model = model
        dataset_recipe_fns = {
            TRAIN: train_dataset_recipe_fn,
            VALIDATION: validation_dataset_recipe_fn,
            TEST: test_dataset_recipe_fn
        }
        self._init_pool()
        self._driver_controller_grpc_client = self._create_driver_controller_grpc_client()
        self._driver_learner_grpc_clients = self._create_driver_learner_grpc_clients()
        self._driver_initilizer = DriverInitializer(
            dataset_recipe_fns=dataset_recipe_fns,
            fed_env=self.federation_environment,
            model=self.model,
            working_dir=working_dir)
        self._monitor = FederationMonitor(
                federation_environment=self.federation_environment,
                driver_controller_grpc_client=self._driver_controller_grpc_client)

    def _init_pool(self):
        # Unix default is "fork", others: "spawn", "forkserver"
        # We use spawn so that the parent process starts a fresh Python interpreter process.
        mp_ctx = mp.get_context("spawn")
        self._executor = ProcessPool(max_workers=self.num_learners + 1, context=mp_ctx)
        self._executor_controller_tasks_q = queue.LifoQueue(maxsize=0)
        self._executor_learners_tasks_q = queue.LifoQueue(maxsize=0)

    def _create_driver_controller_grpc_client(self):
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

    def _ship_model_to_controller(self):
        weights_descriptor = self.model.get_weights_descriptor()
        model_pb = self.homomorphic_encryption.encrypt_np_to_model_pb(weights_descriptor)
        self._driver_controller_grpc_client.replace_community_model(
            num_contributors=self.num_learners,
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
        controller_future = self._executor.schedule(function=self._driver_initilizer.init_controller)
        self._executor_controller_tasks_q.put(controller_future)
        # @stripeli what happens in the else case?
        if self._driver_controller_grpc_client.check_health_status(request_retries=10, request_timeout=30, block=True):
            self._ship_model_to_controller()
            for learner_instance in self.federation_environment.learners.learners:
                learner_future = self._executor.schedule(
                    function=self._driver_initilizer.init_learner,
                    args=(learner_instance,
                          self.federation_environment.controller))
                # TODO If we need to test the pipeline we can force a future return here, i.e., learner_future.result()
                self._executor_learners_tasks_q.put(learner_future)
                # TODO We perform a sleep because if the learners are co-located, e.g., localhost, then an exception
                #  is raised by the SSH client: """ Exception (client): Error reading SSH protocol banner """.
                time.sleep(0.1)
                
    def monitor_federation(self):
        self._monitor.monitor()
        
    def get_federation_statistics(self):
        return self._monitor.get_federation_statistics()
                
    def shutdown_federation(self):
        # Collect all statistics related to learners before sending the shutdown signal.
        self._monitor.collect_local_statistics()
        # Send shutdown signal to all learners in a Round-Robin fashion.
        for learner_id, grpc_client in self._driver_learner_grpc_clients.items():
            # We send a single non-blocking shutdown request to every learner with a 30secs time-to-live.
            grpc_client.shutdown_learner(request_retries=1, request_timeout=30, block=False)
        # Blocking-call, wait for learners shutdown acknowledgment.
        for learner_id, grpc_client in self._driver_learner_grpc_clients.items():
            grpc_client.shutdown()

        # Collect all statistics related to the global execution before sending the shutdown signal.
        self._monitor.collect_global_statistics()

        # Similar to the learners, we also give a bit more time in between requests to
        # the controller since in needs to wrap pending tasks submitted by the learners.
        self._driver_controller_grpc_client.shutdown_controller(request_retries=2, request_timeout=30, block=True)
        self._driver_controller_grpc_client.shutdown()

        self._executor.close()
        self._executor.join()
