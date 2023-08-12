import queue
import time

import multiprocessing as mp

from typing import Callable, List
from pebble import ProcessPool

from metisfl import config
from metisfl.encryption.homomorphic import HomomorphicEncryption
from metisfl.models.metis_model import MetisModel
from metisfl.utils.fedenv import FederationEnvironment
from metisfl.utils.logger import MetisASCIIArt, MetisLogger


from .controller_client import GRPCControllerClient
from .learner_client import GRPCLearnerClient
from .service_initializer import ServiceInitializer
from .service_monitor import ServiceMonitor


class DriverSession(object):
    def __init__(self,
                 fed_env: FederationEnvironment,
                 model: MetisModel,
                 train_dataset_fps: List[str],
                 train_dataset_recipe_fn: Callable,
                 validation_dataset_fps: List[str] = None,
                 validation_dataset_recipe_fn: Callable = None,
                 test_dataset_fps: List[str] = None,
                 test_dataset_recipe_fn: Callable = None):
        """ 
        Entry point for MetisFL Driver Session API.
        """
        # Print welcome message.
        MetisASCIIArt.print()
        self._federation_environment = FederationEnvironment(fed_env)
        self._num_learners = len(self._federation_environment.learners)
        self._model = model
        self._homomorphic_encryption = HomomorphicEncryption(
            he_scheme_pb=self._federation_environment.get_he_scheme_pb(entity="learner"))
        self._init_pool()
        self._driver_controller_grpc_client = \
            self._create_driver_controller_grpc_client()
        self._driver_learner_grpc_clients = \
            self._create_driver_learner_grpc_clients()

        dataset_fps = self._gen_dataset_dict(
            train_dataset_fps, validation_dataset_fps, test_dataset_fps)

        dataset_recipe_fns = self._gen_dataset_dict(
            train_dataset_recipe_fn, 
            validation_dataset_recipe_fn, 
            test_dataset_recipe_fn)

        self._service_initilizer = ServiceInitializer(
            federation_environment=self._federation_environment,
            dataset_recipe_fns=dataset_recipe_fns,
            dataset_fps=dataset_fps,
            model=self._model)

        self._service_monitor = ServiceMonitor(
            federation_environment=self._federation_environment,
            driver_controller_grpc_client=self._driver_controller_grpc_client)
        
    def get_federation_statistics(self):
        return self._service_monitor.get_federation_statistics()

    def initialize_federation(self):
        # NOTE: If we need to test the pipeline we force a future return here,
        # i.e., controller_future.result(). The following initialization futures are
        # always running (status=running) since we need to keep the connections open
        # in order to retrieve logs regarding the execution progress of the federation.
        
        # FIXME(@stripeli): so how would the users be able to see and inform us for any
        # potential errors?
        controller_future = self._executor.schedule(
            function=self._service_initilizer.init_controller)
        self._executor_controller_tasks_q.put(controller_future)
        if self._driver_controller_grpc_client.check_health_status(request_retries=10, request_timeout=30, block=True):
            self._ship_model_to_controller()
            for idx in range(self._num_learners):
                learner_future = self._executor.schedule(
                    function=self._service_initilizer.init_learner,
                    args=(idx, ))  # NOTE: args must be a tuple.
                self._executor_learners_tasks_q.put(learner_future)
                # FIXME(@stripeli): Might need to remove the sleep time in the future.
                # For now, we perform sleep because if the learners are co-located, e.g., localhost, then an 
                # exception is raised by the SSH client: """ Exception (client): Error reading SSH protocol banner """.
                if self._federation_environment.learners[idx].hostname == "localhost":
                    time.sleep(0.1)
                # NOTE: If we need to test the pipeline we can force a future return here, i.e., learner_future.result().
                self._executor_learners_tasks_q.put(learner_future)                  
        else:
            MetisLogger.fatal(
                "Controller is not responsive. Cannot proceed with execution.")

    def monitor_federation(self):
        self._service_monitor.monitor_federation()  # Blocking call.

    def run(self):
        self.initialize_federation()
        self.monitor_federation()  # Blocking call.
        self.shutdown_federation()

    def shutdown_federation(self):
        self._service_monitor.collect_local_statistics()

        for grpc_client in self._driver_learner_grpc_clients.values():
            grpc_client.shutdown_learner(request_timeout=30, block=False)
        for grpc_client in self._driver_learner_grpc_clients.values():
            grpc_client.shutdown()

        self._service_monitor.collect_global_statistics()
        self._driver_controller_grpc_client.shutdown_controller(
            request_retries=2, request_timeout=30, block=True)
        self._driver_controller_grpc_client.shutdown()

        self._executor.close()
        self._executor.join()

    def _create_driver_controller_grpc_client(self):
        controller_server_entity_pb = \
            self._federation_environment.controller.get_server_entity_pb(
                gen_connection_entity=True)
        return GRPCControllerClient(
            controller_server_entity=controller_server_entity_pb,
            max_workers=1)

    def _create_driver_learner_grpc_clients(self):
        grpc_clients = {}
        for idx in range(self._num_learners):
            learner = self._federation_environment.learners[idx]
            learner_server_entity_pb = \
                learner.get_server_entity_pb(gen_connection_entity=True)
            grpc_clients[learner.identifier] = \
                GRPCLearnerClient(
                    learner_server_entity=learner_server_entity_pb, 
                    max_workers=1)
        return grpc_clients

    def _gen_dataset_dict(self,
                          train_val,
                          validation_val=None,
                          test_val=None):
        dataset_dict = {}
        dataset_dict[config.TRAIN] = train_val  # Always required.
        if validation_val:
            dataset_dict[config.VALIDATION] = validation_val
        if test_val:
            dataset_dict[config.TEST] = test_val
        return dataset_dict

    def _init_pool(self):
        mp_ctx = mp.get_context("spawn")
        self._executor = ProcessPool(
            max_workers=self._num_learners + 1, context=mp_ctx)
        self._executor_controller_tasks_q = queue.LifoQueue(maxsize=0)
        self._executor_learners_tasks_q = queue.LifoQueue(maxsize=0)

    def _ship_model_to_controller(self):
        weights_descriptor = self._model.get_weights_descriptor()
        model_pb = self._homomorphic_encryption.encrypt(
            weights_descriptor=weights_descriptor)
        self._driver_controller_grpc_client.replace_community_model(
            num_contributors=self._num_learners,
            model_pb=model_pb)
