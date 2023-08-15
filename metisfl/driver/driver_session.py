import queue
import time

import multiprocessing as mp

from typing import Callable, Dict, List, Union
from pebble import ProcessPool

from metisfl import config
from metisfl.encryption.homomorphic import HomomorphicEncryption
from metisfl.proto import model_pb2
from metisfl.utils.fedenv import FederationEnvironment
from metisfl.utils.logger import MetisASCIIArt, MetisLogger


from .controller_client import GRPCControllerClient
from .learner_client import GRPCLearnerClient
from .federation_monitor import FederationMonitor


class DriverSession(object):
    def __init__(
        self,
        fedenv: Union[str, FederationEnvironment],
    ):
        """Initializes the driver session."""

        MetisASCIIArt.print()

        if isinstance(fedenv, str):
            fed_env = FederationEnvironment.from_yaml(fedenv)

        self._federation_environment = FederationEnvironment(fed_env)
        self._num_learners = len(self._federation_environment.learners)

        global_config = self._federation_environment.global_train_config
        self._homomorphic_encryption = HomomorphicEncryption(
            batch_size=global_config.batch_size,
            scaling_factor_bits=global_config.scaling_factor_bits,
        )

        self._init_pool()
        self._driver_controller_grpc_client = \
            self._create_driver_controller_grpc_client()
        self._driver_learner_grpc_clients = \
            self._create_driver_learner_grpc_clients()

        self._service_monitor = FederationMonitor(
            federation_environment=self._federation_environment,
            driver_controller_grpc_client=self._driver_controller_grpc_client)

    def run(self) -> Dict:
        """Runs the federated training session.

        Returns
        -------
        Dict
            A dictionary containing the statistics of the federated training.
        """
        self.initialize_federation()
    
        statistics = self.monitor_federation()  # Blocking call.
    
        self.shutdown_federation()
        
        return statistics

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

                # NOTE: If we need to test the pipeline we can force a future return here, i.e., learner_future.result().
                self._executor_learners_tasks_q.put(learner_future)
        else:
            MetisLogger.fatal(
                "Controller is not responsive. Cannot proceed with execution.")


    def monitor_federation(self) -> Dict:
        return self._service_monitor.monitor_federation()  # Blocking call.

    def shutdown_federation(self):

        for grpc_client in self._driver_learner_grpc_clients.values():
            grpc_client.shutdown_learner(request_timeout=30, block=False)
            grpc_client.shutdown()

        self._driver_controller_grpc_client.shutdown_controller(
            request_retries=2, request_timeout=30, block=True)

        self._driver_controller_grpc_client.shutdown()

        self._executor.close()
        self._executor.join()

    def _create_driver_controller_grpc_client(self):
        """Creates a GRPC client for the controller."""

        controller = self._federation_environment.controller

        return GRPCControllerClient(
            server_hostname=controller.hostname,
            server_port=controller.port,
            root_certificate=controller.root_certificate,
            max_workers=1
        )

    def _create_driver_learner_grpc_clients(self) -> List[GRPCLearnerClient]:
        """Creates a dictionary of GRPC clients for the learners."""

        grpc_clients: List[GRPCLearnerClient] = [None] * self._num_learners

        for idx in range(self._num_learners):

            learner = self._federation_environment.learners[idx]

            grpc_clients[idx] = GRPCLearnerClient(
                server_hostname=learner.hostname,
                server_port=learner.port,
                root_certificate=learner.root_certificate,
                max_workers=1)

        return grpc_clients

    def _init_pool(self, max_workers: int):
        mp_ctx = mp.get_context("spawn")
        self._executor = ProcessPool(
            max_workers=max_workers, context=mp_ctx)
        self._executor_controller_tasks_q = queue.LifoQueue(maxsize=0)
        self._executor_learners_tasks_q = queue.LifoQueue(maxsize=0)
    

    def _ship_model_to_controller(self, model: model_pb2.Model) -> None:
        """Encrypts and ships the model to the controller.

        Parameters
        ----------
        model : model_pb2.Model
            The Protobuf object containing the model to be shipped.
        """        
        model = self._homomorphic_encryption.encrypt(model=model)
        
        self._driver_controller_grpc_client.set_initial_model(
            model=model,
        )
