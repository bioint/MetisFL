import multiprocessing as mp
import queue
import time
from typing import Callable, List

from pebble import ProcessPool

from metisfl import config
from metisfl.models.utils import construct_model_pb
from metisfl.models.metis_model import MetisModel
from metisfl.utils.fedenv import FederationEnvironment
from metisfl.utils.metis_logger import MetisASCIIArt, MetisLogger

from .controller_client import GRPCControllerClient
from .learner_client import GRPCLearnerClient
from .service_initializer import ServiceInitializer
from .service_monitor import ServiceMonitor


class DriverSession(object):
    def __init__(self,
                 fed_env: str,
                 model: MetisModel,
                 train_dataset_fps: List[str],
                 train_dataset_recipe_fn: Callable,
                 validation_dataset_fps: List[str] = None,
                 validation_dataset_recipe_fn: Callable = None,
                 test_dataset_fps: List[str] = None,
                 test_dataset_recipe_fn: Callable = None):
        """Entry point for MetisFL Driver Session API.

            This class is the entry point of the MetisFL Driver Session API. It is responsible for initializing the federation environment
            defined by the :param:`fed_env` and starting the federated training for the :param:`model` using the datasets defined by
            :param:`train_datset_fps`, :param:`validation_dataset_fps` and :param:`test_dataset_fps`. The dataset recipes given by
            :param:`train_dataset_recipe_fn`, :param:`validation_dataset_recipe_fn` and :param:`test_dataset_recipe_fn` are functions
            that will be used to create the datasets on the remote Learner machine. They must take a single argument, the dataset file path,
            and return a :class: `metisfl.models.model_dataset.ModelDataset` instance. 

            To boostrap the training process, we ssh into the remote Controller and Learners host machines using the authorization
            credentials defined in the federation environment yaml file. The boostrapping process involves the following steps:

                - Copy the intial model weights to the remote Learner machines.
                - Copy the dataset file and the pickled dataset recipe functions to the remote Learner machines.
                - Start the Controller and Learner processes on the remote machines. 

            The boostraping is delegated to the :class:`ServiceInitializer` class, which exposes a :meth:`ServiceInitializer.init_controller`
            and a :meth:`ServiceInitializer.init_learner` method. These methods are invoked in parallel using 
            a [Pebble](https://pypi.org/project/Pebble/) process pool. 

            The present class creates the following GRPC clients

                - one :class:`GRPCControllerClient` for the controller in the federation environment.
                - one :class:`GRPCLearnerClient` for each learner in the federation environment.

            The Controller client ships the initial model weights state to the Controller and is used from the :class:`ServiceMonitor` 
            to monitor the training process and collect statistics upon completion. The method :meth:`DriverSession.run` is used to
            start the federated training process and orderly invokes the methods :meth:`initialize_federation()`, 
            :meth:`ServiceMonitor.monitor_federation`, which blocks execution and waits for the termination signals and 
            :meth:`shutdown_federation()`.

        Args:
            fed_env (str): The path to the federation environment yaml file.
            model (MetisModel): A :class:`MetisModel` instance.
            train_datset_fps (List[str]): A list of file paths to the training datasets (one for each learner)
            train_dataset_recipe_fn (Callable): A function that will be used to create the training datasets on the remote Learner machines.
            validation_dataset_fps (List[str], optional): A list of file paths to the validation datasets (one for each learner). Defaults to None.
            validation_dataset_recipe_fn (Callable, optional): A function that will be used to create the validation datasets on the remote Learner machines. Defaults to None.
            test_dataset_fps (List[str], optional): A list of file paths to the test datasets (one for each learner). Defaults to None.
            test_dataset_recipe_fn (Callable, optional): A function that will be used to create the test datasets on the remote Learner machines. Defaults to None.
        """
        # Print welcome message.
        MetisASCIIArt.print()
        self._federation_environment = FederationEnvironment(fed_env)
        self._homomorphic_encryption = HomomorphicEncryption(
            he_scheme_pb=self._federation_environment.get_he_scheme_pb(entity="learner"))
        self._num_learners = len(
            self._federation_environment.learners)
        self._model = model

        self._init_pool()
        self._controller_server_entity_pb = self._federation_environment.controller.get_server_entity_pb()
        self._learner_server_entities_pb = [
            learner.get_server_entity_pb() for learner in self._federation_environment.learners
        ]
        self._driver_controller_grpc_client = self._create_driver_controller_grpc_client()
        self._driver_learner_grpc_clients = self._create_driver_learner_grpc_clients()

        dataset_fps = self._gen_dataset_dict(
            train_dataset_fps, validation_dataset_fps, test_dataset_fps)

        dataset_recipe_fns = self._gen_dataset_dict(
            train_dataset_recipe_fn, validation_dataset_recipe_fn, test_dataset_recipe_fn)

        self._service_initilizer = ServiceInitializer(
            controller_server_entity_pb=self._controller_server_entity_pb,
            dataset_recipe_fns=dataset_recipe_fns,
            dataset_fps=dataset_fps,
            fed_env=self._federation_environment,
            learner_server_entities_pb=self._learner_server_entities_pb,
            model=self._model
        )

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
            grpc_client.shutdown_learner(
                request_retries=1, request_timeout=30, block=False)
        for grpc_client in self._driver_learner_grpc_clients.values():
            grpc_client.shutdown()

        self._service_monitor.collect_global_statistics()
        self._driver_controller_grpc_client.shutdown_controller(
            request_retries=2, request_timeout=30, block=True)
        self._driver_controller_grpc_client.shutdown()

        self._executor.close()
        self._executor.join()

    def _create_driver_controller_grpc_client(self):
        return GRPCControllerClient(
            controller_server_entity=self._controller_server_entity_pb,
            max_workers=1)

    def _create_driver_learner_grpc_clients(self):
        grpc_clients = {}
        for idx in range(self._num_learners):
            learner = self._federation_environment.learners[idx]
            learner_server_entity_pb = self._learner_server_entities_pb[idx]
            grpc_clients[learner.id] = \
                GRPCLearnerClient(
                    learner_server_entity=learner_server_entity_pb, max_workers=1)
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
        model_pb = construct_model_pb(weights_descriptor, )
        self._driver_controller_grpc_client.replace_community_model(
            num_contributors=self._num_learners,
            model_pb=model_pb)
