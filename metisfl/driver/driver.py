import random
from time import sleep
from typing import Dict, List

from metisfl.common.logger import MetisASCIIArt
from metisfl.common.types import ClientParams, ServerParams, TerminationSingals
from metisfl.driver.clients import GRPCControllerClient, GRPCLearnerClient
from metisfl.driver.monitor import FederationMonitor
from metisfl.proto import model_pb2


class DriverSession(object):
    # TODO: Fix input, driver does not the entire federation environment.

    def __init__(
        self,
        controller: ServerParams,
        learners: List[ServerParams],
        termination_signals: TerminationSingals,
    ) -> None:
        """Initializes a new DriverSession.

        Parameters
        ----------
        controller : ServerParams
            The parameters of the controller.
        learners : List[ServerParams]
            The parameters of the learners.
        termination_signals : TerminationSingals
            The termination signals for the federated training.
        """
        MetisASCIIArt.print()

        self.learners = learners
        self.controller = controller
        self.num_learners = len(self.learners)

        self.controller_client = self.create_controller_client()
        self.learner_clients = self.create_learner_clients()

        self.service_monitor = FederationMonitor(
            controller_client=self.controller_client,
            termination_signals=termination_signals,
        )

    def run(self) -> Dict:
        """Runs the federated training session.

        Returns
        -------
            A dictionary containing the statistics of the federated training.
        """
        self.initialize_federation()

        statistics = self.monitor_federation()  # Blocking call.

        self.shutdown_federation()

        return statistics

    def initialize_federation(self):
        """Initialzes the federation. Picks a random Learner to obtain the intial weights from and
            ships the weights to all other Learners and the Controller.
        """

        learner_index = random.randint(0, self.num_learners - 1)

        model = self.learner_clients[learner_index].get_model(
            request_timeout=30, request_retries=2, block=True)

        self.ship_model_to_learners(model=model, skip_learner=learner_index)
        self.ship_model_to_controller(model=model)

        sleep(1)  # FIXME: Wait for the controller to receive the model.

        self.start_training()

    def start_training(self) -> None:
        """Starts the federated training."""
        # TODO: Ping controller and learners to check if they are alive.
        self.controller_client.start_training()

    def monitor_federation(self) -> Dict:
        """Monitors the federation and returns the statistics. This is a blocking call.

        Returns
        -------
        Dict
            A dictionary containing the statistics of the federated training.
        """
        return self.service_monitor.monitor_federation()  # Blocking call.

    def shutdown_federation(self):
        """Shuts down the Controller and all Learners."""

        for grpc_client in self.learner_clients:
            grpc_client.shutdown_server(request_timeout=30, block=False)
            grpc_client.shutdown_client()

        # Sleep for 1 second to allow the Learners to shutdown.
        sleep(1)

        self.controller_client.shutdown_server(
            request_retries=2, request_timeout=30, block=True)
        self.controller_client.shutdown_client()

    def create_controller_client(self):
        """Creates a GRPC client for the controller."""

        controller = self.controller

        return GRPCControllerClient(
            client_params=ClientParams(
                hostname=controller.hostname,
                port=controller.port,
                root_certificate=controller.root_certificate,
            )
        )

    def create_learner_clients(self) -> List[GRPCLearnerClient]:
        """Creates a dictionary of GRPC clients for the learners."""

        grpc_clients: List[GRPCLearnerClient] = []

        for idx in range(self.num_learners):

            learner = self.learners[idx]

            client = GRPCLearnerClient(
                client_params=ClientParams(
                    hostname=learner.hostname,
                    port=learner.port,
                    root_certificate=learner.root_certificate,
                ),
            )

            grpc_clients.append(client)

        return grpc_clients

    def ship_model_to_controller(self, model: model_pb2.Model) -> None:
        """Encrypts and ships the model to the controller.

        Parameters
        ----------
        model : model_pb2.Model
            The Protobuf object containing the model to be shipped.
        """

        self.controller_client.set_initial_model(
            model=model,
        )

    def ship_model_to_learners(self, model: model_pb2.Model, skip_learner: int = None) -> None:
        """Ships the given model to all Learners.

        Parameters
        ----------
        model : model_pb2.Model
            The Protobuf object containing the model to be shipped.
        skip_learner : Optional[int], (default=None)
            The index of the learner to skip.
        """

        for idx in range(self.num_learners):
            if idx == skip_learner:
                continue

            self.learner_clients[idx].set_initial_model(
                model=model,
            )
