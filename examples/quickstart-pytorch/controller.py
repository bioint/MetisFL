
""" Controller for the Pytorch quickstart example."""

from metisfl.common.types import ControllerConfig, ModelStoreConfig, ServerParams
from metisfl.controller import Controller

# Parameters for the Controller server.
controller_params = ServerParams(
    hostname="localhost",
    port=50051,
)

# Global training configuration.
controller_config = ControllerConfig(
    aggregation_rule="FedAvg",
    scheduler="Synchronous",
    scaling_factor="NumParticipants",
)

# Model store configuration.
model_store_config = ModelStoreConfig(
    model_store="InMemory",
    lineage_length=0
)


def start_controller():
    # Create the controller.
    controller = Controller(
        server_params=controller_params,
        controller_config=controller_config,
        model_store_config=model_store_config,
    )

    # Start the controller. Blocks until Shutdown endpoint is called.
    controller.start()


if __name__ == "__main__":
    start_controller()
