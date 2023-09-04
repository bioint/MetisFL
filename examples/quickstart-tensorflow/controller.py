
""" Controller for the TF quickstart example."""

from metisfl.common.types import GlobalTrainConfig, ModelStoreConfig, ServerParams
from metisfl.controller import Controller

# Parameters for the Controller server.
controller_params = ServerParams(
    hostname="localhost",
    port=50051,
)

# Global training configuration.
global_train_config = GlobalTrainConfig(
    aggregation_rule="FedAvg",
    communication_protocol="Synchronous",
    scaling_factor="NumTrainingExamples",
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
        global_train_config=global_train_config,
        model_store_config=model_store_config,
    )

    # Start the controller. Blocks until Shutdown endpoint is called.
    controller.start()


if __name__ == "__main__":
    start_controller()
