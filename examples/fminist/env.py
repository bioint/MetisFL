
from metisfl.common.types import (FederationEnvironment, GlobalTrainConfig,
                                  LocalTrainConfig, ModelStoreConfig,
                                  ServerParams, TerminationSingals)

controller_params = ServerParams(
    hostname="localhost",
    port=50051,
    root_certificate="/home/panoskyriakis/metisfl/ca-cert.pem",
    server_certificate="/home/panoskyriakis/metisfl/server-cert.pem",
    private_key="/home/panoskyriakis/metisfl/server-key.pem",
)

global_train_config = GlobalTrainConfig(
    aggregation_rule="FedAvg",
    communication_protocol="Synchronous",
    scaling_factor="NumTrainingExamples",
)

model_store_config = ModelStoreConfig(
    model_store="InMemory",
    lineage_length=0
)

learner_1 = ServerParams(
    hostname="localhost",
    port=50052,
    root_certificate="/home/panoskyriakis/metisfl/ca-cert.pem",
    server_certificate="/home/panoskyriakis/metisfl/server-cert.pem",
    private_key="/home/panoskyriakis/metisfl/server-key.pem",
)

learner_2 = ServerParams(
    hostname="localhost",
    port=50053,
    root_certificate="/home/panoskyriakis/metisfl/ca-cert.pem",
    server_certificate="/home/panoskyriakis/metisfl/server-cert.pem",
    private_key="/home/panoskyriakis/metisfl/server-key.pem",
)

learner_3 = ServerParams(
    hostname="localhost",
    port=50054,
    root_certificate="/home/panoskyriakis/metisfl/ca-cert.pem",
    server_certificate="/home/panoskyriakis/metisfl/server-cert.pem",
    private_key="/home/panoskyriakis/metisfl/server-key.pem",
)

env = FederationEnvironment(
    termination_signals=TerminationSingals(
        federation_rounds=5,
    ),
    global_train_config=global_train_config,
    model_store_config=model_store_config,
    controller=controller_params,
    local_train_config=LocalTrainConfig(
        epochs=1,
        batch_size=32,
    ),
    learners=[
        learner_1,
        learner_2,
        learner_3,
    ]
)
