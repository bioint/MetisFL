
from metisfl.common.types import (FederationEnvironment, GlobalTrainConfig,
                                  LocalTrainConfig, ModelStoreConfig,
                                  ServerParams, TerminationSingals)

controller_params = ServerParams(
    hostname="localhost",
    port=50051,
)

global_train_config = GlobalTrainConfig(
    aggregation_rule="FedStride",
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
)

learner_2 = ServerParams(
    hostname="localhost",
    port=50053,
)

env = FederationEnvironment(
    termination_signals=TerminationSingals(
        evaluation_metric="accuracy",
        evaluation_metric_cutoff_score=0.6,
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
    ]
)
