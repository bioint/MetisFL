import argparse

import projectmetis.proto.metis_pb2 as metis_pb2

from projectmetis.python.logging.metis_logger import MetisLogger
from projectmetis.python.utils.proto_messages_factory import MetisProtoMessages, ModelProtoMessages
from pybind.controller.controller_instance import ControllerInstance

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--controller_hostname", type=str,
                        default="[::]",
                        help="controller binding hostname")
    parser.add_argument("-p", "--controller_port", type=int,
                        default=50051,
                        help="controller binding port")
    parser.add_argument("-a", "--aggregation_rule", type=str,
                        default="FED_AVG",
                        help="local models aggregation_rule (i.e, merging function) to create global model")
    parser.add_argument("-r", "--learners_participation_ratio", type=float,
                        default=1.0,
                        help="what is the ratio of participating learners to the community model")
    parser.add_argument("-m", "--communication_protocol_protobuff", type=str,
                        default=None,
                        help="what is the communication protocol for aggregating local models "
                             "(i.e., synchronous, asynchronous, semi_synchronous) and its specifications.")
    parser.add_argument("-y", "--model_hyperparameters_protobuff", type=str,
                        default=None,
                        help="A serialized Model Hyperparameters protobuf message.")
    args = parser.parse_args()

    controller_hostname = args.controller_hostname
    controller_port = args.controller_port
    # Proto message is recognized as capitalized.
    aggregation_rule = args.aggregation_rule.upper()
    learners_participation_ratio = args.learners_participation_ratio
    # Parse serialized model hyperparameters object, 'recover' bytes object
    # If the given protobuff is None then we assign it an empty bytes object.
    # Use parsed protobuff to initialize Metis.CommunicationSpecs() object.
    if args.communication_protocol_protobuff is not None:
        communication_specs_protobuff = eval(args.communication_protocol_protobuff)
        communication_specs = metis_pb2.CommunicationSpecs()
        communication_specs.ParseFromString(communication_specs_protobuff)
    else:
        communication_specs = MetisProtoMessages.construct_communication_specs_pb(protocol="SYNCHRONOUS",
                                                                                  semi_sync_lambda=None,
                                                                                  semi_sync_recompute_num_updates=False)

    # Use parsed protobuff to initialize ControllerParams.ModelHyperparams() object.
    if args.model_hyperparameters_protobuff is not None:
        model_hyperparameters_protobuff = eval(args.model_hyperparameters_protobuff)
        model_hyperparams = metis_pb2.ControllerParams.ModelHyperparams()
        model_hyperparams.ParseFromString(model_hyperparameters_protobuff)
    else:
        model_hyperparams = MetisProtoMessages.construct_controller_modelhyperparams_pb(
            batch_size=100, epochs=5, percent_validation=0.0,
            optimizer_pb=ModelProtoMessages.construct_optimizer_config_pb(
                ModelProtoMessages.construct_vanilla_sgd_optimizer_pb(learning_rate=0.01)))

    controller_params = metis_pb2.ControllerParams(
        server_entity=metis_pb2.ServerEntity(
            hostname=controller_hostname,
            port=controller_port),
        global_model_specs=metis_pb2.GlobalModelSpecs(
            aggregation_rule=aggregation_rule,
            learners_participation_ratio=learners_participation_ratio,
        ),
        communication_specs=communication_specs,
        model_hyperparams=model_hyperparams)

    MetisLogger.info("Controller Parameters: \"\"\"{}\"\"\"".format(controller_params))
    controller_instance = ControllerInstance()
    controller_instance.build_and_start(controller_params)

    controller_instance.wait()
