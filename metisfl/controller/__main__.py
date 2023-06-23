import argparse

import metisfl.proto.metis_pb2 as metis_pb2

from metisfl.utils.metis_logger import MetisLogger
from metisfl.utils.proto_messages_factory import MetisProtoMessages, ModelProtoMessages
from metisfl.controller.controller_instance import ControllerInstance


def init_controller(args):

    # For all incoming hexadecimal representations, we need to first convert them
    # to bytes and later pass them as initialization to the proto message object.
    # If the given protobuff is None then we assign it an empty bytes object.

    if args.controller_server_entity_protobuff_serialized_hexadecimal is not None:
        controller_server_entity_pb = metis_pb2.ServerEntity()
        controller_server_entity_pb_ser = bytes.fromhex(args.controller_server_entity_protobuff_serialized_hexadecimal)
        controller_server_entity_pb.ParseFromString(controller_server_entity_pb_ser)
    else:
        controller_server_entity_pb = MetisProtoMessages.construct_server_entity_pb(
            hostname="[::]", port=50051)

    # Parse serialized model hyperparameters object, 'recover' bytes object.
    # Use parsed protobuff to initialize Metis.CommunicationSpecs() object.
    if args.global_model_specs_protobuff_serialized_hexadecimal is not None:
        global_model_specs_protobuff_ser = bytes.fromhex(args.global_model_specs_protobuff_serialized_hexadecimal)
        global_model_specs_pb = metis_pb2.GlobalModelSpecs()
        global_model_specs_pb.ParseFromString(global_model_specs_protobuff_ser)
    else:
        aggregation_rule_pb = MetisProtoMessages.construct_aggregation_rule_pb(
            rule_name="FEDAVG",
            scaling_factor="NUMTRAININGEXAMPLES",
            stride_length=None,
            he_scheme_pb=None)
        global_model_specs_pb = MetisProtoMessages.construct_global_model_specs(
            aggregation_rule_pb=aggregation_rule_pb,
            learners_participation_ratio=1)

    # Use parsed protobuff to initialize Metis.CommunicationSpecs() object.
    if args.communication_specs_protobuff_serialized_hexadecimal is not None:
        communication_specs_protobuff_ser = bytes.fromhex(args.communication_specs_protobuff_serialized_hexadecimal)
        communication_specs_pb = metis_pb2.CommunicationSpecs()
        communication_specs_pb.ParseFromString(communication_specs_protobuff_ser)
    else:
        communication_specs_pb = MetisProtoMessages.construct_communication_specs_pb(
            protocol="SYNCHRONOUS",
            semi_sync_lambda=None,
            semi_sync_recompute_num_updates=None)

    # Use parsed protobuff to initialize ControllerParams.ModelHyperparams() object.
    if args.model_hyperparameters_protobuff_serialized_hexadecimal is not None:
        model_hyperparameters_protobuff_ser = bytes.fromhex(args.model_hyperparameters_protobuff_serialized_hexadecimal)
        model_hyperparams_pb = metis_pb2.ControllerParams.ModelHyperparams()
        model_hyperparams_pb.ParseFromString(model_hyperparameters_protobuff_ser)
    else:
        model_hyperparams_pb = MetisProtoMessages.construct_controller_modelhyperparams_pb(
            batch_size=100, epochs=5, percent_validation=0.0,
            optimizer_pb=ModelProtoMessages.construct_optimizer_config_pb(
                ModelProtoMessages.construct_vanilla_sgd_optimizer_pb(learning_rate=0.01)))

    # Use parsed protobuff to initialize Metis.ModelStoreConfig() object.
    if args.model_store_config_protobuff_serialized_hexadecimal is not None:
        model_store_config_protobuff_ser = bytes.fromhex(args.model_store_config_protobuff_serialized_hexadecimal)
        model_store_config_pb = metis_pb2.ModelStoreConfig()
        model_store_config_pb.ParseFromString(model_store_config_protobuff_ser)
    else:
        # Default is the in-memory store without any model eviction.
        model_store_config_pb = MetisProtoMessages.construct_model_store_config_pb(
            name="InMemory",
            eviction_policy="NoEviction")

    controller_params_pb = MetisProtoMessages.construct_controller_params_pb(
        controller_server_entity_pb,
        global_model_specs_pb,
        communication_specs_pb,
        model_store_config_pb,
        model_hyperparams_pb)

    MetisLogger.info("Controller Parameters: \"\"\"{}\"\"\"".format(controller_params_pb))
    controller_instance = ControllerInstance()
    controller_instance.build_and_start(controller_params_pb)
    controller_instance.wait()


if __name__ == "__main__":
    # FIXME: the existence of hex-encoded args is not a user-friendly way to start the controller
    # Since the hex encoding is required for sending the args over the wire, let's keep those
    # and add an additional wrapper that accepts user-friendly arg input
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--controller_server_entity_protobuff_serialized_hexadecimal", type=str,
                        default=None,
                        help="Controller server entity.")
    parser.add_argument("-g", "--global_model_specs_protobuff_serialized_hexadecimal", type=str,
                        default=None,
                        help="local models aggregation_rule (i.e, merging function) to create global model.")
    parser.add_argument("-c", "--communication_specs_protobuff_serialized_hexadecimal", type=str,
                        default=None,
                        help="what is the communication protocol for aggregating local models "
                             "(i.e., synchronous, asynchronous, semi_synchronous) and its specifications.")
    parser.add_argument("-m", "--model_hyperparameters_protobuff_serialized_hexadecimal", type=str,
                        default=None,
                        help="A serialized Model Hyperparameters protobuf message.")
    parser.add_argument("-s", "--model_store_config_protobuff_serialized_hexadecimal", type=str,
                        default=None,
                        help="A serialized Model Store Config protobuf message.")

    args = parser.parse_args()
    init_controller(args)
