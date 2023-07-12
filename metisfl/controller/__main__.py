import argparse

import metisfl.proto.metis_pb2 as metis_pb2
from metisfl.controller.controller_instance import ControllerInstance
from metisfl.utils.metis_logger import MetisLogger


def init_controller(args):

    # For all incoming hexadecimal representations, we need to first convert them
    # to bytes and later pass them as initialization to the proto message object.
    # If the given protobuff is None then we assign it an empty bytes object.
    controller_server_entity_pb = metis_pb2.ServerEntity()
    controller_server_entity_pb_ser = bytes.fromhex(
        args.controller_server_entity_protobuff_serialized_hexadecimal)
    controller_server_entity_pb.ParseFromString(
        controller_server_entity_pb_ser)
    
    # Parse serialized model hyperparameters object, 'recover' bytes object.
    # Use parsed protobuff to initialize Metis.CommunicationSpecs() object.
    global_model_specs_protobuff_ser = bytes.fromhex(
        args.global_model_specs_protobuff_serialized_hexadecimal)
    global_model_specs_pb = metis_pb2.GlobalModelSpecs()
    global_model_specs_pb.ParseFromString(global_model_specs_protobuff_ser)

    # Use parsed protobuff to initialize Metis.CommunicationSpecs() object.
    communication_specs_protobuff_ser = bytes.fromhex(
        args.communication_specs_protobuff_serialized_hexadecimal)
    communication_specs_pb = metis_pb2.CommunicationSpecs()
    communication_specs_pb.ParseFromString(
        communication_specs_protobuff_ser)

    # Use parsed protobuff to initialize ControllerParams.ModelHyperparams() object.
    model_hyperparameters_protobuff_ser = bytes.fromhex(
        args.model_hyperparameters_protobuff_serialized_hexadecimal)
    model_hyperparams_pb = metis_pb2.ControllerParams.ModelHyperparams()
    model_hyperparams_pb.ParseFromString(
        model_hyperparameters_protobuff_ser)

    # Use parsed protobuff to initialize Metis.ModelStoreConfig() object.
    model_store_config_protobuff_ser = bytes.fromhex(
        args.model_store_config_protobuff_serialized_hexadecimal)
    model_store_config_pb = metis_pb2.ModelStoreConfig()
    model_store_config_pb.ParseFromString(model_store_config_protobuff_ser)

    controller_params_pb = metis_pb2.ControllerParams(server_entity=controller_server_entity_pb,
                                                      global_model_specs=global_model_specs_pb,
                                                      communication_specs=communication_specs_pb,
                                                      model_store_config=model_store_config_pb,
                                                      model_hyperparams=model_hyperparams_pb)
    MetisLogger.info(
        "Controller Parameters: \"\"\"{}\"\"\"".format(controller_params_pb))

    controller_instance = ControllerInstance()
    controller_instance.start(controller_params_pb)
    controller_instance.shutdown()


if __name__ == "__main__":
    # FIXME(pkyriakis): the existence of hex-encoded args is not a user-friendly way to start the controller.
    # Since the hex encoding is required for sending the args over the wire, let's keep those
    # and add an additional wrapper that accepts user-friendly arg input
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--controller_server_entity_protobuff_serialized_hexadecimal", type=str,
                        required=True,
                        help="Controller server entity.")
    parser.add_argument("-g", "--global_model_specs_protobuff_serialized_hexadecimal", type=str,
                        required=True,
                        help="local models aggregation_rule (i.e, merging function) to create global model.")
    parser.add_argument("-c", "--communication_specs_protobuff_serialized_hexadecimal", type=str,
                        required=True,
                        help="what is the communication protocol for aggregating local models "
                             "(i.e., synchronous, asynchronous, semi_synchronous) and its specifications.")
    parser.add_argument("-m", "--model_hyperparameters_protobuff_serialized_hexadecimal", type=str,
                        required=True,
                        help="A serialized Model Hyperparameters protobuf message.")
    parser.add_argument("-s", "--model_store_config_protobuff_serialized_hexadecimal", type=str,
                        required=True,
                        help="A serialized Model Store Config protobuf message.")

    args = parser.parse_args()
    init_controller(args)
