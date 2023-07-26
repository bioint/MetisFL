import argparse

import metisfl.proto.metis_pb2 as metis_pb2
from metisfl.controller.controller_instance import Controller
from metisfl.utils.logger import MetisLogger

def _get_pb_from_hexadecimal_str(hexadecimal_str, pb_class):
    pb = pb_class()
    pb_ser = bytes.fromhex(hexadecimal_str)
    pb.ParseFromString(pb_ser)
    return pb

def init_controller(args):
    # For all incoming hexadecimal representations, we need to first convert them
    # to bytes and later pass them as initialization to the proto message object.
    # If the given protobuff is None then we assign it an empty bytes object.
    controller_server_entity_pb = _get_pb_from_hexadecimal_str(
        args.controller_server_entity_protobuff_serialized_hexadecimal,
        metis_pb2.ServerEntity)
    
    global_model_specs_pb = _get_pb_from_hexadecimal_str(
        args.global_model_specs_protobuff_serialized_hexadecimal,
        metis_pb2.GlobalModelSpecs)
    
    communication_specs_pb = _get_pb_from_hexadecimal_str(
        args.communication_specs_protobuff_serialized_hexadecimal,
        metis_pb2.CommunicationSpecs)

    model_hyperparams_pb = _get_pb_from_hexadecimal_str(
        args.model_hyperparameters_protobuff_serialized_hexadecimal,
        metis_pb2.ControllerParams.ModelHyperparams)
    
    model_store_config_pb = _get_pb_from_hexadecimal_str(
        args.model_store_config_protobuff_serialized_hexadecimal,
        metis_pb2.ModelStoreConfig)

    controller_params_pb = metis_pb2.ControllerParams(server_entity=controller_server_entity_pb,
                                                      global_model_specs=global_model_specs_pb,
                                                      communication_specs=communication_specs_pb,
                                                      model_store_config=model_store_config_pb,
                                                      model_hyperparams=model_hyperparams_pb)
    MetisLogger.info(
        "Controller Parameters: \"\"\"{}\"\"\"".format(controller_params_pb))

    controller_instance = Controller()
    controller_instance.start(controller_params_pb)
    controller_instance.shutdown()


if __name__ == "__main__":
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
