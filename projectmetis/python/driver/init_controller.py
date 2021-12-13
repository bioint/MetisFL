import argparse
import signal

import projectmetis.proto.metis_pb2 as metis_pb2

from projectmetis.python.logging.metis_logger import MetisLogger
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
    parser.add_argument("-m", "--communication_protocol", type=str,
                        default="SYNCHRONOUS",
                        help="what is the synchronization protocol for aggregation local models "
                             "(i.e., synchronous, asynchronous)")
    parser.add_argument("-y", "--model_hyperparameters_protobuff", type=str,
                        default=None,
                        help="A serialized Model Hyperparameters protobuf message.")
    args = parser.parse_args()

    controller_hostname = args.controller_hostname
    controller_port = args.controller_port
    # Proto message is recognized as capitalized.
    aggregation_rule = args.aggregation_rule.upper()
    learners_participation_ratio = args.learners_participation_ratio
    communication_protocol = args.communication_protocol.upper()
    # Parse serialized model hyperparameters object, 'recover' bytes object
    # If the given protobuff is None then we assign it an empty bytes object.
    # We then use it to initialize ControllerParams.ModelHyperparams() object.
    model_hyperparameters_protobuff = args.model_hyperparameters_protobuff
    if model_hyperparameters_protobuff is None:
        model_hyperparameters_protobuff = "b''"
    model_hyperparameters_protobuff = eval(model_hyperparameters_protobuff)
    model_hyperparams = metis_pb2.ControllerParams.ModelHyperparams()
    model_hyperparams.ParseFromString(model_hyperparameters_protobuff)

    controller_params = metis_pb2.ControllerParams(
        server_entity=metis_pb2.ServerEntity(
            hostname=controller_hostname,
            port=controller_port),
        global_model_specs=metis_pb2.GlobalModelSpecs(
            aggregation_rule=aggregation_rule,
            learners_participation_ratio=learners_participation_ratio,
        ),
        communication_specs=metis_pb2.CommunicationSpecs(
            protocol=communication_protocol
        ),
        model_hyperparams=model_hyperparams)

    MetisLogger.info("Controller Parameters: \"\"\"{}\"\"\"".format(controller_params))
    controller_instance = ControllerInstance()
    controller_instance.build_and_start(controller_params)

    def sigint_handler(signum, frame):
        controller_instance.shutdown()

    signal.signal(signal.SIGINT, sigint_handler)
    controller_instance.wait_until_signaled()
