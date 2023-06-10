import argparse

import metisfl.proto.metis_pb2 as metis_pb2

from metisfl.learner.core import Learner
from metisfl.learner.core.learner_servicer import LearnerServicer
from metisfl.learner.utils.proto_messages_factory import MetisProtoMessages


def init_learner(
    neural_engine="keras",
    learner_hostname="[::]",
    learner_port=50052,
    controller_hostname="[::]",
    controller_port=50051,
    model_definition="/tmp/metis/model/model_definition",
    train_dataset="/tmp/metis/model/model_train_dataset.npz",
    validation_dataset="",
    test_dataset="",
    train_dataset_recipe="/tmp/metis/model/model_train_dataset_ops.pkl",
    validation_dataset_recipe="",
    test_dataset_recipe="",
    he_scheme_protobuff_serialized_hexadecimal=None
):
    # Load train dataset specifications.
    train_dataset_filepath = train_dataset
    train_dataset_recipe_fp_pkl = train_dataset_recipe

    # Load validation dataset specifications.
    validation_dataset_filepath = validation_dataset
    validation_dataset_recipe_fp_pkl = validation_dataset_recipe

    # Load test dataset specifications.
    test_dataset_filepath = test_dataset
    test_dataset_recipe_fp_pkl = test_dataset_recipe

    if he_scheme_protobuff_serialized_hexadecimal is not None:
        # Parse serialized model hyperparameters object, 'recover' bytes object.
        # To do so, we need to convert the incoming hexadecimal representation
        # to bytes and pass it as initialization to the proto message object.
        he_scheme_protobuff_ser = bytes.fromhex(he_scheme_protobuff_serialized_hexadecimal)
        he_scheme_pb = metis_pb2.HEScheme()
        he_scheme_pb.ParseFromString(he_scheme_protobuff_ser)
    else:
        empty_scheme_pb = MetisProtoMessages.construct_empty_he_scheme_pb()
        he_scheme_pb = MetisProtoMessages.construct_he_scheme_pb(
            enabled=False, empty_scheme_pb=empty_scheme_pb)

    learner_credentials_fp = "/tmp/metis/learner_{}_credentials/".format(learner_port)
    learner_server_entity_pb = MetisProtoMessages.construct_server_entity_pb(learner_hostname, learner_port)
    controller_server_entity_pb = MetisProtoMessages.construct_server_entity_pb(controller_hostname, controller_port)
    learner = Learner(
        learner_server_entity=learner_server_entity_pb,
        controller_server_entity=controller_server_entity_pb,
        he_scheme=he_scheme_pb,
        nn_engine=neural_engine,
        model_fp=model_definition,
        train_dataset_fp=train_dataset_filepath,
        train_dataset_recipe_pkl=train_dataset_recipe_fp_pkl,
        test_dataset_fp=test_dataset_filepath,
        test_dataset_recipe_pkl=test_dataset_recipe_fp_pkl,
        validation_dataset_fp=validation_dataset_filepath,
        validation_dataset_recipe_pkl=validation_dataset_recipe_fp_pkl,
        learner_credentials_fp=learner_credentials_fp)
    learner_servicer = LearnerServicer(
        learner=learner,
        learner_server_entity=learner_server_entity_pb,
        servicer_workers=5)
    # First, initialize learner servicer for receiving train/evaluate/inference tasks.
    learner_servicer.init_servicer()
    # Second, block the servicer till a shutdown request is issued and no more requests are received.
    learner_servicer.wait_servicer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--neural_engine", type=str,
                        default="keras",
                        help="neural network training library")
    parser.add_argument("-l", "--learner_hostname", type=str,
                        default="[::]",
                        help="learner binding hostname")
    parser.add_argument("-p", "--learner_port", type=int,
                        default=50052,
                        help="learner binding port")
    parser.add_argument("-c", "--controller_hostname", type=str,
                        default="[::]",
                        help="controller binding hostname")
    parser.add_argument("-o", "--controller_port", type=int,
                        default=50051,
                        help="controller binding port")
    parser.add_argument("-m", "--model_definition", type=str,
                        default="/tmp/metis/model/model_definition",
                        help="model definition filepath")
    parser.add_argument("-t", "--train_dataset", type=str,
                        default="/tmp/metis/model/model_train_dataset.npz",
                        help="train dataset filepath")
    parser.add_argument("-v", "--validation_dataset", type=str,
                        default="",
                        help="validation dataset filepath")
    parser.add_argument("-s", "--test_dataset", type=str,
                        default="",
                        help="test dataset filepath")
    parser.add_argument("-u", "--train_dataset_recipe", type=str,
                        default="/tmp/metis/model/model_train_dataset_ops.pkl",
                        help="train dataset recipe")
    parser.add_argument("-w", "--validation_dataset_recipe", type=str,
                        default="",
                        help="validation dataset recipe")
    parser.add_argument("-z", "--test_dataset_recipe", type=str,
                        default="",
                        help="validation dataset recipe")
    parser.add_argument("-f", "--he_scheme_protobuff_serialized_hexadecimal", type=str,
                        default=None,
                        help="A serialized HE Scheme protobuf message.")

    args = parser.parse_args()

    init_learner(
        neural_engine=args.neural_engine,
        learner_hostname=args.learner_hostname,
        learner_port=args.learner_port,
        controller_port=args.controller_hostname,
        controller_port=args.controller_port,
        model_definition=args.model_definition,
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
        test_dataset=args.test_dataset,
        train_dataset_recipe=args.train_dataset_recipe,
        validation_dataset_recipe=args.validation_dataset_recipe,
        test_dataset_recipe=args.test_dataset_recipe,
        he_scheme_protobuff_serialized_hexadecimal=args.he_scheme_protobuff_serialized_hexadecimalb)

