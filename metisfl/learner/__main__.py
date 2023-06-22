import argparse

import metisfl.proto.metis_pb2 as metis_pb2

from metisfl.learner.learner import Learner
from metisfl.learner.learner_servicer import LearnerServicer
from metisfl.utils.proto_messages_factory import MetisProtoMessages


# @stripeli the default values should not be here
def init_learner(learner_server_entity_protobuff_serialized_hexadecimal,
                 controller_server_entity_protobuff_serialized_hexadecimal,
                 he_scheme_protobuff_serialized_hexadecimal,
                 neural_engine,
                 model_dir,
                 train_dataset="/tmp/metis/model/model_train_dataset.npz",
                 validation_dataset="",
                 test_dataset="",
                 train_dataset_recipe="/tmp/metis/model/model_train_dataset_ops.pkl",
                 validation_dataset_recipe="",
                 test_dataset_recipe=""):
    if learner_server_entity_protobuff_serialized_hexadecimal is not None:
        learner_server_entity_pb = metis_pb2.ServerEntity()
        learner_server_entity_pb_ser = bytes.fromhex(args.learner_server_entity_protobuff_serialized_hexadecimal)
        learner_server_entity_pb.ParseFromString(learner_server_entity_pb_ser)
    else:
        learner_server_entity_pb = MetisProtoMessages.construct_server_entity_pb(
            hostname="[::]", port=50052)

    if controller_server_entity_protobuff_serialized_hexadecimal is not None:
        controller_server_entity_pb = metis_pb2.ServerEntity()
        controller_server_entity_pb_ser = bytes.fromhex(args.controller_server_entity_protobuff_serialized_hexadecimal)
        controller_server_entity_pb.ParseFromString(controller_server_entity_pb_ser)
    else:
        controller_server_entity_pb = MetisProtoMessages.construct_server_entity_pb(
            hostname="[::]", port=50051)

    # Training model engine and architecture definition.
    nn_engine = neural_engine
    model_dir = model_dir

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

    learner_credentials_fp = "/tmp/metis/learner_{}_credentials/".format(learner_server_entity_pb.port)
    learner = Learner(
        learner_server_entity=learner_server_entity_pb,
        controller_server_entity=controller_server_entity_pb,
        he_scheme=he_scheme_pb,
        nn_engine=nn_engine,
        model_dir=model_dir,
        train_dataset_fp=train_dataset_filepath,
        train_dataset_recipe_pkl=train_dataset_recipe_fp_pkl,
        test_dataset_fp=test_dataset_filepath,
        test_dataset_recipe_pkl=test_dataset_recipe_fp_pkl,
        validation_dataset_fp=validation_dataset_filepath,
        validation_dataset_recipe_pkl=validation_dataset_recipe_fp_pkl,
        learner_credentials_fp=learner_credentials_fp)
    learner_servicer = LearnerServicer(
        learner=learner,
        servicer_workers=5)
    # First, initialize learner servicer for receiving train/evaluate/inference tasks.
    learner_servicer.init_servicer()
    # Second, block the servicer till a shutdown request is issued and no more requests are received.
    learner_servicer.wait_servicer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learner_server_entity_protobuff_serialized_hexadecimal", type=str,
                        default="",
                        help="Learner server entity.")
    parser.add_argument("-c", "--controller_server_entity_protobuff_serialized_hexadecimal", type=str,
                        default="",
                        help="Controller server entity.")
    parser.add_argument("-f", "--he_scheme_protobuff_serialized_hexadecimal", type=str,
                        default="",
                        help="A serialized HE Scheme protobuf message.")
    parser.add_argument("-e", "--neural_engine", type=str,
                        default="keras",
                        help="neural network training library")
    parser.add_argument("-m", "--model_dir", type=str,
                        default="",
                        help="model definition directory")
    parser.add_argument("-t", "--train_dataset", type=str,
                        default="",
                        help="train dataset filepath")
    parser.add_argument("-v", "--validation_dataset", type=str,
                        default="",
                        help="validation dataset filepath")
    parser.add_argument("-s", "--test_dataset", type=str,
                        default="",
                        help="test dataset filepath")
    parser.add_argument("-u", "--train_dataset_recipe", type=str,
                        default="",
                        help="train dataset recipe")
    parser.add_argument("-w", "--validation_dataset_recipe", type=str,
                        default="",
                        help="validation dataset recipe")
    parser.add_argument("-z", "--test_dataset_recipe", type=str,
                        default="",
                        help="test dataset recipe")

    args = parser.parse_args()

    init_learner(
        learner_server_entity_protobuff_serialized_hexadecimal=args.learner_server_entity_protobuff_serialized_hexadecimal,
        controller_server_entity_protobuff_serialized_hexadecimal=args.controller_server_entity_protobuff_serialized_hexadecimal,
        he_scheme_protobuff_serialized_hexadecimal=args.he_scheme_protobuff_serialized_hexadecimal,
        neural_engine=args.neural_engine,
        model_dir=args.model_dir,
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
        test_dataset=args.test_dataset,
        train_dataset_recipe=args.train_dataset_recipe,
        validation_dataset_recipe=args.validation_dataset_recipe,
        test_dataset_recipe=args.test_dataset_recipe)
