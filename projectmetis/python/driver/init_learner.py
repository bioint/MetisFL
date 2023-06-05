import argparse

import projectmetis.proto.metis_pb2 as metis_pb2

from projectmetis.python.learner.learner import Learner
from projectmetis.python.learner.learner_servicer import LearnerServicer
from projectmetis.python.utils.proto_messages_factory import MetisProtoMessages

from projectmetis.python.logging.metis_logger import MetisLogger

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
    parser.add_argument("-f", "--fhe_scheme_protobuff", type=str,
                        default=None,
                        help="A serialized FHE Scheme protobuf message.")

    args = parser.parse_args()

    nn_engine = args.neural_engine
    learner_hostname = args.learner_hostname
    learner_port = args.learner_port
    controller_hostname = args.controller_hostname
    controller_port = args.controller_port
    model_filepath = args.model_definition

    # Load train dataset specifications.
    train_dataset_filepath = args.train_dataset
    train_dataset_recipe_fp_pkl = args.train_dataset_recipe

    # Load validation dataset specifications.
    validation_dataset_filepath = args.validation_dataset
    validation_dataset_recipe_fp_pkl = args.validation_dataset_recipe

    # Load test dataset specifications.
    test_dataset_filepath = args.test_dataset
    test_dataset_recipe_fp_pkl = args.test_dataset_recipe

    if args.fhe_scheme_protobuff is not None:
        fhe_scheme_protobuff = eval(args.fhe_scheme_protobuff)
        fhe_scheme = metis_pb2.FHEScheme()
        fhe_scheme.ParseFromString(fhe_scheme_protobuff)
    else:
        fhe_scheme = MetisProtoMessages.construct_fhe_scheme_pb(enabled=False)

    learner_credentials_fp = "/tmp/metis/learner_{}_credentials/".format(learner_port)
    learner_server_entity_pb = MetisProtoMessages.construct_server_entity_pb(learner_hostname, learner_port)
    controller_server_entity_pb = MetisProtoMessages.construct_server_entity_pb(controller_hostname, controller_port)
    learner = Learner(
        learner_server_entity=learner_server_entity_pb,
        controller_server_entity=controller_server_entity_pb,
        fhe_scheme=fhe_scheme,
        nn_engine=nn_engine,
        model_fp=model_filepath,
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
