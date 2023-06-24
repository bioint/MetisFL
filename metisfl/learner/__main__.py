import argparse
from metisfl.models.model_ops import ModelOps

import metisfl.proto.metis_pb2 as metis_pb2
from metisfl.learner.dataset_handler import LearnerDataset
from metisfl.learner.federation_helper import FederationHelper
from metisfl.learner.learner import Learner
from metisfl.learner.learner_evaluator import LearnerEvaluator
from metisfl.learner.learner_servicer import LearnerServicer
from metisfl.utils import fedenv_parser
from metisfl.utils.proto_messages_factory import MetisProtoMessages


DEFAULT_LEARNER_HOST = "[::]"
DEFAULT_LEARNER_PORT = 50052
DEFAULT_CONTROLLER_HOSTNAME = "[::]"
DEFAULT_CONTROLLER_PORT = 50051
LEARNER_CREDENTIALS_FP =  "/tmp/metis/learner_{}_credentials/"

def parse_server_hex(hex_str, default_host, default_port):
    if hex_str is not None:
        server_entity_pb = metis_pb2.ServerEntity()
        server_entity_pb_ser = bytes.fromhex(hex_str)
        server_entity_pb.ParseFromString(server_entity_pb_ser)
    else:
        server_entity_pb = MetisProtoMessages.construct_server_entity_pb(
            hostname=default_host, port=default_port)
    return server_entity_pb       

def parse_he_scheme_hex(hex_str):
    if hex_str is not None:
        he_scheme_pb = metis_pb2.HEScheme()
        he_scheme_pb_ser = bytes.fromhex(hex_str)
        he_scheme_pb.ParseFromString(he_scheme_pb_ser)
    else:
        empty_scheme_pb = MetisProtoMessages.construct_empty_he_scheme_pb()
        he_scheme_pb = MetisProtoMessages.construct_he_scheme_pb(
            enabled=False, empty_scheme_pb=empty_scheme_pb)
    return he_scheme_pb

def get_model_backend(nn_engine, model_dir) -> ModelOps:
    if nn_engine == "keras":
        from metisfl.models.keras.keras_model_ops import KerasModelOps
        return KerasModelOps(model_dir)
    elif nn_engine == "pytorch":
        from metisfl.models.pytorch.pytorch_model_ops import PyTorchModelOps
        return PyTorchModelOps(model_dir)
    else :
        raise ValueError("Unknown neural engine: {}".format(nn_engine))

def create_servers(args):
    learner_server_entity_pb = parse_server_hex(
        args.learner_server_entity_protobuff_serialized_hexadecimal,
         DEFAULT_LEARNER_HOST, DEFAULT_LEARNER_PORT)
    controller_server_entity_pb = parse_server_hex(
        args.controller_server_entity_protobuff_serialized_hexadecimal,
        DEFAULT_CONTROLLER_HOSTNAME, DEFAULT_CONTROLLER_PORT)
        
    return learner_server_entity_pb,controller_server_entity_pb

def init_learner(args):
    learner_server_entity_pb, controller_server_entity_pb = create_servers(args)
    he_scheme_pb = parse_he_scheme_hex(args.he_scheme_protobuff_serialized_hexadecimal)
    homomorphic_encryption = fedenv_parser.HomomorphicEncryption.from_proto(he_scheme_pb) # @stripeli: check this please
    model_backend = get_model_backend(args.neural_engine, args.model_dir)
    learner_credentials_fp =  LEARNER_CREDENTIALS_FP.format(learner_server_entity_pb.port)
    
    learner_dataset = LearnerDataset(
        train_dataset_fp=args.train_dataset,
        validation_dataset_fp=args.validation_dataset,
        test_dataset_fp=args.test_dataset,
        train_dataset_recipe_pkl=args.train_dataset_recipe,
        validation_dataset_recipe_pkl=args.validation_dataset_recipe,
        test_dataset_recipe_pkl=args.test_dataset_recipe,
    )
    
    learner_evaluator = LearnerEvaluator(
        learner_dataset=learner_dataset,
        model_backend=model_backend,
        homomorphic_encryption=homomorphic_encryption,
    )
    
    ## FIXME: dataset should not be passed here
    federation_helper = FederationHelper(
        learner_server_entity=learner_server_entity_pb,
        controller_server_entity=controller_server_entity_pb,
        learner_credentials_fp=learner_credentials_fp,
        learner_dataset=learner_dataset,
    )
           
    learner = Learner(
        federation_helper=federation_helper,
        learner_evaluator=learner_evaluator
    )
    
    learner_servicer = LearnerServicer(
        learner=learner,
        federation_helper=federation_helper,
        servicer_workers=5)
    
    # First, initialize learner servicer for receiving train/evaluate/inference tasks.
    learner_servicer.init_servicer()
    # Second, block the servicer till a shutdown request is issued and no more requests are received.
    learner_servicer.wait_servicer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # @stripeli the first 3-4 args are the most verbose I've ever seen :)
    # and some of them shorthands do not make sense, e.g. -e for neural engine :)
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
    init_learner(args)
