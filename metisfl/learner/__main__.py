import argparse
import metisfl.proto.metis_pb2 as metis_pb2
import metisfl.learner.constants as constants

from metisfl.grpc.grpc_controller_client import GRPCControllerClient
from metisfl.learner.dataset_handler import LearnerDataset
from metisfl.learner.learner_executor import LearnerExecutor
from metisfl.learner.learner_servicer import LearnerServicer
from metisfl.learner.task_executor import TaskExecutor
from metisfl.models import get_model_ops_fn
from metisfl.utils.proto_messages_factory import MetisProtoMessages

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

def create_servers(args):
    learner_server_entity_pb = parse_server_hex(
        args.learner_server_entity_protobuff_serialized_hexadecimal,
         constants.DEFAULT_LEARNER_HOST, constants.DEFAULT_LEARNER_PORT)
    controller_server_entity_pb = parse_server_hex(
        args.controller_server_entity_protobuff_serialized_hexadecimal,
        constants.DEFAULT_CONTROLLER_HOSTNAME, constants.DEFAULT_CONTROLLER_PORT)
        
    return learner_server_entity_pb,controller_server_entity_pb

def init_learner(args):
    learner_server_entity_pb, controller_server_entity_pb = create_servers(args)
    he_scheme_pb = parse_he_scheme_hex(args.he_scheme_protobuff_serialized_hexadecimal)
    model_ops_fn = get_model_ops_fn(args.neural_engine)
    
    learner_dataset = LearnerDataset(
        train_dataset_fp=args.train_dataset,
        validation_dataset_fp=args.validation_dataset,
        test_dataset_fp=args.test_dataset,
        train_dataset_recipe_pkl=args.train_dataset_recipe,
        validation_dataset_recipe_pkl=args.validation_dataset_recipe,
        test_dataset_recipe_pkl=args.test_dataset_recipe,
    )
    task_executor = TaskExecutor(
        learner_dataset=learner_dataset,
        model_ops_fn=model_ops_fn,
        he_scheme_pb=he_scheme_pb,
        model_dir=args.model_dir,
    )          
    learner_executor = LearnerExecutor(task_executor=task_executor)
    learner_controller_client = GRPCControllerClient(
        controller_server_entity=controller_server_entity_pb,
        learner_server_entity=learner_server_entity_pb,
        dataset_metadata=learner_dataset.get_dataset_metadata(),
        learner_id_fp=constants.LEARNER_ID_FP,
        auth_token_fp=constants.AUTH_TOKEN_FP
    )
    learner_servicer = LearnerServicer(
        learner_executor=learner_executor,
        learner_controller_client=learner_controller_client,
        servicer_workers=5
    )
    
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
