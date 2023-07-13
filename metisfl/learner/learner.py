
import metisfl.proto.metis_pb2 as metis_pb2
from metisfl import config
from metisfl.models.utils import get_model_ops_fn
from metisfl.proto.proto_messages_factory import MetisProtoMessages

from .dataset_handler import LearnerDataset
from .learner_executor import LearnerExecutor
from .learner_servicer import LearnerServicer
from .task_executor import TaskExecutor


def parse_server_hex(hex_str, default_host, default_port):
    if hex_str:
        server_entity_pb = metis_pb2.ServerEntity()
        server_entity_pb_ser = bytes.fromhex(hex_str)
        server_entity_pb.ParseFromString(server_entity_pb_ser)
    else:
        server_entity_pb = MetisProtoMessages.construct_server_entity_pb(
            hostname=default_host, port=default_port)
    return server_entity_pb


def parse_he_scheme_hex(hex_str):
    if hex_str:
        he_scheme_protobuff_ser = bytes.fromhex(hex_str)
        he_scheme_config_pb = metis_pb2.HESchemeConfig()
        he_scheme_config_pb.ParseFromString(he_scheme_protobuff_ser)
    else:
        empty_scheme_config_pb = MetisProtoMessages.construct_empty_scheme_config_pb()
        he_scheme_config_pb = MetisProtoMessages.construct_he_scheme_config_pb(
            enabled=False, empty_scheme_config_pb=empty_scheme_config_pb)

    return he_scheme_config_pb


def create_servers(args):
    learner_server_entity_pb = parse_server_hex(
        args.learner_server_entity_protobuff_serialized_hexadecimal,
        config.DEFAULT_LEARNER_HOST, config.DEFAULT_LEARNER_PORT)
    controller_server_entity_pb = parse_server_hex(
        args.controller_server_entity_protobuff_serialized_hexadecimal,
        config.DEFAULT_CONTROLLER_HOSTNAME, config.DEFAULT_CONTROLLER_PORT)
    return learner_server_entity_pb, controller_server_entity_pb


def init_learner(args):
    learner_server_entity_pb, controller_server_entity_pb = create_servers(
        args)
    he_scheme_pb = parse_he_scheme_hex(
        args.he_scheme_protobuff_serialized_hexadecimal)
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
        he_scheme_pb=he_scheme_pb,
        learner_dataset=learner_dataset,
        learner_server_entity_pb=learner_server_entity_pb,
        model_dir=args.model_dir,
        model_ops_fn=model_ops_fn,
    )
    learner_executor = LearnerExecutor(task_executor=task_executor)
    learner_servicer = LearnerServicer(
        controller_server_entity_pb=controller_server_entity_pb,
        dataset_metadata=learner_dataset.get_dataset_metadata(),
        learner_executor=learner_executor,
        learner_server_entity_pb=learner_server_entity_pb,
        servicer_workers=5
    )

    # First, initialize learner servicer for receiving train/evaluate/inference tasks.
    learner_servicer.init_servicer()

    # Second, block the servicer till a shutdown request is issued and no more requests are received.
    # @stripeli: moved this to init_servicer() method
    # learner_servicer.wait_servicer()
