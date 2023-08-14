
import config

from ..driver.controller_client import GRPCControllerClient
from ..grpc.server import get_server
from ..proto import learner_pb2_grpc
from ..utils.fedenv import ClientParams, ServerParams
from .learner import Learner
from .learner_servicer import LearnerServicer


def app(
    learner: Learner,
    controller_params: ServerParams,
    learner_params: ServerParams,
):
    client = GRPCControllerClient(
        client_params=ClientParams(
            hostname=controller_params.hostname,
            port=controller_params.port,
            root_certificate=controller_params.root_certificate,
        ),
        learner_id_fp=config.get_auth_token_fp(learner_params.port),
        auth_token_fp=config.get_auth_token_fp(learner_params.port),
    )

    servicer = LearnerServicer(
        learner=learner,
    )

    server = get_server(
        server_params=learner_params,
        servicer=servicer,
        add_servicer_to_server_fn=learner_pb2_grpc.add_LearnerServiceServicer_to_server,
    )
