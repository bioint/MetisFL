
from typing import Optional

from ..config import get_auth_token_fp
from ..common.types import ClientParams, ServerParams
from .controller_client import GRPCClient
from .learner import Learner
from .learner_server import LearnerServer
from .task_manager import TaskManager


def app(
    learner: Learner,
    client_params: ClientParams,
    server_params: ServerParams,
    num_training_examples: Optional[int] = None,
):
    """Entry point for the MetisFL Learner application.

    Parameters
    ----------
    learner : Learner
        The Learner object. Must impliment the Learner interface.
    client_params : ClientParams
        The client parameters of the Learner client. 
    server_params : ServerParams
        The server parameters of the Learner server. 
    num_training_examples : Optional[int], (default=None)
        TODO: complete this docstring
        The number of training examples. Used in certain aggregaction strategies by the Controller.
        If None, the aggregation strategies that need this parameter will not run. By default None
    """

    port = client_params.port

    # Create the gRPC client to communicate with the Controller
    client = GRPCClient(
        client_params=client_params,
        learner_id_fp=get_auth_token_fp(port),
    )

    # Create the gRPC server for the Controller to communicate with the Learner
    server = LearnerServer(
        learner=learner,
        server_params=server_params,
        task_manager=TaskManager(),
        client=client,
    )

    # Register with the Controller
    client.join_federation(
        num_training_examples=num_training_examples,
        server_params=server_params,
    )

    # Blocking until Shutdown endpoint is called
    server.start()
