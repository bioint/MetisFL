
from typing import Optional

from ..config import get_auth_token_fp, get_learner_id_fp
from ..utils.fedenv import ClientParams, ServerParams
from .controller_client import GRPCControllerClient
from .learner import Learner
from .learner_server import LearnerServer
from .task_manager import TaskManager

def app(
    learner: Learner,
    controller_params: ServerParams,
    learner_params: ServerParams,
    num_training_examples: Optional[int] = None,
):
    """Entry point for the MetisFL Learner application.

    Parameters
    ----------
    learner : Learner
        The Learner object. Must impliment the Learner interface.
    controller_params : ServerParams
        The server parameters of the Controller server. Used by the Learner to connect to the Controller.
    learner_params : ServerParams
        The server parameters of the Learner server.
    num_training_examples : Optional[int], (default=None)
        TODO: complete this docstring
        The number of training examples. Used in certain aggregaction strategies by the Controller.
        If None, the aggregation strategies that need this parameter will not run. By default None
    """    
    
    client_params = ClientParams(
        hostname=controller_params.hostname,
        port=controller_params.port,
        root_certificate=controller_params.root_certificate,
    )        
    
    port = learner_params.port
    
    # Create the gRPC client to communicate with the Controller
    client = GRPCControllerClient(
        client_params=client_params,
        learner_id_fp=get_auth_token_fp(port),
        auth_token_fp=get_learner_id_fp(port),
    )

    # Register with the Controller
    client.join_federation(
        num_training_examples=num_training_examples,
    )
    
    # Create the gRPC server for the Controller to communicate with the Learner
    server = LearnerServer(
        learner=learner,
        controller_params=controller_params,
        task_manager=TaskManager(),
        client=client,
    )

    # Blocking until Shutdown endpoint is called
    server.start()
