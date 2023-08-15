
from typing import Optional

import config

from ..utils.fedenv import ClientParams, ServerParams
from .controller_client import GRPCControllerClient
from .learner import Learner
from .learner_server import LearnerServer


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
    
    client = GRPCControllerClient(
        client_params=ClientParams(
            hostname=controller_params.hostname,
            port=controller_params.port,
            root_certificate=controller_params.root_certificate,
        ),
        learner_id_fp=config.get_auth_token_fp(learner_params.port),
        auth_token_fp=config.get_auth_token_fp(learner_params.port),
    )

    server = LearnerServer(
        learner=learner,
        controller_params=controller_params,
    )

    server.start()

    client.join_federation(
        num_training_examples=num_training_examples,
    )