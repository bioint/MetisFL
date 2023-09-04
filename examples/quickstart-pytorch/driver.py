
""" Driver for the quickstart example. """

import argparse
import json

from controller import controller_params, global_train_config
from learner import get_learner_server_params

from metisfl.common.types import TerminationSingals
from metisfl.driver import DriverSession

if __name__ == "__main__":
    """ Entry point for the driver. """

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max-learners', type=int, default=1)
    args = parser.parse_args()
    max_learners = args.max_learners

    # Setup the environment.
    termination_signals = TerminationSingals(
        federation_rounds=5)
    learners = [get_learner_server_params(i) for i in range(max_learners)]
    is_async = global_train_config.communication_protocol == 'Asynchronous'

    # Start the driver session.
    session = DriverSession(
        controller=controller_params,
        learners=learners,
        termination_signals=termination_signals,
        is_async=is_async,
    )

    # Run the driver session.
    logs = session.run()

    # Save the results.
    with open('result.json', 'w') as f:
        json.dump(logs, f)
