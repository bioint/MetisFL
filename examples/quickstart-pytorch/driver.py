
""" Driver for the pytorch quickstart example. """

import argparse
import json

from controller import controller_params
from learner import get_learner_server_params

from metisfl.common.types import TerminationSingals
from metisfl.driver import DriverSession

if __name__ == "__main__":
    """ Entry point for the driver. """

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max-learners', type=int, default=3)
    args = parser.parse_args()
    max_learners = args.max_learners

    # Setup the environment.
    termination_signals = TerminationSingals(
        federation_rounds=5)
    learners = [get_learner_server_params(
        i, max_learners) for i in range(max_learners)]

    # Start the driver session.
    session = DriverSession(
        controller=controller_params,
        learners=learners,
        termination_signals=termination_signals,
    )

    # Run
    logs = session.run()

    # Save the results.
    with open('result.json', 'w') as f:
        json.dump(logs, f)
