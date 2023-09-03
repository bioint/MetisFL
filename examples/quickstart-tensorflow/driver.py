
import argparse
import json


from metisfl.common.types import FederationEnvironment, LocalTrainConfig, TerminationSingals
from metisfl.driver.driver_session import DriverSession

from controller import (controller_params, global_train_config,
                        model_store_config)
from learner import get_learner_server_params

if __name__ == "__main__":
    """ Entry point for the driver. """

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max-learners', type=int, default=1)
    args = parser.parse_args()
    max_learners = args.max_learners

    # Setup the environment.
    env = FederationEnvironment(
        termination_signals=TerminationSingals(
            federation_rounds=5,
        ),
        local_train_config=LocalTrainConfig(
            epochs=10,
            batch_size=64,
        ),
        global_train_config=global_train_config,
        model_store_config=model_store_config,
        controller=controller_params,
        learners=[
            get_learner_server_params(i) for i in range(max_learners)
        ]
    )

    # Start the driver session.
    session = DriverSession(env)

    # Run the driver session.
    res = session.run()

    # Save the results.
    with open('result.json', 'w') as f:
        json.dump(res, f)
