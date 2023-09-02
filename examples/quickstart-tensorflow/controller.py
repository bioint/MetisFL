from metisfl.controller.controller_instance import Controller

from env import controller_params, global_train_config, model_store_config

controller_instance = Controller(
    server_params=controller_params,
    global_train_config=global_train_config,
    model_store_config=model_store_config,
)

controller_instance.start()
