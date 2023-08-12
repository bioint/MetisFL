import signal
import time

from metisfl.controller import controller

from ..utils.fedenv import GlobalTrainConfig, ModelStoreConfig, ServerParams


class Controller(object):

    def __init__(
        self,
        server_params: ServerParams,
        global_train_config: GlobalTrainConfig,
        model_store_config: ModelStoreConfig
    ):
        self._server_params = server_params
        self._global_train_config = global_train_config
        self._model_store_config = model_store_config
        
        self._controller_wrapper = controller.ControllerWrapper()
        self._shutdown_signal_received = False

    def start(self):
        # TODO: need to pass the server params to the controller wrapper
        self._controller_wrapper.start()

    def shutdown(self, instantly=False):

        def sigint_handler():
            self._shutdown_signal_received = True

        signal.signal(signal.SIGTERM, sigint_handler)
        signal.signal(signal.SIGINT, sigint_handler)

        while True:
            shutdown_condition = \
                instantly or \
                self._shutdown_signal_received or \
                self._controller_wrapper.shutdown_request_received()
            if shutdown_condition:
                break
            time.sleep(0.01)

        if not self._controller_wrapper.shutdown_request_received():
            self._controller_wrapper.shutdown()
