import signal
import time
from typing import Optional

from metisfl.common.types import (GlobalTrainConfig, ModelStoreConfig,
                                  ServerParams)
from metisfl.controller import controller


class Controller(object):

    def __init__(
        self,
        server_params: ServerParams,
        global_train_config: GlobalTrainConfig,
        model_store_config: ModelStoreConfig
    ):
        """Initializes the MetisFL Controller.

        Parameters
        ----------
        server_params : ServerParams
            The server parameters.
        global_train_config : GlobalTrainConfig
            Configuration for the global training.
        model_store_config : ModelStoreConfig
            Configuration for the model store.
        """
        self._server_params = server_params
        self._global_train_config = global_train_config
        self._model_store_config = model_store_config
        self._controller_wrapper = controller.ControllerWrapper()
        self._shutdown_signal_received = False

    def start(self):
        """Starts the controller."""

        server = self._server_params
        global_train = self._global_train_config
        model_store = self._model_store_config

        # Optional parameters are passed as empty strings or -1.
        self._controller_wrapper.start(
            hostname=server.hostname,
            port=server.port,
            root_certificate=server.root_certificate or "",
            server_certificate=server.server_certificate or "",
            private_key=server.private_key or "",

            aggregation_rule=global_train.aggregation_rule,
            communication_protocol=global_train.communication_protocol,
            scaling_factor=global_train.scaling_factor,
            participation_ratio=global_train.participation_ratio,
            stride_length=global_train.stride_length or -1,
            he_batch_size=global_train.he_batch_size or -1,
            he_scaling_factor_bits=global_train.he_scaling_factor_bits or -1,
            he_crypto_context_file=global_train.he_crypto_context_file or "",
            semi_sync_lambda=global_train.semi_sync_lambda or -1,
            semi_sync_recompute_num_updates=global_train.semi_sync_recompute_num_updates or -1,

            model_store=model_store.model_store,
            lineage_length=model_store.lineage_length,
            model_store_hostname=model_store.model_store_hostname or "",
            model_store_port=model_store.model_store_port or -1,
        )

        # Wait for the controller to shut down.
        self.wait()

    def wait(self, stop_instantly: Optional[bool] = False):
        """Waits for the shutdown signal.

        Parameters
        ----------
        instantly : Optional[bool], (default=False)
            If True, the controller will be shut down instantly.
            Otherwise, the controller will wait either for a shutdown request
            from the server or for a SIGINT/SIGTERM signal.
        """
        def sigint_handler(signum, frame):
            self._shutdown_signal_received = True

        signal.signal(signal.SIGTERM, sigint_handler)
        signal.signal(signal.SIGINT, sigint_handler)

        while True:
            shutdown_condition = \
                stop_instantly or \
                self._shutdown_signal_received or \
                self._controller_wrapper.shutdown_request_received()
            if shutdown_condition:
                break
            time.sleep(0.01)

        if not self._controller_wrapper.shutdown_request_received():
            self._controller_wrapper.shutdown()
