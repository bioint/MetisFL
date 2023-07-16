import signal
import time
from metisfl.proto.metis_pb2 import ControllerParams
import metisfl.proto.metis_pb2 as metis_pb2

# This imports the controller python module defined inside the `pybind/controller_pybind.cc` script.
from metisfl.controller import controller


class Controller(object):

    def __init__(self):
        self.__controller_wrapper = controller.ControllerWrapper()
        self.__shutdown_signal_received = False

    def start(self, controller_params_pb):
        assert isinstance(controller_params_pb, ControllerParams)
        controller_params_ser = controller_params_pb.SerializeToString()
        self.__controller_wrapper.start(controller_params_ser)

    def shutdown(self, instantly=False):

        def sigint_handler(signum, frame):
            self.__shutdown_signal_received = True

        # Registering signal termination/shutdown calls.
        signal.signal(signal.SIGTERM, sigint_handler)
        signal.signal(signal.SIGINT, sigint_handler)

        # Infinite loop till shutdown signal is triggered.
        while True:
            shutdown_condition = \
                instantly or \
                self.__shutdown_signal_received or \
                self.__controller_wrapper.shutdown_request_received()
            if shutdown_condition:
                break
            time.sleep(0.01)

        self.__controller_wrapper.shutdown()
        

# class ControllerWrapper(object):

#     def __init__(self,                 
#                  hostname: str,
#                  port: int,
#                  aggregation_rule: str,
#                  scale_factor: int,
#                  stride_length: int,
#                  communication_protocol: str,
#                  ckks_params: list[int] = [], # scalling bits and batch size
#                  ckks_crypto_context_file: str = None,
#                  public_certificate_file: str = None, 
#                  private_key_file: str = None,
#                  learners_participation_ratio: float = 1.0,
#                  semi_sync_lambda: float = 0.0,
#                  semi_sync_recompute_num_updates: bool = False,
#                  model_store_config: str = "InMemory",
#                  percent_validation: float = 0.0):
#         pass


        

            
            