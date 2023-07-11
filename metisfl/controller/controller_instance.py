import signal
import time
from threading import Thread
from metisfl.proto.metis_pb2 import ControllerParams
# This imports the controller python module defined inside the `pybind/controller_pybind.cc` script.
from metisfl.controller import controller


class ControllerInstance(object):

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