import signal
import time

from metisfl.proto.metis_pb2 import ControllerParams
# This imports the controller python module defined inside the `pybind/controller_pybind.cc` script.
from metisfl.controller import controller


class ControllerInstance(object):

    def __init__(self):
        self.__service_wrapper = None
        self.__should_stop = False

    def build_and_start(self, controller_params_pb):
        assert isinstance(controller_params_pb, ControllerParams)
        controller_params_ser = controller_params_pb.SerializeToString()
        self.__service_wrapper = controller.BuildAndStart(
            controller_params_ser)
        return self.__service_wrapper

    def wait(self):
        if self.__service_wrapper is None:
            raise RuntimeError("Controller needs to be initialized.")
        controller.Wait(self.__service_wrapper)

    def wait_until_signaled(self):

        if self.__service_wrapper is None:
            raise RuntimeError("Controller needs to be initialized.")

        def sigint_handler(signum, frame):
            self.shutdown()

        # Registering signal termination/shutdown calls.
        signal.signal(signal.SIGTERM, sigint_handler)
        signal.signal(signal.SIGINT, sigint_handler)

        # Infinite loop till shutdown signal is triggered.
        while not self.__should_stop:
            time.sleep(3)

    def shutdown(self):
        if self.__service_wrapper is None:
            raise RuntimeError("Controller needs to be initialized.")
        self.__should_stop = True
        controller.Shutdown(self.__service_wrapper)
