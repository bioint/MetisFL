import signal
import time
from threading import Thread
from metisfl.proto.metis_pb2 import ControllerParams
# This imports the controller python module defined inside the `pybind/controller_pybind.cc` script.
from metisfl.controller import controller


class ControllerInstance(object):

    def __init__(self):
        self.__service_wrapper = controller.ServicerWrapper()
        self.__shutdown_signal_received = False

    def build_and_start(self, controller_params_pb):
        assert isinstance(controller_params_pb, ControllerParams)
        controller_params_ser = controller_params_pb.SerializeToString()
        self.__service_wrapper.BuildAndStart(controller_params_ser)

    def wait(self):

        self.__service_wrapper.Wait()
        # def sigint_handler(signum, frame):
        #     self.__shutdown_signal_received = True
        #     self.shutdown()
        #
        # # Registering signal termination/shutdown calls.
        # signal.signal(signal.SIGTERM, sigint_handler)
        # signal.signal(signal.SIGINT, sigint_handler)
        #
        # # Infinite loop till shutdown signal is triggered.
        # while not self.__shutdown_signal_received:
        #     time.sleep(3)

    def shutdown(self):
        self.__service_wrapper.Shutdown()
