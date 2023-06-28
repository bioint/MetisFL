import signal
import time
from threading import Thread
from metisfl.proto.metis_pb2 import ControllerParams
# This imports the controller python module defined inside the `pybind/controller_pybind.cc` script.
from metisfl.controller import controller


class ControllerInstance(object):

    def __init__(self):
        self.__controller_wrapper = controller.ControllerWrapper()
        self.__shutdown_signal = False

    def start(self, controller_params_pb):
        assert isinstance(controller_params_pb, ControllerParams)
        controller_params_ser = controller_params_pb.SerializeToString()
        self.__controller_wrapper.start(controller_params_ser)

    def shutdown(self, instantly=False):
        """
        The function registers the termination signals that will modify the
        internal state of the controller instance and trigger a shutdown() event.

        The function also checks whether the controller has received any shutdown
        requests at the servicer level; recall this instance is a wrapper of the
        actual controller class. In such a case we only need to wait() for any
        resources to be released by the main thread that started the controller.

        Parameters
        ----------
        instantly : bool, optional
            Whether to stop/shutdown the controller instance right away.
        """
        def sigint_handler(signum, frame):
            self.__shutdown_signal = True

        # Registering signal termination/shutdown calls.
        signal.signal(signal.SIGTERM, sigint_handler)
        signal.signal(signal.SIGINT, sigint_handler)

        # Infinite loop till shutdown signal is triggered
        # or shutdown request is received. The
        while True:
            if instantly or self.__shutdown_signal:
                self.__controller_wrapper.shutdown()
                break
            if self.__controller_wrapper.shutdown_request_received():
                self.__controller_wrapper.wait()
                break
            time.sleep(0.01)
