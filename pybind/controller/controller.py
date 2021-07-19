import time

from projectmetis.proto.controller_pb2 import ControllerParams
from pybind.controller import controller


class Controller(object):

    def __init__(self):
        self.__service_wrapper = None
        self.__should_stop = False
        pass

    def build_and_start(self, controller_params_pb):
        assert isinstance(controller_params_pb, ControllerParams)
        controller_params_ser = controller_params_pb.SerializeToString()
        self.__service_wrapper = controller.BuildAndStart(controller_params_ser)
        return self.__service_wrapper

    def wait(self):
        if self.__service_wrapper is None:
            raise RuntimeError("Controller needs to be initialized.")
        controller.Wait(self.__service_wrapper)

    def wait_until_signaled(self):
        if self.__service_wrapper is None:
            raise RuntimeError("Controller needs to be initialized.")

        while not self.__should_stop:
            time.sleep(3)

    def shutdown(self):
        if self.__service_wrapper is None:
            raise RuntimeError("Controller needs to be initialized.")
        self.__should_stop = True
        controller.Shutdown(self.__service_wrapper)




