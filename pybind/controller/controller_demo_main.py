import os
import signal
import sys

from controller import Controller
from pybind.controller import controller_demo as demo

if __name__ == "__main__":
    print('\n'.join(os.environ['PYTHONPATH'].split(':')), flush=True)
    print('', flush=True)

    print("=== Bazel Python ===", flush=True)
    print(sys.executable, flush=True)
    print(sys.version, flush=True)

    print("=== Host Python ===", flush=True)
    print(demo.cmd(["which", "python3"]), flush=True)
    print(demo.cmd(["python3", "-c", "import sys; print(sys.version)"]), flush=True)

    params = demo.default_controller_params()
    controller_instance = Controller()
    controller_instance.build_and_start(params)

    def sigint_handler(signum, frame):
        controller_instance.shutdown()

    signal.signal(signal.SIGINT, sigint_handler)

    controller_instance.wait_until_signaled()
