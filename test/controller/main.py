import os
import signal
import sys

from metisfl.controller.controller_instance import ControllerInstance
from utils import utils
from metisfl.utils.metis_logger import MetisLogger

if __name__ == "__main__":
    MetisLogger.info('\n'.join(os.environ['PYTHONPATH'].split(':')))
    MetisLogger.info('')

    MetisLogger.info("=== Bazel Python ===")
    MetisLogger.info(sys.executable)
    MetisLogger.info(sys.version)

    MetisLogger.info("=== Host Python ===")
    MetisLogger.info(utils.cmd(["which", "python3"]))
    MetisLogger.info(utils.cmd(["python3", "-c", "import sys; print(sys.version)"]))

    params = utils.default_controller_params()
    controller_instance = ControllerInstance()
    controller_instance.build_and_start(params)

    def sigint_handler(signum, frame):
        controller_instance.shutdown()

    signal.signal(signal.SIGINT, sigint_handler)
    controller_instance.wait_until_signaled()
