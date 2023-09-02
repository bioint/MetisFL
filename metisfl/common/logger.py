import inspect
import threading
import logging
import os

import datetime as dt

from termcolor import cprint
from pyfiglet import figlet_format


class MetisASCIIArt(object):

    @classmethod
    def print(cls):
        # Print 'METIS Federated Learning' on console as an ASCII-Art pattern.
        cprint(figlet_format('METIS', font='greek'),
               'blue', None, attrs=['bold'], flush=True)
        cprint(figlet_format('Federated Learning Framework', width=150),
               'blue', None, attrs=['bold'], flush=True)


class MyFormatter(logging.Formatter):
    """
    Code for microseconds logging found at: 
    https://stackoverflow.com/questions/6290739/python-logging-use-milliseconds-in-time-format
    """
    converter = dt.datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s


class MetisLogger(object):

    log_formatter = MyFormatter(
        '%(asctime)s %(name)s [%(levelname)s] %(message)s',
        datefmt="%H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    __logger = logging.getLogger('MetisFL')
    __logger.setLevel('INFO')
    __logger.addHandler(console_handler)
    __logger.propagate = False
    __logger_lock = threading.Lock()

    @classmethod
    def getlogger(cls):
        MetisLogger.__logger_lock.acquire()
        try:
            return MetisLogger.__logger
        finally:
            MetisLogger.__logger_lock.release()

    @classmethod
    def log_with_filename(cls, level, msg):
        caller_frame = inspect.stack()[2]  # Get the caller's frame
        filename = os.path.basename(caller_frame.filename)
        lineno = caller_frame.lineno
        log_msg = f"{filename}:{lineno}: {msg}"
        cls.getlogger().log(level, log_msg)

    @classmethod
    def debug(cls, msg):
        cls.log_with_filename(logging.DEBUG, msg)

    @classmethod
    def info(cls, msg):
        cls.log_with_filename(logging.INFO, msg)

    @classmethod
    def warning(cls, msg):
        cls.log_with_filename(logging.WARNING, msg)

    @classmethod
    def error(cls, msg):
        cls.log_with_filename(logging.ERROR, msg)

    @classmethod
    def fatal(cls, msg):
        cls.log_with_filename(logging.CRITICAL, msg)
