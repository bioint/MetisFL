import datetime as dt
import threading
import logging
import sys


class MyFormatter(logging.Formatter):
	"""
	Code for microseconds logging found at: https://stackoverflow.com/questions/6290739/python-logging-use-milliseconds-in-time-format
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

	log_formatter = MyFormatter("%(asctime)s: %(name)s: %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S.%f")
	ch = logging.StreamHandler(stream=sys.stderr)
	ch.setFormatter(log_formatter)
	__logger = logging.getLogger('Metis')
	__logger.setLevel('INFO')
	__logger.addHandler(ch)
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
	def log(cls, msg, *args, **kwargs):
		MetisLogger.getlogger().log(msg, *args, **kwargs)

	@classmethod
	def debug(cls, msg, *args, **kwargs):
		MetisLogger.getlogger().debug(msg, *args, **kwargs)

	@classmethod
	def error(cls, msg, *args, **kwargs):
		MetisLogger.getlogger().error(msg, *args, **kwargs)

	@classmethod
	def fatal(cls, msg, *args, **kwargs):
		MetisLogger.getlogger().fatal(msg, *args, **kwargs)

	@classmethod
	def info(cls, msg, *args, **kwargs):
		MetisLogger.getlogger().info(msg, *args, **kwargs)

	@classmethod
	def warning(cls, msg, *args, **kwargs):
		MetisLogger.getlogger().warning(msg, *args, **kwargs)