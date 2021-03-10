import socket


class NetOpsUtil(object):

	@classmethod
	def is_endpoint_listening(cls, host, port):
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		try:
			s.connect((host, int(port)))
			s.shutdown(2)
			return True
		except:
			return False

	@classmethod
	def get_hostname(cls):
		return socket.gethostname()