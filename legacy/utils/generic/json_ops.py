import re

from federation.fed_execution import FedRoundExecutionResults, FedHostExecutionResultsStats
from simplejson import JSONEncoder


class JSONUtil(object):

	class FedRoundsExecutionResultsJsonEncoder(JSONEncoder):

		def default(self, o):
			if isinstance(o, FedRoundExecutionResults) or isinstance(o, FedHostExecutionResultsStats):
				return o.toJSON_representation()
			else:
				return o.__dict__

	@classmethod
	def lower_keys(cls, x, replace_capital_with_underscore=True):
		if isinstance(x, list):
			return [cls.lower_keys(v) for v in x]
		elif isinstance(x, dict):
			if replace_capital_with_underscore:
				return dict((re.sub('(?<!^)(?=[A-Z])', '_', k).lower(), cls.lower_keys(v)) for k, v in x.items())
			else:
				return dict((k.lower(), cls.lower_keys(v)) for k, v in x.items())
		else:
			return x
