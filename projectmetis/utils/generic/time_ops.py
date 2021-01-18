import time

from datetime import datetime


class TimeUtil(object):

	@classmethod
	def current_datetime(cls):
		return datetime.now()

	@classmethod
	def current_milli_time(cls):
		return int(round(time.time() * 1000))

	@classmethod
	def delta_diff_in_ms(cls, dt1, dt2):
		return abs(dt1 - dt2)

	@classmethod
	def delta_diff_in_secs(cls, dt1, dt2):
		milli_diff = cls.delta_diff_in_ms(dt1, dt2)
		sec_diff = milli_diff / 1000
		return sec_diff

	@classmethod
	def delta_diff_in_mins(cls, dt1, dt2):
		milli_diff = cls.delta_diff_in_ms(dt1, dt2)
		min_diff = (milli_diff / 1000) / 60
		return min_diff

	@classmethod
	def current_process_time(cls):
		return time.process_time() * 1000 # return as milliseconds

