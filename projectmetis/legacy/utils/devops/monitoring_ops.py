import tensorflow as tf


def run_session_with_location_trace(sess, op):
	# From:
	# 1. https://stackoverflow.com/a/41525764/7832197
	# 2. http://amid.fish/distributed-tensorflow-a-gentle-introduction#placement
	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	result = sess.run(op, options=run_options, run_metadata=run_metadata)
	for device in run_metadata.step_stats.dev_stats:
		print('Device: %s' % device.device)
		for node_stat in device.node_stats:
			print(node_stat)
			# print("  ", node.node_name)

	return result