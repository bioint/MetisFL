import argparse
import ast
import os
import pickle

import tensorflow as tf


tf.logging.set_verbosity(tf.logging.DEBUG)


def main():

	SERVER_INSTANCE_JOB_NAME = str(FLAGS.job_name)
	SERVER_INSTANCE_TASK_IDX = int(FLAGS.task_index)

	SERVER_INSTANCE_CLUSTER_SPEC = str(FLAGS.cluster_spec)
	# Convert cluster specification to a dictionary representation
	SERVER_INSTANCE_CLUSTER_SPEC = pickle.loads(ast.literal_eval(SERVER_INSTANCE_CLUSTER_SPEC))
	# Convert dictionary to a tensorflow native tf.train.ClusterSpec representation
	SERVER_INSTANCE_CLUSTER_SPEC = tf.train.ClusterSpec(SERVER_INSTANCE_CLUSTER_SPEC)

	SERVER_INSTANCE_INTER_OP_THREADS = int(FLAGS.intra_op_threads)
	SERVER_INSTANCE_INTRA_OP_THREADS = int(FLAGS.inter_op_threads)
	SERVER_INSTANCE_IS_GPU = bool(FLAGS.is_gpu)
	SERVER_INSTANCE_GPU_MEMORY_FRACTION = float(FLAGS.gpu_memory_fraction)

	tf.logging.info("Cluster Specification: {}".format(SERVER_INSTANCE_CLUSTER_SPEC))

	if 'CUDA_VISIBLE_DEVICES' in os.environ:
		tf.logging.info("CUDA_DEVICE: {}, SERVER_TYPE: {}, TASK_INDEX: {}".format(os.environ['CUDA_VISIBLE_DEVICES'],
																				  SERVER_INSTANCE_JOB_NAME,
																				  SERVER_INSTANCE_TASK_IDX))
	else:
		tf.logging.info("SERVER_TYPE: {}, TASK_INDEX: {}".format(SERVER_INSTANCE_JOB_NAME,
																 SERVER_INSTANCE_TASK_IDX))

	gpu_options = None
	if SERVER_INSTANCE_IS_GPU:
		gpu_options = tf.GPUOptions(allow_growth=True,
									per_process_gpu_memory_fraction=SERVER_INSTANCE_GPU_MEMORY_FRACTION)

	config = tf.ConfigProto(gpu_options=gpu_options,
							inter_op_parallelism_threads=SERVER_INSTANCE_INTER_OP_THREADS,
							intra_op_parallelism_threads=SERVER_INSTANCE_INTRA_OP_THREADS,
							allow_soft_placement=True)

	server = tf.train.Server(SERVER_INSTANCE_CLUSTER_SPEC,
							 job_name=SERVER_INSTANCE_JOB_NAME,
							 task_index=SERVER_INSTANCE_TASK_IDX,
							 config=config,
							 start=True)
	server.join()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--job_name",
		type=str,
		default="",
		help="One of 'ps', 'worker'",
		required=True
	)
	parser.add_argument(
		"--task_index",
		type=int,
		default=0,
		help="Index of task within the job",
		required=True
	)
	parser.add_argument(
		"--cluster_spec",
		type=str,
		default=None,
		help="Tensorflow Cluster Specification (tf.train.ClusterSpec) given as a pickled string",
		required=True
	)
	parser.add_argument(
		"--inter_op_threads",
		type=int,
		default=3,
		help="Number of threads for inter_op_parallelism",
		required=False
	)
	parser.add_argument(
		"--intra_op_threads",
		type=int,
		default=3,
		help="Number of threads for intra_op_parallelism",
		required=False
	)
	parser.add_argument(
		"--is_gpu",
		type=bool,
		default=False,
		help="Whether this is a gpu server or not",
		required=False
	)
	parser.add_argument(
		"--gpu_memory_fraction",
		type=float,
		default=0.3,
		help="Size of memory allocation per session process",
		required=False
	)

	FLAGS, unparsed = parser.parse_known_args()
	main()