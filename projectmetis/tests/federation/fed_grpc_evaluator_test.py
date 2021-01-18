from federation.fed_grpc_evaluator_client import FedModelEvaluatorClient
from federation.fed_grpc_evaluator import FedModelEvaluator
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.objectdetection.imgdata_client import MetisDBSession
from utils.generic.time_ops import TimeUtil

from experiments.tf_fedmodels.cnn.cnn2_cifar10_model import Cifar10FedModel

import federation.fed_cluster_env as fed_cluster_env

import federation.fed_model as fedmodel
import os


# import sys
# sys.path.append("pydevd-pycharm.egg")
# import pydevd
# pydevd.settrace('bdnf.isi.edu', port=47001, stdoutToServer=True, stderrToServer=True)

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

scriptDirectory = os.path.dirname(os.path.realpath(__file__))

TRAINING_EXAMPLES_NUM = 6000
DEV_EXAMPLES_NUM = 0
TEST_EXAMPLES_NUM = 1000
IID_DATA_PARTITIONING = True
CLASSES_PER_PARTITION = 3
PARTITIONS_NUM = 3
SHARDS_NUM = None
RUN_WITH_DISTORTED_IMAGES = True
EVALUATOR_HOST_PORT = "bdnf.isi.edu:8787"

if __name__=="__main__":

	tf_fedcluster = fed_cluster_env.FedEnvironment.init_multi_localhost_tf_clusters(hostname="bdnf.isi.edu",
																					clusters_num=PARTITIONS_NUM,
																					federation_rounds=10,
																					target_local_epochs=3,
																					batch_size_per_worker=50,
																					starting_port=5221)

	# Fill Data Cache, start DB Session
	st = TimeUtil.current_milli_time()
	metis_logger.info(msg='Initializing Data Cache...')
	metis_db_session = MetisDBSession(cifar10_session=True)
	metis_db_session.load_session_dataset(train_examples=TRAINING_EXAMPLES_NUM, dev_examples=DEV_EXAMPLES_NUM, test_examples=TEST_EXAMPLES_NUM, distort_images=RUN_WITH_DISTORTED_IMAGES)
	metis_db_session.partition_session_training_data(partitions_num=PARTITIONS_NUM, classes_per_partition=CLASSES_PER_PARTITION)
	metis_logger.info('Data Partitioning Scheme: %s' % metis_db_session.partition_policy)
	metis_logger.info('Data Partitions: %s' % metis_db_session.partitions_num)
	metis_logger.info('Classes Per Partition: %s' % CLASSES_PER_PARTITION)
	metis_logger.info('Training Data Num: %s' % metis_db_session.train_examples_num)
	metis_logger.info('Testing Data Num: %s' % metis_db_session.test_examples_num)
	metis_logger.info(msg='Data Cache Filled')
	et = TimeUtil.current_milli_time()
	metis_logger.info('Data Cache Fill Time: %s ms' % TimeUtil.delta_diff_in_ms(et, st))

	nnmodel = Cifar10FedModel(learning_rate=0.0015, momentum=0.0, run_with_distorted_images=RUN_WITH_DISTORTED_IMAGES)
	federated_variables = fedmodel.FedModelDef.construct_model_federated_variables(nnmodel)

	evaluator = FedModelEvaluator(grpc_servicer_host_port=EVALUATOR_HOST_PORT, fed_environment=tf_fedcluster, federated_variables=federated_variables, federation_model_obj=nnmodel,
								  metis_db_session=metis_db_session, target_stat_name='accuracy', validation_proportion=0.05, max_workers=10)
	evaluator.start()

	print("Init GRPC-Evaluator Client 1")
	grpc_client1 = FedModelEvaluatorClient(client_id=tf_fedcluster.partition_idx_host_catalog[0], evaluator_host_port=EVALUATOR_HOST_PORT)
	learner_model = [x.value for x in federated_variables]
	evaluation_score = grpc_client1.request_model_evaluation(model_variables=learner_model, is_community_model=False, block=True)
	print(grpc_client1.request_is_federation_loss_threshold_reached_for_all_learners())
	print(evaluation_score)
	grpc_client1.shutdown()

	print("Init GRPC-Evaluator Client 2")
	grpc_client2 = FedModelEvaluatorClient(client_id=tf_fedcluster.partition_idx_host_catalog[1], evaluator_host_port=EVALUATOR_HOST_PORT)
	learner_model = [x.value for x in federated_variables]
	evaluation_score = grpc_client2.request_model_evaluation(model_variables=learner_model, is_community_model=False)
	print(evaluation_score)
	print(grpc_client2.request_is_federation_loss_threshold_reached_for_all_learners())
	print(grpc_client2.retrieve_evaluator_metadata())
	grpc_client2.shutdown()

	evaluator.stop()


