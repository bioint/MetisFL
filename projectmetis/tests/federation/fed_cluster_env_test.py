from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import federation.fed_cluster_env as fed_cluster_env
# from tfcodetemplates.mnistdb_driver import MnistDBSession
# import federation.fed_execution as fedexec
# import federation.fed_model as fedmodel
# import tensorflow as tf
# import glob
import os

scriptDirectory = os.path.dirname(os.path.realpath(__file__))

if __name__=="__main__":

	# Cluster Settings
	SECURE_CLUSTER_SETTINGS_FILEPATH = '../../resources/config/experiments_cluster_config/cifar10/asynchronous/cifar10.cnn2.federation.10Learners.5Fast_atBDNF_5Slow_atLEARN.AsyncDVWInvertedLoss.run2.yaml'
	# CLUSTER_SETTINGS_FILE_TEMPLATE = '../../resources/config/tensorflow.federation.execution.configs.secure.bdnf.isi.edu.10hosts.yaml'
	secure_cluster_settings_file = os.path.join(scriptDirectory, SECURE_CLUSTER_SETTINGS_FILEPATH)
	tf_fedcluster = fed_cluster_env.FedEnvironment.tf_federated_cluster_from_yaml(cluster_specs_file=secure_cluster_settings_file, init_cluster=True)
	fed_cluster_env.FedEnvironment.shutdown_yaml_created_tf_federated_cluster(tf_fedcluster)
	# conn = RemoteTFServerUtils.get_remote_server_connection("bdnf.isi.edu", "stripeli")
	# conn.run("kill -9 $(cat /tmp/metis_project/tf_servers_pids/worker5223.pid)")
	# conn.run("kill -9 $(cat /tmp/metis_project/tf_servers_pids/ps5222.pid)")
	# conn.run("kill -9 $(cat /tmp/metis_project/tf_servers_pids/ps9898.pid)")