from experiments.tf_fedmodels.resnet.resnet_cifar_fedmodel import ResNetCifarFedModel
from federation.fed_grpc_controller import FedController
from federation import fed_grpc_controller_client
from federation import fed_model as fedmodel
from utils.generic.time_ops import TimeUtil

import numpy as np
import random
import time
import ast
import os

scriptDirectory = os.path.dirname(os.path.realpath(__file__))
FEDCONTROLLER_HOST_PORT = 'bdnf.isi.edu:560051'
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

WITH_CACHING_LAYER = ast.literal_eval(os.environ['WITH_CACHING_LAYER'])
NUMBER_OF_UPDATE_REQUESTS = int(os.environ['NUMBER_OF_UPDATE_REQUESTS'])
RESNET_NUMBER_OF_LAYERS = int(os.environ['RESNET_NUMBER_OF_LAYERS'])
NUMBER_OF_CLIENTS = int(os.environ['NUMBER_OF_CLIENTS'])

if __name__=="__main__":

	experiments_log_filepath = scriptDirectory + "/../../resources/logs/grpc_controller_caching_layer/{}.jsonl".format("Grpc_Controller_CachingLayer_ResNet_Cifar_Testing")
	fexperiments_writer = open(experiments_log_filepath, 'a')

	nnmodel = ResNetCifarFedModel(num_classes=10,
								  learning_rate=0.05,
								  momentum=0.7,
								  resnet_size=RESNET_NUMBER_OF_LAYERS,
								  run_with_distorted_images=True)
	federated_variables = fedmodel.FedModelDef.construct_model_federated_variables(nnmodel)
	network_size = 0
	for fed_var in federated_variables:
		fedarray_mb = np.divide(np.divide(fed_var.value.nbytes, 1024), 1024)  # MBs
		network_size += fedarray_mb
	network_size = "{0:.2f}".format(network_size)

	participating_hosts = ['Driver{}'.format(idx) for idx in range(1, NUMBER_OF_CLIENTS + 1)]
	synchronous_execution = not WITH_CACHING_LAYER # Synchronous execution does not employ any caching mechanism
	fed_controller = FedController(grpc_servicer_host_port=FEDCONTROLLER_HOST_PORT,
								   participating_hosts_ids=participating_hosts,
								   synchronous_execution=synchronous_execution,
								   max_workers=5)
	fed_controller.start()
	time.sleep(5) # Wait till server is up and running

	primary_channel = fed_grpc_controller_client.GRPCChannel(host_port=FEDCONTROLLER_HOST_PORT).channel

	# INITIALIZE SERVER COMMUNITY COLLECTIONS.
	learners_controller_grpc_clients = list()
	for learner_id in range(0, NUMBER_OF_CLIENTS):
		# TODO Careful, the background listening threads of the grpc clients saturate server's open connections
		#  Therefore, we start one driver per client, register its pseudo trained weights and then shut it down.
		learner_grpc_client = fed_grpc_controller_client.FedClient(client_id='Driver{}'.format(learner_id),
																   controller_host_port=FEDCONTROLLER_HOST_PORT,
																   listen_for_federation_round_signals=False,
																   thread_pool_workers=1)
		print("Initialized GRPC Client: {}".format(learner_grpc_client.client_id))
		learner_grpc_client.update_completed_epochs((learner_id+1) * 100)
		learner_grpc_client.update_num_training_examples((learner_id+1) * 200)
		learner_grpc_client.update_latest_validation_score(0.0)

		learner_trained_variables = []
		for fed_var in federated_variables:
			learner_trained_variables.append(fed_var.value)
		learner_grpc_client.update_trained_variables(learner_trained_variables)
		learner_grpc_client.send_model_local_trained_variables_to_controller(block=True)
		learners_controller_grpc_clients.append(learner_id)
		learner_grpc_client.shutdown(wait=True)

	# REQUEST COMMUNITY UPDATES WITH NEW DATA
	comm_reqs_exec_time = list()
	for request_id in range(NUMBER_OF_UPDATE_REQUESTS):
		random_idx = random.randint(0, NUMBER_OF_CLIENTS-1) # randint is inclusive
		learner_id = learners_controller_grpc_clients[random_idx]
		# TODO Since we have already registered the drivers with the controller we create the driver again.
		learner_grpc_client = fed_grpc_controller_client.FedClient(client_id='Driver{}'.format(learner_id),
																   controller_host_port=FEDCONTROLLER_HOST_PORT,
																   listen_for_federation_round_signals=False,
																   thread_pool_workers=1)
		print("Request ID: {}, Requesting Client: {}".format(request_id, learner_id))
		req_start_time_in_ms = TimeUtil.current_milli_time()
		l_res = learner_grpc_client.request_current_community(send_learner_state=False, block=True)
		req_end_time_in_ms = TimeUtil.current_milli_time()
		comm_reqs_exec_time.append(TimeUtil.delta_diff_in_ms(req_start_time_in_ms, req_end_time_in_ms))
		learner_grpc_client.shutdown(wait=True)

	fed_controller.stop()

	comm_reqs_exec_time = comm_reqs_exec_time[1:] # exclude the first request since it is solely used to materialize the cache
	AVG_RESPONSE_TIME = sum(comm_reqs_exec_time) / (len(comm_reqs_exec_time))
	if WITH_CACHING_LAYER:
		json_output = """{"WITH_CACHING_LAYER": "%s" ,"NUMBER_OF_CLIENTS": %s, "RESNET_NUMBER_OF_LAYERS": %s, "NETWORK_SIZE_IN_MB": %s, "NUMBER_OF_UPDATE_REQUESTS": %s, "AVG_RESPONSE_TIME": %s, "RESPONSE_TIMES": %s}\n""" % (
			True,
			NUMBER_OF_CLIENTS,
			RESNET_NUMBER_OF_LAYERS,
			network_size,
			len(comm_reqs_exec_time),
			AVG_RESPONSE_TIME,
			comm_reqs_exec_time)
		print(json_output)
	else:
		json_output = """{"WITH_CACHING_LAYER": "%s" ,"NUMBER_OF_CLIENTS": %s, "RESNET_NUMBER_OF_LAYERS": %s, "NETWORK_SIZE_IN_MB": %s, "NUMBER_OF_UPDATE_REQUESTS": %s, "AVG_RESPONSE_TIME": %s, "RESPONSE_TIMES": %s}\n""" % (
			False,
			NUMBER_OF_CLIENTS,
			RESNET_NUMBER_OF_LAYERS,
			network_size,
			len(comm_reqs_exec_time),
			AVG_RESPONSE_TIME,
			comm_reqs_exec_time)
		print(json_output)
	fexperiments_writer.write(json_output)
	fexperiments_writer.close()
