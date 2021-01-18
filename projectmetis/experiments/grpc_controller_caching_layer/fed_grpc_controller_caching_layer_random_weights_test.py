from federation.fed_grpc_controller import FedController
from utils.generic.time_ops import TimeUtil

import federation.fed_grpc_client as fed_grpc_client
import numpy as np
import random
import time
import os

scriptDirectory = os.path.dirname(os.path.realpath(__file__))
FEDCONTROLLER_HOST_PORT = 'bdnf.isi.edu:560051'

NUMBER_OF_UPDATE_REQUESTS = 10
experiments_log_filepath = scriptDirectory + "/../../resources/logs/testing_producibility/{}.jsonl".format("Grpc_Controller_WithCachingLayer_Testing")
fexperiments_writer = open(experiments_log_filepath, 'w+')

if __name__=="__main__":

	for WITH_CACHING_LAYER in [True]:
		for NUMBER_OF_WEIGHTS in [10, 25, 50, 100, 150]:
			for NUMBER_OF_CLIENTS in [10, 25, 50, 100, 250, 500, 750, 1000]:

				participating_hosts = ['Driver{}'.format(idx) for idx in range(1, NUMBER_OF_CLIENTS + 1)]
				synchronous_execution = not WITH_CACHING_LAYER # Synchronous execution does not employ any caching mechanism
				fed_controller = FedController(grpc_servicer_host_port=FEDCONTROLLER_HOST_PORT, participating_hosts_ids=participating_hosts,
											   max_workers=1, synchronous_execution=synchronous_execution)
				fed_controller.start()
				time.sleep(5) # Wait till server is up and running

				primary_channel = fed_grpc_client.GRPCChannel(host_port=FEDCONTROLLER_HOST_PORT).channel

				img1 = np.ones((50, 50, 3), dtype=np.float64)

				# INITIALIZE SERVER COMMUNITY COLLECTIONS
				for learner_id in range(1, NUMBER_OF_CLIENTS + 1):
					# TODO Careful, the background listening threads of the grpc clients saturate server's open connections
					learner = fed_grpc_client.FedClient(client_id='Driver{}'.format(learner_id), channel=primary_channel, listen_for_federation_round_signals=False)
					print(learner.client_id)
					learner.update_completed_epochs(learner_id*100)
					learner.update_num_training_examples(learner_id * 200)

					trained_variables = list()
					for wid in range(1, NUMBER_OF_WEIGHTS):
						trained_variables.append(img1 * np.random.normal())

					learner.update_trained_variables(trained_variables)
					learner.send_model_local_trained_variables_to_controller(block=True)
					learner.stop()

				# REQUEST # COMMUNITY UPDATES WITH NEW DATA
				comm_reqs_exec_time = list()
				for request_id in range(NUMBER_OF_UPDATE_REQUESTS):
					learner_id = random.randint(0, NUMBER_OF_CLIENTS - 1)
					learner = fed_grpc_client.FedClient(client_id='Driver{}'.format(learner_id), channel=primary_channel)
					learner.update_completed_epochs(learner_id*100)
					learner.update_num_training_examples(learner_id * 200)
					trained_variables = list()
					for wid in range(1, NUMBER_OF_WEIGHTS):
						trained_variables.append(img1 * np.random.normal())
					learner.update_trained_variables(trained_variables)
					learner.send_model_local_trained_variables_to_controller(block=True)

					req_start_time_in_ms = TimeUtil.current_milli_time()
					l_res = learner.request_current_community(send_learner_state=False, block=True)
					req_end_time_in_ms = TimeUtil.current_milli_time()
					comm_reqs_exec_time.append(TimeUtil.delta_diff_in_ms(req_start_time_in_ms, req_end_time_in_ms))

					# time.sleep(0.1) # ensure we send requests to the server with some delay
					learner.stop()

				AVG_RESPONSE_TIME = sum(comm_reqs_exec_time[1:]) / (len(comm_reqs_exec_time) - 1)  # excluding the first computation which is the materialization of the cache
				print(AVG_RESPONSE_TIME)
				if WITH_CACHING_LAYER:
					json_output = """{"WITH_CACHING_LAYER": "%s" ,"NUMBER_OF_CLIENTS": %s, "RESNET_NUMBER_OF_LAYERS": %s, "NUMBER_OF_UPDATE_REQUESTS": %s, "AVG_RESPONSE_TIME": %s, "RESPONSE_TIMES": %s}\n""" % (
						True,
						NUMBER_OF_CLIENTS,
						NUMBER_OF_WEIGHTS,
						NUMBER_OF_UPDATE_REQUESTS,
						AVG_RESPONSE_TIME,
						comm_reqs_exec_time)
					print(json_output)
				else:
					json_output = """{"WITH_CACHING_LAYER": "%s" ,"NUMBER_OF_CLIENTS": %s, "RESNET_NUMBER_OF_LAYERS": %s, "NUMBER_OF_UPDATE_REQUESTS": %s, "AVG_RESPONSE_TIME": %s, "RESPONSE_TIMES": %s}\n""" % (
						False,
						NUMBER_OF_CLIENTS,
						NUMBER_OF_WEIGHTS,
						NUMBER_OF_UPDATE_REQUESTS,
						AVG_RESPONSE_TIME,
						comm_reqs_exec_time)
					print(json_output)
				fexperiments_writer.write(json_output)

				fed_controller.stop()
				time.sleep(5)  # ensure there is some delay between controller shutdowns

fexperiments_writer.close()
