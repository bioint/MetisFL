from federation.fed_grpc_controller import FedController

import federation.fed_grpc_controller_client as fed_grpc_client
import numpy as np

NUMBER_OF_CLIENTS = 3
NUMBER_OF_WEIGHTS = 5
FEDCONTROLLER_HOST_PORT = 'bdnf.isi.edu:60051'

if __name__=="__main__":

	participating_hosts = ['Driver{}'.format(idx) for idx in range(1, NUMBER_OF_CLIENTS + 1)]
	fed_controller = FedController(host_port=FEDCONTROLLER_HOST_PORT, participating_hosts_ids=participating_hosts, max_workers=10)
	fed_controller.start()

	primary_channel = fed_grpc_client.GRPCChannel(host_port=FEDCONTROLLER_HOST_PORT).channel

	img1 = np.ones((50, 50, 3), dtype=np.float64)

	# INITIALIZE SERVER COMMUNITY COLLECTIONS
	for learner_id in range(1, NUMBER_OF_CLIENTS + 1):
		# TODO Careful, the background listening threads of the grpc clients saturate server's open connections
		learner = fed_grpc_client.FedClient(client_id='Driver{}'.format(learner_id),
											channel=primary_channel,
											controller_host_port=FEDCONTROLLER_HOST_PORT,
											listen_for_federation_round_signals=True)
		# print(learner.client_id)
		learner.update_completed_batches(50)
		learner.update_completed_epochs(learner_id*100)
		learner.update_num_training_examples(1)
		learner.request_eligibility_for_community_request_based_on_staleness(validation_phase_stalenesses=[0, 2, 3],
																			 current_update_steps=None)

		trained_variables = list()
		for wid in range(1, NUMBER_OF_WEIGHTS):
			trained_variables.append(img1 * np.random.normal())

		learner.update_trained_variables(trained_variables)
		learner.send_model_local_trained_variables_to_controller(block=True)
		learner.request_current_community(send_learner_state=True, block=True)
		learner.shutdown()

	# REQUEST A COMMUNITY UPDATE WITH NEW DATA
	comm_reqs_exec_time = list()
	for learner_id in range(1, NUMBER_OF_CLIENTS + 1):
		learner = fed_grpc_client.FedClient(client_id='Driver{}'.format(learner_id),
											channel=primary_channel,
											controller_host_port=FEDCONTROLLER_HOST_PORT,
											listen_for_federation_round_signals=True)
		print(learner.client_id)
		learner.update_completed_epochs(learner_id*100)
		learner.update_num_training_examples(learner_id * 200)
		learner.request_eligibility_for_community_request_based_on_staleness(validation_phase_stalenesses=[0],
																			 current_update_steps=50)

		# trained_variables = list()
		# for wid in range(1, RESNET_NUMBER_OF_LAYERS):
		# 	trained_variables.append(img1 * np.random.normal())
		#
		# learner.update_trained_variables(trained_variables)
		# req_start_time_in_ms = TimeUtil.current_milli_time()
		# l_res = learner.request_current_community(send_learner_state=True, block=False)
		# req_end_time_in_ms = TimeUtil.current_milli_time()
		# comm_reqs_exec_time.append(TimeUtil.delta_diff_in_ms(req_start_time_in_ms, req_end_time_in_ms))
		# # time.sleep(0.1) # ensure we send requests to the server with some delay
		# print(learner.request_community_and_learner_global_community_steps())
		learner.shutdown()

	# print(comm_reqs_exec_time)
	# print(sum(comm_reqs_exec_time) / len(comm_reqs_exec_time))