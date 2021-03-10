import json

from concurrent.futures import ThreadPoolExecutor
from federation.fed_protobuff import model_evaluation_serving_pb2_grpc
from utils.devops.grpc_services import GRPCChannel
from utils.devops.proto_buff_exchange_ops import ProtoBuffExchangeOps
from utils.logging.metis_logger import MetisLogger as metis_logger


class FedModelEvaluatorClient(object):


	def __init__(self, client_id, evaluator_host_port, max_workers=3, *args, **kwargs):
		self.client_id = client_id
		self.channel = GRPCChannel(evaluator_host_port).channel
		self.__stub = model_evaluation_serving_pb2_grpc.EvalServingStub(channel=self.channel)
		self.__thread_executor = ThreadPoolExecutor(max_workers=max_workers,
													thread_name_prefix='PoolExecutorOf_{}'.format(self.client_id))


	def __generate_model_evaluation_request(self, model_variables, is_community_model):
		model_evaluation_request_pb = ProtoBuffExchangeOps.construct_model_evaluation_request(
			learner_id=self.client_id, matrices=model_variables, num_training_examples=0, num_validation_examples=0,
			latest_train_score=0.0, latest_validation_score=0.0, comp_epochs=0.0, comp_batches=0, batch_size=0,
			processing_ms_per_epoch=0.0, target_local_epochs=0, is_community_model=is_community_model)
		yield model_evaluation_request_pb


	def __request_model_evaluation(self, model_variables, is_community_model):
		model_evaluation_request_pb = self.__generate_model_evaluation_request(model_variables=model_variables,
																			   is_community_model=is_community_model)
		double_value = self.__stub.ModelEvaluationOnFederationValidationSets(model_evaluation_request_pb)
		metis_logger.info("Learner: {}, Successfully sent and evaluated its model with the evaluator"
						  .format(self.client_id))
		model_evaluation_score = double_value.value
		return model_evaluation_score


	def request_model_evaluation(self, model_variables, is_community_model, block=True):
		future = self.__thread_executor.submit(self.__request_model_evaluation, model_variables, is_community_model)
		if block:
			model_evaluation_score = future.result()
			return model_evaluation_score


	def retrieve_evaluator_metadata(self):
		learner_pb = ProtoBuffExchangeOps.construct_learner_pb(self.client_id)
		import logging
		import traceback
		try:
			jsonstring_pb = self.__stub.RetrieveValidationMetadataFromEvaluator(learner_pb)
		except Exception as e:
			metis_logger.info("CONTROLLER ERROR KK2: {}".format(e))
			logging.error(traceback.format_exc())

		return json.loads(jsonstring_pb.value) # result should be in json


	def shutdown(self):
		metis_logger.info("Evaluator GRPC Client: {}, is shutting down".format(self.client_id))
		self.__thread_executor.shutdown(wait=False) # Graceful shutdown, wait for threads to join
		self.channel.close()
		metis_logger.info("Evaluator GRPC Client: {}, shut down".format(self.client_id))
