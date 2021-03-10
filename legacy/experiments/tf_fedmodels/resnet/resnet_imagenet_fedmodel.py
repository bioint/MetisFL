import federation.fed_model as fedmodel
import experiments.tfmodels.resnet.resnet_model as resnet_model
import tensorflow as tf

class ResNetImagenetFedModel(fedmodel.FedModelDef):

	def __init__(self):
		pass

	def input_tensors_datatype(self):
		pass

	def output_tensors_datatype(self):
		pass

	def model_architecture(self, input_tensors, output_tensors, global_step, batch_size, dataset_size, **kwargs):
		pass