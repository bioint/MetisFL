import gc
from metisfl.utils.formatting import DictionaryFormatter
from metisfl.utils.proto_messages_factory import MetisProtoMessages

import torch

from metisfl.models.torch.helper import construct_dataset_pipeline
from metisfl.models.model_dataset import ModelDataset
from metisfl.models.model_ops import LearningTaskStats, ModelOps
from metisfl.models.torch.torch_model import MetisModelTorch
from metisfl.proto import metis_pb2
from metisfl.utils.metis_logger import MetisLogger
from metisfl.models.utils import get_num_of_epochs

class TorchModelOps(ModelOps):
    
    def __init__(self, model_dir: str):
        self._metis_model = MetisModelTorch.load(model_dir)

    def train_model(self,
                    train_dataset: ModelDataset,
                    learning_task_pb: metis_pb2.LearningTask,
                    hyperparameters_pb: metis_pb2.Hyperparameters):
        if not train_dataset:
            raise RuntimeError("Provided `dataset` for training is None.")
        MetisLogger.info("Starting model training.")

        total_steps = learning_task_pb.num_local_updates
        batch_size = hyperparameters_pb.batch_size
        dataset_size = train_dataset.get_size()
        epochs_num = get_num_of_epochs(total_steps, dataset_size, batch_size)
        dataset = construct_dataset_pipeline(train_dataset) #TODO: this is inconsistent with tf counterpart
        
        self._metis_model._backend_model.train() # set to training mode
        train_res = self._metis_model.fit(dataset, epochs=epochs_num)
        
        # TODO (dstripelis) Need to add the metrics for computing the execution time
        model_weights_descriptor = self.get_model_weights()
        learning_task_stats = LearningTaskStats(
            train_stats=train_res,
            completed_epochs=epochs_num,
            global_iteration=learning_task_pb.global_iteration)
        MetisLogger.info("Model training is complete.")
        return model_weights_descriptor, learning_task_stats

    def evaluate_model(self, eval_dataset: ModelDataset):
        if not eval_dataset:
            raise RuntimeError("Provided `dataset` for evaluation is None.")
        MetisLogger.info("Starting model evaluation.")
        dataset = construct_dataset_pipeline(eval_dataset)
        self._metis_model._backend_model.eval() # set to evaluation mode
        eval_res = self._metis_model.evaluate(dataset)            
        MetisLogger.info("Model evaluation is complete.")
        metric_values = DictionaryFormatter.stringify(eval_res, stringify_nan=True)
        return MetisProtoMessages.construct_model_evaluation_pb(metric_values)
    
    def infer_model(self):
        # Set model to evaluation state.
        # FIXME @panoskyriakis: check this
        self._metis_model._backend_model.eval()
        pass

    # @stripeli do we really need this?
    def cleanup(self):
        del self._metis_model
        torch.cuda.empty_cache()
        gc.collect()
