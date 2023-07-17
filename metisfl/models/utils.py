import math

import numpy as np

from metisfl import config
from metisfl.encryption.encryption import Encryption
from metisfl.models.types import LearningTaskStats, ModelWeightsDescriptor
from metisfl.proto import metis_pb2, model_pb2
from metisfl.utils.formatting import DictionaryFormatter
from metisfl.proto.proto_messages_factory import MetisProtoMessages, ModelProtoMessages

from .model_ops import ModelOps

def calc_mean_wall_clock(wall_clock):
    return np.mean(wall_clock) * 1000

def construct_model_pb(weights: ModelWeightsDescriptor, 
                       encryption_scheme: metis_pb2.EncryptionScheme = None):
    
    if encryption_scheme:
        return Encryption.from_proto(encryption_scheme).encrypt_model(weights)
    else:
        return _construct_model_pb(weights)

def _construct_model_pb(weights: ModelWeightsDescriptor):
    weights_names = weights.weights_names
    weights_trainable = weights.weights_trainable
    weights_values = weights.weights_values
    variables_pb = []
    for w_n, w_t, w_v in zip(weights_names, weights_trainable, weights_values):
        # If we have a ciphertext we prioritize it over the plaintext.
        tensor_pb = ModelProtoMessages.construct_tensor_pb(nparray=w_v)
        model_var = ModelProtoMessages.construct_model_variable_pb(name=w_n,
                                                                   trainable=w_t,
                                                                   tensor_pb=tensor_pb)
        variables_pb.append(model_var)
    
    model_pb = model_pb2.Model(variables=variables_pb)
    return model_pb

def get_weights_from_model_pb(model_pb: model_pb2.Model,
                              encryption_scheme: metis_pb2.EncryptionScheme = None) -> ModelWeightsDescriptor:
    if encryption_scheme:
        return Encryption.from_proto(encryption_scheme).decrypt_model(model_pb)
    else:
        return _get_weights_from_model_pb(model_pb)

def _get_weights_from_model_pb(model_pb: model_pb2.Model) -> ModelWeightsDescriptor:
    variables = model_pb.variables
    var_names, var_trainables, var_nps = list(), list(), list()
    for var in variables:
        # Variable specifications.
        var_name = var.name
        var_trainable = var.trainable   
        tensor_spec = var.plaintext_tensor.tensor_spec
        # If the tensor is a plaintext tensor, then we need to read the byte buffer
        # and load the tensor as a numpy array casting it to the specified data type.
        np_array = ModelProtoMessages.TensorSpecProto.proto_tensor_spec_to_numpy_array(
            tensor_spec)

        # Append variable specifications to model's variable list.
        var_names.append(var_name)
        var_trainables.append(var_trainable)
        var_nps.append(np_array)

    return ModelWeightsDescriptor(weights_names=var_names,
                                  weights_trainable=var_trainables,
                                  weights_values=var_nps)

def get_completed_learning_task_pb(model_pb: model_pb2.Model,
                                   learning_task_stats: LearningTaskStats,
                                   aux_metadata=None):
    task_execution_meta_pb = _construct_task_execution_metadata_pb(
        learning_task_stats)
    completed_learning_task_pb = metis_pb2.CompletedLearningTask(model=model_pb,
                                                                 execution_metadata=task_execution_meta_pb,
                                                                 aux_metadata=aux_metadata)
    return completed_learning_task_pb

def get_model_ops_fn(nn_engine) -> ModelOps:
    if nn_engine == config.KERAS_NN_ENGINE:
        from metisfl.models.keras.keras_model_ops import KerasModelOps
        return KerasModelOps
    elif nn_engine == config.PYTORCH_NN_ENGINE:
        from metisfl.models.torch.torch_model_ops import TorchModelOps
        return TorchModelOps
    else:
        raise ValueError("Unknown neural engine: {}".format(nn_engine))

def get_num_of_epochs(dataset_size: int, batch_size: int, total_steps: int) -> int:
    steps_per_epoch = np.ceil(np.divide(dataset_size, batch_size))
    epochs_num = 1
    if total_steps > steps_per_epoch:
        epochs_num = int(np.ceil(np.divide(total_steps, steps_per_epoch)))
    return epochs_num

def _construct_task_evaluation_pb(collection, completed_epochs):
    """
    Input collection follows the following syntax:
    {'loss': [0.3419254422187805, 0.33265018463134766],
    'accuracy': [0.8773148059844971, 0.8805555701255798]}

    Since we might have evaluated the last model and not the model trained at every epoch,
    we construct the epoch evaluation protos starting from the last epoch, else we collect
    the evaluations across all epochs. A metric value is computed at every evaluation step.
    For instance, if we evaluate a model and its evaluation metrics are loss and accuracy,
    then the recorded evaluation result will contain one value for the loss and another
    for the accuracy. By convention, if the number of recorded evaluation values are not
    equal to the number of completed epochs, then we return the value of the last evaluation
    else we return one value for each completed epoch.

    :param collection:
    :return:
    """
    _completed_epochs = int(math.ceil(completed_epochs))
    values_len = [len(v) for v in collection.values()]
    is_model_evaluated_at_every_epoch = all(
        [_completed_epochs == length for length in values_len])
    epoch_evaluations_pb = []
    if is_model_evaluated_at_every_epoch:
        # Loop over the evaluation for each epoch.
        for e_idx in range(0, _completed_epochs):
            epoch_evaluation_stats = dict()
            for k, v in collection.items():
                epoch_evaluation_stats[k] = v[e_idx]
            epoch_evaluation_stats = DictionaryFormatter\
                .stringify(epoch_evaluation_stats, stringify_nan=True)
            model_evaluation_pb = MetisProtoMessages \
                .construct_model_evaluation_pb(metric_values=epoch_evaluation_stats)
            # Need to store the actual epoch id hence the +1.
            epoch_evaluations_pb.append(
                MetisProtoMessages.construct_epoch_evaluation_pb(
                    epoch_id=e_idx + 1, model_evaluation_pb=model_evaluation_pb))
    else:
        # Grab the results of the last evaluation.
        epoch_evaluation_stats = {k: v[-1] for k, v in collection.items()}
        epoch_evaluation_stats = DictionaryFormatter\
            .stringify(epoch_evaluation_stats, stringify_nan=True)
        model_evaluation_pb = MetisProtoMessages \
            .construct_model_evaluation_pb(metric_values=epoch_evaluation_stats)
        # Need to store the index/value of the last epoch.
        epoch_evaluations_pb.append(
            MetisProtoMessages.construct_epoch_evaluation_pb(
                epoch_id=_completed_epochs, model_evaluation_pb=model_evaluation_pb))
    return epoch_evaluations_pb

def _formater(stats):
    return DictionaryFormatter.listify_values(stats) \
        if stats else dict()

def _construct_task_execution_metadata_pb(learning_task_stats: LearningTaskStats):
    completed_epochs = learning_task_stats.completed_epochs
    epoch_training_evaluations_pbs = \
        _construct_task_evaluation_pb(
            collection=learning_task_stats.train_stats, completed_epochs=completed_epochs)
    epoch_validation_evaluations_pbs = \
        _construct_task_evaluation_pb(collection=_formater(learning_task_stats.validation_stats),
                                      completed_epochs=completed_epochs)
    epoch_test_evaluations_pbs = \
        _construct_task_evaluation_pb(collection=_formater(learning_task_stats.test_stats),
                                      completed_epochs=completed_epochs)

    task_evaluation_pb= metis_pb2.TaskEvaluation(training_evaluation=epoch_training_evaluations_pbs,
                                        validation_evaluation=epoch_validation_evaluations_pbs,
                                        test_evaluation=epoch_test_evaluations_pbs)
    
    task_execution_pb = metis_pb2.TaskExecutionMetadata(global_iteration=learning_task_stats.global_iteration,
                                               task_evaluation=task_evaluation_pb,
                                               completed_epochs=learning_task_stats.completed_epochs,
                                               completed_batches=learning_task_stats.completes_batches,
                                               batch_size=learning_task_stats.batch_size,
                                               processing_ms_per_epoch=learning_task_stats.processing_ms_per_epoch,
                                               processing_ms_per_batch=learning_task_stats.processing_ms_per_batch)
    return task_execution_pb
