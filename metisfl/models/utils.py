import math

import numpy as np

from metisfl import config
from metisfl.utils.logger import MetisLogger
from metisfl.encryption.encryption_scheme import EncryptionScheme
from metisfl.models.types import LearningTaskStats, ModelWeightsDescriptor
from metisfl.utils.formatting import DataTypeFormatter
from metisfl.proto.proto_messages_factory import MetisProtoMessages, ModelProtoMessages
from metisfl.proto import metis_pb2, model_pb2

from .model_ops import ModelOps

def calc_mean_wall_clock(wall_clock):
    return np.mean(wall_clock) * 1000

def construct_model_pb(
        weights: ModelWeightsDescriptor,
        encryption_scheme_pb: metis_pb2.EncryptionScheme = None) -> model_pb2.Model:
    
    encryption = None if encryption_scheme_pb is None else \
        EncryptionScheme().from_proto(encryption_scheme_pb)
    weights_names = weights.weights_names
    weights_trainable = weights.weights_trainable
    weights_values = weights.weights_values
    variables_pb = []
    for w_n, w_t, w_v in zip(weights_names, weights_trainable, weights_values):        
        ciphertext = None
        if encryption:
            ciphertext = encryption.encrypt_data(w_v)
        tensor_pb = ModelProtoMessages.construct_tensor_pb(
            nparray=w_v, ciphertext=ciphertext)
        model_var = ModelProtoMessages.construct_model_variable_pb(name=w_n,
                                                                   trainable=w_t,
                                                                   tensor_pb=tensor_pb)
        variables_pb.append(model_var)
    
    model_pb = model_pb2.Model(variables=variables_pb)
    return model_pb

def get_weights_from_model_pb(
        model_pb: model_pb2.Model,
        encryption_scheme_pb: metis_pb2.EncryptionScheme = None) -> ModelWeightsDescriptor:
    
    encryption = None if encryption_scheme_pb is None else \
        EncryptionScheme().from_proto(encryption_scheme_pb)
    variables = model_pb.variables
    var_names = [var.name for var in variables]
    var_trainables = [var.trainable for var in variables]
    var_nps = list()
    for var in variables:
        if var.HasField("ciphertext_tensor"):            
            tensor_spec = var.ciphertext_tensor.tensor_spec
            if encryption:
                decoded_value = encryption.decrypt_data(
                    var.ciphertext_tensor.ciphertext, 
                    tensor_spec.length)
                np_array = ModelProtoMessages.TensorSpecProto\
                    .np_array_from_cipherext_tensor_spec(tensor_spec, decoded_value)
            else:
                MetisLogger.fatal("Encryption is not defined.")
        elif var.HasField('plaintext_tensor'):
            tensor_spec = var.plaintext_tensor.tensor_spec
            np_array = ModelProtoMessages.TensorSpecProto\
                .np_array_from_plaintext_tensor_spec(tensor_spec)
        else:
            MetisLogger.fatal("Not a supported tensor type.")
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
        MetisLogger.fatal("Unknown neural engine: {}".format(nn_engine))

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
            epoch_evaluation_stats = DataTypeFormatter\
                .stringify_dict(epoch_evaluation_stats, stringify_nan=True)
            model_evaluation_pb = MetisProtoMessages \
                .construct_model_evaluation_pb(metric_values=epoch_evaluation_stats)
            # Need to store the actual epoch id hence the +1.
            epoch_evaluations_pb.append(
                MetisProtoMessages.construct_epoch_evaluation_pb(
                    epoch_id=e_idx + 1, model_evaluation_pb=model_evaluation_pb))
    else:
        # Grab the results of the last evaluation.
        epoch_evaluation_stats = {k: v[-1] for k, v in collection.items()}
        epoch_evaluation_stats = DataTypeFormatter\
            .stringify_dict(epoch_evaluation_stats, stringify_nan=True)
        model_evaluation_pb = MetisProtoMessages \
            .construct_model_evaluation_pb(metric_values=epoch_evaluation_stats)
        # Need to store the index/value of the last epoch.
        epoch_evaluations_pb.append(
            MetisProtoMessages.construct_epoch_evaluation_pb(
                epoch_id=_completed_epochs, model_evaluation_pb=model_evaluation_pb))
    return epoch_evaluations_pb

def _formater(stats):
    return DataTypeFormatter.listify_dict_values(stats) \
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

    task_evaluation_pb= metis_pb2.TaskEvaluation(
        training_evaluation=epoch_training_evaluations_pbs,
        validation_evaluation=epoch_validation_evaluations_pbs,
        test_evaluation=epoch_test_evaluations_pbs)
    
    task_execution_pb = metis_pb2.TaskExecutionMetadata(
        global_iteration=learning_task_stats.global_iteration,
        task_evaluation=task_evaluation_pb,
        completed_epochs=learning_task_stats.completed_epochs,
        completed_batches=learning_task_stats.completes_batches,
        batch_size=learning_task_stats.batch_size,
        processing_ms_per_epoch=learning_task_stats.processing_ms_per_epoch,
        processing_ms_per_batch=learning_task_stats.processing_ms_per_batch)
    return task_execution_pb
