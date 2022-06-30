import math

from projectmetis.python.utils.formatting import DictionaryFormatter
from projectmetis.python.utils.proto_messages_factory import MetisProtoMessages, ModelProtoMessages


class KerasProtoFactory:

    class CompletedLearningTaskProtoMessage(object):

        def __init__(self,
                     model,
                     train_stats,
                     completed_epochs,
                     global_iteration,
                     validation_stats=None,
                     test_stats=None,
                     completes_batches=0,
                     batch_size=0,
                     processing_ms_per_epoch=0.0,
                     processing_ms_per_batch=0.0,
                     *args, **kwargs):

            self._model = model
            self._train_stats = train_stats
            self._completed_epochs = completed_epochs
            self._global_iteration = global_iteration
            self._validation_stats = DictionaryFormatter.listify_values(validation_stats) \
                if validation_stats else dict()
            self._test_stats = DictionaryFormatter.listify_values(test_stats) \
                if test_stats else dict()
            self._completes_batches = completes_batches
            self._batch_size = batch_size
            self._processing_ms_per_epoch = processing_ms_per_epoch
            self._processing_ms_per_batch = processing_ms_per_batch

        def _construct_task_evaluation_pb(self, collection):
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
            _completed_epochs = int(math.ceil(self._completed_epochs))
            values_len = [len(v) for v in collection.values()]
            is_model_evaluated_at_every_epoch = all([_completed_epochs == length for length in values_len])
            epoch_evaluations_pb = []
            if is_model_evaluated_at_every_epoch:
                # Loop over the evaluation for each epoch.
                for e_idx in range(0, _completed_epochs):
                    epoch_evaluation_stats = dict()
                    for k, v in collection.items():
                        epoch_evaluation_stats[k] = v[e_idx]
                    epoch_evaluation_stats = \
                        DictionaryFormatter.stringify(epoch_evaluation_stats)
                    model_evaluation_pb = MetisProtoMessages \
                        .construct_model_evaluation_pb(metric_values=epoch_evaluation_stats)
                    # Need to store the actual epoch id hence the +1.
                    epoch_evaluations_pb.append(
                        MetisProtoMessages.construct_epoch_evaluation_pb(
                            epoch_id=e_idx + 1, model_evaluation_pb=model_evaluation_pb))
            else:
                # Grab the results of the last evaluation.
                epoch_evaluation_stats = {k: v[-1] for k, v in collection.items()}
                epoch_evaluation_stats = DictionaryFormatter.stringify(epoch_evaluation_stats)
                model_evaluation_pb = MetisProtoMessages \
                    .construct_model_evaluation_pb(metric_values=epoch_evaluation_stats)
                # Need to store the index/value of the last epoch.
                epoch_evaluations_pb.append(
                    MetisProtoMessages.construct_epoch_evaluation_pb(
                        epoch_id=_completed_epochs, model_evaluation_pb=model_evaluation_pb))
            return epoch_evaluations_pb

        def construct_task_execution_metadata_pb(self):
            epoch_training_evaluations_pbs = \
                self._construct_task_evaluation_pb(collection=self._train_stats)
            epoch_validation_evaluations_pbs = \
                self._construct_task_evaluation_pb(collection=self._validation_stats)
            epoch_test_evaluations_pbs = \
                self._construct_task_evaluation_pb(collection=self._test_stats)
            task_evaluation_pb = \
                MetisProtoMessages.construct_task_evaluation_pb(
                    epoch_training_evaluations_pbs=epoch_training_evaluations_pbs,
                    epoch_validation_evaluations_pbs=epoch_validation_evaluations_pbs,
                    epoch_test_evaluations_pbs=epoch_test_evaluations_pbs)
            task_execution_pb = MetisProtoMessages.construct_task_execution_metadata_pb(
                self._global_iteration, task_evaluation_pb, self._completed_epochs,
                self._completes_batches, self._batch_size, self._processing_ms_per_epoch,
                self._processing_ms_per_batch)
            return task_execution_pb

        def construct_completed_learning_task_pb(self, aux_metadata="", encryption_scheme=None):
            model_vars = []
            trainable_vars_names = [v.name for v in self._model.trainable_variables]
            for w in self._model.weights:
                is_weight_trainable = True if w.name in trainable_vars_names else False
                ciphertext = None
                if encryption_scheme is not None:
                    ciphertext = encryption_scheme.encrypt(w.numpy().flatten(), 1)
                tensor_pb = ModelProtoMessages.construct_tensor_pb_from_nparray(w.numpy(), ciphertext=ciphertext)
                model_var = ModelProtoMessages.construct_model_variable_pb(name=w.name,
                                                                           trainable=is_weight_trainable,
                                                                           tensor_pb=tensor_pb)
                model_vars.append(model_var)
            model_pb = ModelProtoMessages.construct_model_pb(model_vars)
            task_execution_meta_pb = self.construct_task_execution_metadata_pb()
            completed_learning_task_pb = MetisProtoMessages.construct_completed_learning_task_pb(
                model_pb=model_pb, task_execution_metadata_pb=task_execution_meta_pb, aux_metadata=aux_metadata)
            return completed_learning_task_pb

    class ModelEvaluationProtoMessage(object):

        def __init__(self, metric_values):
            self.metric_values = metric_values

        def construct_model_evaluation_pb(self):
            metric_values = DictionaryFormatter.stringify(self.metric_values)
            return MetisProtoMessages.construct_model_evaluation_pb(metric_values)
