import abc
import itertools

from projectmetis.python.utils.proto_messages_factory import MetisProtoMessages


class ModelMeta:

    def __init__(self,
                 training_scores=None,
                 training_losses=None,
                 validation_scores=None,
                 validation_losses=None,
                 test_scores=None,
                 test_losses=None,
                 completed_epochs=0.0,
                 completes_batches=0,
                 batch_size=0,
                 processing_ms_per_epoch=0.0,
                 processing_ms_per_batch=0.0,
                 *args, **kwargs):

        self._training_scores = training_scores
        if self._training_scores is None:
            self._training_scores = list()
        self._training_losses = training_losses
        if self._training_losses is None:
            self._training_losses = list()
        self._validation_scores = validation_scores
        if self._validation_scores is None:
            self._validation_scores = list()
        self._validation_losses = validation_losses
        if self._validation_losses is None:
            self._validation_losses = list()
        self._test_losses = test_losses
        if self._test_losses is None:
            self._test_losses = list()
        self._test_scores = test_scores
        if self._test_scores is None:
            self._test_scores = list()
        self._completed_epochs = completed_epochs
        self._completes_batches = completes_batches
        self._batch_size = batch_size
        self._processing_ms_per_epoch = processing_ms_per_epoch
        self._processing_ms_per_batch = processing_ms_per_batch

    def _construct_evaluation_collection_pbs(self, scores, losses):
        epoch_training_evaluations_pbs = []
        scores_losses = list(itertools.zip_longest(scores, losses, fillvalue=0.0))
        for idx, val in enumerate(scores_losses, start=1):
            epoch_score, epoch_loss = val[0], val[1]
            epoch_eval_pb = MetisProtoMessages.construct_epoch_evaluation_pb(
                epoch_id=idx, epoch_score=epoch_score, epoch_loss=epoch_loss)
            epoch_training_evaluations_pbs.append(epoch_eval_pb)
        return epoch_training_evaluations_pbs

    def get_task_execution_metadata_pb(self):
        epoch_training_evaluations_pbs = self._construct_evaluation_collection_pbs(
            scores=self._training_scores, losses=self._training_losses)
        epoch_validation_evaluations_pbs = self._construct_evaluation_collection_pbs(
            scores=self._validation_scores, losses=self._validation_losses)
        epoch_test_evaluations_pbs = self._construct_evaluation_collection_pbs(
            scores=self._test_scores, losses=self._test_losses)
        task_execution_pb = MetisProtoMessages.construct_task_execution_metadata_pb(
            epoch_training_evaluations_pbs, epoch_validation_evaluations_pbs, epoch_test_evaluations_pbs,
            self._completed_epochs, self._completes_batches, self._batch_size, self._processing_ms_per_epoch,
            self._processing_ms_per_batch)
        return task_execution_pb


