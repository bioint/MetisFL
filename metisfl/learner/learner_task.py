from .learner import Learner
from ..encryption.homomorphic import HomomorphicEncryption
from ..proto import metis_pb2, model_pb2


class LearnerTask(object):

    def __init__(
        self,
        learner: Learner,
        he_batch_size: int,
        he_scaling_factor_bits: int
    ):
        self._learner = learner
        self._homomorphic_encryption = HomomorphicEncryption(
            batch_size=he_batch_size,
            scaling_factor_bits=he_scaling_factor_bits
        )

    def evaluate(
            self,
            model: model_pb2.Model,
            batch_size: int,
    ) -> metis_pb2.ModelEvaluations:
        self._set_weights(model)
        self._learner.evaluate()
        pass

    def train(
        self,
        model: model_pb2.Model,
        tark_params: metis_pb2.TaskParams,
    ) -> metis_pb2.CompletedLearningTask:
        self._set_weights(model)
        pass

    def _set_weights(self, model: model_pb2.Model):
        weights_descriptor = self._homomorphic_encryption.decrypt(
            model.variables)
        self._learner.set_weights(weights_descriptor)
