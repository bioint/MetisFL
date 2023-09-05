
import time
from typing import Dict, List, Tuple
from metisfl.common.types import GlobalTrainConfig
from metisfl.proto import controller_pb2, model_pb2
from metisfl.common.utils import random_id_generator
from metisfl.server.aggregation import Aggregator
from metisfl.server.core import LearnerManager
from metisfl.server.selection.scheduled_cardinality import ScheduledCardinality
from metisfl.server.store import ModelStore
from metisfl.server.scaling import batches_scaling, participants_scaling, dataset_scaling


class ModelManager:

    aggregator: Aggregator = None

    model: model_pb2.Model = None
    model_store: ModelStore = None
    global_train_config: GlobalTrainConfig = None
    metadata: controller_pb2.ModelMetadata = {}
    selector: ScheduledCardinality = None
    learner_manager: LearnerManager = None
    is_initialized: bool = False

    def __init__(
        self,
        global_train_config: GlobalTrainConfig,
        learner_manager: LearnerManager,
        model_store: ModelStore,
    ) -> None:
        """Initializes the model manager.

        Parameters
        ----------
        learner_manager : LearnerManager
            The learner manager to be used.
        global_train_config : GlobalTrainConfig
            The global training configuration.
        model_store_config : ModelStoreConfig
            The model store configuration.
        """
        self.learner_manager = learner_manager
        self.global_train_config = global_train_config
        self.model_store = model_store

    def set_initial_model(self, model: model_pb2.Model) -> None:
        """Sets the initial model for the controller.

        Parameters
        ----------
        model : model_pb2.Model
            The initial model.
        """

        if self.is_initialized:
            raise Exception("Model already initialized")

        self.model = model
        self.is_initialized = True

    def insert_model(self, learner_id: str, model: model_pb2.Model) -> None:
        """Inserts a model into the model manager.

        Parameters
        ----------
        learner_id : str
            The learner id.
        model : model_pb2.Model
            The model to be inserted.
        """
        self.model_store.insert(
            [(learner_id, model)],
        )

    def update_model(
        self,
        to_schedule: List[str],
        learner_ids: List[str],
    ) -> None:
        """Updates the model performing an aggregation step.

        Parameters
        ----------
        to_schedule : List[str]
            The learners to be scheduled.
        learner_ids : List[str]
            The learners ids.
        """
        selected_ids = self.selector.select(to_schedule, learner_ids)
        scaling_factors = self.compute_scaling_factor(learner_ids)
        stride_length = self.get_stride_length(len(learner_ids))

        update_id = self.init_metadata()
        aggregation_start_time = time.time()

        # FIXME: continue this

    def erase_models(self, learner_ids: List[str]) -> None:
        """Erases the models of the learners.

        Parameters
        ----------
        learner_ids : List[str]
            The learners ids.
        """
        self.model_store.erase(learner_ids)

    def get_model(self) -> model_pb2.Model:
        """Gets the model.

        Returns
        -------
        model_pb2.Model
            The model.
        """
        return self.model

    def init_metadata(self) -> str:
        """Initializes the metadata."""

        update_id = str(random_id_generator())
        self.metadata[update_id] = controller_pb2.ModelMetadata()

        return

    def get_stride_length(self, num_learners: int) -> int:
        """Returns the stride length.

        Parameters
        ----------
        num_learners : int
            The number of learners.

        Returns
        -------
        int
            The stride length.
        """

        stride_length = num_learners

        if self.global_train_config.aggregation_rule == "FedStride":
            fed_stride_length = self.global_train_config.stride_length
            if fed_stride_length > 0:
                stride_length = fed_stride_length

        return stride_length

    def get_lineage_length(self, learner_id: str) -> int:
        """Returns the lineage length for the given learner id.

        Parameters
        ----------
        learner_id : str
            The learner id.

        Returns
        -------
        int
            The lineage length for the given learner id.
        """
        pass

    def compute_scaling_factor(
        self,
        learner_ids: List[str],
    ) -> Dict[str, float]:
        """Computes the scaling factor for the given learners.

        Parameters
        ----------
        learner_ids : List[str]
            The learner ids.

        Returns
        -------
        Dict[str, float]
            The scaling factor for each learner.
        """

        scaling_factor = self.global_train_config.scaling_factor

        if scaling_factor == "NumCompletedBatches":
            num_completed_batches = self.learner_manager.get_num_completed_batches(
                learner_ids
            )
            return batches_scaling(num_completed_batches)
        elif scaling_factor == "NumTrainingExamples":
            num_training_examples = self.learner_manager.get_num_training_examples(
                learner_ids
            )
            return dataset_scaling(num_training_examples)
        elif scaling_factor == "NumParticipants":
            return participants_scaling(learner_ids)
        else:
            raise Exception("Invalid scaling factor")

    def get_aggregation_pairs(
        self,
        selected_models: Dict[str, List[model_pb2.Model]],
        scaling_factors: Dict[str, float],
    ) -> List[List[Tuple[model_pb2.Model, float]]]:
        """Returns the aggregation pairs. """

    def aggregate(
        self,
        update_id: str,
        to_aggregate_block: List[List[Tuple[model_pb2.Model, float]]],
    ) -> None:
        """Aggregates the models.

        Parameters
        ----------
        update_id : str
            The update id.
        to_aggregate_block : List[List[Tuple[model_pb2.Model, float]]]
            The models to be aggregated.
        """
        pass

    def record_block_size(self, update_id: str, block_size: int) -> None:
        """Records the block size.

        Parameters
        ----------
        update_id : str
            The update id.
        block_size : int
            The block size.
        """
        pass

    def record_aggregation_time(self, update_id: str, start_time: float) -> None:
        """Records the aggregation time.

        Parameters
        ----------
        update_id : str
            The update id.
        start_time : float
            The start time.
        """
        pass

    def record_model_size(self, update_id: str) -> None:
        """Records the model size.

        Parameters
        ----------
        update_id : str
            The update id.
        """
        pass
