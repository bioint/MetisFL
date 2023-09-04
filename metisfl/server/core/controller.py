
from typing import List
from metisfl.proto import model_pb2, controller_pb2
from metisfl.server.core.learner_manager import LearnerManager
from metisfl.server.core.model_manager import ModelManager
from metisfl.server.scheduling.scheduler import Scheduler
from metisfl.server.selection.scheduled_cardinality import ScheduledCardinality


class Controller:

    learner_manager: LearnerManager = None
    model_manager: ModelManager = None
    selector: ScheduledCardinality = None
    scheduler: Scheduler = None

    def __init__(
        self,
        learner_manager: LearnerManager,
        model_manager: ModelManager,
        selector: ScheduledCardinality,
        scheduler: Scheduler
    ) -> None:
        """Initializes the controller.

        Parameters
        ----------
        learner_manager : LearnerManager
            The learner manager to be used.
        model_manager : ModelManager
            The model manager to be used.
        selector : ScheduledCardinality
            The selector to be used.
        """
        self.learner_manager = learner_manager
        self.model_manager = model_manager
        self.selector = selector
        self.scheduler = scheduler

    def add_learner(self, learner: controller_pb2.Learner) -> str:
        """Adds a learner to the controller.

        Parameters
        ----------
        learner : controller_pb2.Learner
            The learner to be added.
        """
        return self.learner_manager.add_learner(learner=learner)

    def remove_learner(self, learner_id: str) -> None:
        """Removes a learner from the controller.

        Parameters
        ----------
        learner_id : str
            The learner id.
        """
        self.learner_manager.remove_learner(learner_id=learner_id)

    def set_initial_model(self, model: model_pb2.Model) -> None:
        """Sets the initial model for the controller.

        Parameters
        ----------
        model : model_pb2.Model
            The initial model.
        """
        self.model_manager.set_initial_model(model=model)

    def start_training(self) -> None:
        """Starts the training process."""

        learner_ids = self.learner_manager.get_learner_ids()
        for learner_id in learner_ids:
            to_schedule = self.scheduler.schedule(learner_id=learner_id)

            if len(to_schedule) == 0:
                continue

            self.learner_manager.schedule_train(
                learner_ids=to_schedule,
                model=self.model_manager.get_model(),
            )

    def train_done(self, request: controller_pb2.TrainDoneRequest) -> None:
        """Notifies the controller that the training is done.

        Parameters
        ----------
        request : controller_pb2.TrainDoneRequest
            The request containing the learner id and the model.
        """

        task = request.task
        learner_id = self.learner_manager.get_learner_id(task=task)
        model = request.model
        train_results = request.train_result

        self.model_manager.insert_model(
            learner_id=learner_id,
            model=model,
        )

        self.learner_manager.schedule_evaluate(
            learner_ids=[learner_id],
            model=self.model_manager.get_model(),
        )

        self.learner_manager.update_train_result(
            task=task,
            learner_id=learner_id,
            train_results=train_results,
        )

        learner_ids = self.learner_manager.get_learner_ids()
        to_schedule = self.scheduler.schedule(learner_id=learner_id)

        if len(to_schedule) == 0:
            return

        self.model_manager.update_model(
            to_schedule=to_schedule,
            learner_ids=learner_ids,
        )

        self.learner_manager.schedule_train(
            learner_ids=to_schedule,
            model=self.model_manager.get_model(),
        )
