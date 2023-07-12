import gc
import multiprocessing as mp
import queue
from typing import Callable

from pebble import ProcessPool

from metisfl import config
from metisfl.learner.task_executor import TaskExecutor
from metisfl.proto import metis_pb2


class LearnerExecutor(object):

    def __init__(self, task_executor: TaskExecutor, recreate_queue_task_worker=False):
        self.task_executor = task_executor
        self._init_tasks_pools(recreate_queue_task_worker)

    def _init_tasks_pools(self, recreate_queue_task_worker=False):
        mp_ctx = mp.get_context("spawn")
        max_tasks = 1 if recreate_queue_task_worker else 0
        self.pool = dict()
        for task in config.TASKS:
            self.pool[task] = self._init_task_pool(max_tasks, mp_ctx)

    def _init_task_pool(self, max_tasks, mp_ctx):
        # @stripeli: why maxsize=1?
        return ProcessPool(max_workers=1, max_tasks=max_tasks, context=mp_ctx), \
            queue.Queue(maxsize=1)

    def _empty_tasks_q(self, task, force=False):
        # Get the tasks queue; the second element of the tuple.
        future_tasks_q = self.pool[task][1]
        while not future_tasks_q.empty():
            future_tasks_q.get(block=False).cancel(
            ) if force else future_tasks_q.get().result()

    def run_evaluation_task(self, block=False, **kwargs):
        future = self._run_task(
            task_name=config.EVALUATION_TASK,
            task_fn=self.task_executor.evaluate_model,
            callback=None,
            **kwargs
        )
        model_evaluations_pb = future.result() if block else metis_pb2.ModelEvaluations()
        return model_evaluations_pb

    def run_inference_task(self, block=False, **kwargs):
        future = self._run_task(
            task_name=config.INFERENCE_TASK,
            task_fn=self.task_executor.infer_model,
            callback=None,
            **kwargs
        )
        model_predictions_pb = future.result() if block else None  # FIXME: @stripeli
        return model_predictions_pb

    def run_learning_task(self,
                          callback: Callable = None,
                          block=False, **kwargs):
        future = self._run_task(
            task_name=config.LEARNING_TASK,
            task_fn=self.task_executor.train_model,
            callback=callback,
            **kwargs
        )
        # This will return the completed_task_pb.
        _ = future.result() if block else None
        # Which is not used as we're alway return the acknowledgement.
        # TODO: We need to return the completed_task_pb.
        return not future.cancelled()

    def shutdown(self, CANCEL_RUNNING: dict = {
        config.LEARNING_TASK: True,
        config.EVALUATION_TASK: True,
        config.INFERENCE_TASK: True
    }):
        for task, (pool, _) in self.pool.items():
            pool.close()
            self._empty_tasks_q(config.LEARNING_TASK,
                                force=CANCEL_RUNNING[task])
            pool.join()
        gc.collect()
        return True  # FIXME: We need to capture any failures.

    def _run_task(self,
                  task_name: str,
                  task_fn: Callable,
                  callback: Callable = None,
                  cancel_running_tasks=False,
                  **kwargs):
        self._empty_tasks_q(task_name, force=cancel_running_tasks)
        tasks_pool, tasks_futures_q = self.pool[task_name]
        future = tasks_pool.schedule(function=task_fn, kwargs={**kwargs})
        future.add_done_callback(
            self._callback_wrapper(callback)
        ) if callback else None
        tasks_futures_q.put(future)
        return future

    def _callback_wrapper(self, callback: Callable):
        def callback_wrapper(future):
            if future.done() and not future.cancelled():
                completed_task_pb = future.result()
                callback(completed_task_pb)
        return callback_wrapper
