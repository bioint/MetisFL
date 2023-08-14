import gc
import multiprocessing as mp
import queue
from typing import Callable

from pebble import ProcessPool

from metisfl import config

from .learner_task import LearnerTask
from ..proto import metis_pb2


class TaskManager(object):

    """Manages the tasks of the Learner."""

    def __init__(self, learner_task: LearnerTask, recreate_queue_task_worker=False):
        self._learner_task = learner_task
        self._pool, self._queue = self._init_pools(
            recreate_queue_task_worker=recreate_queue_task_worker)

    def run_evaluation_task(self, block=False, **kwargs):
        future = self._run_task(
            task_name=config.EVALUATION_TASK,
            task_fn=self._learner_task.evaluate,
            callback=None,
            **kwargs
        )

        model_evaluations_pb = future.result() if block else metis_pb2.ModelEvaluations()

        return model_evaluations_pb

    def run_learning_task(self,
                          callback: Callable = None,
                          block=False, **kwargs):
        future = self._run_task(
            task_name=config.LEARNING_TASK,
            task_fn=self._learner_task.train_model,
            callback=callback,
            **kwargs
        )
        # This will return the completed_task_pb.
        _ = future.result() if block else None
        # Which is not used as we're alway return the acknowledgement.
        # TODO: We need to return the completed_task_pb.
        return not future.cancelled()

    def shutdown(self, cancel_running: dict = {
        config.LEARNING_TASK: True,
        config.EVALUATION_TASK: True,
    }):
        for task, (pool, _) in self.pool.items():
            pool.close()
            self._empty_tasks_q(config.LEARNING_TASK,
                                force=cancel_running[task])
            pool.join()
        gc.collect()
        return True  # FIXME: We need to capture any failures.

    def _init_pools(self, recreate_queue_task_worker=False):
        mp_ctx = mp.get_context("spawn")
        max_tasks = 1 if recreate_queue_task_worker else 0
        p, q = {}, {}
        for task in config.TASKS:
            p[task] = ProcessPool(
                max_workers=1, max_tasks=max_tasks, context=mp_ctx)
            q[task] = queue.Queue(maxsize=1)
            
        return p, q


    def _empty_tasks_q(self, task, force=False):
        # Get the tasks queue; the second element of the tuple.
        future_tasks_q = self.pool[task][1]
        while not future_tasks_q.empty():
            future_tasks_q.get(block=False).cancel(
            ) if force else future_tasks_q.get().result()

    def _run_task(
        self,
        task_name: str,
        task_fn: Callable,
        callback: Callable = None,
        cancel_running_tasks=False,
        **kwargs
    ):
        self._empty_tasks_q(task_name, force=cancel_running_tasks)
        tasks_pool, tasks_futures_q = self.pool[task_name]
        
        future = tasks_pool.schedule(function=task_fn, kwargs={**kwargs})
        if callback:
            future.add_done_callback(
                _callback_wrapper(callback)
            )
        tasks_futures_q.put(future)
        
        return future

def _callback_wrapper(callback: Callable):
    def callback_wrapper(future):
        if future.done() and not future.cancelled():
            result = future.result()
            callback(result)
    return callback_wrapper
