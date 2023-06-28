import gc
import queue
import multiprocessing as mp

from typing import Callable
from pebble import ProcessPool

import metisfl.learner.constants as constants
from metisfl.learner.task_executor import TaskExecutor
from metisfl.proto import metis_pb2
    
class LearnerExecutor(object):

    def __init__(self, task_executor: TaskExecutor, recreate_queue_task_worker=False):
        self.task_executor = task_executor
        self._init_tasks_pools(recreate_queue_task_worker) 
        
    def _init_tasks_pools(self, recreate_queue_task_worker=False):
        mp_ctx = mp.get_context("spawn")
        max_tasks = 1 if recreate_queue_task_worker else 0
        self.pool = ()
        for task in constants.TASKS:
            self.pool[task] = self._init_task_pool(max_tasks, mp_ctx)
                            
    def _init_task_pool(self, max_tasks, mp_ctx):
        return ProcessPool(max_workers=1, max_tasks=max_tasks, context=mp_ctx), \
                queue.Queue(maxsize=1)
        
    def _empty_tasks_q(self, task, force=False):
        future_tasks_q = self.pool[task][1] # Get the tasks queue; the second element of the tuple.
        while not future_tasks_q.empty():
            future_tasks_q.get(block=False).cancel() if force else future_tasks_q.get().result()
                
    def run_evaluation_task(self, cancel_running=False, block=False, **kwargs):
        future = self._run_task(
            task_name=constants.EVALUATION_TASK,
            task_fun=self.task_executor.evaluate_model,
            cancel_running=cancel_running,
            callback=None,
            kwargs=kwargs
        )
        model_evaluations_pb = future.result() if block else metis_pb2.ModelEvaluations()
        return model_evaluations_pb

    def run_inference_task(self, cancel_running=False, block=False, **kwargs):
        future = self._run_task(
            task_name=constants.INFERENCE_TASK,
            task_fn=self.task_executor.infer_model,
            cancel_running=cancel_running,
            callback=None,
            kwargs=kwargs
        )
        model_predictions_pb = future.result() if block else None # FIXME: @stripeli
        return model_predictions_pb

    def run_learning_task(self, 
                          callback: Callable = None,
                          cancel_running=False, 
                          block=False, **kwargs):
        future = self._run_task(
            task_name=constants.LEARNING_TASK,
            task_fn=self.task_executor.train_model,
            callback=callback,
            cancel_running=cancel_running,
            kwargs=kwargs
        )
        _ = future.result() if block else None # This will return the completed_task_pb.
                                               # Which is not used as we're alway return the acknowledgement.
                                               # TODO: We need to return the completed_task_pb.
        return not future.cancelled()

    def shutdown(self, CANCEL_RUNNING: dict = {
        constants.LEARNING_TASK: True,
        constants.EVALUATION_TASK: True,
        constants.INFERENCE_TASK: True
    }):
        # If graceful is True, it will allow all pending tasks to be completed,
        # else it will stop immediately all active tasks. At first, we close the
        # tasks pool so that no more tasks can be submitted, and then we wait
        # gracefully or non-gracefully (cancel future) for their completion.
        for task, (pool, _) in self.pool.items():
            pool.close()
            self._empty_tasks_q(constants.LEARNING_TASK, force=CANCEL_RUNNING[task])
            pool.join()       
        gc.collect()
        return True # FIXME: We need to capture any failures. 

    def _run_task(self, 
                task_name: str, 
                task_fn: Callable, 
                callback: Callable = None,
                cancel_running=False, 
                **kwargs):
        self._empty_tasks_q(task_name, force=cancel_running)
        tasks_pool, tasks_futures_q = self.pool[task_name]
        future = tasks_pool.schedule(function=task_fn, kwargs=kwargs)
        future.add_done_callback(callback) if callback else None
        tasks_futures_q.put(future)
        return future

