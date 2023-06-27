import gc
import queue
 
import multiprocessing as mp
from typing import Callable
from pebble import ProcessPool

import metisfl.learner.constants as constants
from metisfl.learner.federation_helper import FederationHelper
from metisfl.learner.task_executor import TaskExecutor
from metisfl.proto import metis_pb2
    
class LearnerExecutor(object):
    """
    Any invocation to the public functions of the Learner instance need to be wrapped inside a process,
    since the body of every function is generating a new Neural Network registry/context.

    In order to be able to run the training/evaluation/prediction functions as independent
    processes, we needed to create a ModelOperations class factory, depending on the neural network engine
    being used. For instance, for Keras we define the `_keras_model_ops_factory()` that internally imports
    the KerasModelOps class and for PyTorch we follow the same design by importing the PyTorchModeOps
    inside the body of the `_pytorch_model_ops_factory()` function.

    Specifically, no ModelOps subclass should be imported in the global scope of the learner but rather
    within the local scope (i.e., namespace) of each neural network model operations factory function.
    
    We use pebble.ProcessPool() and not the native concurrent.futures.ProcessPoolExecutor() so that
    when a termination signal (SIGTERM) is received we stop immediately the active task. This was
    not possible with jobs/tasks submitted to the concurrent.futures.ProcessPoolExecutor()
    because we had to wait for the active task to complete.
    A single Process per training/evaluation/inference task.

    If recreate_queue_task_worker is True, then we will create the entire model training backend
    from scratch. For instance, in the case of Tensorflow, if this flag is True, then the
    dependency graph holding the tensors and the variables definitions will be created again.
    This re-creation adds substantial delay during model training and evaluation. If we do not
    re-create, that is recreate_queue_task_worker is set to False, then we can re-use the existing
    graph and avoid graph creation delays.
    Unix default context is "fork", others: "spawn", "forkserver".
    We use "spawn" because it initiates a new Python interpreter, and
    it is more stable when running multiple processes in the same machine.
    """

    def __init__(self,
                 task_executor: TaskExecutor,
                 recreate_queue_task_worker=False):
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
                
    def run_evaluation_task(self, 
                            cancel_running=False, 
                            block=False, 
                            **kwargs):
        future = self._run_task(
            task_name=constants.EVALUATION_TASK,
            task_fun=self.task_executor.evaluate_model,
            cancel_running=cancel_running,
            callback=None,
            kwargs=kwargs
        )
        model_evaluations_pb = future.result() if block else metis_pb2.ModelEvaluations()
        return model_evaluations_pb

    def run_inference_task(self,
                            cancel_running=False,
                            block=False,
                            **kwargs):
        future = self._run_task(
            task_name=constants.INFERENCE_TASK,
            task_fn=self.task_executor.predict,
            cancel_running=cancel_running,
            callback=None,
            kwargs=kwargs
        )
        model_predictions_pb = future.result() if block else metis_pb2.ModelPredictions()
        return model_predictions_pb

    def run_learning_task(self, 
                          cancel_running=False, 
                          block=False,
                          **kwargs):
        future = self._run_task(
            task_name=constants.LEARNING_TASK,
            task_fn=self.task_executor.train_model,
            callback=self.federation_helper.mark_learning_task_completed,
            cancel_running=cancel_running,
            kwargs=kwargs
        )
        _ = future.result() if block else None # This will return the completed_task_pb.
                                               # Which is not used as we're alway return the acknowledgement.
                                               # TODO: We need to return the completed_task_pb.
        is_task_submitted = not future.cancelled()
        return is_task_submitted

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
        # TODO - we always return True, but we need to capture any failures that may occur while terminating.
        return True

    def _run_task(self, 
                task_name: str, 
                task_fn: Callable, 
                cancel_running=False, 
                callback: Callable = None,
                **kwargs):
        self._empty_tasks_q(task_name, force=cancel_running)
        tasks_pool, tasks_futures_q = self.pool[task_name]
        future = tasks_pool.schedule(function=task_fn, kwargs=kwargs)
        future.add_done_callback(callback) if callback else None
        tasks_futures_q.put(future)
        return future

