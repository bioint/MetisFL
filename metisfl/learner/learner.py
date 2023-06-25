import gc
import queue
 
import multiprocessing as mp
from pebble import ProcessPool

from metisfl.learner.federation_helper import FederationHelper
from metisfl.learner.learner_evaluator import LearnerEvaluator
from metisfl.proto import learner_pb2, model_pb2, metis_pb2
from metisfl.utils.metis_logger import MetisLogger

LEARNING_TASK = "learning"
EVALUATION_TASK = "evaluation"
INFERENCE_TASK = "inference"
TASKS = [LEARNING_TASK, EVALUATION_TASK, INFERENCE_TASK]
    
class Learner(object):
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
                 learner_evaluator: LearnerEvaluator,
                 federation_helper: FederationHelper,
                 recreate_queue_task_worker=False):
        self.learner_evaluator = learner_evaluator
        self.federation_helper = federation_helper
        self._init_tasks_pools(recreate_queue_task_worker) 
        
    def _init_tasks_pools(self, recreate_queue_task_worker=False):
        mp_ctx = mp.get_context("spawn")
        worker_max_tasks = 1 if recreate_queue_task_worker else 0
        self.pool = ()
        for task in TASKS:
            self.pool[task] = self._init_single_task_pool(worker_max_tasks, mp_ctx)
                            
    def _init_single_task_pool(self, worker_max_tasks, mp_ctx):
        return ProcessPool(max_workers=1, max_tasks=worker_max_tasks, context=mp_ctx), \
                queue.Queue(maxsize=1)
        
    def _empty_tasks_q(self, task, forceful=False):
        future_tasks_q = self.pool[task][1] # Get the tasks queue; the second element of the tuple.
        while not future_tasks_q.empty():
            if forceful:
                # Forceful Shutdown. Non-blocking retrieval
                # of AsyncResult from futures queue.
                future_tasks_q.get(block=False).cancel()
            else:
                # Graceful Shutdown. Await for the underlying
                # future inside the queue to complete.
                future_tasks_q.get().result()
                
    def _log(self, state, task):
        host_port = self.federation_helper.host_port_identifier()
        MetisLogger.info("Learner {} {} {} task on requested datasets."
                         .format(host_port, state, task))

    def run_evaluation_task(self, 
                            model_pb: model_pb2.Model, 
                            batch_size: int,
                            evaluation_dataset_pb: [learner_pb2.EvaluateModelRequest.dataset_to_eval],
                            metrics_pb: metis_pb2.EvaluationMetrics,
                            cancel_running_tasks=False, 
                            block=False, 
                            verbose=False):
        # If `cancel_running_tasks` is True, we perform a forceful shutdown of running tasks, else graceful.
        self._empty_tasks_q(future_tasks_q=self._evaluation_tasks_futures_q, forceful=cancel_running_tasks)
        # If we submit the datasets and the metrics as is (i.e., as repeated fields) pickle cannot
        # serialize the repeated messages, and it requires converting the repeated messages into a list.
        self._log(state="starts", task="evaluation")
        evaluation_datasets_pb = [d for d in evaluation_dataset_pb]
        evaluation_tasks_pool, evaluation_tasks_futures_q = self.pool[EVALUATION_TASK]
        future = evaluation_tasks_pool.schedule(
            function=self.learner_evaluator.evaluate_model,
            args=(model_pb, batch_size, evaluation_datasets_pb, metrics_pb, verbose))
        evaluation_tasks_futures_q.put(future)
        # @stripeli is this going to instantiate a model eval pb? 
        # anf if so, why is this empty pb returned in case block==False?
        model_evaluations_pb = metis_pb2.ModelEvaluations()
        if block:
            model_evaluations_pb = future.result()
        self._log(state="completed", task="evaluation")
        return model_evaluations_pb

    def run_inference_task(self):
        raise NotImplementedError("Not yet implemented.")

    def run_learning_task(self, 
                          learning_task_pb: metis_pb2.LearningTask,
                          hyperparameters_pb: metis_pb2.Hyperparameters, 
                          model_pb: model_pb2.Model,
                          cancel_running_tasks=False, 
                          block=False, 
                          verbose=False):
        # If `cancel_running_tasks` is True, we perform a forceful shutdown of running tasks, else graceful.
        self._empty_tasks_q(future_tasks_q=self._training_tasks_futures_q, forceful=cancel_running_tasks)
        self._log(state="starts", task="learning")
        # Submit the learning/training task to the Process Pool and add a callback to send the
        # trained local model to the controller when the learning task is complete. Given that
        # local training could span from seconds to hours, we cannot keep the grpc connection
        # open indefinitely and therefore the callback will collect the training result and
        # forward it accordingly to the controller.
        learning_tasks_pool, learning_tasks_futures_q = self.pool[LEARNING_TASK]
        future = learning_tasks_pool.schedule(
            function=self.learner_evaluator.train_model,
            args=(model_pb, learning_task_pb, hyperparameters_pb, verbose))
        
        # The following callback will trigger the request to the controller to receive the next task.
        future.add_done_callback(self.federation_helper.mark_learning_task_completed)
        learning_tasks_futures_q.put(future)
        if block:
            future.result()
        # If the task is submitted for processing then it is not cancelled.
        is_task_submitted = not future.cancelled()
        self._log(state="completed", task="learning")
        return is_task_submitted

    def shutdown(self,
                 cancel_train_running_tasks=True,
                 cancel_eval_running_tasks=True,
                 cancel_infer_running_tasks=True):
        # If graceful is True, it will allow all pending tasks to be completed,
        # else it will stop immediately all active tasks. At first, we close the
        # tasks pool so that no more tasks can be submitted, and then we wait
        # gracefully or non-gracefully (cancel future) for their completion.

        # TODO: pass args in a list or something to avoid code duplication.
        for _, (pool, _) in self.pool.items():
            pool.close()
        
        self._empty_tasks_q(LEARNING_TASK, forceful=cancel_train_running_tasks)
        self._empty_tasks_q(EVALUATION_TASK, forceful=cancel_eval_running_tasks)
        self._empty_tasks_q(INFERENCE_TASK, forceful=cancel_infer_running_tasks)
       
        for _, (pool, _) in self.pool.items():
            pool.join()       
       
        gc.collect()
        # TODO - we always return True, but we need to capture any failures that may occur while terminating.
        return True