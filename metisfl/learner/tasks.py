import multiprocessing as mp
import queue
from typing import Any, Callable, Optional, Tuple

from pebble import ProcessFuture, ProcessPool


class TaskManager(object):

    """Manages the execution of tasks in a pool of workers."""

    def __init__(
        self,
        max_workers: Optional[int] = 1,
        max_tasks: Optional[int] = 1,
        max_queue_size: Optional[int] = 1
    ):
        """Initializes a TaskManager object.

        Parameters
        ----------
        max_workers : Optional[int], (default=1)
            The maximum number of workers in the pool, by default 1
        max_tasks : Optional[int], (default=1)
            The maximum number of tasks that can be scheduled in each worker before it is restarted, by default 1
        max_queue_size : Optional[int], (default=1)
            The maximum size of the future queue, by default 1
        """
        mp_ctx = mp.get_context("spawn")
        self.worker_pool = ProcessPool(max_workers=max_workers,
                                       max_tasks=max_tasks,
                                       context=mp_ctx)
        self.future_queue = queue.Queue(maxsize=max_queue_size)

    def run_task(
        self,
        task_fn: Callable,
        task_kwargs: Optional[dict] = {},
        callback: Optional[Callable] = lambda x: None,
        task_out_to_callback_fn: Optional[Callable[[
            Any], Tuple]] = lambda x: x,
        cancel_running: Optional[bool] = False
    ) -> None:
        """Runs a task in the pool of workers.

        Parameters
        ----------
        task_fn : Callable
            A Callable object that represents the task to be run.
        task_kwargs : Optional[dict], (default={})
            The keyword arguments to be passed to the task function, by default {}
        callback : Optional[Callable], (default=None)
            A Callable object that represents the callback function to be run after the task is completed, by default None
        task_results_to_callback_fn : Optional[Callable], (default=lambda x: x)
            A Callable object that represents the function to be run on the task output before passing them to the callback function, by default lambda x: x  
        cancel_running : Optional[bool], (default=False)
            Whether to cancel the running task before running the new one, by default False

        """
        self.empty_tasks_q(force=cancel_running)

        future = self.worker_pool.schedule(function=task_fn,
                                           kwargs={**task_kwargs})
        if callback:
            future.add_done_callback(
                self.callback_wrapper(
                    callback=callback,
                    task_out_to_callback_fn=task_out_to_callback_fn
                )
            )

        self.future_queue.put(future)

    def callback_wrapper(
        self,
        callback: Callable,
        task_out_to_callback_fn: Callable = lambda x: x,
    ) -> Callable:
        """Wraps the callback function to be run after the task is completed.

        Parameters
        ----------
        callback : Callable
            A Callable object to be called after the task is completed.
        task_out_to_callback_fn : Callable, (default=lambda x: x)
            A Callable object that represents the function to be run on the task output before passing them to the callback function, by default lambda x: x

        Returns
        -------
        Callable
            A Callable object that represents the wrapped callback function.
        """

        def wrapper(future: ProcessFuture) -> None:
            if future.done() and not future.cancelled():
                # FIXME: need to catch errors here
                callback(*task_out_to_callback_fn(future.result()))

        return wrapper

    def shutdown(self, force: Optional[bool] = False) -> None:
        """Shuts down the pool of workers and empties the task queue.

        Parameters
        ----------
        force : Optional[bool], (default=False)
            Whether to force shutdown the running tasks.

        """
        self.worker_pool.close()
        self.empty_tasks_q(force=force)
        self.worker_pool.join()

    def empty_tasks_q(self, force: Optional[bool] = False) -> None:
        """Empties the task queue.

        Parameters
        ----------
        force : Optional[bool], (default=False)
            Whether to force empty the task queue.
        """
        while not self.future_queue.empty():
            if force:
                self.future_queue.get(block=False).cancel()
            else:
                self.future_queue.get().result()
