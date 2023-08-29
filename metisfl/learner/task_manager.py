import multiprocessing as mp
import queue
from typing import Callable, Optional

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
        self._worker_pool = ProcessPool(max_workers=max_workers,
                                        max_tasks=max_tasks,
                                        context=mp_ctx)
        self._future_queue = queue.Queue(maxsize=max_queue_size)

    def run_task(
        self,
        task_fn: Callable,
        task_args: Optional[tuple] = None,
        task_kwargs: Optional[dict] = None,
        callback: Optional[Callable] = None,
        cancel_running: Optional[bool] = False
    ) -> None:
        """Runs a task in the pool of workers.

        Parameters
        ----------
        task_fn : Callable
            A Callable object that represents the task to be run.
        task_args : Optional[tuple], (default=None)
            The arguments to be passed to the task function, by default None
        task_kwargs : Optional[dict], (default=None)
            The keyword arguments to be passed to the task function, by default None
        callback : Optional[Callable], (default=None)
            A Callable object that represents the callback function to be run after the task is completed, by default None
        cancel_running : Optional[bool], (default=False)
            Whether to cancel the running task before running the new one, by default False

        """
        self._empty_tasks_q(force=cancel_running)

        future = self._worker_pool.schedule(function=task_fn,
                                            args=task_args,
                                            kwargs={**task_kwargs})
        if callback:
            future.add_done_callback(
                self._callback_wrapper(callback)
            )

        self._future_queue.put(future)

    def _callback_wrapper(self, callback: Callable) -> Callable:

        def callback_wrapper(future: ProcessFuture) -> None:
            if future.done() and not future.cancelled():
                # FIXME: need to catch errors here
                callback(future.result())

        return callback_wrapper

    def shutdown(self, force: Optional[bool] = False) -> None:
        """Shuts down the pool of workers and empties the task queue.

        Parameters
        ----------
        force : Optional[bool], (default=False)
            Whether to force shutdown the running tasks.

        """
        self._worker_pool.close()
        self._empty_tasks_q(force=force)
        self._worker_pool.join()

    def _empty_tasks_q(self, force: Optional[bool] = False) -> None:
        """Empties the task queue.

        Parameters
        ----------
        force : Optional[bool], (default=False)
            Whether to force empty the task queue.
        """
        while not self._future_queue.empty():
            if force:
                self._future_queue.get(block=False).cancel()
            else:
                self._future_queue.get().result()
