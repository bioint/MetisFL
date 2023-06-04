import time
import tensorflow as tf


class PerformanceProfiler(tf.keras.callbacks.Callback):

    def __init__(self):
        super(PerformanceProfiler, self).__init__()
        self.epochs_wall_clock_time_sec = list()
        self.batches_wall_clock_time_sec = list()

        self._epoch_start_wt = None
        self._epoch_end_wt = None
        self._batch_start_wt = None
        self._batch_end_wt = None

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_wt = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_end_wt = time.time()
        wt_delta = self._epoch_end_wt - self._epoch_start_wt
        self.epochs_wall_clock_time_sec.append(wt_delta)

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        self._batch_start_wt = time.time()

    def on_train_batch_end(self, batch, logs=None):
        """ On every local update we increase the counter by 1 and check whether the
        total number of steps is reached in order to stop the training of the model. """
        self._batch_end_wt = time.time()
        delta_wt = self._batch_end_wt - self._batch_start_wt
        self.batches_wall_clock_time_sec.append(delta_wt)

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass
