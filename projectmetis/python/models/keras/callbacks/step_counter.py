import tensorflow as tf


class StepCounter(tf.keras.callbacks.Callback):

    def __init__(self, total_steps):
        super(StepCounter, self).__init__()
        self.total_steps = total_steps
        self.steps_counter = 0
        self.epochs_counter = 0

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_counter += 1

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        """ On every local update we increase the counter by 1 and check whether the
        total number of steps is reached in order to stop the training of the model. """
        self.steps_counter += 1
        if self.steps_counter > self.total_steps:
            self.model.stop_training = True

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass
