import tensorflow as tf

from src.python.models.model_def import ModelDef
from src.python.models.keras.optimizers.fed_prox import FedProx


class CifarCNN(ModelDef):

    def __init__(self, metrics=["accuracy"], optimizer_name="MomentumSGD"):
        super(CifarCNN, self).__init__()
        self.metrics = metrics
        self.optimizer_name = optimizer_name

    def get_model(self):
        """
        Prepare CNN model
        :return:
        """
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=(32, 32, 3),
                                         activation='relu', padding='same'))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=(32, 32, 3),
                                         activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2))

        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2))

        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        if self.optimizer_name.lower() in ["vanillasgd", "momentumsgd"]:
            optimizer = tf.keras.optimizers.SGD()
        elif self.optimizer_name.lower() == "fedprox":
            optimizer = FedProx()
        else:
            raise RuntimeError("Not supported optimizer.")
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=self.metrics)
        return model
