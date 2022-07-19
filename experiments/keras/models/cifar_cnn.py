import tensorflow as tf

from projectmetis.python.models.model_def import ModelDef


class CifarCNN(ModelDef):

    def __init__(self, learning_rate=0.005, metrics=["accuracy"]):
        super(CifarCNN, self).__init__()
        self.learning_rate = learning_rate
        self.metrics = metrics

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

        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.75)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=self.metrics)
        return model
