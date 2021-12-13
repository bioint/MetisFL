import tensorflow as tf

from projectmetis.python.models.model_def import ModelDef


class CifarCNN(ModelDef):

    def __init__(self, learning_rate=0.005):
        super(CifarCNN, self).__init__()
        self.learning_rate = learning_rate

    def get_model(self):
        """
        Prepare CNN model
        :return:
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.0),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=["accuracy"])

        return model
