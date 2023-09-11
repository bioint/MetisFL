import tensorflow as tf

# from tensorflow.keras import layers, models, regularizers
from metisfl.models.keras.keras_model import MetisModelKeras


class IMDB_LSTM(MetisModelKeras):

    def __init__(self, learning_rate=0.01, max_features=25000):
        super(IMDB_LSTM, self).__init__()
        self.learning_rate = learning_rate
        self.max_features = max_features  # Only consider the top X words

    def get_model(self):
        """
        Prepare CNN model
        :return:
        """
        model = tf.keras.models.Sequential()
        # Embed each integer in a 128-dimensional vector
        model.add(tf.keras.layers.Embedding(self.max_features, 128))
        # Add 2 bidirectional LSTMs
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
        # Add a classifier
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.0),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

        return model
