import tensorflow as tf

from metisfl.models.keras.keras_model import MetisModelKeras


class HousingMLP(MetisModelKeras):

    def __init__(self, params_per_layer=10, hidden_layers_num=1, learning_rate=0.0, data_type="float32"):
        super(HousingMLP, self).__init__()
        self.params_per_layer = params_per_layer
        self.hidden_layers_num = hidden_layers_num
        # We set the default value to 0.0 so not to train,
        # since this model is solely used for stress testing.
        self.learning_rate = learning_rate

        if data_type == "float32":
            self.data_type = tf.float32
        elif data_type == "float64":
            self.data_type = tf.float64
        else:
            raise RuntimeError("Not a supported data type. Please pass float32 or float64")

    def get_model(self):
        model = tf.keras.models.Sequential()
        # This layer outputs 14x10, 14x100, 14x1000, etc...
        model.add(tf.keras.layers.Dense(self.params_per_layer,
                                        input_shape=(13,),
                                        kernel_initializer='normal',
                                        activation='relu',
                                        dtype=self.data_type))
        for i in range(self.hidden_layers_num):
            model.add(tf.keras.layers.Dense(self.params_per_layer,
                                            input_shape=(self.params_per_layer,),
                                            kernel_initializer='normal',
                                            activation='relu',
                                            dtype=self.data_type))
        model.add(tf.keras.layers.Dense(1,
                                        kernel_initializer='normal',
                                        dtype=self.data_type))
        # We set a very
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model
