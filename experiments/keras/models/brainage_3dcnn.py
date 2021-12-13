import tensorflow as tf

from tensorflow.keras.layers import Input
from projectmetis.python.models.model_def import ModelDef


class BrainAge3DCNN(ModelDef):

    # Def for standard conv block:
    # 1. Conv (3x3) + relu
    # 2. Conv (3x3)
    # 3. Batch normalization
    # 4. Relu
    # 5. Maxpool (3x3)

    # Model def
    def __init__(self, original_input=Input(shape=(91, 109, 91, 1), name='input'), learning_rate=5e-5):
        self.original_input = original_input
        self.learning_rate = learning_rate
        super(BrainAge3DCNN, self).__init__()

    def get_model(self, *args, **kwargs):

        def conv_block(inputs, num_filters, scope):
            inputs = tf.keras.layers.Conv3D(num_filters, 3, strides=1, padding="same", name=scope + "_conv")(inputs)
            # inputs = tf.contrib.layers.instance_norm(inputs, center=False, scale=False)
            inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid", name=scope + "_max_pool")(inputs)
            inputs = tf.nn.relu(inputs, name=scope + "_relu")
            return inputs

        # Series of conv blocks
        inputs = conv_block(self.original_input, 32,  "conv_block1")
        inputs = conv_block(inputs, 64,  "conv_block2")
        inputs = conv_block(inputs, 128, "conv_block3")
        inputs = conv_block(inputs, 256, "conv_block4")
        inputs = conv_block(inputs, 256, "conv_block5")

        inputs = tf.keras.layers.Conv3D(
            64, 1, strides=1, name="post_conv1")(inputs)
        # inputs = tf.contrib.layers.instance_norm(inputs, center=False, scale=False)
        inputs = tf.nn.relu(inputs, name="post_relu")
        inputs = tf.keras.layers.AveragePooling3D(pool_size=(2, 3, 2))(inputs)

        # model.add(tf.keras.layers.Dropout(0.5))
        # Default rate: 0.5
        # drop = tf.layers.dropout(inputs, training=is_training, name="drop")

        outputs = tf.keras.layers.Conv3D(
            1, 1, strides=1, name="reg_conv",
            bias_initializer=tf.constant_initializer(62.68))(inputs)
        # outputs = tf.squeeze(outputs)

        model = tf.keras.Model(inputs=self.original_input, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, nesterov=False),
                      loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
                      metrics=["mse"])
        return model