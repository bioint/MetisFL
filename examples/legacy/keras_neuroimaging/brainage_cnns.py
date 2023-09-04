import tensorflow as tf

from tensorflow.keras.layers import Input
from metisfl.models.keras.keras_model import MetisModelKeras


class BrainAge2DCNN(MetisModelKeras):

    # Def for standard conv block:
    # 1. Conv (3x3) + relu
    # 2. Conv (3x3)
    # 3. Batch normalization
    # 4. Relu
    # 5. Maxpool (3x3)

    # Model def
    def __init__(self, learning_rate=5e-4, batch_size=1):
        self.original_input = tf.keras.layers.Input(shape=(91, 109, 91), batch_size=batch_size, name='input')
        self.learning_rate = learning_rate
        super(BrainAge2DCNN, self).__init__()

    def get_model(self):
        def conv_block(inputs, num_filters, name):
            inputs = tf.keras.layers.Conv2D(
                num_filters, 3, strides=1, padding="same", name=name + "_conv")(inputs)
            # inputs = tf.keras.layers.BatchNormalization(
            # axis=[0, 3], name=name + "_batch_norm", center=True, scale=True)(inputs)
            inputs = tf.keras.layers.BatchNormalization(
                axis=[0, 3], name=name + "_batch_norm", center=False, scale=False)(inputs, training=True)
            inputs = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid", name=name + "_max_pool")(inputs)
            inputs = tf.keras.layers.ReLU(name=name + "_relu")(inputs)
            return inputs

        def encoder(inputs):
            inputs = conv_block(inputs, 32, "conv_block1")
            inputs = conv_block(inputs, 64, "conv_block2")
            inputs = conv_block(inputs, 128, "conv_block3")
            inputs = conv_block(inputs, 256, "conv_block4")
            inputs = conv_block(inputs, 256, "conv_block5")
            return inputs

        def compute_embedding(x):
            x = encoder(x)
            x = tf.keras.layers.Conv2D(
                64, 1, strides=1, name="post_conv1")(x)
            # x = tf.keras.layers.BatchNormalization(center=True, scale=True)(x)
            x = tf.keras.layers.BatchNormalization(
                name="post_batch_norm", axis=[0, 3], center=False, scale=False)(x, training=True)
            x = tf.keras.layers.ReLU(name="post_relu")(x)
            x = tf.keras.layers.AveragePooling2D(pool_size=(3, 2))(x)

            # Default rate: 0.5
            # x = tf.keras.layers.Dropout(0.5)(x, training=True)
            x = tf.keras.layers.Conv2D(
                32, 1, strides=1, name="post_conv2")(x)
            return tf.squeeze(x)

        def infer_ages(images):
            # Map function computes embeddings for every image,
            # however, an TypeError is raised due to KerasTensor.
            # embeddings = tf.map_fn(lambda x: compute_embedding(x), images)
            # This is only applicable to a single image.
            images = tf.transpose(images, [1, 2, 3, 0])
            embeddings = compute_embedding(images)
            # Again, This is only applicable to a single image.
            embeddings = tf.transpose(embeddings, [1, 0])
            # print("Embeddings Shape: ", embeddings.shape.dims, flush=True)
            embedding = tf.reduce_mean(embeddings, axis=1, keepdims=True)
            # print("Embeddings 2 Shape: ", embedding.shape.dims, flush=True)
            inputs = tf.keras.layers.Dense(64)(embedding)
            inputs = tf.nn.relu(inputs)

            # Regression
            output = tf.keras.layers.Dense(1, bias_initializer=tf.constant_initializer(62.68))(inputs)

            output = tf.squeeze(output)
            return output

        model = tf.keras.Model(inputs=self.original_input,
                               outputs=infer_ages(self.original_input),
                               name="BrainAge-2DCNN")
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
                      metrics=["mse", "mae"])
        return model


class BrainAge3DCNN(MetisModelKeras):

    # Def for standard conv block:
    # 1. Conv (3x3) + relu
    # 2. Conv (3x3)
    # 3. Batch normalization
    # 4. Relu
    # 5. Maxpool (3x3)

    # Model def
    def __init__(self, learning_rate=5e-5, batch_size=1):
        self.original_input = tf.keras.layers.Input(shape=(91, 109, 91, 1), batch_size=batch_size, name='input')
        self.learning_rate = learning_rate
        super(BrainAge3DCNN, self).__init__()

    def get_model(self, *args, **kwargs):
        def conv_block(inputs, num_filters, scope):
            inputs = tf.keras.layers.Conv3D(num_filters, 3, strides=1, padding="same", name=scope + "_conv")(inputs)
            # since we use BatchNorm as InstanceNorm, we need to keep training=True
            inputs = tf.keras.layers.BatchNormalization(center=False, scale=False, axis=[0, 4])(inputs, training=True)
            # inputs = tf.keras.layers.BatchNormalization(center=False, scale=False, axis=[0, 4])(inputs)
            inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid", name=scope + "_max_pool")(inputs)
            inputs = tf.nn.relu(inputs, name=scope + "_relu")
            return inputs

        # Series of conv blocks
        inputs = conv_block(self.original_input, 32, "conv_block1")
        inputs = conv_block(inputs, 64, "conv_block2")
        inputs = conv_block(inputs, 128, "conv_block3")
        inputs = conv_block(inputs, 256, "conv_block4")
        inputs = conv_block(inputs, 256, "conv_block5")

        inputs = tf.keras.layers.Conv3D(
            64, 1, strides=1, name="post_conv1")(inputs)
        # since we use BatchNorm as InstanceNorm, we need to keep training=True
        inputs = tf.keras.layers.BatchNormalization(center=False, scale=False, axis=[0, 4])(inputs, training=True)
        # inputs = tf.keras.layers.BatchNormalization(center=False, scale=False, axis=[0, 4])(inputs)
        inputs = tf.nn.relu(inputs, name="post_relu")
        inputs = tf.keras.layers.AveragePooling3D(pool_size=(2, 3, 2))(inputs)

        # Default rate: 0.5
        # inputs = tf.keras.layers.Dropout(0.5)(inputs, training=True)

        outputs = tf.keras.layers.Conv3D(
            1, 1, strides=1, name="reg_conv",
            bias_initializer=tf.constant_initializer(62.68))(inputs)
        outputs = tf.squeeze(outputs, axis=[1, 2, 3, 4])

        model = tf.keras.Model(inputs=self.original_input, outputs=outputs, name="BrainAge-3DCNN")
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
                      metrics=["mse", "mae"])
        return model
