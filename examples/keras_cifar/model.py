import tensorflow as tf


def get_model(input_shape=(32, 32, 3)):
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

    return model
