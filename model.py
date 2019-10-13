import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, UpSampling2D
from tensorflow.keras.layers import Input, Flatten, BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Lambda, Reshape
from tensorflow.keras.layers import LeakyReLU, Conv2DTranspose
from tensorflow.keras.models import load_model, Model


def discriminator():
    """
    """
    _input = Input((28, 28, 1))

    x = Conv2D(filters=64, kernel_size=5, strides=2,
               padding="same")(_input)
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filters=64, kernel_size=5, strides=2,
               padding="same")(x)
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filters=128, kernel_size=5, strides=2,
               padding="same")(x)
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filters=128, kernel_size=5, strides=2,
               padding="same")(x)
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    output = Dense(units=1, activation="relu")(x)

    model = Model(inputs=_input, outputs=output)
    return model


def discriminator_bn():
    """
    """
    _input = Input((28, 28, 1))

    x = Conv2D(filters=64, kernel_size=5, strides=2,
               padding="same")(_input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters=64, kernel_size=5, strides=2,
               padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters=128, kernel_size=5, strides=2,
               padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters=128, kernel_size=5, strides=2,
               padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    output = Dense(units=1, activation="relu")(x)

    model = Model(inputs=_input, outputs=output)
    return model


def generator():
    """
    """
    _input = Input((100))

    x = Dense(units=7 * 7 * 512)(_input)
    x = LeakyReLU()(x)

    x = Reshape((7, 7, 512))(x)

    x  = Conv2DTranspose(filters=256, kernel_size=(3,3),
                         padding='same', strides=2)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU()(x)

    x  = Conv2DTranspose(filters=128, kernel_size=(3,3),
                         padding='same', strides=2)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU()(x)

    x  = Conv2DTranspose(filters=64, kernel_size=(3,3),
                         padding='same', strides=1)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters=1, kernel_size=3, strides=1,
               padding="same")(x)
    output = Activation("tanh")(x) * 255

    model = Model(inputs=_input, outputs=output)
    return model

