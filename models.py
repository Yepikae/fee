"""Keras models modules."""

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
from keras import optimizers
import numpy as np
from keras.layers import Activation, Convolution2D, Dropout, Input, Flatten
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import concatenate as Kconcatenate
from keras.models import Model


class SimpleDenseFactory:
    """A simple Dense model class."""

    def __init__(self):
        """Constructor."""
        self.__input_shape__ = None
        self.__optimizer__ = optimizers.RMSprop(lr=0.001)
        self.__loss__ = 'categorical_crossentropy'
        self.__layers__ = []

    def set_input_shape(self, input_shape):
        """Set __input_shape__."""
        self.__input_shape__ = input_shape

    def set_loss(self, loss):
        """Set __loss__."""
        self.__loss__ = loss

    def set_learning_rate(self, learning_rate):
        """Set __optimizer__."""
        self.__optimizer__ = optimizers.RMSprop(lr=learning_rate)

    def add_layer(self, units=124, activation='relu', dropout=None):
        """Add a layer to the model."""
        if len(self.__layers__) == 0:
            self.__layers__.append(Dense(units, activation=activation,
                                         input_shape=self.__input_shape__))
        else:
            self.__layers__.append(Dense(units, activation=activation))
        if dropout is not None:
            self.__layers__.append(Dropout(dropout))

    def reset(self):
        """Empty __layers__."""
        self.__layers__ = []

    def generate_model(self):
        """Generate a model."""
        model = Sequential()
        for i, l in enumerate(self.__layers__):
            model.add(l)
        model.compile(optimizer=self.__optimizer__,
                      loss=self.__loss__,
                      metrics=['accuracy'])
        return model


def simple_CNN(input_shape, num_classes=2):
    """Return a simple CNN."""
    # First, define the vision modules
    pic_input = Input(shape=input_shape)

    x = Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                      name='image_array')(pic_input)
    x = BatchNormalization()(x)
    x = Convolution2D(filters=16, kernel_size=(7, 7), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(.5)(x)

    x = Convolution2D(filters=32, kernel_size=(5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(filters=32, kernel_size=(5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(.5)(x)

    x = Convolution2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(.5)(x)

    x = Convolution2D(filters=128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(filters=128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    out = Flatten()(x)

    vision_model = Model(pic_input, out)

    # Then define the tell-digits-apart model
    pic_a = Input(shape=input_shape)
    pic_b = Input(shape=input_shape)

    # The vision model will be shared, weights and all
    out_a = vision_model(pic_a)
    out_b = vision_model(pic_b)

    concatenated = Kconcatenate([out_a, out_b])
    out = Dense(1, activation='sigmoid')(concatenated)

    classification_model = Model([pic_a, pic_b], out)
    return classification_model
