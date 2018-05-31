"""Keras models modules."""

from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import classification_report
from keras import optimizers
import numpy as np
from keras.layers import Activation, Convolution2D, Dropout, Input, Flatten
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import concatenate as Kconcatenate
from keras.models import Model
from keras.optimizers import SGD
from fee.classification import Expression as Exp


class BITBOTSFaceClassification:
    """A model from https://github.com/oarriaga/face_classification."""

    def __init__(self, path):
        """Contructor."""
        if path is None:
            path = './face_classification/trained_models/emotion_models/'
            path += 'fer2013_mini_XCEPTION.102-0.66.hdf5'
        self.model = load_model(path)
        self.labels = [Exp.ANGER, Exp.DISGUST, Exp.FEAR, Exp.HAPPINESS,
                       Exp.SADNESS, Exp.SURPRISE, Exp.NEUTRAL]

    def get_model_and_labels(self):
        """Return (model, labels)."""
        return self.model, self.labels


class DANN:
    """A DANN model generator.

    Inspired the article https://arxiv.org/pdf/1505.07818.pdf.
    """

    def __init__(self, input_layer=None, labels_layers=None,
                 domains_layers=None, features_layers=None):
        """Constructor."""
        self.input_layer     = input_layer
        self.features_layers = self.__connect_layers__(self.input_layer,
                                                       features_layers)
        self.labels_layers   = self.__connect_layers__(self.features_layers,
                                                       labels_layers)
        self.domains_layers   = self.__connect_layers__(self.features_layers,
                                                        domains_layers)
        self.linear_model       = None
        self.supervised_model   = None
        self.unsupervised_model = None

    def build_models_from_source(self, source_model=None,
                                 features_output_index=None,
                                 domain_layers=None, learning_rate=0.1):
        """Doc to do."""
        self.linear_model = source_model
        # Init the domain layers
        output = self.linear_model.get_layer(index=features_output_index)
        output = output.output
        x = domain_layers[0](output)
        for i in range(1, len(domain_layers)):
            x = domain_layers[i](x)
        domain_layers = x
        # Init the supervised model
        outputs = source_model.outputs
        outputs.append(domain_layers)
        s_model = self.__build_model__(learning_rate=learning_rate,
                                       inputs=source_model.inputs,
                                       outputs=outputs)
        self.supervised_model = s_model
        # Init the unsupervised model
        u_model = self.__build_model__(learning_rate=learning_rate,
                                       inputs=source_model.inputs,
                                       outputs=domain_layers)
        self.unsupervised_model = u_model

    def build_model(self, model, learning_rate=0.1):
        """Initialise the different models."""
        if model == 'linear':
            self.linear_model = self.__build_model__(learning_rate,
                                                     self.input_layer,
                                                     self.labels_layers)
        elif model == "supervised":
            self.supervised_model = self.__build_model__(learning_rate,
                                                         self.input_layer,
                                                         [self.domains_layers,
                                                          self.labels_layers])
        elif model == "unsupervised":
            self.unsupervised_model = self.__build_model__(learning_rate,
                                                           self.input_layer,
                                                           self.domains_layers)
        else:
            print('Model '+model+' doesn\'t exist.')
            exit()

    def build_models(self, learning_rate=0.1, verbose=True):
        """Doc to do."""
        self.build_model('linear')
        self.build_model('supervised')
        self.build_model('unsupervised')
        if verbose:
            print('== Linear Model (inputs+features+labels) =================')
            self.linear_model.summary()
            print('== Supervised Model (inputs+features+domains+labels) =====')
            self.supervised_model.summary()
            print('== Unsupervised Model (inputs+features+domains) ==========')
            self.unsupervised_model.summary()

    def get_models(self):
        """Doc to do."""
        t = (self.linear_model, self.supervised_model, self.unsupervised_model)
        return t

    def __build_model__(self, learning_rate, inputs, outputs):
        """Doc to do."""
        sgd = SGD(lr=learning_rate)
        model = Model(inputs=inputs, outputs=outputs)
        # model.compile(loss='categorical_crossentropy', optimizer=sgd,
        #               metrics=['accuracy'])
        return model

    def __connect_layers__(self, input, layers):
        """Doc to do."""
        if layers is None:
            return None
        x = layers[0](input)
        for i in range(1, len(layers)):
            x = layers[i](x)
        return x


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
