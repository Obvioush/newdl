from __future__ import print_function
import tensorflow as tf
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K


class MyDense(Layer):
    def __init__(self, units=32):
        super(MyDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
            initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,),
            initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.nn.sigmoid(tf.matmul(inputs, self.w) + self.b)


class MyEmbedding(Layer):
    def __init__(self, input_unit, output_unit):
        super(MyEmbedding, self).__init__()
        self.input_unit = input_unit
        self.output_unit = output_unit

    def build(self, input_shape):
        self.embedding = self.add_weight(shape=(self.input_unit, self.output_unit),
                                         initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embedding, inputs)