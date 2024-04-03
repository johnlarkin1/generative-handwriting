from typing import Tuple
from model.lstm_peephole_cell import LSTMCellWithPeepholes
import tensorflow as tf
from model.handwriting_mdn import MDNLayer, mdn_loss


class SimplestHandwritingPredictionModel(tf.keras.Model):
    def __init__(self, units=400, feature_size=3):
        super(SimplestHandwritingPredictionModel, self).__init__()
        self.lstm_cell = LSTMCellWithPeepholes(units)
        self.rnn_layer = tf.keras.layers.RNN(self.lstm_cell, return_sequences=True)
        self.dense_layer = tf.keras.layers.Dense(feature_size)

    def call(self, inputs):
        x = self.rnn_layer(inputs)
        return self.dense_layer(x)


class SimpleHandwritingPredictionModel(tf.keras.Model):
    def __init__(self, units=400, num_layers=3, feature_size=3):
        super(SimpleHandwritingPredictionModel, self).__init__()
        self.lstm_cells = [LSTMCellWithPeepholes(units) for _ in range(num_layers)]
        self.rnn_layers = [tf.keras.layers.RNN(cell, return_sequences=True) for cell in self.lstm_cells]
        self.dense_layer = tf.keras.layers.Dense(feature_size)

    def call(self, inputs):
        x = inputs
        for rnn in self.rnn_layers:
            x = rnn(x)
        x = self.dense_layer(x)
        return x


@tf.keras.utils.register_keras_serializable()
class DeepHandwritingPredictionModel(tf.keras.Model):
    def __init__(self, units=400, num_layers=3, num_mixture_components=20, **kwargs):
        super(DeepHandwritingPredictionModel, self).__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.num_mixture_components = num_mixture_components
        self.lstm_cells = [LSTMCellWithPeepholes(units) for _ in range(num_layers)]
        self.rnn_layers = [tf.keras.layers.RNN(cell, return_sequences=True) for cell in self.lstm_cells]
        self.mdn_layer = MDNLayer(num_mixture_components)

    def call(self, inputs):
        x = inputs
        for rnn in self.rnn_layers:
            x = rnn(x)
        return self.mdn_layer(x)

    def get_config(self):
        config = super(DeepHandwritingPredictionModel, self).get_config()
        config.update(
            {"units": self.units, "num_layers": self.num_layers, "num_mixture_components": self.num_mixture_components}
        )
        return config

    @classmethod
    def from_config(cls, config):
        units = config.pop("units")
        num_layers = config.pop("num_layers")
        num_mixture_components = config.pop("num_mixture_components")
        return cls(units, num_layers, num_mixture_components, **config)
