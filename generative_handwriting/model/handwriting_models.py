from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf

from generative_handwriting.alphabet import ALPHABET_SIZE
from generative_handwriting.constants import (
    GRADIENT_CLIP_VALUE,
    NUM_ATTENTION_GAUSSIAN_COMPONENTS,
    NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
    NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
    NUM_LSTM_HIDDEN_LAYERS,
)
from generative_handwriting.model.attention_mechanism import AttentionMechanism
from generative_handwriting.model.attention_rnn_cell import AttentionRNNCell
from generative_handwriting.model.lstm_peephole_cell import LSTMPeepholeCell
from generative_handwriting.model.mixture_density_network import MixtureDensityLayer, mdn_loss


class SimpleLSTMModel(tf.keras.Model):
    def __init__(self, units=400, num_layers=None, feature_size=3):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True)
        self.dense = tf.keras.layers.Dense(feature_size)

    def call(self, inputs, training=None, mask=None):
        x = self.lstm(inputs)
        return self.dense(x)


class SimplestHandwritingPredictionModel(tf.keras.Model):
    def __init__(self, units=400, feature_size=3):
        super(SimplestHandwritingPredictionModel, self).__init__()
        self.lstm_cell = LSTMPeepholeCell(units, 0)
        self.rnn_layer = tf.keras.layers.RNN(self.lstm_cell, return_sequences=True)
        self.dense_layer = tf.keras.layers.Dense(feature_size)

    def call(self, inputs, training=None, mask=None):
        x = self.rnn_layer(inputs)
        return self.dense_layer(x)


class SimpleHandwritingPredictionModel(tf.keras.Model):
    def __init__(self, units=400, num_layers=3, feature_size=3):
        super(SimpleHandwritingPredictionModel, self).__init__()
        self.lstm_cells = [LSTMPeepholeCell(units, i) for i in range(num_layers)]
        self.rnn_layers = [tf.keras.layers.RNN(cell, return_sequences=True) for cell in self.lstm_cells]
        self.dense_layer = tf.keras.layers.Dense(feature_size)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for rnn in self.rnn_layers:
            x = rnn(x)
        x = self.dense_layer(x)
        return x


@tf.keras.utils.register_keras_serializable()
class DeepHandwritingPredictionModel(tf.keras.Model):
    def __init__(
        self,
        units=400,
        num_layers=3,
        num_mixture_components=20,
        **kwargs,
    ) -> None:
        super(DeepHandwritingPredictionModel, self).__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.num_mixture_components = num_mixture_components
        self.lstm_cells = [LSTMPeepholeCell(units, i) for i in range(num_layers)]
        self.rnn_layers = [tf.keras.layers.RNN(cell, return_sequences=True) for cell in self.lstm_cells]
        self.mdn_layer = MixtureDensityLayer(num_mixture_components)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i, lstm_layer in enumerate(self.rnn_layers):
            if i == 0:
                x = lstm_layer(x)
            else:
                combined_input = tf.concat([inputs, x], axis=-1)
                x = lstm_layer(combined_input)
        output = self.mdn_layer(x)
        return output

    def get_config(self):
        config = super(DeepHandwritingPredictionModel, self).get_config()
        config.update(
            {
                "units": self.units,
                "num_layers": self.num_layers,
                "num_mixture_components": self.num_mixture_components,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class DeepHandwritingSynthesisModel(tf.keras.Model):
    """
    A similar implementation to the previous model, but with a different approach to the attention mechanism.
    This is batched for efficiency
    """

    def __init__(
        self,
        units=NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
        num_layers=NUM_LSTM_HIDDEN_LAYERS,
        num_mixture_components=NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
        num_chars=ALPHABET_SIZE,
        num_attention_gaussians=NUM_ATTENTION_GAUSSIAN_COMPONENTS,
        gradient_clip_value=GRADIENT_CLIP_VALUE,
        enable_mdn_regularization=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.num_mixture_components = num_mixture_components
        self.num_chars = num_chars
        self.num_attention_gaussians = num_attention_gaussians
        self.gradient_clip_value = gradient_clip_value
        self.enable_mdn_regularization = enable_mdn_regularization
        self.lstm_cells = [LSTMPeepholeCell(units, idx) for idx in range(num_layers)]
        self.attention_mechanism = AttentionMechanism(num_gaussians=num_attention_gaussians, num_chars=num_chars)
        self.attention_rnn_cell = AttentionRNNCell(self.lstm_cells, self.attention_mechanism, self.num_chars)
        self.rnn_layer = tf.keras.layers.RNN(self.attention_rnn_cell, return_sequences=True)
        self.mdn_layer = MixtureDensityLayer(num_mixture_components, enable_regularization=enable_mdn_regularization)

        # Metric trackers for proper loss aggregation
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.nll_tracker = tf.keras.metrics.Mean(name="nll")

    def call(
        self, inputs: Dict[str, tf.Tensor], training: Optional[bool] = None, mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        # This call is optimized for using tf functions and
        # having a dictionary of batched items passed in as an input
        input_strokes = inputs["input_strokes"]
        input_chars = inputs["input_chars"]
        input_char_lens = inputs["input_char_lens"]

        # Convert inputs to one-hot and set RNN cell attributes
        char_seq_one_hot = tf.one_hot(input_chars, depth=self.num_chars)
        self.attention_rnn_cell.char_seq_one_hot = char_seq_one_hot
        self.attention_rnn_cell.char_seq_len = input_char_lens

        # Get initial states
        batch_size = tf.shape(input_strokes)[0]
        initial_states = self.attention_rnn_cell.get_initial_state(batch_size=batch_size, dtype=input_strokes.dtype)
        initial_states_list = [
            initial_states["lstm_0_h"],
            initial_states["lstm_0_c"],
            initial_states["lstm_1_h"],
            initial_states["lstm_1_c"],
            initial_states["lstm_2_h"],
            initial_states["lstm_2_c"],
            initial_states["kappa"],
            initial_states["w"],
        ]

        # Process through RNN and MDN layers
        outputs = self.rnn_layer(input_strokes, initial_state=initial_states_list, training=training)
        final_output = self.mdn_layer(outputs)
        return final_output

    @property
    def metrics(self):
        # Tells Keras what metrics to reset/aggregate each epoch
        return [self.loss_tracker, self.nll_tracker]

    def train_step(self, data: Tuple[Dict[str, tf.Tensor], tf.Tensor]) -> Dict[str, tf.Tensor]:
        inputs, y_true = data
        target_stroke_lens = inputs["target_stroke_lens"]
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            nll = mdn_loss(y_true, y_pred, target_stroke_lens, self.num_mixture_components)
            # Add layer regularization losses (e.g., from MixtureDensityLayer)
            reg = tf.add_n(self.losses) if self.losses else 0.0
            loss = nll + reg

        gradients = tape.gradient(loss, self.trainable_variables)
        # Use optimizer's global_clipnorm instead of manual clipping to avoid double clipping
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics so callbacks/History see real numbers
        self.nll_tracker.update_state(nll)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "nll": self.nll_tracker.result()}

    def build(self, inputs_by_name_shape: Union[Dict[str, Tuple[int, ...]], Tuple[int, ...]]) -> None:
        # Input here is going to look like:
        # input_shape {'input_strokes': (None, 567, 3), 'input_chars': (None, 30), 'input_char_lens': (None,)}
        if isinstance(inputs_by_name_shape, dict):
            strokes_input_shape = inputs_by_name_shape["input_strokes"]
        else:
            # Handle the case where TensorFlow passes a TensorShape directly
            strokes_input_shape = inputs_by_name_shape

        # Build the MDN layer first
        mdn_input_shape = tf.TensorShape([strokes_input_shape[0], strokes_input_shape[1], self.units])
        self.mdn_layer.build(mdn_input_shape)

        # Build the RNN layer
        self.rnn_layer.build(tf.TensorShape(strokes_input_shape))

        # Set the model's built flag
        self._is_built = True

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "num_layers": self.num_layers,
                "num_mixture_components": self.num_mixture_components,
                "num_chars": self.num_chars,
                "num_attention_gaussians": self.num_attention_gaussians,
                "enable_mdn_regularization": self.enable_mdn_regularization,
                "debug": self.attention_mechanism.debug,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DeepHandwritingSynthesisModel":
        return cls(**config)
