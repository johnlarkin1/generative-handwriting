from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf

from generative_handwriting.alphabet import ALPHABET_SIZE
from generative_handwriting.constants import (
    GRADIENT_CLIP_VALUE,
    NUM_ATTENTION_GAUSSIAN_COMPONENTS,
    NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
    NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
    NUM_LSTM_HIDDEN_LAYERS,
    NUM_MIXTURE_COMPONENTS_PER_COMPONENT,
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
        units: int = 400,
        num_layers: int = 3,
        num_mixture_components: int = 20,
        **kwargs,
    ) -> None:
        super(DeepHandwritingPredictionModel, self).__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.num_mixture_components = num_mixture_components
        self.lstm_cells: list[LSTMPeepholeCell] = [LSTMPeepholeCell(units, i) for i in range(num_layers)]
        self.rnn_layers: list[tf.keras.layers.RNN] = [
            tf.keras.layers.RNN(cell, return_sequences=True) for cell in self.lstm_cells
        ]
        self.mdn_layer = MixtureDensityLayer(num_mixture_components)

        # loss metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.nll_tracker = tf.keras.metrics.Mean(name="nll")

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

    @property
    def metrics(self):
        return [self.loss_tracker, self.nll_tracker]

    def train_step(self, data):
        # trying to make this flexible if we do not have the lengths of the sequences
        # note that x and y are tensors of shape [batch_size, seq_length, 3]
        # it's just if we have `data` as a tuple of (x, y) or (x, y, lengths)
        if len(data) == 2:
            x, y = data
            lengths = None
        else:
            x, y, lengths = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            nll = mdn_loss(y, y_pred, lengths, self.num_mixture_components)
            # todo: @larkin - look at regularization?
            loss = nll

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.nll_tracker.update_state(nll)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result(), "nll": self.nll_tracker.result()}

    @property
    def output_shape(self):
        return (None, None, self.num_mixture_components * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + 1)

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
    A similar implementation to the previous model,
    but now we're throwing the good old attention mechanism back into the mix.
    """

    def __init__(
        self,
        units: int = NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
        num_layers: int = NUM_LSTM_HIDDEN_LAYERS,
        num_mixture_components: int = NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
        num_chars: int = ALPHABET_SIZE,
        num_attention_gaussians: int = NUM_ATTENTION_GAUSSIAN_COMPONENTS,
        gradient_clip_value: float = GRADIENT_CLIP_VALUE,
        enable_mdn_regularization: bool = False,
        debug=False,
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
        self.debug = debug

        # metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.nll_tracker = tf.keras.metrics.Mean(name="nll")
        self.eos_accuracy_tracker = tf.keras.metrics.Mean(name="eos_accuracy")
        self.eos_prob_tracker = tf.keras.metrics.Mean(name="eos_prob")

    def call(
        self, inputs: Dict[str, tf.Tensor], training: Optional[bool] = None, mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        input_strokes = inputs["input_strokes"]
        input_chars = inputs["input_chars"]
        input_char_lens = inputs["input_char_lens"]

        # one-hot encode the character sequence and set RNN cell attributes
        char_seq_one_hot = tf.one_hot(input_chars, depth=self.num_chars)
        self.attention_rnn_cell.char_seq_one_hot = char_seq_one_hot
        self.attention_rnn_cell.char_seq_len = input_char_lens

        # initial states
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

        # then through our RNN (which wraps stacked LSTM cells + attention mechanism)
        # and then through our MDN layer
        outputs = self.rnn_layer(input_strokes, initial_state=initial_states_list, training=training)
        final_output = self.mdn_layer(outputs)
        return final_output

    @property
    def metrics(self):
        return [self.loss_tracker, self.nll_tracker, self.eos_accuracy_tracker, self.eos_prob_tracker]

    def train_step(self, data: Tuple[Dict[str, tf.Tensor], tf.Tensor]) -> Dict[str, tf.Tensor]:
        inputs, y_true = data
        target_stroke_lens = inputs["target_stroke_lens"]
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            nll = mdn_loss(y_true, y_pred, target_stroke_lens, self.num_mixture_components)
            reg = tf.add_n(self.losses) if self.losses else 0.0
            loss = nll + reg

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        eos_logits = y_pred[:, :, -1]  # Last column contains EOS logits
        eos_probs = tf.sigmoid(eos_logits)
        eos_predictions = tf.round(eos_probs)
        eos_targets = y_true[:, :, 2]  # EOS is the 3rd column (index 2)

        mask = tf.sequence_mask(target_stroke_lens, maxlen=tf.shape(y_true)[1], dtype=tf.float32)
        masked_correct = tf.cast(tf.equal(eos_predictions, eos_targets), tf.float32) * mask
        eos_accuracy = tf.reduce_sum(masked_correct) / tf.maximum(tf.reduce_sum(mask), 1.0)
        mean_eos_prob = tf.reduce_sum(eos_probs * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)

        # update metrics
        self.nll_tracker.update_state(nll)
        self.loss_tracker.update_state(loss)
        self.eos_accuracy_tracker.update_state(eos_accuracy)
        self.eos_prob_tracker.update_state(mean_eos_prob)

        return {
            "loss": self.loss_tracker.result(),
            "nll": self.nll_tracker.result(),
            "eos_accuracy": self.eos_accuracy_tracker.result(),
            "eos_prob": self.eos_prob_tracker.result(),
        }

    def build(self, inputs_by_name_shape: Union[Dict[str, Tuple[int, ...]], Tuple[int, ...]]) -> None:
        # Input here is going to look like:
        # input_shape {'input_strokes': (None, 567, 3), 'input_chars': (None, 30), 'input_char_lens': (None,)}
        if isinstance(inputs_by_name_shape, dict):
            strokes_input_shape = inputs_by_name_shape["input_strokes"]
        else:
            # Handle the case where TensorFlow passes a TensorShape directly
            strokes_input_shape = inputs_by_name_shape

        mdn_input_shape = tf.TensorShape([strokes_input_shape[0], strokes_input_shape[1], self.units])
        self.mdn_layer.build(mdn_input_shape)
        self.rnn_layer.build(tf.TensorShape(strokes_input_shape))
        self._is_built = True

    @property
    def output_shape(self):
        return (None, None, self.num_mixture_components * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + 1)

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
