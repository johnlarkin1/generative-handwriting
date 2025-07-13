from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf
from alphabet import ALPHABET_SIZE
from constants import (
    BATCH_SIZE,
    GRADIENT_CLIP_VALUE,
    NUM_ATTENTION_GAUSSIAN_COMPONENTS,
    NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
    NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
    NUM_LSTM_HIDDEN_LAYERS,
)

from model.attention_mechanism import AttentionMechanism
from model.attention_rnn_cell import AttentionRNNCell
from model.lstm_peephole_cell import LSTMPeepholeCell
from model.mixture_density_network import MixtureDensityLayer, mdn_loss


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


class DeepHandwritingSynthesisExplicitModel(tf.keras.Model):
    def __init__(
        self,
        batch_size=BATCH_SIZE,
        units=400,
        num_layers=3,
        num_mixture_components=20,
        num_chars=73,
        num_attention_gaussians=10,
        **kwargs,
    ):
        super(DeepHandwritingSynthesisExplicitModel, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.units = units
        self.num_layers = num_layers
        self.num_mixture_components = num_mixture_components
        self.num_chars = num_chars
        self.num_attention_gaussians = num_attention_gaussians
        self.lstm_cells = [LSTMPeepholeCell(units, idx) for idx in range(num_layers)]
        self.attention_mechanism = AttentionMechanism(
            num_gaussians=num_attention_gaussians,
            num_chars=num_chars,
            batch_size=batch_size,
        )
        self.mdn_layer = MixtureDensityLayer(num_mixture_components)

    def initial_state(self, batch_size):
        states = {
            f"s{i + 1}": self.lstm_cells[i].get_initial_state(batch_size=batch_size) for i in range(self.num_layers)
        }
        states["kappa"] = tf.zeros((batch_size, self.num_attention_gaussians))
        states["w"] = tf.zeros((batch_size, self.num_chars))
        return states

    def call(self, inputs, state, char_seq, char_seq_lengths, training=None):
        char_seq_one_hot = tf.one_hot(char_seq, depth=self.num_chars)
        outputs = []
        for step in range(inputs.shape[1]):
            x_t = inputs[:, step, :]
            s1_in = tf.concat([state["w"], x_t], axis=1)
            s1_out, s1_state = self.lstm_cells[0](s1_in, states=state["s1"])
            state["s1"] = s1_state

            attention_inputs = tf.concat([state["w"], x_t, s1_out], axis=1)
            w, kappa = self.attention_mechanism(
                attention_inputs,
                state["kappa"],
                char_seq_one_hot,
                char_seq_lengths,
            )
            state["w"] = w
            state["kappa"] = kappa

            s2_in = tf.concat([x_t, s1_out, w], axis=1)
            s2_out, s2_state = self.lstm_cells[1](s2_in, states=state["s2"])
            state["s2"] = s2_state

            s3_in = tf.concat([x_t, s2_out, w], axis=1)
            s3_out, s3_state = self.lstm_cells[2](s3_in, states=state["s3"])
            state["s3"] = s3_state

            outputs.append(s3_out)

        outputs = tf.stack(outputs, axis=1)
        final_output = self.mdn_layer(outputs)
        return final_output, state


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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.num_mixture_components = num_mixture_components
        self.num_chars = num_chars
        self.num_attention_gaussians = num_attention_gaussians
        self.gradient_clip_value = gradient_clip_value
        self.lstm_cells = [LSTMPeepholeCell(units, idx) for idx in range(num_layers)]
        self.attention_mechanism = AttentionMechanism(num_gaussians=num_attention_gaussians, num_chars=num_chars)
        self.attention_rnn_cell = AttentionRNNCell(self.lstm_cells, self.attention_mechanism, self.num_chars)
        self.rnn_layer = tf.keras.layers.RNN(self.attention_rnn_cell, return_sequences=True)
        self.mdn_layer = MixtureDensityLayer(num_mixture_components)

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

    def train_step(self, data: Tuple[Dict[str, tf.Tensor], tf.Tensor]) -> Dict[str, tf.Tensor]:
        inputs, y_true = data
        x_train_len = inputs["input_stroke_lens"]
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = mdn_loss(y_true, y_pred, x_train_len, self.num_mixture_components)

        gradients = tape.gradient(loss, self.trainable_variables)
        clipped_grads_and_vars = [
            (tf.clip_by_value(g, -1 * self.gradient_clip_value, self.gradient_clip_value), v_)
            for g, v_ in zip(gradients, self.trainable_variables)
        ]

        self.optimizer.apply_gradients(clipped_grads_and_vars)
        # We only really care about the log likelihood loss
        # here as tracking. The other metrics are not as important
        # Doing anything like comparing y_true and y_pred is not
        # really useful here because the MDN output shape
        # is not the same as the y_true shape.
        return {"loss": loss}

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
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DeepHandwritingSynthesisModel":
        return cls(**config)
