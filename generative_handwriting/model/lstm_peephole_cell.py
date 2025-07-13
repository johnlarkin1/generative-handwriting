from typing import Tuple

import tensorflow as tf


class LSTMPeepholeCell(tf.keras.layers.Layer):
    def __init__(
        self,
        num_lstm_units: int,
        idx: int,
        clip_value=1.0,
        should_apply_peephole: bool = True,
        should_clip_gradients: bool = False,
        **kwargs,
    ) -> None:
        super(LSTMPeepholeCell, self).__init__(**kwargs)
        self.num_lstm_units = num_lstm_units
        self.clip_value = clip_value
        self.should_apply_peephole = should_apply_peephole
        self.should_clip_gradients = should_clip_gradients
        self.state_size = [
            tf.TensorShape([self.num_lstm_units]),
            tf.TensorShape([self.num_lstm_units]),
        ]
        self.output_size = tf.TensorShape([self.num_lstm_units])
        self.idx = idx + 1

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Returns an initial state consisting of zeros for both the hidden state and the cell state.
        """
        if batch_size is None:
            batch_size = tf.shape(inputs)[0] if inputs is not None else None
        if dtype is None:
            dtype = inputs.dtype if inputs is not None else tf.float32

        # (hidden state (h), cell state (c))
        return (
            tf.zeros([batch_size, self.num_lstm_units], dtype=dtype),
            tf.zeros([batch_size, self.num_lstm_units], dtype=dtype),
        )

    def build(self, input_shape):
        """
        Building the LSTM cell with peephole connections.
        Basically defining all of the appropriate weights, biases, and peephole weights.
        """
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.num_lstm_units * 4),
            initializer="glorot_uniform",
            name=f"lstm_peephole_kernel{self.idx}",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.num_lstm_units, self.num_lstm_units * 4),
            initializer="glorot_uniform",
            name=f"lstm_peephole_recurrent_kernel{self.idx}",
        )

        # Peephole weights for input, forget, and output gates
        self.peephole_weights = self.add_weight(
            shape=(self.num_lstm_units, 3),
            initializer="glorot_uniform",
            name=f"lstm_peephole_weights{self.idx}",
        )
        self.bias = self.add_weight(
            shape=(self.num_lstm_units * 4,),
            initializer="zeros",
            name=f"lstm_peephole_bias{self.idx}",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, state: Tuple[tf.Tensor, tf.Tensor]):
        """
        This is basically implementing Graves's equations on page 5
        https://www.cs.toronto.edu/~graves/preprint.pdf
        equations 5-11.

        From the paper,
        * sigma is the logistic sigmoid function
        * i -> input gate
        * f -> forget gate
        * o -> output gate
        * c -> cell state
        * W_{hi} - hidden-input gate matrix
        * W_{xo} - input-output gate matrix
        * W_{ci} - are diagonal
          + so element m in each gate vector only receives input from
          + element m of the cell vector
        """
        # Both of these are going to be shape (?, num_lstm_units)
        h_tm1, c_tm1 = state
        # Compute linear combinations for input, forget, and output gates, and cell candidate
        # Basically the meat of eq, 7, 8, 9, 10
        z = tf.matmul(inputs, self.kernel) + tf.matmul(h_tm1, self.recurrent_kernel) + self.bias
        # Split the transformations into input, forget, cell, and output components
        i, f, c_candidate, o = tf.split(z, num_or_size_splits=4, axis=1)

        if self.should_apply_peephole:
            # Peephole connections before the activation functions
            peephole_i = tf.expand_dims(self.peephole_weights[:, 0], axis=0)
            peephole_f = tf.expand_dims(self.peephole_weights[:, 1], axis=0)
            i += c_tm1 * peephole_i
            f += c_tm1 * peephole_f

        # apply the activations - first step for eq. 7, eq. 8. eq. 10
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)

        if self.should_clip_gradients:
            # Per Graves, we need to apply gradient clipping to still fight off
            # the exploding derivative issue. It's a bit weird
            # to do it here maybe so that's why this bool defaults to off.
            i = tf.clip_by_value(i, -self.clip_value, self.clip_value)
            f = tf.clip_by_value(f, -self.clip_value, self.clip_value)
            o = tf.clip_by_value(o, -self.clip_value, self.clip_value)
            c_candidate = tf.clip_by_value(c_candidate, -self.clip_value, self.clip_value)

        c_candidate = tf.tanh(c_candidate)
        c = f * c_tm1 + i * c_candidate
        if self.should_apply_peephole:
            # Adjusting the output gate with peephole connection after computing new cell state
            peephole_o = tf.expand_dims(self.peephole_weights[:, 2], axis=0)
            o += c * peephole_o

        # Compute final hidden state -> Equation 11
        h = o * tf.tanh(c)
        return h, [h, c]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_lstm_units": self.num_lstm_units,
                "clip_value": self.clip_value,
                "should_apply_peephole": self.should_apply_peephole,
                "should_clip_gradients": self.should_clip_gradients,
                "idx": self.idx,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
