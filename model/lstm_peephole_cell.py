from typing import Tuple
import tensorflow as tf


class LSTMCellWithPeepholes(tf.keras.layers.Layer):
    def __init__(
        self,
        num_lstm_units: int,
        clip_value=1.0,
        should_apply_peephole: bool = True,
        should_clip_gradients: bool = False,
        **kwargs,
    ) -> None:
        super(LSTMCellWithPeepholes, self).__init__(**kwargs)
        self.num_lstm_units = num_lstm_units
        self.clip_value = clip_value
        self.should_apply_peephole = should_apply_peephole
        self.should_clip_gradients = should_clip_gradients
        self.state_size = [self.num_lstm_units, self.num_lstm_units]

    def build(self, input_shape):
        """
        Building the LSTM cell with peephole connections.
        Basically defining all of the appropriate weights, biases, and peephole weights.
        """
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.num_lstm_units * 4), initializer="glorot_uniform", name="kernel"
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.num_lstm_units, self.num_lstm_units * 4), initializer="glorot_uniform", name="recurrent_kernel"
        )

        # Peephole weights for input, forget, and output gates
        self.peephole_weights = self.add_weight(
            shape=(self.num_lstm_units, 3), initializer="glorot_uniform", name="peephole_weights"
        )
        self.bias = self.add_weight(shape=(self.num_lstm_units * 4,), initializer="zeros", name="bias")

        # Apparently - and again struggles of Tensorflow - this is imperative because if you define
        # a built method then it's more or less lazy loaded, and so you need to set this
        # property so tensorflow knows that you're totally live for your Layer.
        self.built = True

    def call(self, inputs: tf.Tensor, states: Tuple[tf.Tensor, tf.Tensor]):
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
        h_tm1, c_tm1 = states
        # Compute linear combinations for input, forget, and output gates, and cell candidate
        # Basically the meat of eq, 7, 8, 9, 10
        z = tf.matmul(inputs, self.kernel) + tf.matmul(h_tm1, self.recurrent_kernel) + self.bias
        # Split the transformations into input, forget, cell, and output components
        i, f, c_candidate, o = tf.split(z, num_or_size_splits=4, axis=1)

        if self.should_apply_peephole:
            # Peephole connections before the activation functions
            i += c_tm1 * self.peephole_weights[:, 0]
            f += c_tm1 * self.peephole_weights[:, 1]

        # apply the activations - first step for eq. 7, eq. 8. eq. 10
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)

        if self.should_clip_gradients:
            # Per Graves, we need to apply gradient clipping to still fight off
            # the exploding derivative issue
            i = tf.clip_by_value(i, -self.clip_value, self.clip_value)
            f = tf.clip_by_value(f, -self.clip_value, self.clip_value)
            o = tf.clip_by_value(o, -self.clip_value, self.clip_value)
            c_candidate = tf.clip_by_value(c_candidate, -self.clip_value, self.clip_value)

        c_candidate = tf.tanh(c_candidate)
        c = f * c_tm1 + i * c_candidate
        if self.should_apply_peephole:
            # Adjusting the output gate with peephole connection after computing new cell state
            o += c * self.peephole_weights[:, 2]

        # Compute final hidden state -> Equation 11
        h = o * tf.tanh(c)
        return h, [h, c]
