from typing import Tuple

import tensorflow as tf


class LSTMPeepholeCell(tf.keras.layers.Layer):
    def __init__(
        self,
        num_lstm_units: int,
        idx: int,
        clip_value=10.0,  # from the paper
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

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None) -> tuple[tf.Tensor, tf.Tensor]:
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
        # ok this is in the paper:
        # > The network was trained with rmsprop, using the same parameters as in
        # > the previous section. The network was retrained with adaptive weight noise,
        # > initial standard deviation 0.075, and the output and LSTM gradients were again
        # > clipped in the range [−100,100] and [−10,10] respectively.
        graves_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.075)

        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.num_lstm_units * 4),
            initializer=graves_initializer,
            name=f"lstm_peephole_kernel{self.idx}",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.num_lstm_units, self.num_lstm_units * 4),
            initializer=graves_initializer,
            name=f"lstm_peephole_recurrent_kernel{self.idx}",
        )

        # peephole weights
        self.peephole_weights = self.add_weight(
            shape=(self.num_lstm_units, 3),
            initializer=graves_initializer,
            name=f"lstm_peephole_weights{self.idx}",
        )
        forget_bias_init = tf.constant_initializer(
            [0.0] * self.num_lstm_units + [1.0] * self.num_lstm_units + [0.0] * self.num_lstm_units * 2
        )
        self.bias = self.add_weight(
            shape=(self.num_lstm_units * 4,),
            initializer=forget_bias_init,
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
        # basically the meat of eq, 7, 8, 9, 10
        z = tf.matmul(inputs, self.kernel) + tf.matmul(h_tm1, self.recurrent_kernel) + self.bias
        i_lin, f_lin, g_lin, o_lin = tf.split(z, num_or_size_splits=4, axis=1)

        if self.should_apply_peephole:
            # this was a bug I had!! we want peephole before activation
            pw_i = tf.expand_dims(self.peephole_weights[:, 0], axis=0)
            pw_f = tf.expand_dims(self.peephole_weights[:, 1], axis=0)
            i_lin = i_lin + c_tm1 * pw_i
            f_lin = f_lin + c_tm1 * pw_f

        # Note: gradient clipping is now handled globally via clipnorm in optimizer

        # apply activation functions! throwback to biomedical signals
        i = tf.sigmoid(i_lin)
        f = tf.sigmoid(f_lin)
        g = tf.tanh(g_lin)
        c = f * c_tm1 + i * g

        if self.should_apply_peephole:
            pw_o = tf.expand_dims(self.peephole_weights[:, 2], axis=0)
            o_lin = o_lin + c * pw_o

        o = tf.sigmoid(o_lin)
        # final hidden state -> eq. 11
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
