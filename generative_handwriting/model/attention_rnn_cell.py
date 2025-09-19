import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class AttentionRNNCell(tf.keras.layers.Layer):
    def __init__(self, lstm_cells, attention_mechanism, num_chars, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.lstm_cells = lstm_cells  # Should be a list of 3 LSTM cells
        self.attention_mechanism = attention_mechanism
        self.num_chars = num_chars
        self.debug = debug

        # Flatten the state sizes
        self.state_size = []
        for cell in self.lstm_cells:
            self.state_size.extend([cell.state_size[0], cell.state_size[1]])
        self.state_size.extend(
            [
                tf.TensorShape([self.attention_mechanism.num_gaussians]),  # kappa
                tf.TensorShape([self.num_chars]),  # w (window vector)
            ]
        )
        # The one-hot encoding of the character sequence
        # will be set by the model before calling the cell
        self.char_seq_one_hot = None
        # Same for the length of the character sequence
        self.char_seq_len = None

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, states):
        x_t = inputs

        (
            s1_state_h,
            s1_state_c,
            s2_state_h,
            s2_state_c,
            s3_state_h,
            s3_state_c,
            kappa,
            w,
        ) = states

        # LSTM layer 1
        s1_in = tf.concat([w, x_t], axis=1)
        s1_out, [s1_state_h_new, s1_state_c_new] = self.lstm_cells[0](s1_in, [s1_state_h, s1_state_c])

        # Attention
        attention_inputs = tf.concat([w, x_t, s1_out], axis=1)
        w_new, kappa_new = self.attention_mechanism(attention_inputs, kappa, self.char_seq_one_hot, self.char_seq_len)

        # LSTM layer 2
        s2_in = tf.concat([x_t, s1_out, w_new], axis=1)
        s2_out, [s2_state_h_new, s2_state_c_new] = self.lstm_cells[1](s2_in, [s2_state_h, s2_state_c])

        # LSTM layer 3
        s3_in = tf.concat([x_t, s2_out, w_new], axis=1)
        s3_out, [s3_state_h_new, s3_state_c_new] = self.lstm_cells[2](s3_in, [s3_state_h, s3_state_c])

        # Preparing new states as a list to return
        new_states = [
            s1_state_h_new,
            s1_state_c_new,
            s2_state_h_new,
            s2_state_c_new,
            s3_state_h_new,
            s3_state_c_new,
            kappa_new,
            w_new,
        ]

        return s3_out, new_states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None:
            batch_size = tf.shape(inputs)[0] if inputs is not None else None
        if dtype is None:
            dtype = tf.float32

        initial_states = {}
        for idx, cell in enumerate(self.lstm_cells):
            h, c = cell.get_initial_state(batch_size=batch_size, dtype=dtype)
            initial_states[f"lstm_{idx}_h"] = h
            initial_states[f"lstm_{idx}_c"] = c

        initial_states["kappa"] = tf.zeros((batch_size, self.attention_mechanism.num_gaussians), dtype=dtype)
        initial_states["w"] = tf.zeros((batch_size, self.num_chars), dtype=dtype)
        return initial_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lstm_cells": self.lstm_cells,
                "attention_mechanism": self.attention_mechanism,
                "num_chars": self.num_chars,
            }
        )
        return config
