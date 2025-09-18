import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class AttentionMechanism(tf.keras.layers.Layer):
    """
    Attention mechanism for the handwriting synthesis model.
    This is a version of the attention mechanism used in
    the original paper by Alex Graves. It uses a Gaussian
    window to focus on different parts of the character sequence
    at each time step.
    """

    def __init__(self, num_gaussians, num_chars, name="attention", kappa_scale=1 / 25.0, **kwargs) -> None:
        super(AttentionMechanism, self).__init__(**kwargs)
        self.num_gaussians = num_gaussians
        self.num_chars = num_chars
        self.name_mod = name
        self.kappa_scale = float(kappa_scale)

    def build(self, input_shape):
        graves_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.075)
        window_bias_initializer = tf.keras.initializers.TruncatedNormal(mean=-3.0, stddev=0.25)

        self.dense_attention = tf.keras.layers.Dense(
            units=3 * self.num_gaussians,
            activation=None,  # No activation - we'll apply exp manually
            kernel_initializer=graves_initializer,
            bias_initializer=window_bias_initializer,
            name=f"{self.name_mod}_dense",
        )
        super().build(input_shape)

    def call(self, inputs, prev_kappa, char_seq_one_hot, sequence_lengths):
        # Generate concatenated attention parameters - use raw outputs, apply exp manually
        raw = self.dense_attention(inputs)
        alpha_hat, beta_hat, kappa_hat = tf.split(raw, 3, axis=1)

        # Apply exp activation as in Graves eq. 48-51
        alpha = tf.exp(alpha_hat)
        beta = tf.exp(beta_hat)
        kappa = prev_kappa + tf.exp(kappa_hat) * self.kappa_scale

        # Tiling 'enum' across batch size and number of attention mixture components
        char_len = tf.shape(char_seq_one_hot)[1]
        batch_size = tf.shape(inputs)[0]
        u = tf.cast(tf.range(0, char_len), tf.float32)  # Shape: [char_len]
        u = tf.reshape(u, [1, 1, -1])  # Shape: [1, 1, char_len]
        u = tf.tile(u, [batch_size, self.num_gaussians, 1])  # Shape: [batch_size, num_gaussians, char_len]

        # gaussian window
        alpha = tf.expand_dims(alpha, axis=-1)  # Shape: [batch_size, num_gaussians, 1]
        beta = tf.expand_dims(beta, axis=-1)  # Shape: [batch_size, num_gaussians, 1]
        kappa = tf.expand_dims(kappa, axis=-1)  # Shape: [batch_size, num_gaussians, 1]

        # phi - attention weights with numerical stability
        # Clip the exponent to prevent overflow
        exponent = -beta * tf.square(kappa - u)
        exponent = tf.clip_by_value(exponent, -50.0, 50.0)
        phi = alpha * tf.exp(exponent)  # Shape: [batch_size, num_gaussians, char_len]
        phi = tf.reduce_sum(phi, axis=1)  # Sum over the gaussians: [batch_size, char_len]

        # sequence mask
        sequence_mask = tf.sequence_mask(sequence_lengths, maxlen=char_len, dtype=tf.float32)
        phi = phi * sequence_mask  # Apply mask to attention weights

        # normalize with stronger epsilon for numerical stability
        phi_sum = tf.reduce_sum(phi, axis=1, keepdims=True) + 1e-6
        phi = phi / phi_sum

        # window vec
        phi = tf.expand_dims(phi, axis=-1)  # Shape: [batch_size, char_len, 1]
        w = tf.reduce_sum(phi * char_seq_one_hot, axis=1)  # Shape: [batch_size, num_chars]

        return w, kappa[:, :, 0]  # Return updated kappa

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_gaussians": self.num_gaussians,
                "num_chars": self.num_chars,
                "name": self.name_mod,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class AttentionMechanismBatch(tf.keras.layers.Layer):
    """
    Deprecated. Does not make sense to use the attention
    mechanism in a batched manner. This is kept for reference
    """

    def __init__(self, num_gaussians, num_chars, name="attention", **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_gaussians = num_gaussians
        self.num_chars = num_chars
        self.name = name
        self.dense_attention = tf.keras.layers.Dense(
            units=3 * num_gaussians,
            activation="softplus",
            name=f"{self.name}_dense",
        )

    def call(self, inputs, prev_kappa, char_seq_one_hot, char_seq_len):
        # Assumes inputs is of shape [batch_size, timesteps, features]
        attention_params = self.dense_attention(inputs)
        alpha, beta, kappa_increment = tf.split(attention_params, 3, axis=-1)

        kappa = prev_kappa + kappa_increment / 25.0
        beta = tf.clip_by_value(beta, 0.01, np.inf)

        char_len = tf.shape(char_seq_one_hot)[1]
        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]

        enum = tf.reshape(tf.range(char_len), (1, 1, 1, char_len))
        u = tf.cast(tf.tile(enum, (batch_size, timesteps, self.num_gaussians, 1)), tf.float32)

        kappa = tf.expand_dims(kappa, axis=-1)
        alpha = tf.expand_dims(alpha, axis=-1)
        beta = tf.expand_dims(beta, axis=-1)

        phi = tf.reduce_sum(alpha * tf.exp(-tf.square(kappa - u) / beta), axis=2)
        phi = tf.expand_dims(phi, axis=3)

        # Create a sequence mask based on char_seq_len, expanded for broadcasting
        sequence_mask = tf.sequence_mask(char_seq_len, maxlen=char_len, dtype=tf.float32)
        # [batch_size, timesteps, 1, 1] for broadcasting
        sequence_mask = tf.expand_dims(sequence_mask, axis=-1)

        # Apply the mask to phi before summing over the characters
        # This ensures attention is only applied to valid parts of the sequence
        phi_masked = phi * sequence_mask

        # Calculate weighted sum of character sequences, applying the mask
        w = tf.reduce_sum(phi_masked * char_seq_one_hot, axis=2)

        return w, kappa
