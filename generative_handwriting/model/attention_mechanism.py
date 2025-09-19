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

    def __init__(self, num_gaussians, num_chars, name="attention", debug=False, **kwargs) -> None:
        super(AttentionMechanism, self).__init__(**kwargs)
        self.num_gaussians = num_gaussians
        self.num_chars = num_chars
        self.name_mod = name
        self.debug = debug

    @staticmethod
    def _center_of_mass(phi: tf.Tensor) -> tf.Tensor:
        """
        phi: [B, L] (non-normalized, masked)
        returns: [B] center of mass in character index space (1..L)
        """
        L = tf.shape(phi)[1]
        u = tf.cast(tf.range(1, L + 1), tf.float32)  # [L], 1-based
        u = tf.reshape(u, [1, L])  # [1, L]
        Z = tf.reduce_sum(phi, axis=1, keepdims=True) + 1e-8  # [B,1]
        phi_norm = phi / Z  # [B, L]
        com = tf.reduce_sum(phi_norm * u, axis=1)  # [B]
        return com

    def build(self, input_shape):
        graves_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.075)
        window_bias_initializer = tf.keras.initializers.TruncatedNormal(mean=-1.0, stddev=0.25)

        self.dense_attention = tf.keras.layers.Dense(
            units=3 * self.num_gaussians,
            activation=None,  # No activation - we'll apply exp manually
            kernel_initializer=graves_initializer,
            bias_initializer=window_bias_initializer,
            name=f"{self.name_mod}_dense",
        )

        # learnable scale to help out how we're moving
        self.kappa_scale = self.add_weight(
            name="kappa_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(-2.5),  # exp(-2.5) ≈ 0.082
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, prev_kappa, char_seq_one_hot, sequence_lengths):
        raw = self.dense_attention(inputs)
        alpha_hat, beta_hat, kappa_hat = tf.split(raw, 3, axis=1)

        # apply exp activation as in Graves eq. 48-51
        alpha = tf.exp(alpha_hat)
        beta = tf.exp(beta_hat)
        delta_kappa = tf.exp(kappa_hat + self.kappa_scale)  # smaller initial step with learnable scale
        kappa = prev_kappa + delta_kappa  # Eq. 51 cumulative monotonic

        char_len = tf.shape(char_seq_one_hot)[1]
        batch_size = tf.shape(inputs)[0]
        u = tf.cast(tf.range(1, char_len + 1), tf.float32)  # Shape: [char_len] - 1-based indexing as in Graves
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
        phi = tf.reduce_sum(phi, axis=1)  # Sum over gaussians: [B, L]

        # sequence mask
        sequence_mask = tf.sequence_mask(sequence_lengths, maxlen=char_len, dtype=tf.float32)
        phi = phi * sequence_mask  # mask paddings
        # we don't normalize here - Graves calls that out specifically

        # window vec
        phi = tf.expand_dims(phi, axis=-1)  # Shape: [batch_size, char_len, 1]
        w = tf.reduce_sum(phi * char_seq_one_hot, axis=1)  # Shape: [batch_size, num_chars]
        return w, kappa[:, :, 0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_gaussians": self.num_gaussians,
                "num_chars": self.num_chars,
                "name": self.name_mod,
                "debug": self.debug,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
