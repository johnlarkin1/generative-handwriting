import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class AttentionMechanism(tf.keras.layers.Layer):
    """
    Attention mechanism for the handwriting synthesis model.
    This is a version of the attention mechanism used in
    the original paper by Alex Graves. It uses a Gaussian
    window to focus on different parts of the character sequence
    at each time step.

    See section: 5.0 / 5.1
    """

    def __init__(self, num_gaussians, num_chars, name="attention", debug=False, **kwargs) -> None:
        super(AttentionMechanism, self).__init__(**kwargs)
        self.num_gaussians = num_gaussians
        self.num_chars = num_chars
        self.name_mod = name
        self.debug = debug

    def build(self, input_shape):
        # so Graves didn't really say in his paper about attention mechanism initialization
        # so I'm going to match svasquez's implementation
        variance_scaling_initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_avg", distribution="truncated_normal"
        )
        zero_bias_initializer = tf.keras.initializers.Zeros()

        # no activation, we'll apply manually
        self.dense_attention = tf.keras.layers.Dense(
            units=3 * self.num_gaussians,
            activation=None,
            kernel_initializer=variance_scaling_initializer,
            bias_initializer=zero_bias_initializer,
            name=f"{self.name_mod}_dense",
        )

        # learnable scale to help out how we're moving
        #   exp(-2.5) ≈ 0.082
        # which is kinda close to what svasquez had
        # with his static / 25
        self.kappa_scale = self.add_weight(
            name="kappa_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(-2.5),
            trainable=True,
        )
        super().build(input_shape)

    def call(
        self,
        inputs,  # shape: [batch_size, num_gaussians, 3]
        prev_kappa,  # shape: [batch_size, num_gaussians]
        char_seq_one_hot,  # shape: [batch_size, char_len, num_chars]
        sequence_lengths,  # shape: [batch_size]
    ) -> tuple[tf.Tensor, tf.Tensor]:
        raw = self.dense_attention(inputs)
        alpha_hat, beta_hat, kappa_hat = tf.split(raw, 3, axis=1)  # shape: [batch_size, num_gaussians, 1]

        # apply exp activation as in Graves eq. 48-51 with numerical safety
        alpha = tf.exp(tf.clip_by_value(alpha_hat, -8.0, 8.0))
        beta = tf.exp(tf.clip_by_value(beta_hat, -8.0, 8.0))
        # e^(kappa_hat + kappa_scale) = e^(kappa_hat) * e^(kappa_scale)
        delta_kappa = tf.exp(tf.clip_by_value(kappa_hat + self.kappa_scale, -8.0, 5.0))
        kappa = prev_kappa + delta_kappa  # Eq. 51 cumulative monotonic

        char_len = tf.shape(char_seq_one_hot)[1]
        batch_size = tf.shape(inputs)[0]
        u = tf.cast(tf.range(1, char_len + 1), tf.float32)  # shape: [char_len] -> 1-based indexing as in Graves
        u = tf.reshape(u, [1, 1, -1])  # shape: [1, 1, char_len]
        u = tf.tile(u, [batch_size, self.num_gaussians, 1])  # shape: [batch_size, num_gaussians, char_len]

        # gaussian window
        alpha = tf.expand_dims(alpha, axis=-1)  # shape: [batch_size, num_gaussians, 1]
        beta = tf.expand_dims(beta, axis=-1)  # shape: [batch_size, num_gaussians, 1]
        kappa = tf.expand_dims(kappa, axis=-1)  # shape: [batch_size, num_gaussians, 1]

        # phi - attention weights with numerical stability
        exponent = -beta * tf.square(kappa - u)
        # Keep exponent non-positive and bounded (Gaussian exponent should be ≤ 0)
        exponent = tf.clip_by_value(exponent, -50.0, 0.0)
        phi = alpha * tf.exp(exponent)  # shape: [batch_size, num_gaussians, char_len]
        phi = tf.reduce_sum(phi, axis=1)  # Sum over gaussians: [B, L]

        # sequence mask with inf*0 safeguards
        sequence_mask = tf.sequence_mask(sequence_lengths, maxlen=char_len, dtype=tf.float32)
        phi = phi * sequence_mask  # mask paddings

        # Sanitize non-finites from inf*0 scenarios
        phi = tf.where(tf.math.is_finite(phi), phi, tf.zeros_like(phi))
        # we don't normalize here - Graves calls that out specifically!
        # > Note that the window mixture is not normalised
        # > and hence does not determine a probability distribution; however the window
        # > weight φ(t,u) can be loosely interpreted as the network's belief that it is writ-
        # > ing character cu at time t.
        # still section 5.1

        # window vec
        phi = tf.expand_dims(phi, axis=-1)  # shape: [batch_size, char_len, 1]
        w = tf.reduce_sum(phi * char_seq_one_hot, axis=1)  # shape: [batch_size, num_chars]

        # Final safeguard for window vector
        w = tf.where(tf.math.is_finite(w), w, tf.zeros_like(w))

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
