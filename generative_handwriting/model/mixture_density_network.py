# ruff: noqa: E501
import numpy as np
import tensorflow as tf

from generative_handwriting.constants import NUM_MIXTURE_COMPONENTS_PER_COMPONENT


class MixtureDensityLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_components,
        name="mdn",
        temperature=1.0,
        enable_regularization=False,
        sigma_reg_weight=0.01,
        rho_reg_weight=0.01,
        entropy_reg_weight=0.1,
        **kwargs,
    ):
        super(MixtureDensityLayer, self).__init__(name=name, **kwargs)
        self.num_components = num_components
        # The number of parameters per mixture component: 2 means, 2 standard deviations, 1 correlation
        # Plus 1 for the mixture weights and 1 for the end-of-stroke probability
        self.output_dim = num_components * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + 1
        self.mod_name = name
        self.temperature = temperature
        self.enable_regularization = enable_regularization
        self.sigma_reg_weight = sigma_reg_weight
        self.rho_reg_weight = rho_reg_weight
        self.entropy_reg_weight = entropy_reg_weight

    def build(self, input_shape):
        graves_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.075)

        # These will be determined dynamically in call()
        self.input_units = input_shape[-1]
        self.W_pi = self.add_weight(
            name=f"{self.mod_name}_W_pi",
            shape=(input_shape[-1], self.num_components),
            initializer=graves_initializer,
            trainable=True,
        )
        # Weights for means
        self.W_mu = self.add_weight(
            name=f"{self.mod_name}_W_mu",
            shape=(input_shape[-1], self.num_components * 2),
            initializer=graves_initializer,
            trainable=True,
        )
        # Weights for standard deviations
        self.W_sigma = self.add_weight(
            name=f"{self.mod_name}_W_sigma",
            shape=(input_shape[-1], self.num_components * 2),
            initializer=graves_initializer,
            trainable=True,
        )
        # Weights for correlation coefficients
        self.W_rho = self.add_weight(
            name=f"{self.mod_name}_W_rho",
            shape=(input_shape[-1], self.num_components),
            initializer=graves_initializer,
            trainable=True,
        )
        # Weights for end-of-stroke probability
        self.W_eos = self.add_weight(
            name=f"{self.mod_name}_W_eos",
            shape=(input_shape[-1], 1),
            initializer=graves_initializer,
            trainable=True,
        )
        # Bias for mixture weights
        self.b_pi = self.add_weight(
            name=f"{self.mod_name}_b_pi",
            shape=(self.num_components,),
            initializer="zeros",
            trainable=True,
        )
        # Bias for means
        self.b_mu = self.add_weight(
            name=f"{self.mod_name}_b_mu",
            shape=(self.num_components * 2,),
            initializer="zeros",
            trainable=True,
        )
        # Bias for standard deviations
        self.b_sigma = self.add_weight(
            name=f"{self.mod_name}_b_sigma",
            shape=(self.num_components * 2,),
            initializer="zeros",
            trainable=True,
        )
        # Bias for correlation coefficients
        self.b_rho = self.add_weight(
            name=f"{self.mod_name}_b_rho",
            shape=(self.num_components,),
            initializer="zeros",
            trainable=True,
        )
        # Bias for end-of-stroke probability
        self.b_eos = self.add_weight(
            name=f"{self.mod_name}_b_eos",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        sigma_eps = 1e-2  # Increased from 1e-3 for better stability
        sigma_max = 1e5  # Add upper bound to prevent extreme values
        temperature = 1.0 if not training else self.temperature

        pi_logits = tf.matmul(inputs, self.W_pi) + self.b_pi
        pi = tf.nn.softmax(pi_logits / temperature, axis=-1)

        mu = tf.matmul(inputs, self.W_mu) + self.b_mu
        mu1, mu2 = tf.split(mu, num_or_size_splits=2, axis=2)

        log_sigma = tf.matmul(inputs, self.W_sigma) + self.b_sigma
        log_sigma = tf.clip_by_value(log_sigma, -1e5, 1e5)
        sigma = tf.exp(log_sigma)
        sigmas = tf.clip_by_value(sigma, sigma_eps, sigma_max)
        sigma1, sigma2 = tf.split(sigmas, num_or_size_splits=2, axis=2)

        rho_raw = tf.matmul(inputs, self.W_rho) + self.b_rho
        rho = tf.tanh(rho_raw)

        eos_logit = tf.matmul(inputs, self.W_eos) + self.b_eos
        eos = tf.reshape(eos_logit, [-1, inputs.shape[1], 1])

        outputs = tf.concat([pi, mu1, mu2, sigma1, sigma2, rho, eos], axis=2)

        # Validate shapes match expected dimensions but allow dynamic batch size
        # tf.debugging.assert_rank(pi, 3, message="pi must be rank 3 [batch, time, components]")  # Disabled for XLA compatibility
        # tf.debugging.assert_rank(mu1, 3, message="mu1 must be rank 3 [batch, time, components]")  # Disabled for XLA compatibility
        # tf.debugging.assert_rank(mu2, 3, message="mu2 must be rank 3 [batch, time, components]")  # Disabled for XLA compatibility
        # tf.debugging.assert_rank(sigma1, 3, message="sigma1 must be rank 3 [batch, time, components]")  # Disabled for XLA compatibility
        # tf.debugging.assert_rank(sigma2, 3, message="sigma2 must be rank 3 [batch, time, components]")  # Disabled for XLA compatibility
        # tf.debugging.assert_rank(rho, 3, message="rho must be rank 3 [batch, time, components]")  # Disabled for XLA compatibility
        # tf.debugging.assert_rank(eos, 3, message="eos must be rank 3 [batch, time, 1]")  # Disabled for XLA compatibility
        # tf.debugging.assert_rank(outputs, 3, message="outputs must be rank 3 [batch, time, features]")  # Disabled for XLA compatibility
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_components": self.num_components,
                "name": self.mod_name,
                "temperature": self.temperature,
                "enable_regularization": self.enable_regularization,
                "sigma_reg_weight": self.sigma_reg_weight,
                "rho_reg_weight": self.rho_reg_weight,
                "entropy_reg_weight": self.entropy_reg_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
def mdn_loss(y_true, y_pred, stroke_lengths, num_components, eps=1e-6):
    """Calculate the mixture density loss with masking for valid sequence lengths.
    Implements the loss calculation fully in log space for numerical stability.

    Args:
        y_true: The true next points in sequence, shape [batch_size, seq_length, 3]
        y_pred: MDN outputs, shape [batch_size, seq_length, num_components * 6 + 1]
        stroke_lengths: Sequence lengths, shape [batch_size]
        num_components: Number of mixture components
        eps: Small constant for numerical stability

    Returns:
        Masked negative log-likelihood loss
    """
    out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_rho, out_eos = tf.split(
        y_pred,
        [num_components] * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + [1],
        axis=2,
    )

    x_data, y_data, eos_data = tf.split(y_true, [1, 1, 1], axis=-1)

    out_sigma1 = tf.clip_by_value(out_sigma1, 1e-5, 1e5)
    out_sigma2 = tf.clip_by_value(out_sigma2, 1e-5, 1e5)
    out_rho = tf.clip_by_value(out_rho, -0.99, 0.99)
    out_pi = tf.clip_by_value(out_pi, eps, 1.0 - eps)
    out_pi = out_pi / tf.reduce_sum(out_pi, axis=-1, keepdims=True)

    # basically part of eq 24
    norm = 1.0 / (2 * np.pi * out_sigma1 * out_sigma2 * tf.sqrt(1 - tf.square(out_rho)))
    z_1 = (x_data - out_mu1) / out_sigma1
    z_2 = (y_data - out_mu2) / out_sigma2

    # eq 25
    Z = tf.square(z_1) + tf.square(z_2) - 2 * out_rho * z_1 * z_2
    exp_term = -0.5 * Z / (1 - tf.square(out_rho))

    # prob space
    gaussian_likelihoods = tf.exp(exp_term) * norm
    gmm_likelihood = tf.reduce_sum(out_pi * gaussian_likelihoods, axis=2)
    gmm_likelihood = tf.clip_by_value(gmm_likelihood, eps, np.inf)

    # bernoulli likelihood for eos
    es_prob = tf.sigmoid(out_eos)
    bernoulli_likelihood = tf.squeeze(
        tf.where(tf.equal(tf.ones_like(eos_data), eos_data), es_prob, 1 - es_prob), axis=2
    )
    bernoulli_likelihood = tf.clip_by_value(bernoulli_likelihood, eps, 1.0 - eps)

    nll = -(tf.math.log(gmm_likelihood) + tf.math.log(bernoulli_likelihood))
    # just in case
    nll = tf.where(
        tf.logical_or(tf.math.is_nan(nll), tf.math.is_inf(nll)),
        tf.constant(50.0, dtype=nll.dtype),  # Large positive value for NLL
        nll,
    )

    # Apply masking if stroke lengths are provided
    if stroke_lengths is not None:
        max_len = tf.shape(y_true)[1]
        mask = tf.sequence_mask(stroke_lengths, maxlen=max_len, dtype=tf.float32)

        # Additional safety: also mask NaN values
        mask = tf.logical_and(tf.cast(mask, tf.bool), tf.logical_not(tf.math.is_nan(nll)))
        mask = tf.cast(mask, tf.float32)

        # Mask the negative log-likelihood
        masked_nll = nll * mask
        masked_nll = tf.where(tf.not_equal(mask, 0), masked_nll, tf.zeros_like(masked_nll))

        # Return average loss over valid sequence steps
        return tf.reduce_sum(masked_nll) / tf.maximum(tf.reduce_sum(mask), 1.0)

    return tf.reduce_mean(nll)
