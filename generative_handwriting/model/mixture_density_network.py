import numpy as np
import tensorflow as tf
from constants import NUM_MIXTURE_COMPONENTS_PER_COMPONENT


class MixtureDensityLayer(tf.keras.layers.Layer):
    def __init__(self, num_components, name="mdn", temperature=1.0, **kwargs):
        super(MixtureDensityLayer, self).__init__(name=name, **kwargs)
        self.num_components = num_components
        # The number of parameters per mixture component: 2 means, 2 standard deviations, 1 correlation
        # Plus 1 for the mixture weights and 1 for the end-of-stroke probability
        self.output_dim = num_components * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + 1
        self.mod_name = name
        self.temperature = temperature

    def build(self, input_shape):
        # Weights for mixture weights
        # These will be determined dynamically in call()
        self.input_units = input_shape[-1]
        self.W_pi = self.add_weight(
            name=f"{self.mod_name}_W_pi",
            shape=(input_shape[-1], self.num_components),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )
        # Weights for means
        self.W_mu = self.add_weight(
            name=f"{self.mod_name}_W_mu",
            shape=(input_shape[-1], self.num_components * 2),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            trainable=True,
        )
        # Weights for standard deviations
        self.W_sigma = self.add_weight(
            name=f"{self.mod_name}_W_sigma",
            shape=(input_shape[-1], self.num_components * 2),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            trainable=True,
        )
        # Weights for correlation coefficients
        self.W_rho = self.add_weight(
            name=f"{self.mod_name}_W_rho",
            shape=(input_shape[-1], self.num_components),
            initializer="uniform",
            trainable=True,
        )
        # Weights for end-of-stroke probability
        self.W_eos = self.add_weight(
            name=f"{self.mod_name}_W_eos",
            shape=(input_shape[-1], 1),
            initializer="uniform",
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
        self.b_eos = self.add_weight(name=f"{self.mod_name}_b_eos", shape=(1,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def _compute_regularization(self, pi, sigma, rho):
        """Add regularization to prevent degenerate solutions."""
        # L2 regularization on sigmas to prevent them from getting too large
        sigma_reg = 0.01 * tf.reduce_mean(tf.square(tf.math.log(sigma)))

        # Regularization to encourage diverse mixture components
        pi_entropy = -tf.reduce_mean(tf.reduce_sum(pi * tf.math.log(pi + 1e-6), axis=-1))

        # Correlation regularization to prevent components from becoming too correlated
        rho_reg = 0.01 * tf.reduce_mean(tf.square(rho))

        return sigma_reg - 0.1 * pi_entropy + rho_reg

    def call(self, inputs, training=None):
        sigma_eps = 1e-2  # Increased from 1e-3 for better stability
        sigma_max = 100.0  # Add upper bound to prevent extreme values
        temperature = 1.0 if not training else self.temperature

        pi_logits = tf.matmul(inputs, self.W_pi) + self.b_pi
        pi = tf.nn.softmax(pi_logits / temperature, axis=-1)

        mu = tf.matmul(inputs, self.W_mu) + self.b_mu
        mu1, mu2 = tf.split(mu, num_or_size_splits=2, axis=2)

        log_sigma = tf.matmul(inputs, self.W_sigma) + self.b_sigma
        # Clip log_sigma to prevent extreme values
        log_sigma = tf.clip_by_value(log_sigma, -10.0, 10.0)
        sigma = tf.exp(log_sigma)
        sigmas = tf.clip_by_value(sigma, sigma_eps, sigma_max)
        sigma1, sigma2 = tf.split(sigmas, num_or_size_splits=2, axis=2)

        rho_raw = tf.matmul(inputs, self.W_rho) + self.b_rho
        rho = tf.tanh(rho_raw) * 0.95  # More conservative to prevent numerical issues

        reg_loss = self._compute_regularization(pi, sigma, rho)
        self.add_loss(reg_loss)

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
    # Split the predictions
    out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_rho, out_eos = tf.split(
        y_pred,
        [num_components] * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + [1],
        axis=2,
    )

    # Split the targets
    x_data, y_data, eos_data = tf.split(y_true, [1, 1, 1], axis=-1)

    # Ensure sigma values are properly bounded
    out_sigma1 = tf.clip_by_value(out_sigma1, 1e-2, 100.0)
    out_sigma2 = tf.clip_by_value(out_sigma2, 1e-2, 100.0)

    # Ensure rho is properly bounded to avoid numerical issues
    out_rho = tf.clip_by_value(out_rho, -0.95, 0.95)

    # Ensure pi values are properly normalized and bounded
    out_pi = tf.clip_by_value(out_pi, eps, 1.0 - eps)
    out_pi = out_pi / tf.reduce_sum(out_pi, axis=-1, keepdims=True)

    # Calculate log probabilities for the bivariate normal distribution
    # Log of normalization constant with safer log operations
    log_norm = -tf.math.log(2 * np.pi) - tf.math.log(out_sigma1) - tf.math.log(out_sigma2)

    # Handle correlation term in log space with bounds check
    rho_squared = tf.square(out_rho)
    rho_term = tf.math.log(tf.maximum(1.0 - rho_squared, eps))
    log_norm = log_norm - 0.5 * rho_term

    # Calculate Z-score terms with clipping to prevent extreme values
    z_1 = tf.clip_by_value((x_data - out_mu1) / out_sigma1, -10.0, 10.0)
    z_2 = tf.clip_by_value((y_data - out_mu2) / out_sigma2, -10.0, 10.0)
    z_12 = z_1 * z_2

    # Calculate exponent term in log space with protection against division by small values
    denominator = tf.maximum(1.0 - rho_squared, eps)
    exponent = -0.5 / denominator * (tf.square(z_1) + tf.square(z_2) - 2 * out_rho * z_12)

    # Clip exponent to prevent overflow/underflow
    exponent = tf.clip_by_value(exponent, -50.0, 50.0)

    # Complete log probability for each component
    log_probs = log_norm + exponent

    # Calculate log mixture probabilities using log-sum-exp trick
    log_pi = tf.math.log(out_pi)
    weighted_log_probs = log_pi + log_probs

    # Use stable log-sum-exp
    max_weighted = tf.reduce_max(weighted_log_probs, axis=2, keepdims=True)
    log_mixture = max_weighted + tf.math.log(tf.reduce_sum(
        tf.exp(weighted_log_probs - max_weighted), axis=2, keepdims=True
    ))
    log_mixture = tf.squeeze(log_mixture, axis=2)

    # Calculate bernoulli log likelihood for end-of-stroke using logits
    eos_logit = out_eos  # rename for clarity - this is now logits, not probabilities
    bernoulli_nll = tf.nn.sigmoid_cross_entropy_with_logits(labels=eos_data, logits=eos_logit)
    log_bernoulli = -tf.squeeze(bernoulli_nll)  # negative because our nll is -(log_mixture + log_bernoulli)

    # Combine log likelihoods
    total_log_likelihood = log_mixture + log_bernoulli

    # Check for NaN/Inf and replace with large finite value
    total_log_likelihood = tf.where(
        tf.logical_or(tf.math.is_nan(total_log_likelihood), tf.math.is_inf(total_log_likelihood)),
        tf.constant(-50.0, dtype=total_log_likelihood.dtype),
        total_log_likelihood
    )

    nll = -total_log_likelihood

    # Apply masking if stroke lengths are provided
    if stroke_lengths is not None:
        max_len = tf.shape(y_true)[1]
        mask = tf.sequence_mask(stroke_lengths, maxlen=max_len, dtype=tf.float32)

        # Mask the negative log-likelihood
        masked_nll = nll * mask
        masked_nll = tf.where(tf.not_equal(mask, 0), masked_nll, tf.zeros_like(masked_nll))

        # Return average loss over valid sequence steps
        return tf.reduce_sum(masked_nll) / tf.maximum(tf.reduce_sum(mask), 1.0)

    return tf.reduce_mean(nll)
