import tensorflow as tf
import numpy as np

from constants import NUM_MIXTURE_COMPONENTS_PER_COMPONENT, EPSILON


class MDNLayer(tf.keras.layers.Layer):
    def __init__(self, num_components, **kwargs):
        super(MDNLayer, self).__init__(**kwargs)
        self.num_components = num_components
        # The number of parameters per mixture component: 2 means, 2 standard deviations, 1 correlation
        # Plus 1 for the mixture weights and 1 for the end-of-stroke probability
        self.output_dim = num_components * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + 1

    def build(self, input_shape):
        # Weights for mixture weights
        self.W_pi = self.add_weight(
            name="W_pi", shape=(input_shape[-1], self.num_components), initializer="uniform", trainable=True
        )
        # Weights for means
        self.W_mu = self.add_weight(
            name="W_mu", shape=(input_shape[-1], self.num_components * 2), initializer="uniform", trainable=True
        )
        # Weights for standard deviations
        self.W_sigma = self.add_weight(
            name="W_sigma",
            shape=(input_shape[-1], self.num_components * 2),
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),  # Small stddev
            trainable=True,
        )
        # Weights for correlation coefficients
        self.W_rho = self.add_weight(
            name="W_rho", shape=(input_shape[-1], self.num_components), initializer="uniform", trainable=True
        )
        # Weights for end-of-stroke probability
        self.W_eos = self.add_weight(name="W_eos", shape=(input_shape[-1], 1), initializer="uniform", trainable=True)
        # Bias for mixture weights
        self.b_pi = self.add_weight(name="b_pi", shape=(self.num_components,), initializer="zeros", trainable=True)
        # Bias for means
        self.b_mu = self.add_weight(name="b_mu", shape=(self.num_components * 2,), initializer="zeros", trainable=True)
        # Bias for standard deviations
        self.b_sigma = self.add_weight(
            name="b_sigma", shape=(self.num_components * 2,), initializer="zeros", trainable=True
        )
        # Bias for correlation coefficients
        self.b_rho = self.add_weight(name="b_rho", shape=(self.num_components,), initializer="zeros", trainable=True)
        # Bias for end-of-stroke probability
        self.b_eos = self.add_weight(name="b_eos", shape=(1,), initializer="zeros", trainable=True)

    def call(self, inputs):
        eps = 1e-8
        sigma_eps = 1e-4

        pi = tf.nn.softmax(tf.matmul(inputs, self.W_pi) + self.b_pi)

        mu = tf.matmul(inputs, self.W_mu) + self.b_mu
        mu1, mu2 = tf.split(mu, num_or_size_splits=2, axis=2)

        sigma = tf.exp(tf.matmul(inputs, self.W_sigma) + self.b_sigma)
        sigmas = tf.clip_by_value(sigma, sigma_eps, np.inf)
        sigma1, sigma2 = tf.split(sigmas, num_or_size_splits=2, axis=2)

        rho = tf.tanh(tf.matmul(inputs, self.W_rho) + self.b_rho)
        rho = tf.clip_by_value(rho, eps - 1.0, 1.0 - eps)

        eos = tf.sigmoid(tf.matmul(inputs, self.W_eos) + self.b_eos)
        eos = tf.clip_by_value(eos, eps, 1.0 - eps)
        eos = tf.reshape(eos, [-1, inputs.shape[1], 1])

        outputs = tf.concat([pi, mu1, mu2, sigma1, sigma2, rho, eos], axis=2)
        return outputs


def mdn_loss(y_true, y_pred, stroke_lengths, num_components, eps=1e-8):
    """Calculate the mixture density loss with masking for valid sequence lengths.

    Args:
    - y_true: The true next points in the sequence, with shape [batch_size, seq_length, 3].
    - y_pred: The concatenated MDN outputs, with shape [batch_size, seq_length, num_components * 6 + 1].
    - stroke_lengths: The actual lengths of each sequence in the batch, with shape [batch_size].
    - num_components: The number of mixture components.

    Returns:
    - The calculated loss.
    """

    out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_rho, out_eos = tf.split(
        y_pred,
        [num_components] * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + [1],
        axis=-1,
    )

    x_data, y_data, eos_data = tf.split(y_true, [1, 1, 1], axis=-1)

    norm = 1.0 / (2 * np.pi * out_sigma1 * out_sigma2 * tf.sqrt(1 - tf.square(out_rho)))
    Z = (
        tf.square((x_data - out_mu1) / (out_sigma1))
        + tf.square((y_data - out_mu2) / (out_sigma2))
        - 2 * out_rho * (x_data - out_mu1) * (y_data - out_mu2) / (out_sigma1 * out_sigma2)
    )

    exp = -Z / (2 * (1 - tf.square(out_rho)))
    gaussian_likelihoods = tf.exp(exp) * norm
    gmm_likelihood = tf.reduce_sum(out_pi * gaussian_likelihoods, axis=2)
    gmm_likelihood = tf.clip_by_value(gmm_likelihood, eps, np.inf)

    bernoulli_likelihood = tf.squeeze(tf.where(tf.equal(tf.ones_like(eos_data), eos_data), out_eos, 1 - out_eos))

    nll = -(tf.math.log(gmm_likelihood) + tf.math.log(bernoulli_likelihood))

    # Create a mask for valid sequence lengths
    max_len = tf.shape(y_true)[1]
    mask = tf.sequence_mask(stroke_lengths, maxlen=max_len, dtype=tf.float32)
    mask = tf.tile(tf.expand_dims(mask, 2), [1, 1, 3])  # Ensure the mask has the same shape as nll

    # Apply the mask to the negative log-likelihood
    masked_nll = nll * mask
    masked_nll = tf.where(tf.not_equal(mask, 0), masked_nll, tf.zeros_like(masked_nll))

    # Calculate the loss, considering only the valid parts of each sequence
    loss = tf.reduce_sum(masked_nll) / tf.reduce_sum(mask)
    return loss
