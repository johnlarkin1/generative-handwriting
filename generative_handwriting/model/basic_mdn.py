import numpy as np
import tensorflow as tf
from constants import EPSILON


class BasicMixtureDensityNetwork(tf.keras.Model):
    def __init__(self, num_mixtures, hidden_units):
        super(BasicMixtureDensityNetwork, self).__init__()
        self.num_mixtures = num_mixtures
        self.hidden_units = hidden_units
        self.hidden_layer = tf.keras.layers.Dense(hidden_units, activation="relu")
        # Each mixture component has parameters for mean (x, y), variance (x, y), and mixture weight.
        number_of_params_per_mixture = 2 + 2 + 1
        expected_output_size = num_mixtures * number_of_params_per_mixture
        self.mdn_layer = tf.keras.layers.Dense(expected_output_size)

    def call(self, inputs):
        hidden_outputs = self.hidden_layer(inputs)
        mdn_outputs = self.mdn_layer(hidden_outputs)
        return mdn_outputs


def gaussian_pdf(y, mean, std):
    """
    Compute the probability density of y in the Gaussian distribution
    defined by mean and std. This assumes independence between dimensions
    to simplify the computation.
    """
    # Assuming y, mean, and std have shapes [batch_size, num_mixtures, 2]
    # where the last dimension is for the (x, y) coordinates.
    # Sum over the x and y dimensions.
    norm = tf.reduce_sum((y - mean) ** 2 / (std**2 + EPSILON), axis=2)
    return tf.exp(-0.5 * norm) / (2.0 * np.pi * tf.reduce_prod(std, axis=2) + EPSILON)


def mdn_loss_function(actual, model_outputs, num_mixtures):
    # Reshape model outputs to separate mixture components
    output_shape = model_outputs.shape[-1] // num_mixtures
    reshaped_outputs = tf.reshape(model_outputs, [-1, num_mixtures, output_shape])

    # Extract parameters
    means = reshaped_outputs[:, :, :2]  # Means for x and y
    stds = tf.exp(reshaped_outputs[:, :, 2:4])  # Log standard deviations for stability
    weights = tf.nn.softmax(reshaped_outputs[:, :, 4], axis=-1)  # Mixture weights

    # Expand actual values for PDF calculation
    expanded_actual = tf.expand_dims(actual, 1)  # Shape: [batch_size, 1, 2]

    # Calculate the PDF for each mixture component
    pdf_vals = gaussian_pdf(expanded_actual, means, stds)  # Shape: [batch_size, num_mixtures]

    # Weighted sum of PDFs for each mixture component
    weighted_pdf = tf.reduce_sum(weights * pdf_vals, axis=-1)  # Shape: [batch_size]

    # Negative log-likelihood
    loss = -tf.reduce_mean(tf.math.log(weighted_pdf + EPSILON))

    return loss
