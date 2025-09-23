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
        # The number of parameters per mixture component: 2 means, 2 standard deviations, 1 correlation, 1 weight , 1 for eos
        # so that's our constant num_mixture_components_per_component
        self.output_dim = num_components * NUM_MIXTURE_COMPONENTS_PER_COMPONENT + 1
        self.mod_name = name
        self.temperature = temperature
        self.enable_regularization = enable_regularization
        self.sigma_reg_weight = sigma_reg_weight
        self.rho_reg_weight = rho_reg_weight
        self.entropy_reg_weight = entropy_reg_weight

    def build(self, input_shape):
        graves_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.075)

        self.input_units = input_shape[-1]
        # weights
        self.W_pi = self.add_weight(
            name=f"{self.mod_name}_W_pi",
            shape=(input_shape[-1], self.num_components),
            initializer=graves_initializer,
            trainable=True,
        )
        self.W_mu = self.add_weight(
            name=f"{self.mod_name}_W_mu",
            shape=(input_shape[-1], self.num_components * 2),
            initializer=graves_initializer,
            trainable=True,
        )
        self.W_sigma = self.add_weight(
            name=f"{self.mod_name}_W_sigma",
            shape=(input_shape[-1], self.num_components * 2),
            initializer=graves_initializer,
            trainable=True,
        )
        self.W_rho = self.add_weight(
            name=f"{self.mod_name}_W_rho",
            shape=(input_shape[-1], self.num_components),
            initializer=graves_initializer,
            trainable=True,
        )
        self.W_eos = self.add_weight(
            name=f"{self.mod_name}_W_eos",
            shape=(input_shape[-1], 1),
            initializer=graves_initializer,
            trainable=True,
        )
        # biases
        self.b_pi = self.add_weight(
            name=f"{self.mod_name}_b_pi",
            shape=(self.num_components,),
            initializer="zeros",
            trainable=True,
        )
        self.b_mu = self.add_weight(
            name=f"{self.mod_name}_b_mu",
            shape=(self.num_components * 2,),
            initializer="zeros",
            trainable=True,
        )
        self.b_sigma = self.add_weight(
            name=f"{self.mod_name}_b_sigma",
            shape=(self.num_components * 2,),
            initializer="zeros",
            trainable=True,
        )
        self.b_rho = self.add_weight(
            name=f"{self.mod_name}_b_rho",
            shape=(self.num_components,),
            initializer="zeros",
            trainable=True,
        )
        self.b_eos = self.add_weight(
            name=f"{self.mod_name}_b_eos",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        temperature = 1.0 if not training else self.temperature

        pi_logits = tf.matmul(inputs, self.W_pi) + self.b_pi
        pi = tf.nn.softmax(pi_logits / temperature, axis=-1)  # [B, T, K]
        mu = tf.matmul(inputs, self.W_mu) + self.b_mu  # [B, T, 2K]
        mu1, mu2 = tf.split(mu, 2, axis=2)

        # log sigma more stable
        log_sigma = tf.matmul(inputs, self.W_sigma) + self.b_sigma  # [B, T, 2K]
        log_sigma = tf.clip_by_value(log_sigma, -7.0, 4.0)  # σ ∈ [~1e-3, ~55]
        sigma = tf.exp(log_sigma)
        sigma1, sigma2 = tf.split(sigma, 2, axis=2)
        rho_raw = tf.matmul(inputs, self.W_rho) + self.b_rho
        rho = tf.tanh(rho_raw)  # (-1, 1)
        eos_logit = tf.matmul(inputs, self.W_eos) + self.b_eos

        return tf.concat([pi, mu1, mu2, sigma1, sigma2, rho, tf.reshape(eos_logit, [-1, inputs.shape[1], 1])], axis=2)

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
def mdn_loss(y_true, y_pred, stroke_lengths, num_components, eps=1e-8):
    """
    Mixture density negative log-likelihood computed fully in log-space.

    y_true: [B, T, 3]  -> (x, y, eos ∈ {0,1})
    y_pred: [B, T, 6*K + 1] -> (pi, mu1, mu2, sigma1, sigma2, rho, eos_logit)

    The log space change was because I was getting absolutely torched by the
    gradients when using the normal space.
    """
    out_pi, mu1, mu2, sigma1, sigma2, rho, eos_logits = tf.split(
        y_pred,
        [num_components] * 6 + [1],
        axis=2,
    )

    x, y, eos_targets = tf.split(y_true, [1, 1, 1], axis=-1)

    sigma1 = tf.clip_by_value(sigma1, 1e-3, 1e5)
    sigma2 = tf.clip_by_value(sigma2, 1e-3, 1e7)
    rho = tf.clip_by_value(rho, -0.95, 0.95)

    log_2pi = tf.constant(np.log(2.0 * np.pi), dtype=y_pred.dtype)
    one_minus_rho2 = 1.0 - tf.square(rho)
    log_one_minus_rho2 = tf.math.log(one_minus_rho2)
    z1 = (x - mu1) / sigma1
    z2 = (y - mu2) / sigma2

    quad = tf.square(z1) + tf.square(z2) - 2.0 * rho * z1 * z2
    log_norm = -(log_2pi + tf.math.log(sigma1) + tf.math.log(sigma2) + 0.5 * log_one_minus_rho2)
    log_gauss = log_norm - 0.5 * quad / one_minus_rho2  # [B, T, K]

    # log mixture via log-sum-exp
    log_pi = tf.math.log(tf.clip_by_value(out_pi, eps, 1.0))  # [B, T, K]
    log_gmm = tf.reduce_logsumexp(log_pi + log_gauss, axis=-1)  # [B, T]

    # bce (bernoulli cross entropy) to help out with stability
    eos_nll = tf.nn.sigmoid_cross_entropy_with_logits(labels=eos_targets, logits=eos_logits)  # [B, T, 1]
    eos_nll = tf.squeeze(eos_nll, axis=-1)  # [B, T]

    # Total per-timestep NLL
    nll = -log_gmm + eos_nll  # [B, T]

    # Mask by sequence lengths if provided
    if stroke_lengths is not None:
        mask = tf.sequence_mask(stroke_lengths, maxlen=tf.shape(y_true)[1], dtype=nll.dtype)
        nll = nll * mask
        denom = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(nll) / denom

    return tf.reduce_mean(nll)
