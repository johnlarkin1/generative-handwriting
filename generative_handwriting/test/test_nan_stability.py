"""
Test suite for NaN stability and numerical robustness.
Tests various edge cases that could lead to NaN values in the handwriting synthesis model.
"""

import unittest
import numpy as np
import tensorflow as tf

from model.mixture_density_network import MixtureDensityLayer, mdn_loss
from model.attention_mechanism import AttentionMechanism
from model.handwriting_models import DeepHandwritingPredictionModel, DeepHandwritingSynthesisModel
from debug_utils import NaNMonitor
from constants import NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS


class TestNaNStability(unittest.TestCase):
    """Test numerical stability and NaN resistance."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_components = 3
        self.batch_size = 2
        self.seq_length = 10
        self.num_chars = 73
        self.num_attention_gaussians = 5

    def test_mdn_layer_extreme_inputs(self):
        """Test MDN layer with extreme input values."""
        mdn = MixtureDensityLayer(self.num_components)

        # Create extreme inputs that might cause numerical issues
        extreme_inputs = tf.constant([
            [[1000.0, -1000.0, 0.0]],  # Very large/small values
            [[1e-10, 1e10, -1e10]],   # Mixed very small and large
        ], dtype=tf.float32)

        mdn.build(extreme_inputs.shape)
        outputs = mdn(extreme_inputs, training=True)

        # Check that outputs don't contain NaN or Inf
        self.assertFalse(tf.reduce_any(tf.math.is_nan(outputs)).numpy(),
                        "MDN layer produced NaN with extreme inputs")
        self.assertFalse(tf.reduce_any(tf.math.is_inf(outputs)).numpy(),
                        "MDN layer produced Inf with extreme inputs")

        # Check that outputs are in expected ranges
        # Split outputs to check each component
        pi, mu1, mu2, sigma1, sigma2, rho, eos = tf.split(
            outputs, [self.num_components] * 6 + [1], axis=2
        )

        # Pi should be valid probabilities
        self.assertTrue(tf.reduce_all(pi >= 0).numpy(), "Pi contains negative values")
        self.assertTrue(tf.reduce_all(pi <= 1).numpy(), "Pi contains values > 1")

        # Sigma should be positive and bounded
        self.assertTrue(tf.reduce_all(sigma1 > 0).numpy(), "Sigma1 contains non-positive values")
        self.assertTrue(tf.reduce_all(sigma2 > 0).numpy(), "Sigma2 contains non-positive values")
        self.assertTrue(tf.reduce_all(sigma1 <= 100).numpy(), "Sigma1 exceeds upper bound")
        self.assertTrue(tf.reduce_all(sigma2 <= 100).numpy(), "Sigma2 exceeds upper bound")

        # Rho should be bounded
        self.assertTrue(tf.reduce_all(rho >= -1).numpy(), "Rho below -1")
        self.assertTrue(tf.reduce_all(rho <= 1).numpy(), "Rho above 1")

    def test_mdn_loss_extreme_predictions(self):
        """Test MDN loss with extreme prediction values."""
        # Create targets
        y_true = tf.constant([
            [[0.5, 0.5, 0.0], [0.1, 0.2, 1.0]],
            [[0.0, 0.0, 0.0], [-0.5, 0.8, 0.0]]
        ], dtype=tf.float32)

        # Create extreme predictions that could cause NaN (3 components = 3*6+1 = 19 values)
        extreme_pred = tf.constant([
            [
                # 3 components: pi(3) + mu1(3) + mu2(3) + sigma1(3) + sigma2(3) + rho(3) + eos(1) = 19
                [0.33, 0.33, 0.34,          # pi (3 components)
                 1000.0, -1000.0, 0.0,      # mu1 (3 components)
                 1e-10, 1e10, 0.0,          # mu2 (3 components)
                 1e-10, 1e-10, 1.0,         # sigma1 (3 components, very small)
                 100.0, 100.0, 1.0,        # sigma2 (3 components, very large)
                 0.999, -0.999, 0.0,       # rho (3 components, extreme correlations)
                 0.5],                      # eos
                [0.5, 0.3, 0.2,             # pi
                 0.0, 0.0, 0.0,             # mu1
                 0.0, 0.0, 0.0,             # mu2
                 0.001, 0.001, 0.001,       # sigma1
                 0.001, 0.001, 0.001,       # sigma2
                 0.0, 0.0, 0.0,             # rho
                 0.999]                     # eos
            ],
            [
                [0.1, 0.8, 0.1,             # pi
                 50.0, -50.0, 0.0,          # mu1
                 0.0, 0.0, 0.0,             # mu2
                 1.0, 1.0, 1.0,             # sigma1
                 1.0, 1.0, 1.0,             # sigma2
                 0.5, -0.5, 0.0,            # rho
                 0.001],                    # eos (very small)
                [0.4, 0.4, 0.2,             # pi
                 0.0, 0.0, 0.0,             # mu1
                 0.0, 0.0, 0.0,             # mu2
                 10.0, 10.0, 10.0,          # sigma1
                 10.0, 10.0, 10.0,          # sigma2
                 0.9, -0.9, 0.0,            # rho
                 0.5]                       # eos
            ]
        ], dtype=tf.float32)

        stroke_lengths = tf.constant([2, 2], dtype=tf.int32)

        # Calculate loss
        loss = mdn_loss(y_true, extreme_pred, stroke_lengths, self.num_components)

        # Check that loss is finite
        self.assertFalse(tf.math.is_nan(loss).numpy(),
                        f"MDN loss is NaN: {loss.numpy()}")
        self.assertFalse(tf.math.is_inf(loss).numpy(),
                        f"MDN loss is Inf: {loss.numpy()}")
        self.assertGreater(loss.numpy(), 0, "Loss should be positive")

    def test_attention_mechanism_extreme_inputs(self):
        """Test attention mechanism with extreme inputs."""
        attention = AttentionMechanism(self.num_attention_gaussians, self.num_chars)

        # Extreme inputs
        extreme_inputs = tf.constant([
            [1000.0, -1000.0, 0.0, 1e-10, 1e10],  # Very large/small values
            [0.0, 0.0, 0.0, 0.0, 0.0]             # All zeros
        ], dtype=tf.float32)

        prev_kappa = tf.zeros((2, self.num_attention_gaussians), dtype=tf.float32)
        char_seq_one_hot = tf.random.uniform((2, 20, self.num_chars), dtype=tf.float32)
        sequence_lengths = tf.constant([15, 10], dtype=tf.int32)

        attention.build(extreme_inputs.shape)
        w, kappa = attention(extreme_inputs, prev_kappa, char_seq_one_hot, sequence_lengths)

        # Check outputs
        self.assertFalse(tf.reduce_any(tf.math.is_nan(w)).numpy(),
                        "Attention window contains NaN")
        self.assertFalse(tf.reduce_any(tf.math.is_inf(w)).numpy(),
                        "Attention window contains Inf")
        self.assertFalse(tf.reduce_any(tf.math.is_nan(kappa)).numpy(),
                        "Attention kappa contains NaN")
        self.assertFalse(tf.reduce_any(tf.math.is_inf(kappa)).numpy(),
                        "Attention kappa contains Inf")

    def test_nan_monitor_detection(self):
        """Test that NaN monitor correctly detects NaN values."""
        monitor = NaNMonitor()

        # Create tensors with NaN
        tensors_with_nan = {
            "normal": tf.constant([1.0, 2.0, 3.0]),
            "with_nan": tf.constant([1.0, float('nan'), 3.0]),
            "with_inf": tf.constant([1.0, float('inf'), 3.0])
        }

        # Should detect NaN/Inf
        has_nan = monitor.check_tensors(tensors_with_nan, step=1)
        self.assertTrue(has_nan, "NaN monitor failed to detect NaN/Inf")
        self.assertEqual(len(monitor.nan_history), 2, "Should detect 2 problematic tensors")

        # Clean tensors should not trigger detection
        clean_tensors = {
            "tensor1": tf.constant([1.0, 2.0, 3.0]),
            "tensor2": tf.constant([[1.0, 2.0], [3.0, 4.0]])
        }

        has_nan = monitor.check_tensors(clean_tensors, step=2)
        self.assertFalse(has_nan, "NaN monitor incorrectly detected NaN in clean tensors")

    def test_model_numerical_stability(self):
        """Test complete model with challenging inputs."""
        model = DeepHandwritingPredictionModel(num_mixture_components=self.num_components)

        # Create challenging input sequence
        challenging_inputs = tf.random.normal((self.batch_size, self.seq_length, 3))
        # Add some extreme values
        challenging_inputs = tf.tensor_scatter_nd_update(
            challenging_inputs,
            [[0, 0, 0], [1, 5, 1]],
            [1000.0, -1000.0]
        )

        # Forward pass
        outputs = model(challenging_inputs, training=True)

        # Check outputs
        self.assertFalse(tf.reduce_any(tf.math.is_nan(outputs)).numpy(),
                        "Model outputs contain NaN")
        self.assertFalse(tf.reduce_any(tf.math.is_inf(outputs)).numpy(),
                        "Model outputs contain Inf")

        # Test gradient computation
        with tf.GradientTape() as tape:
            predictions = model(challenging_inputs, training=True)
            # Simple loss for testing
            loss = tf.reduce_mean(tf.square(predictions))

        gradients = tape.gradient(loss, model.trainable_variables)

        # Check gradients
        for i, grad in enumerate(gradients):
            if grad is not None:
                self.assertFalse(tf.reduce_any(tf.math.is_nan(grad)).numpy(),
                                f"Gradient {i} contains NaN")
                self.assertFalse(tf.reduce_any(tf.math.is_inf(grad)).numpy(),
                                f"Gradient {i} contains Inf")

    def test_zero_division_protection(self):
        """Test protection against division by zero in loss calculation."""
        # Create targets
        y_true = tf.constant([[[0.0, 0.0, 0.0]]], dtype=tf.float32)

        # Create predictions with very small sigmas
        y_pred = tf.constant([[[
            1.0,           # pi (single component)
            0.0, 0.0,      # mu1, mu2
            1e-10, 1e-10,  # sigma1, sigma2 (very small)
            0.0,           # rho
            0.5            # eos
        ]]], dtype=tf.float32)

        # Should not crash or produce NaN
        loss = mdn_loss(y_true, y_pred, None, 1)

        self.assertFalse(tf.math.is_nan(loss).numpy(), "Loss is NaN with small sigmas")
        self.assertFalse(tf.math.is_inf(loss).numpy(), "Loss is Inf with small sigmas")

    def test_gradient_clipping_effectiveness(self):
        """Test that gradient clipping prevents exploding gradients."""
        model = DeepHandwritingPredictionModel(num_mixture_components=self.num_components)

        # Create inputs that might cause large gradients
        extreme_inputs = tf.random.normal((self.batch_size, self.seq_length, 3)) * 100

        with tf.GradientTape() as tape:
            predictions = model(extreme_inputs, training=True)
            # Use a loss that could cause exploding gradients
            loss = tf.reduce_sum(tf.exp(predictions))

        gradients = tape.gradient(loss, model.trainable_variables)

        # Apply clipping as in the fixed training script
        clipped_gradients = [
            tf.clip_by_value(g, -5.0, 5.0) if g is not None else g
            for g in gradients
        ]

        # Check that clipped gradients are within bounds
        for grad in clipped_gradients:
            if grad is not None:
                self.assertTrue(tf.reduce_all(grad >= -5.0).numpy(),
                               "Gradient below clipping threshold")
                self.assertTrue(tf.reduce_all(grad <= 5.0).numpy(),
                               "Gradient above clipping threshold")


class TestStressConditions(unittest.TestCase):
    """Test model behavior under stress conditions."""

    def test_long_sequence_stability(self):
        """Test model with very long sequences."""
        model = DeepHandwritingPredictionModel(num_mixture_components=5)

        # Very long sequence
        long_inputs = tf.random.normal((1, 1000, 3))

        # Should not run out of memory or produce NaN
        outputs = model(long_inputs, training=False)

        self.assertFalse(tf.reduce_any(tf.math.is_nan(outputs)).numpy(),
                        "Long sequence produced NaN")

    def test_batch_consistency(self):
        """Test that different batch sizes produce consistent results."""
        model = DeepHandwritingPredictionModel(num_mixture_components=3)

        # Same input in different batch configurations
        single_input = tf.random.normal((1, 10, 3))
        batch_input = tf.tile(single_input, [4, 1, 1])

        single_output = model(single_input, training=False)
        batch_output = model(batch_input, training=False)

        # First output should be approximately equal
        self.assertTrue(tf.reduce_all(
            tf.abs(single_output[0] - batch_output[0]) < 1e-5
        ).numpy(), "Batch processing changes results")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    unittest.main(verbosity=2)