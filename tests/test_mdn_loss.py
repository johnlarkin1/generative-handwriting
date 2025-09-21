import numpy as np
import pytest
import tensorflow as tf

from generative_handwriting.model.mixture_density_network import mdn_loss


class TestMDNLoss:
    """Unit tests for the MDN loss function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_length = 5
        self.num_components = 3
        self.eps = 1e-6

        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

    def _create_mock_predictions(self, batch_size=None, seq_length=None, num_components=None):
        """Create mock MDN predictions."""
        batch_size = batch_size or self.batch_size
        seq_length = seq_length or self.seq_length
        num_components = num_components or self.num_components

        # Create predictions with proper shape [batch, seq, num_components * 6 + 1]
        # Components: pi, mu1, mu2, sigma1, sigma2, rho, eos
        output_size = num_components * 6 + 1

        # Initialize with reasonable values
        predictions = np.zeros((batch_size, seq_length, output_size), dtype=np.float32)

        # Mixture weights (softmax will normalize)
        pi_start = 0
        predictions[:, :, pi_start : pi_start + num_components] = np.random.randn(
            batch_size, seq_length, num_components
        )

        # Means (can be any value)
        mu1_start = num_components
        mu2_start = mu1_start + num_components
        predictions[:, :, mu1_start : mu1_start + num_components] = (
            np.random.randn(batch_size, seq_length, num_components) * 0.1
        )
        predictions[:, :, mu2_start : mu2_start + num_components] = (
            np.random.randn(batch_size, seq_length, num_components) * 0.1
        )

        # Standard deviations (must be positive)
        sigma1_start = mu2_start + num_components
        sigma2_start = sigma1_start + num_components
        predictions[:, :, sigma1_start : sigma1_start + num_components] = np.exp(
            np.random.randn(batch_size, seq_length, num_components) * 0.5
        )
        predictions[:, :, sigma2_start : sigma2_start + num_components] = np.exp(
            np.random.randn(batch_size, seq_length, num_components) * 0.5
        )

        # Correlation coefficients (must be in [-1, 1])
        rho_start = sigma2_start + num_components
        predictions[:, :, rho_start : rho_start + num_components] = np.tanh(
            np.random.randn(batch_size, seq_length, num_components) * 0.5
        )

        # End-of-stroke logits
        eos_start = rho_start + num_components
        predictions[:, :, eos_start : eos_start + 1] = np.random.randn(batch_size, seq_length, 1) * 0.1

        return tf.constant(predictions)

    def _create_mock_targets(self, batch_size=None, seq_length=None):
        """Create mock target sequences."""
        batch_size = batch_size or self.batch_size
        seq_length = seq_length or self.seq_length

        # Targets: [x, y, eos]
        targets = np.random.randn(batch_size, seq_length, 3).astype(np.float32) * 0.1
        # EOS is binary
        targets[:, :, 2] = np.random.randint(0, 2, size=(batch_size, seq_length)).astype(np.float32)

        return tf.constant(targets)

    def test_basic_loss_calculation(self):
        """Test basic loss calculation without masking."""
        y_pred = self._create_mock_predictions()
        y_true = self._create_mock_targets()

        loss = mdn_loss(y_true, y_pred, stroke_lengths=None, num_components=self.num_components, eps=self.eps)

        # Check that loss is a scalar
        assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"

        # Check that loss is finite
        assert tf.math.is_finite(loss), f"Loss is not finite: {loss}"

        # Check that loss is positive (negative log likelihood)
        assert loss > 0, f"Loss should be positive, got {loss}"

    def test_perfect_prediction(self):
        """Test loss when prediction perfectly matches target."""
        batch_size = 2
        seq_length = 3
        num_components = 2

        # Create perfect predictions
        y_true = tf.constant(
            [
                [[0.0, 0.0, 0.0], [0.1, 0.1, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [-0.1, -0.1, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=tf.float32,
        )

        # Create predictions that put all weight on one component centered at true values
        output_size = num_components * 6 + 1
        y_pred_np = np.zeros((batch_size, seq_length, output_size), dtype=np.float32)

        # Set first component to have all weight (others will be near 0 after softmax)
        y_pred_np[:, :, 0] = 10.0  # High logit for first component

        # Set means to match targets
        y_pred_np[0, :, num_components : num_components + 1] = [[0.0], [0.1], [0.0]]  # mu1
        y_pred_np[0, :, 2 * num_components : 2 * num_components + 1] = [[0.0], [0.1], [0.0]]  # mu2
        y_pred_np[1, :, num_components : num_components + 1] = [[0.0], [-0.1], [0.0]]  # mu1
        y_pred_np[1, :, 2 * num_components : 2 * num_components + 1] = [[0.0], [-0.1], [0.0]]  # mu2

        # Set small standard deviations
        y_pred_np[:, :, 3 * num_components : 4 * num_components] = 0.1  # sigma1
        y_pred_np[:, :, 4 * num_components : 5 * num_components] = 0.1  # sigma2

        # Set zero correlation
        y_pred_np[:, :, 5 * num_components : 6 * num_components] = 0.0  # rho

        # Set EOS logits
        y_pred_np[:, :, -1] = [[-10.0, -10.0, 10.0], [-10.0, -10.0, 10.0]]  # Match EOS targets

        y_pred = tf.constant(y_pred_np)

        loss = mdn_loss(y_true, y_pred, stroke_lengths=None, num_components=num_components, eps=self.eps)

        # Loss should be relatively small for perfect prediction
        assert loss < 10.0, f"Loss too high for near-perfect prediction: {loss}"
        assert tf.math.is_finite(loss), f"Loss is not finite: {loss}"

    def test_masking_functionality(self):
        """Test that masking works correctly with variable-length sequences."""
        y_pred = self._create_mock_predictions()
        y_true = self._create_mock_targets()

        # Create stroke lengths that mask out later timesteps
        stroke_lengths = tf.constant([3, 2], dtype=tf.int32)  # First seq: 3 steps, second: 2 steps

        loss_with_mask = mdn_loss(
            y_true, y_pred, stroke_lengths=stroke_lengths, num_components=self.num_components, eps=self.eps
        )

        loss_without_mask = mdn_loss(
            y_true, y_pred, stroke_lengths=None, num_components=self.num_components, eps=self.eps
        )

        # Losses should be different when masking is applied
        assert not tf.reduce_all(tf.equal(loss_with_mask, loss_without_mask)), (
            "Masked and unmasked losses should be different"
        )

        # Both should be finite
        assert tf.math.is_finite(loss_with_mask), f"Masked loss is not finite: {loss_with_mask}"
        assert tf.math.is_finite(loss_without_mask), f"Unmasked loss is not finite: {loss_without_mask}"

    def test_extreme_values_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very large standard deviations
        y_pred = self._create_mock_predictions()
        y_true = self._create_mock_targets()

        # Manually set extreme values
        y_pred_np = y_pred.numpy()
        # Set very large sigmas
        sigma1_start = 3 * self.num_components
        sigma2_start = 4 * self.num_components
        y_pred_np[:, :, sigma1_start : sigma1_start + self.num_components] = 100.0
        y_pred_np[:, :, sigma2_start : sigma2_start + self.num_components] = 100.0

        y_pred_extreme = tf.constant(y_pred_np)

        loss = mdn_loss(y_true, y_pred_extreme, stroke_lengths=None, num_components=self.num_components, eps=self.eps)

        # Should handle extreme values without NaN/Inf
        assert tf.math.is_finite(loss), f"Loss is not finite with extreme sigmas: {loss}"
        assert loss > 0, f"Loss should be positive, got {loss}"

    def test_zero_standard_deviation(self):
        """Test handling of near-zero standard deviations."""
        y_pred = self._create_mock_predictions()
        y_true = self._create_mock_targets()

        # Set very small sigmas (should be clipped internally)
        y_pred_np = y_pred.numpy()
        sigma1_start = 3 * self.num_components
        sigma2_start = 4 * self.num_components
        y_pred_np[:, :, sigma1_start : sigma1_start + self.num_components] = 1e-10
        y_pred_np[:, :, sigma2_start : sigma2_start + self.num_components] = 1e-10

        y_pred_small_sigma = tf.constant(y_pred_np)

        loss = mdn_loss(
            y_true, y_pred_small_sigma, stroke_lengths=None, num_components=self.num_components, eps=self.eps
        )

        # Should handle small sigmas without NaN/Inf
        assert tf.math.is_finite(loss), f"Loss is not finite with small sigmas: {loss}"

    def test_extreme_correlation(self):
        """Test handling of extreme correlation values."""
        y_pred = self._create_mock_predictions()
        y_true = self._create_mock_targets()

        # Test with correlation near ±1 (should be clipped internally)
        y_pred_np = y_pred.numpy()
        rho_start = 5 * self.num_components

        # Test near +1
        y_pred_np[:, :, rho_start : rho_start + self.num_components] = 0.999
        y_pred_high_rho = tf.constant(y_pred_np)

        loss_high = mdn_loss(
            y_true, y_pred_high_rho, stroke_lengths=None, num_components=self.num_components, eps=self.eps
        )

        # Test near -1
        y_pred_np[:, :, rho_start : rho_start + self.num_components] = -0.999
        y_pred_low_rho = tf.constant(y_pred_np)

        loss_low = mdn_loss(
            y_true, y_pred_low_rho, stroke_lengths=None, num_components=self.num_components, eps=self.eps
        )

        # Both should be finite
        assert tf.math.is_finite(loss_high), f"Loss not finite with high rho: {loss_high}"
        assert tf.math.is_finite(loss_low), f"Loss not finite with low rho: {loss_low}"

    def test_batch_independence(self):
        """Test that loss calculation is independent across batch items."""
        # Create predictions with different patterns for each batch item
        batch_size = 3
        y_pred = self._create_mock_predictions(batch_size=batch_size)
        y_true = self._create_mock_targets(batch_size=batch_size)

        # Calculate loss for full batch
        full_loss = mdn_loss(y_true, y_pred, stroke_lengths=None, num_components=self.num_components, eps=self.eps)

        # Calculate loss for each item separately
        individual_losses = []
        for i in range(batch_size):
            single_pred = tf.expand_dims(y_pred[i], 0)
            single_true = tf.expand_dims(y_true[i], 0)
            single_loss = mdn_loss(
                single_true, single_pred, stroke_lengths=None, num_components=self.num_components, eps=self.eps
            )
            individual_losses.append(single_loss)

        # Average of individual losses should be close to batch loss
        avg_individual = tf.reduce_mean(tf.stack(individual_losses))

        # Allow some numerical difference due to floating point operations
        assert tf.abs(full_loss - avg_individual) < 0.1, (
            f"Batch loss {full_loss} differs significantly from average of individual losses {avg_individual}"
        )

    def test_gradient_flow(self):
        """Test that gradients flow properly through the loss function."""
        y_pred = tf.Variable(self._create_mock_predictions())
        y_true = self._create_mock_targets()

        with tf.GradientTape() as tape:
            loss = mdn_loss(y_true, y_pred, stroke_lengths=None, num_components=self.num_components, eps=self.eps)

        gradients = tape.gradient(loss, y_pred)

        # Check that gradients exist
        assert gradients is not None, "Gradients are None"

        # Check that gradients have the same shape as predictions
        assert gradients.shape == y_pred.shape, (
            f"Gradient shape {gradients.shape} doesn't match prediction shape {y_pred.shape}"
        )

        # Check that gradients are finite
        assert tf.reduce_all(tf.math.is_finite(gradients)), "Gradients contain NaN or Inf"

        # Check that gradients are not all zero
        assert not tf.reduce_all(tf.equal(gradients, 0)), "All gradients are zero"

    def test_different_num_components(self):
        """Test loss calculation with different numbers of mixture components."""
        for num_components in [1, 5, 10]:
            y_pred = self._create_mock_predictions(num_components=num_components)
            y_true = self._create_mock_targets()

            loss = mdn_loss(y_true, y_pred, stroke_lengths=None, num_components=num_components, eps=self.eps)

            assert tf.math.is_finite(loss), f"Loss is not finite with {num_components} components: {loss}"
            assert loss > 0, f"Loss should be positive with {num_components} components, got {loss}"

    def test_empty_sequence_handling(self):
        """Test handling of sequences with zero length."""
        y_pred = self._create_mock_predictions()
        y_true = self._create_mock_targets()

        # All sequences have zero length
        stroke_lengths = tf.constant([0, 0], dtype=tf.int32)

        loss = mdn_loss(y_true, y_pred, stroke_lengths=stroke_lengths, num_components=self.num_components, eps=self.eps)

        # Should handle zero-length sequences without error
        assert tf.math.is_finite(loss) or loss == 0, f"Loss should be finite or zero for empty sequences, got {loss}"

    def test_single_timestep(self):
        """Test loss calculation with single timestep sequences."""
        batch_size = 3
        seq_length = 1

        y_pred = self._create_mock_predictions(batch_size=batch_size, seq_length=seq_length)
        y_true = self._create_mock_targets(batch_size=batch_size, seq_length=seq_length)

        loss = mdn_loss(y_true, y_pred, stroke_lengths=None, num_components=self.num_components, eps=self.eps)

        assert tf.math.is_finite(loss), f"Loss is not finite for single timestep: {loss}"
        assert loss > 0, f"Loss should be positive for single timestep, got {loss}"

    @pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
    def test_dtype_compatibility(self, dtype):
        """Test that loss function works with different data types."""
        y_pred = tf.cast(self._create_mock_predictions(), dtype)
        y_true = tf.cast(self._create_mock_targets(), dtype)

        loss = mdn_loss(y_true, y_pred, stroke_lengths=None, num_components=self.num_components, eps=self.eps)

        assert tf.math.is_finite(loss), f"Loss is not finite with dtype {dtype}: {loss}"
        assert loss.dtype == dtype, f"Loss dtype {loss.dtype} doesn't match input dtype {dtype}"

    def test_pi_normalization(self):
        """Test that mixture weights are properly normalized."""
        y_pred = self._create_mock_predictions()
        y_true = self._create_mock_targets()

        # Set non-normalized pi values
        y_pred_np = y_pred.numpy()
        y_pred_np[:, :, : self.num_components] = (
            np.random.rand(self.batch_size, self.seq_length, self.num_components) * 10
        )  # Random positive values

        y_pred_unnorm = tf.constant(y_pred_np)

        loss = mdn_loss(y_true, y_pred_unnorm, stroke_lengths=None, num_components=self.num_components, eps=self.eps)

        # Should handle unnormalized pi values
        assert tf.math.is_finite(loss), f"Loss is not finite with unnormalized pi: {loss}"
        assert loss > 0, f"Loss should be positive, got {loss}"

    def test_bernoulli_loss_component(self):
        """Test that EOS (end-of-stroke) Bernoulli loss is calculated correctly."""
        batch_size = 2
        seq_length = 3
        num_components = 2

        # Create simple predictions focusing on EOS
        y_true = tf.constant(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # EOS at positions 0 and 2
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],  # EOS at position 1
            ],
            dtype=tf.float32,
        )

        output_size = num_components * 6 + 1
        y_pred_np = np.zeros((batch_size, seq_length, output_size), dtype=np.float32)

        # Set reasonable MDN parameters
        y_pred_np[:, :, 0] = 1.0  # First component weight
        y_pred_np[:, :, 3 * num_components : 4 * num_components] = 0.5  # sigma1
        y_pred_np[:, :, 4 * num_components : 5 * num_components] = 0.5  # sigma2

        # Test with correct EOS predictions (high logit when EOS=1, low when EOS=0)
        y_pred_np[0, :, -1] = [10.0, -10.0, 10.0]  # Match first batch EOS pattern
        y_pred_np[1, :, -1] = [-10.0, 10.0, -10.0]  # Match second batch EOS pattern

        y_pred_correct = tf.constant(y_pred_np)

        loss_correct = mdn_loss(
            y_true, y_pred_correct, stroke_lengths=None, num_components=num_components, eps=self.eps
        )

        # Test with incorrect EOS predictions
        y_pred_np[0, :, -1] = [-10.0, 10.0, -10.0]  # Opposite of true EOS
        y_pred_np[1, :, -1] = [10.0, -10.0, 10.0]  # Opposite of true EOS

        y_pred_incorrect = tf.constant(y_pred_np)

        loss_incorrect = mdn_loss(
            y_true, y_pred_incorrect, stroke_lengths=None, num_components=num_components, eps=self.eps
        )

        # Loss should be higher with incorrect EOS predictions
        assert loss_correct < loss_incorrect, (
            f"Loss with correct EOS {loss_correct} should be less than with incorrect EOS {loss_incorrect}"
        )

        # Both should be finite
        assert tf.math.is_finite(loss_correct), f"Loss not finite with correct EOS: {loss_correct}"
        assert tf.math.is_finite(loss_incorrect), f"Loss not finite with incorrect EOS: {loss_incorrect}"
