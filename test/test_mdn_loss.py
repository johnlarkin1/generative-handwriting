import tensorflow as tf
import numpy as np
import unittest
from model.handwriting_mdn import mdn_loss


class TestMDNLoss(unittest.TestCase):
    def setUp(self):
        # Common setup code if needed
        pass

    def test_loss_value_known_input(self):
        """Test the MDN loss function with a known input and expected output."""
        # Define a simple input for y_true and y_pred with known expected loss
        y_true = tf.constant([[[0.1, 0.2, 1]]], dtype=tf.float32)  # Shape: (1, 1, 3)
        # Assuming num_components = 1 for simplicity; adjust if testing more components
        y_pred = tf.constant([[[0.5, 0.1, 0.1, 0.2, 0.2, 0.0, 0.7]]], dtype=tf.float32)  # Shape: (1, 1, 7)
        num_components = 1

        # Calculate loss
        loss = mdn_loss(y_true, y_pred, num_components)

        # Assert (using an expected loss value based on manual calculation or intuition)
        print("loss.numpy()", loss.numpy())
        self.assertTrue(np.isclose(loss.numpy(), 0, atol=1e-6))

    def test_loss_stability_extreme_values(self):
        """Test the MDN loss function for stability with extreme values."""
        y_true = tf.constant([[[100, -100, 1]]], dtype=tf.float32)
        y_pred = tf.constant([[[100, -100, 100, -100, 100, -1, 0.1]]], dtype=tf.float32)
        num_components = 1

        # Calculate loss
        loss = mdn_loss(y_true, y_pred, num_components)

        # Check for NaN or inf values
        self.assertFalse(tf.math.is_nan(loss))
        self.assertFalse(tf.math.is_inf(loss))

    def test_loss_with_all_eos_data(self):
        """Test the MDN loss function when all eos_data is 1."""
        y_true = tf.constant([[[0.1, 0.2, 1], [0.1, 0.2, 1]]], dtype=tf.float32)  # Shape: (1, 2, 3)
        y_pred = tf.constant(
            [[[0.5, 0.1, 0.1, 0.2, 0.2, 0.0, 0.7], [0.5, 0.1, 0.1, 0.2, 0.2, 0.0, 0.7]]], dtype=tf.float32
        )  # Shape: (1, 2, 7)
        num_components = 1

        # Calculate loss
        loss = mdn_loss(y_true, y_pred, num_components)

        # Basic check to ensure loss is calculated
        self.assertTrue(loss.numpy() > 0)

    def test_loss_value_known_input_multiple_components(self):
        """Test the MDN loss function with multiple mixture components and a known expected output."""
        # Example with 5 components, adjust y_pred to include 5 sets of parameters + 1 eos
        y_true = tf.constant([[[0.1, 0.2, 1]]], dtype=tf.float32)  # Shape: (1, 1, 3)
        y_pred = tf.constant(
            [
                [
                    # Assuming mixture weights are normalized (they sum up to 1 across the components)
                    [
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        # Means for each component
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        # Log standard deviations (assuming some positive value after exp)
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        # Correlations (all zeros for simplicity)
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        # End-of-stroke probability
                        0.7,
                    ]
                ]
            ],
            dtype=tf.float32,
        )  # Shape: (1, 1, 31)
        num_components = 5

        # Calculate loss
        loss = mdn_loss(y_true, y_pred, num_components)

        # Assert (using an expected loss value based on manual calculation or intuition)
        self.assertTrue(np.isclose(loss.numpy(), 1, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
