import tensorflow as tf
import numpy as np
import unittest
from model.mixture_density_network import mdn_loss


class TestMDNLoss(unittest.TestCase):
    def setUp(self):
        # Common setup code if needed
        pass

    def test_loss_value_known_input(self):
        """Test the MDN loss function with a known input and expected output."""
        # Define a simple input for y_true and y_pred with known expected loss
        y_true = tf.constant([[[0.1, 0.2, 1]]], dtype=tf.float32)  # Shape: (1, 1, 3)
        # Assuming num_components = 1 for simplicity; adjust if testing more components
        y_pred = tf.constant(
            [[[0.5, 0.1, 0.1, 0.2, 0.2, 0.0, 0.7]]], dtype=tf.float32
        )  # Shape: (1, 1, 7)
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
        y_true = tf.constant(
            [[[0.1, 0.2, 1], [0.1, 0.2, 1]]], dtype=tf.float32
        )  # Shape: (1, 2, 3)
        y_pred = tf.constant(
            [
                [
                    [0.5, 0.1, 0.1, 0.2, 0.2, 0.0, 0.7],
                    [0.5, 0.1, 0.1, 0.2, 0.2, 0.0, 0.7],
                ]
            ],
            dtype=tf.float32,
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


def main():
    x = np.array(
        [
            [-8.3443660e-01, -1.9042810e00, 0.0000000e00],
            [-8.8908482e-01, -1.9883552e00, 0.0000000e00],
            [-7.8609389e-01, -1.8349197e00, 0.0000000e00],
            [-5.5909353e-01, -1.4271598e00, 0.0000000e00],
            [-3.9094511e-01, -8.1972361e-01, 0.0000000e00],
            [-1.9547255e-01, -3.9514881e-01, 0.0000000e00],
            [-6.3055661e-03, 4.2037107e-03, 0.0000000e00],
            [1.8916698e-01, 3.3419502e-01, 0.0000000e00],
            [3.4890798e-01, 6.7889929e-01, 0.0000000e00],
            [5.2546382e-01, 9.7526091e-01, 0.0000000e00],
            [5.1075083e-01, 1.4439746e00, 0.0000000e00],
            [4.7712117e-01, 1.6772805e00, 0.0000000e00],
            [3.9725065e-01, 1.8496327e00, 0.0000000e00],
            [3.8463953e-01, 1.8012900e00, 0.0000000e00],
            [3.7623212e-01, 1.6310397e00, 0.0000000e00],
            [3.6362097e-01, 1.4250579e00, 0.0000000e00],
            [2.7324119e-01, 1.2989466e00, 0.0000000e00],
            [2.5432450e-01, 1.0025851e00, 0.0000000e00],
            [2.0177811e-01, 6.4947331e-01, 0.0000000e00],
            [1.5343544e-01, 3.2578757e-01, 0.0000000e00],
            [1.4923173e-01, -1.2611133e-01, 0.0000000e00],
            [1.9337070e-01, -6.0743618e-01, 0.0000000e00],
            [3.4050056e-01, -1.0824555e00, 0.0000000e00],
            [5.7801020e-01, -1.6079193e00, 0.0000000e00],
            [7.7768648e-01, -1.9841515e00, 0.0000000e00],
            [9.2691821e-01, -2.1838276e00, 0.0000000e00],
            [9.8787200e-01, -2.3120408e00, 0.0000000e00],
            [1.0404184e00, -2.4024208e00, 0.0000000e00],
            [8.6386257e-01, -2.0345960e00, 0.0000000e00],
            [6.0113066e-01, -1.4124469e00, 0.0000000e00],
            [3.0897275e-01, -6.8730670e-01, 0.0000000e00],
            [9.2481636e-02, -1.8496327e-01, 0.0000000e00],
            [-5.2546386e-02, 1.7235214e-01, 0.0000000e00],
            [-1.6184287e-01, 4.5610261e-01, 0.0000000e00],
            [-2.5012079e-01, 6.0113066e-01, 0.0000000e00],
            [-2.2279666e-01, 5.4227871e-01, 1.0000000e00],
            [-1.4187524e00, 4.5358038e00, 0.0000000e00],
            [-3.1527832e-02, 1.0299091e-01, 0.0000000e00],
            [-8.6176068e-02, 8.4074214e-02, 0.0000000e00],
            [-2.5642636e-01, 8.4074214e-02, 0.0000000e00],
            [-4.7081560e-01, 3.1527832e-02, 0.0000000e00],
            [-7.2934383e-01, 1.8916698e-02, 0.0000000e00],
            [-1.0067887e00, 2.1018554e-03, 0.0000000e00],
            [-1.1413075e00, -6.5157518e-02, 0.0000000e00],
            [-1.1497148e00, -1.3031504e-01, 0.0000000e00],
            [-1.0193999e00, -1.7655586e-01, 0.0000000e00],
            [-7.7558464e-01, -1.1139833e-01, 0.0000000e00],
            [-4.2247292e-01, -7.3564939e-02, 0.0000000e00],
            [-1.4292617e-01, -4.8342675e-02, 0.0000000e00],
            [1.0509277e-02, -2.1018554e-02, 1.0000000e00],
        ]
    )
    y_true = np.array(
        [
            [-8.8908482e-01, -1.9883552e00, 0.0000000e00],
            [-7.8609389e-01, -1.8349197e00, 0.0000000e00],
            [-5.5909353e-01, -1.4271598e00, 0.0000000e00],
            [-3.9094511e-01, -8.1972361e-01, 0.0000000e00],
            [-1.9547255e-01, -3.9514881e-01, 0.0000000e00],
            [-6.3055661e-03, 4.2037107e-03, 0.0000000e00],
            [1.8916698e-01, 3.3419502e-01, 0.0000000e00],
            [3.4890798e-01, 6.7889929e-01, 0.0000000e00],
            [5.2546382e-01, 9.7526091e-01, 0.0000000e00],
            [5.1075083e-01, 1.4439746e00, 0.0000000e00],
            [4.7712117e-01, 1.6772805e00, 0.0000000e00],
            [3.9725065e-01, 1.8496327e00, 0.0000000e00],
            [3.8463953e-01, 1.8012900e00, 0.0000000e00],
            [3.7623212e-01, 1.6310397e00, 0.0000000e00],
            [3.6362097e-01, 1.4250579e00, 0.0000000e00],
            [2.7324119e-01, 1.2989466e00, 0.0000000e00],
            [2.5432450e-01, 1.0025851e00, 0.0000000e00],
            [2.0177811e-01, 6.4947331e-01, 0.0000000e00],
            [1.5343544e-01, 3.2578757e-01, 0.0000000e00],
            [1.4923173e-01, -1.2611133e-01, 0.0000000e00],
            [1.9337070e-01, -6.0743618e-01, 0.0000000e00],
            [3.4050056e-01, -1.0824555e00, 0.0000000e00],
            [5.7801020e-01, -1.6079193e00, 0.0000000e00],
            [7.7768648e-01, -1.9841515e00, 0.0000000e00],
            [9.2691821e-01, -2.1838276e00, 0.0000000e00],
            [9.8787200e-01, -2.3120408e00, 0.0000000e00],
            [1.0404184e00, -2.4024208e00, 0.0000000e00],
            [8.6386257e-01, -2.0345960e00, 0.0000000e00],
            [6.0113066e-01, -1.4124469e00, 0.0000000e00],
            [3.0897275e-01, -6.8730670e-01, 0.0000000e00],
            [9.2481636e-02, -1.8496327e-01, 0.0000000e00],
            [-5.2546386e-02, 1.7235214e-01, 0.0000000e00],
            [-1.6184287e-01, 4.5610261e-01, 0.0000000e00],
            [-2.5012079e-01, 6.0113066e-01, 0.0000000e00],
            [-2.2279666e-01, 5.4227871e-01, 1.0000000e00],
            [-1.4187524e00, 4.5358038e00, 0.0000000e00],
            [-3.1527832e-02, 1.0299091e-01, 0.0000000e00],
            [-8.6176068e-02, 8.4074214e-02, 0.0000000e00],
            [-2.5642636e-01, 8.4074214e-02, 0.0000000e00],
            [-4.7081560e-01, 3.1527832e-02, 0.0000000e00],
            [-7.2934383e-01, 1.8916698e-02, 0.0000000e00],
            [-1.0067887e00, 2.1018554e-03, 0.0000000e00],
            [-1.1413075e00, -6.5157518e-02, 0.0000000e00],
            [-1.1497148e00, -1.3031504e-01, 0.0000000e00],
            [-1.0193999e00, -1.7655586e-01, 0.0000000e00],
            [-7.7558464e-01, -1.1139833e-01, 0.0000000e00],
            [-4.2247292e-01, -7.3564939e-02, 0.0000000e00],
            [-1.4292617e-01, -4.8342675e-02, 0.0000000e00],
            [1.0509277e-02, -2.1018554e-02, 1.0000000e00],
            [2.6594776e01, 9.9922209e00, 0.0000000e00],
        ]
    )

    # Close values (likely to result in low MDN loss)
    # Assuming mean values are very close to y_true values, with small standard deviations
    mu1_close = y_true[:, :1] + np.random.normal(
        loc=0, scale=0.1, size=(50, 1)
    )  # Slightly perturbed x
    mu2_close = y_true[:, 1:2] + np.random.normal(
        loc=0, scale=0.1, size=(50, 1)
    )  # Slightly perturbed y
    sigma1_close = np.full((50, 1), 0.1)  # Small std dev for x
    sigma2_close = np.full((50, 1), 0.1)  # Small std dev for y
    rho_close = np.full((50, 1), 0)  # Minimal correlation
    pi_close = np.full((50, 1), 1)  # Assuming a single mixture component
    eos_close = y_true[:, 2:3]  # Same eos as true
    eos_all_zero = np.full((50, 1), 0)

    # Concatenate to get the final close mock output
    y_actual_close = np.concatenate(
        (
            pi_close,
            mu1_close,
            mu2_close,
            sigma1_close,
            sigma2_close,
            rho_close,
            eos_close,
        ),
        axis=1,
    )
    y_actual_close = y_actual_close.reshape(1, 50, 7)
    y_actual_less_close = np.concatenate(
        (
            pi_close,
            mu1_close,
            mu2_close,
            sigma1_close,
            sigma2_close,
            rho_close,
            eos_all_zero,
        ),
        axis=1,
    )
    y_actual_less_close = y_actual_less_close.reshape(1, 50, 7)

    # Far values (likely to result in high MDN loss)
    # Mean values are set arbitrarily far from y_true values, with larger standard deviations
    mu1_far = y_true[:, :1] + np.random.normal(
        loc=5, scale=2, size=(50, 1)
    )  # Far perturbed x
    mu2_far = y_true[:, 1:2] + np.random.normal(
        loc=5, scale=2, size=(50, 1)
    )  # Far perturbed y
    sigma1_far = np.full((50, 1), 2)  # Large std dev for x
    sigma2_far = np.full((50, 1), 2)  # Large std dev for y
    rho_far = np.full((50, 1), 0.5)  # Some arbitrary correlation
    pi_far = np.full((50, 1), 1)  # Arbitrary mixture component weight
    eos_far = np.random.uniform(low=0, high=1, size=(50, 1))  # Random eos

    # Concatenate to get the final far mock output
    y_actual_far = np.concatenate(
        (pi_far, mu1_far, mu2_far, sigma1_far, sigma2_far, rho_far, eos_far), axis=1
    )
    y_actual_far = y_actual_far.reshape(1, 50, 7)

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_actual_far = tf.cast(y_actual_far, dtype=tf.float32)
    y_actual_less_close = tf.cast(y_actual_less_close, dtype=tf.float32)
    y_actual_close = tf.cast(y_actual_close, dtype=tf.float32)

    # print(
    #     f"For y actual less close, mdn loss: {mdn_loss(y_true, y_actual_less_close, None, 1)}"
    # )
    print(f"For y actual close, mdn loss: {mdn_loss(y_true, y_actual_close, None, 1)}")
    print(f"For y actual far, mdn loss: {mdn_loss(y_true, y_actual_far, None, 1)}")


def test_with_two_components():
    num_components = 2
    num_points = 50
    batch_size = 1

    y_true = np.array(
        [
            [-8.8908482e-01, -1.9883552e00, 0.0000000e00],
            [-7.8609389e-01, -1.8349197e00, 0.0000000e00],
            [-5.5909353e-01, -1.4271598e00, 0.0000000e00],
            [-3.9094511e-01, -8.1972361e-01, 0.0000000e00],
            [-1.9547255e-01, -3.9514881e-01, 0.0000000e00],
            [-6.3055661e-03, 4.2037107e-03, 0.0000000e00],
            [1.8916698e-01, 3.3419502e-01, 0.0000000e00],
            [3.4890798e-01, 6.7889929e-01, 0.0000000e00],
            [5.2546382e-01, 9.7526091e-01, 0.0000000e00],
            [5.1075083e-01, 1.4439746e00, 0.0000000e00],
            [4.7712117e-01, 1.6772805e00, 0.0000000e00],
            [3.9725065e-01, 1.8496327e00, 0.0000000e00],
            [3.8463953e-01, 1.8012900e00, 0.0000000e00],
            [3.7623212e-01, 1.6310397e00, 0.0000000e00],
            [3.6362097e-01, 1.4250579e00, 0.0000000e00],
            [2.7324119e-01, 1.2989466e00, 0.0000000e00],
            [2.5432450e-01, 1.0025851e00, 0.0000000e00],
            [2.0177811e-01, 6.4947331e-01, 0.0000000e00],
            [1.5343544e-01, 3.2578757e-01, 0.0000000e00],
            [1.4923173e-01, -1.2611133e-01, 0.0000000e00],
            [1.9337070e-01, -6.0743618e-01, 0.0000000e00],
            [3.4050056e-01, -1.0824555e00, 0.0000000e00],
            [5.7801020e-01, -1.6079193e00, 0.0000000e00],
            [7.7768648e-01, -1.9841515e00, 0.0000000e00],
            [9.2691821e-01, -2.1838276e00, 0.0000000e00],
            [9.8787200e-01, -2.3120408e00, 0.0000000e00],
            [1.0404184e00, -2.4024208e00, 0.0000000e00],
            [8.6386257e-01, -2.0345960e00, 0.0000000e00],
            [6.0113066e-01, -1.4124469e00, 0.0000000e00],
            [3.0897275e-01, -6.8730670e-01, 0.0000000e00],
            [9.2481636e-02, -1.8496327e-01, 0.0000000e00],
            [-5.2546386e-02, 1.7235214e-01, 0.0000000e00],
            [-1.6184287e-01, 4.5610261e-01, 0.0000000e00],
            [-2.5012079e-01, 6.0113066e-01, 0.0000000e00],
            [-2.2279666e-01, 5.4227871e-01, 1.0000000e00],
            [-1.4187524e00, 4.5358038e00, 0.0000000e00],
            [-3.1527832e-02, 1.0299091e-01, 0.0000000e00],
            [-8.6176068e-02, 8.4074214e-02, 0.0000000e00],
            [-2.5642636e-01, 8.4074214e-02, 0.0000000e00],
            [-4.7081560e-01, 3.1527832e-02, 0.0000000e00],
            [-7.2934383e-01, 1.8916698e-02, 0.0000000e00],
            [-1.0067887e00, 2.1018554e-03, 0.0000000e00],
            [-1.1413075e00, -6.5157518e-02, 0.0000000e00],
            [-1.1497148e00, -1.3031504e-01, 0.0000000e00],
            [-1.0193999e00, -1.7655586e-01, 0.0000000e00],
            [-7.7558464e-01, -1.1139833e-01, 0.0000000e00],
            [-4.2247292e-01, -7.3564939e-02, 0.0000000e00],
            [-1.4292617e-01, -4.8342675e-02, 0.0000000e00],
            [1.0509277e-02, -2.1018554e-02, 1.0000000e00],
            [2.6594776e01, 9.9922209e00, 0.0000000e00],
        ]
    )

    # Function to generate parameters for "close" predictions
    def generate_close_parameters(num_points, num_components):
        pi_components = np.full(
            (num_points, num_components), 0.5
        )  # Evenly split pi values for simplicity
        mu1 = y_true[:, :1] + np.random.normal(
            loc=0, scale=0.1, size=(num_points, num_components)
        )
        mu2 = y_true[:, 1:2] + np.random.normal(
            loc=0, scale=0.1, size=(num_points, num_components)
        )
        sigma1 = np.full((num_points, num_components), 0.1)
        sigma2 = np.full((num_points, num_components), 0.1)
        rho = np.full((num_points, num_components), 0)
        return pi_components, mu1, mu2, sigma1, sigma2, rho

    # Function to generate parameters for "far" predictions
    def generate_far_parameters(num_points, num_components):
        pi_components = np.full(
            (num_points, num_components), 0.5
        )  # Evenly split pi values for simplicity
        mu1 = y_true[:, :1] + np.random.normal(
            loc=5, scale=2, size=(num_points, num_components)
        )
        mu2 = y_true[:, 1:2] + np.random.normal(
            loc=5, scale=2, size=(num_points, num_components)
        )
        sigma1 = np.full((num_points, num_components), 2)
        sigma2 = np.full((num_points, num_components), 2)
        rho = np.full((num_points, num_components), 0.5)
        return pi_components, mu1, mu2, sigma1, sigma2, rho

    # Generate close and far parameters
    (
        pi_close,
        mu1_close,
        mu2_close,
        sigma1_close,
        sigma2_close,
        rho_close,
    ) = generate_close_parameters(num_points, num_components)
    pi_far, mu1_far, mu2_far, sigma1_far, sigma2_far, rho_far = generate_far_parameters(
        num_points, num_components
    )

    # EOS data - same for both close and far to simplify
    eos = y_true[:, 2:3].reshape(
        batch_size, num_points, 1
    )  # Ensure it matches the shape [batch_size, num_points, 1]

    # Reshape and concatenate parameters to form y_actual_close and y_actual_far
    y_actual_close = np.concatenate(
        [pi_close, mu1_close, mu2_close, sigma1_close, sigma2_close, rho_close], axis=-1
    ).reshape(batch_size, num_points, -1)
    y_actual_close = np.concatenate(
        [y_actual_close, eos], axis=-1
    )  # Add EOS at the end

    y_actual_far = np.concatenate(
        [pi_far, mu1_far, mu2_far, sigma1_far, sigma2_far, rho_far], axis=-1
    ).reshape(batch_size, num_points, -1)
    y_actual_far = np.concatenate([y_actual_far, eos], axis=-1)  # Add EOS at the end

    # Ensure to cast your numpy arrays to TensorFlow tensors before using them with TensorFlow operations
    y_actual_close = tf.convert_to_tensor(y_actual_close, dtype=tf.float32)
    y_actual_far = tf.convert_to_tensor(y_actual_far, dtype=tf.float32)
    y_true_reshaped = tf.convert_to_tensor(
        y_true.reshape([1, num_points, 3]), dtype=tf.float32
    )

    print(
        f"For y_actual_close, MDN loss: {mdn_loss(y_true_reshaped, y_actual_close, None, num_components)}"
    )
    print(
        f"For y_actual_far, MDN loss: {mdn_loss(y_true_reshaped, y_actual_far, None, num_components)}"
    )


if __name__ == "__main__":
    # unittest.main()
    # main()
    test_with_two_components()
