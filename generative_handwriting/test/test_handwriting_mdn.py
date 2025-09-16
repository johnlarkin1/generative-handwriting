import numpy as np
import tensorflow as tf

from generative_handwriting.model.mixture_density_network import mdn_loss


def test_mdn_loss():
    # Test case 1
    y_true = tf.constant([[[0.1, 0.2, 0.0], [0.3, 0.4, 1.0], [0.5, 0.6, 0.0]]])
    y_pred = tf.constant([[[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1]]])
    num_components = 1
    expected_loss = 0.6931472  # Expected negative log likelihood

    loss = mdn_loss(y_true, y_pred, num_components)
    assert np.isclose(loss, expected_loss)

    # Test case 2
    y_true = tf.constant([[[0.1, 0.2, 0.0], [0.3, 0.4, 1.0], [0.5, 0.6, 0.0]]])
    y_pred = tf.constant(
        [
            [
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1],
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1],
            ]
        ]
    )
    num_components = 2
    expected_loss = 0.6931472  # Expected negative log likelihood

    loss = mdn_loss(y_true, y_pred, num_components)
    assert np.isclose(loss, expected_loss)

    # Test case 3
    y_true = tf.constant([[[0.1, 0.2, 0.0], [0.3, 0.4, 1.0], [0.5, 0.6, 0.0]]])
    y_pred = tf.constant(
        [
            [
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1],
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1],
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1],
            ]
        ]
    )
    num_components = 3
    expected_loss = 0.6931472  # Expected negative log likelihood

    loss = mdn_loss(y_true, y_pred, num_components)
    assert np.isclose(loss, expected_loss)

    print("All tests passed!")


if __name__ == "__main__":
    test_mdn_loss()
