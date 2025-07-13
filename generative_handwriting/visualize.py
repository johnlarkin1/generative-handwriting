import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mdn import bivariate_gaussian


def compute_density_grid_tf(
    pi: tf.Tensor,
    mus: tf.Tensor,
    sigmas: tf.Tensor,
    rhos: tf.Tensor,
    x1_vals: np.ndarray,
    x2_vals: np.ndarray,
) -> np.ndarray:
    """Compute the probability density for each point in a grid using TensorFlow."""
    x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
    x1_grid_tf = tf.constant(x1_grid.reshape(-1), dtype=tf.float32)
    x2_grid_tf = tf.constant(x2_grid.reshape(-1), dtype=tf.float32)

    # Assuming a single MDNOutput for simplicity, expand dimensions if necessary
    density = tf.zeros_like(x1_grid_tf)

    for i in range(pi.shape[-1]):  # Iterate over components
        mu1, mu2 = mus[:, i * 2], mus[:, i * 2 + 1]
        sigma1, sigma2 = sigmas[:, i * 2], sigmas[:, i * 2 + 1]
        rho = rhos[:, i]
        density += pi[:, i] * bivariate_gaussian(
            x1_grid_tf, x2_grid_tf, mu1, mu2, sigma1, sigma2, rho
        )

    density_grid = tf.reshape(density, x1_grid.shape).numpy()
    return density_grid


def plot_density_heatmap_tf(
    density_grid: np.ndarray, x1_vals: np.ndarray, x2_vals: np.ndarray
) -> None:
    """Plot a heatmap of the predicted probability density using matplotlib."""
    plt.figure(figsize=(8, 6))
    x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
    contour = plt.contourf(x1_grid, x2_grid, density_grid, levels=50, cmap="viridis")
    plt.colorbar(contour, label="Probability Density")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Predicted Probability Density Heatmap")
    plt.show()
