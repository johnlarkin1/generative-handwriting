import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from common import mdn_to_heatmap


def plot_predictions_through_sequence(
    model,
    model_name,
    training_data,
    full_reconstructed_data,
    num_components,
    sequence_length,
    save_dir,
):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Perform a single prediction for the entire input sequence
    full_sequence_prediction = model.predict(training_data)

    # Set a buffer for the y-axis
    buffer = 5

    # Iterate through the desired sequence indices
    for seq_index in range(sequence_length, len(full_reconstructed_data) - sequence_length):
        # Extract the MDN parameters for the current subsequence index
        last_step_output = full_sequence_prediction[:, seq_index - 1, :]  # seq_index-1 because it's zero-indexed

        # Establish the range of x and y to plot, including a buffer
        x_min, x_max = (
            full_reconstructed_data[seq_index - sequence_length : seq_index, 0].min() - buffer,
            full_reconstructed_data[seq_index - sequence_length : seq_index + 10, 0].max() + buffer,
        )
        y_min, y_max = (
            full_reconstructed_data[seq_index - sequence_length : seq_index, 1].min() - buffer,
            full_reconstructed_data[seq_index - sequence_length : seq_index, 1].max() + buffer,
        )
        last_known_point = full_reconstructed_data[seq_index - 1, :2]

        grid_x = np.linspace(x_min, x_max, 50)
        grid_y = np.linspace(y_min, y_max, 50)

        # Generate the heatmap data from the model's predicted output
        pdf_total = mdn_to_heatmap(last_step_output, num_components, grid_x, grid_y, last_known_point)

        # Plotting
        print(f"Plotting sequence index {seq_index}")
        plt.figure(figsize=(10, 6))
        plt.axis("equal")
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.scatter(
            full_reconstructed_data[:, 0],
            full_reconstructed_data[:, 1],
            alpha=0.2,
            label="Full Dataset",
            color="grey",
        )
        plt.scatter(
            full_reconstructed_data[seq_index - sequence_length : seq_index, 0],
            full_reconstructed_data[seq_index - sequence_length : seq_index, 1],
            color="blue",
            label="Subsequence",
        )
        plt.scatter(
            last_known_point[0],
            last_known_point[1],
            color="green",
            label="Last Subsequence Point",
            zorder=50,
        )
        plt.scatter(
            full_reconstructed_data[seq_index, 0],
            full_reconstructed_data[seq_index, 1],
            color="red",
            label="Actual Next Point",
            zorder=100,
        )
        plt.contourf(grid_x, grid_y, pdf_total, levels=10, cmap="viridis", alpha=0.5)
        plt.colorbar(label="Probability Density")
        plt.legend()
        plt.title(f"Predictive Heatmap for Sequence Index {seq_index}")
        filename = f"{model_name}_heatmap_seq_{seq_index}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()


def calculate_pdf_vectorized(grid_x, grid_y, pis, mus1, mus2, sigmas1, sigmas2, rhos):
    """
    Calculate the PDF over a grid for a vector of mixture components, considering broadcasting.
    """
    # Adding new axes to parameters for broadcasting
    # Shape adjustments: we're considering the parameters to have shapes like (1, num_components),
    # thus, adding an extra dimension at the start for broadcasting with grid_x and grid_y
    pis = pis[tf.newaxis, tf.newaxis, :]
    mus1 = mus1[tf.newaxis, tf.newaxis, :]
    mus2 = mus2[tf.newaxis, tf.newaxis, :]
    sigmas1 = sigmas1[tf.newaxis, tf.newaxis, :]
    sigmas2 = sigmas2[tf.newaxis, tf.newaxis, :]
    rhos = rhos[tf.newaxis, tf.newaxis, :]

    # Ensuring grid_x and grid_y are compatible for broadcasting with parameters
    grid_x = grid_x[:, :, tf.newaxis]
    grid_y = grid_y[:, :, tf.newaxis]

    # PDF calculation as before, now should broadcast correctly
    z_x = ((grid_x - mus1) / sigmas1) ** 2
    z_y = ((grid_y - mus2) / sigmas2) ** 2
    z_xy = (2 * rhos * (grid_x - mus1) * (grid_y - mus2)) / (sigmas1 * sigmas2)

    z = z_x + z_y - z_xy
    denom = 2 * np.pi * sigmas1 * sigmas2 * tf.sqrt(1 - tf.square(rhos))
    pdf = tf.exp(-z / (2 * (1 - tf.square(rhos)))) / denom

    # Weight the PDF by the mixture coefficients and sum across the last dimension (components)
    weighted_pdf = tf.reduce_sum(pdf * pis, axis=-1)

    return weighted_pdf


def generate_and_plot_heatmap(
    model,
    model_name,
    training_data,
    reconstructed_data,
    num_components,
    model_pred_dir,
    grid_size=100,
):
    model_output = model.predict(training_data)
    model_output = model_output[0]  # Assuming model.predict returns batch first

    min_x, max_x = np.min(reconstructed_data[:, 0]), np.max(reconstructed_data[:, 0])
    min_y, max_y = np.min(reconstructed_data[:, 1]), np.max(reconstructed_data[:, 1])

    # Initialize an empty aggregate heatmap
    _aggregate_heatmap = np.zeros((grid_size, grid_size))

    cum_mu1 = np.cumsum(model_output[:, num_components : num_components * 2])
    cum_mu2 = np.cumsum(model_output[:, num_components * 2 : num_components * 3])

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(20, 8))

    # Plot reconstructed data
    axs[0].scatter(reconstructed_data[:, 0], reconstructed_data[:, 1])
    axs[0].set_title("Reconstructed Stroke Data")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_aspect("equal")

    # Plot raw mu predictions
    axs[1].scatter(cum_mu1, cum_mu2)
    axs[1].set_title("Reconstructed from Solely Means from MDN")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].set_xlim(min_x, max_x)
    axs[1].set_ylim(min_y, max_y)
    axs[1].set_aspect("equal")
    plt.show()


def generate_and_plot_heatmap_full(
    model,
    x_stroke,
    chars,
    char_len,
    reconstructed_data,
    num_components,
    grid_size=100,
):
    predictions = model(x_stroke, chars, char_len, training=False)
    print("predictions.shape", predictions.shape)
    model_output = predictions[0]  # Assuming model.predict returns batch first
    print("model_output", model_output)

    min_x, max_x = np.min(reconstructed_data[:, 0]), np.max(reconstructed_data[:, 0])
    min_y, max_y = np.min(reconstructed_data[:, 1]), np.max(reconstructed_data[:, 1])

    # Initialize an empty aggregate heatmap
    _aggregate_heatmap = np.zeros((grid_size, grid_size))

    cum_mu1 = np.cumsum(model_output[:, num_components : num_components * 2])
    cum_mu2 = np.cumsum(model_output[:, num_components * 2 : num_components * 3])

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(20, 8))

    # Plot reconstructed data
    axs[0].scatter(reconstructed_data[:, 0], reconstructed_data[:, 1])
    axs[0].set_title("Reconstructed Stroke Data")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_aspect("equal")

    # Plot raw mu predictions
    axs[1].scatter(cum_mu1, cum_mu2)
    axs[1].set_title("Reconstructed from Solely Means from MDN")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].set_xlim(min_x, max_x)
    axs[1].set_ylim(min_y, max_y)
    axs[1].set_aspect("equal")
    plt.show()
