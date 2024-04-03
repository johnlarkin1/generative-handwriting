import os
from common import (
    mdn_to_heatmap,
)
from model.lstm_peephole_cell import LSTMCellWithPeepholes
from model.handwriting_mdn import MDNLayer, mdn_loss
from model.handwriting_prediction import DeepHandwritingPredictionModel
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np


def plot_predictions_through_sequence(
    model, model_name, training_data, full_reconstructed_data, num_components, sequence_length, save_dir
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
            full_reconstructed_data[:, 0], full_reconstructed_data[:, 1], alpha=0.2, label="Full Dataset", color="grey"
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
