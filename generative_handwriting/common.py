import glob
import os
import xml.etree.ElementTree as ET

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def check_files_exist(file_paths: list[str]) -> bool:
    """Check if all the required files exist."""
    return all(os.path.isfile(file_path) for file_path in file_paths)


def create_sequences(data, sequence_length):
    """
    Creates sequences for both inputs and targets with the specified sequence length.

    Args:
    data: Numpy array of shape (num_points, features), where features include (x, y) and possibly EoS.
    sequence_length: The length of each input sequence for the LSTM.

    Returns:
    x_sequences: Input sequences of shape [num_samples, sequence_length, features].
    y_sequences: Target sequences of shape [num_samples, sequence_length, features].
    """
    x_sequences, y_sequences = [], []
    for i in range(len(data) - sequence_length - 1):  # Adjusted to avoid index error
        x_seq = data[i : i + sequence_length]
        y_seq = data[i + 1 : i + sequence_length + 1]  # Next sequence as target
        x_sequences.append(x_seq)
        y_sequences.append(y_seq)
    return np.array(x_sequences), np.array(y_sequences)


def plot_strokes_from_dx_dy(stroke_data, show_image: bool = False, title="Handwriting Stroke Plot"):
    """
    Plot the handwriting strokes.

    Args:
        stroke_data: A numpy array of shape (n, 3) where each row contains dx, dy, and eos (end-of-stroke flag).
    """
    # Initialize the plot
    plt.figure(figsize=(8, 5))

    # Reconstruct the absolute coordinates
    x = np.cumsum(stroke_data[:, 0])
    y = np.cumsum(stroke_data[:, 1])

    # Start a new stroke
    stroke_start = 0

    # Iterate through the strokes to plot them
    for i in range(1, len(stroke_data)):
        if stroke_data[i, 2] == 1:  # If it's the end of a stroke
            if show_image:
                plt.plot(x[stroke_start : i + 1], y[stroke_start : i + 1])
            stroke_start = i + 1  # Set the next stroke start

    # Ensure the last segment is plotted if not finished by an end-of-stroke flag
    if show_image and stroke_start < len(stroke_data):
        plt.plot(x[stroke_start:], y[stroke_start:])

    if show_image:
        plt.axis("equal")
        plt.title(title)
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.show()
    plt.close()
    return np.column_stack((x, y, stroke_data[:, 2]))


def plot_original_strokes_from_xml(filename, show_image: bool = False) -> None:
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/lineStrokes", filename)
    root = ET.parse(file_path)

    # Extract stroke points
    points = []
    for stroke in root.find("StrokeSet"):
        stroke_points = []
        for point in stroke:
            x = int(point.get("x"))
            y = int(point.get("y"))
            stroke_points.append((x, y))
        if stroke_points:
            points.append(stroke_points)

    # Plotting
    plt.figure(figsize=(10, 6))

    for stroke in points:
        x, y = zip(*stroke, strict=False)  # This unzips into x and y coordinates
        plt.plot(x, y)

    plt.axis("equal")
    plt.gca().invert_yaxis()  # Invert y axis to match the expected handwriting orientation
    plt.title("Handwriting Stroke Data")
    plt.show()


def create_gif(image_folder, matching_source_file_pattern, gif_name, fps: int = 5):
    images = []
    for filename in sorted(
        glob.glob(f"{image_folder}/{matching_source_file_pattern}.png"),
        key=os.path.getmtime,
    ):
        images.append(imageio.imread(filename))
    imageio.mimsave(f"{gif_name}.gif", images, fps=fps)


def calculate_pdf(x, y, mu1, mu2, sigma1, sigma2, rho):
    """
    Calculate the value of a bivariate Gaussian PDF.
    """
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    mu1 = tf.cast(mu1, tf.float32)
    mu2 = tf.cast(mu2, tf.float32)
    sigma1 = tf.cast(sigma1, tf.float32)
    sigma2 = tf.cast(sigma2, tf.float32)
    rho = tf.cast(rho, tf.float32)

    z = (
        tf.square((x - mu1) / sigma1)
        + tf.square((y - mu2) / sigma2)
        - 2 * rho * (x - mu1) * (y - mu2) / (sigma1 * sigma2)
    )
    denom = 2 * np.pi * sigma1 * sigma2 * tf.sqrt(1 - tf.square(rho))
    result = tf.exp(-z / (2 * (1 - tf.square(rho)))) / denom
    return result


def mdn_to_heatmap(last_step_output, num_components, grid_x, grid_y, offset):
    # Extract parameters
    pis = last_step_output[0, :num_components]  # Mixture weights
    mu1 = last_step_output[0, num_components : num_components * 2]  # Means for x
    mu2 = last_step_output[0, num_components * 2 : num_components * 3]  # Means for y
    sigma1 = last_step_output[0, num_components * 3 : num_components * 4]  # Standard deviations for x
    sigma2 = last_step_output[0, num_components * 4 : num_components * 5]  # Standard deviations for y
    rhos = last_step_output[0, num_components * 5 : num_components * 6]  # Correlation coefficients
    eos = last_step_output[0, -1]  # End-of-stroke probabilities

    X, Y = np.meshgrid(grid_x, grid_y)
    pdf_total = np.zeros_like(X)

    print("pis:", pis)
    print("mu1:", mu1)
    print("mu2:", mu2)
    print("sigma1:", sigma1)
    print("sigma2:", sigma2)
    print("rhos:", rhos)
    print("eos:", eos)
    print("offset:", offset)
    # Calculate the PDF for each grid point
    for j in range(X.shape[0]):
        for k in range(Y.shape[1]):
            x = X[j, k]
            y = Y[j, k]
            pdf_sum = 0  # Sum of PDFs for this grid point
            for i in range(num_components):
                # Apply the offset to the means
                mu1_offset = mu1[i] + offset[0]
                mu2_offset = mu2[i] + offset[1]

                # Calculate the PDF for the i-th mixture component
                pdf = calculate_pdf(x, y, mu1_offset, mu2_offset, sigma1[i], sigma2[i], rhos[i]) * pis[i]
                pdf_sum += pdf  # Aggregate the PDF over all mixture components

            pdf_total[j, k] = pdf_sum

    pdf_sum = np.sum(pdf_total)
    if pdf_sum != 0:
        normalized_pdf_total = pdf_total / pdf_sum
    else:
        normalized_pdf_total = np.zeros_like(pdf_total)
    return normalized_pdf_total


def calculate_individual_mixture_pdf(x, y, mu1, mu2, sigma1, sigma2, rho):
    """Calculate the probability density function of a bivariate Gaussian."""
    z = (
        ((x - mu1) ** 2 / sigma1**2)
        + ((y - mu2) ** 2 / sigma2**2)
        - (2 * rho * (x - mu1) * (y - mu2) / (sigma1 * sigma2))
    )
    denom = 2 * np.pi * sigma1 * sigma2 * np.sqrt(1 - rho**2)
    return np.exp(-z / (2 * (1 - rho**2))) / denom


def mdn_to_heatmap_many(last_step_output, num_components, grid_x, grid_y, sequence):
    # Calculate the last known point from the sequence
    last_known_point = np.cumsum(sequence, axis=0)[-1, :2]  # Assuming sequence is [dx, dy, eos]

    # Extract parameters from the last step output
    pis = last_step_output[0, :num_components]
    mu1 = last_step_output[0, num_components : num_components * 2]
    mu2 = last_step_output[0, num_components * 2 : num_components * 3]
    sigma1 = last_step_output[0, num_components * 3 : num_components * 4]
    sigma2 = last_step_output[0, num_components * 4 : num_components * 5]
    rhos = last_step_output[0, num_components * 5 : num_components * 6]

    X, Y = np.meshgrid(grid_x, grid_y)

    # Initialize a list to hold the PDF for each mixture component
    pdfs = []

    for i in range(num_components):
        # Apply the offset to the means
        mu1_offset = mu1[i] + last_known_point[0]
        mu2_offset = mu2[i] + last_known_point[1]

        pdf = np.zeros_like(X)
        for j in range(X.shape[0]):
            for k in range(Y.shape[1]):
                x = X[j, k]
                y = Y[j, k]
                pdf[j, k] = (
                    calculate_individual_mixture_pdf(x, y, mu1_offset, mu2_offset, sigma1[i], sigma2[i], rhos[i])
                    * pis[i]
                )

        pdfs.append(pdf)
    return pdfs  # Return a list of PDFs for each mixture component


def print_model_parameters(model):
    def print_title(title):
        print()
        print(title)
        print("=" * len(title))

    print_title("All parameters:")
    for var in model.variables:
        print(f"[[ {var.name} ]] shape: {var.shape}")

    print_title("Trainable parameters:")
    for var in model.trainable_variables:
        print(f"[[ {var.name} ]] shape: {var.shape}")

    trainable_params = np.sum([np.prod(v.shape.as_list()) for v in model.trainable_variables])
    print_title("Trainable parameter count:")
    print(trainable_params)


