#!/usr/bin/env python3
"""Minimal script to overfit the prediction model on example-stroke.xml.

The goal is to teach the DeepHandwritingPredictionModel to perfectly memorize
one sequence (the word "hello") and then autoregressively resample it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model.handwriting_models import DeepHandwritingPredictionModel

# Force CPU – this run is tiny and we want consistent behaviour across machines.
tf.config.set_visible_devices([], "GPU")
tf.keras.mixed_precision.set_global_policy("float32")

np.random.seed(42)
tf.random.set_seed(42)


@dataclass
class StrokeDataset:
    """Container for the processed stroke data."""

    text: str
    deltas: np.ndarray  # shape (seq_len, 3) in absolute deltas (not normalised)
    norm_deltas: np.ndarray  # shape (seq_len, 3)
    norm_mu: np.ndarray  # shape (2,)
    norm_sigma: np.ndarray  # shape (2,)
    stroke_centroid: Tuple[float, float]


@dataclass
class GeneratedSample:
    """Output from the autoregressive sampler."""

    deltas: np.ndarray  # normalised deltas
    coords: np.ndarray  # absolute coordinates (centred back to original frame)


# ---------------------------------------------------------------------------
# Data preparation utilities
# ---------------------------------------------------------------------------


def parse_example_stroke_xml(xml_file: str = "example-stroke.xml") -> List[Tuple[float, float, bool]]:
    """Parse IAM-style XML and return a list of (x, y, pen_up) points."""

    import xml.etree.ElementTree as ET

    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"Example stroke file not found: {xml_file}")

    tree = ET.parse(xml_file)
    root = tree.getroot()
    stroke_points: list[tuple[float, float, bool]] = []

    for stroke in root.findall(".//Stroke"):
        points = []
        for point in stroke.findall("Point"):
            x = float(point.get("x"))
            y = float(point.get("y"))
            points.append((x, y))

        for i, (x, y) in enumerate(points):
            pen_up = i == len(points) - 1
            stroke_points.append((x, y, pen_up))

    return stroke_points


def absolute_to_deltas(points: Sequence[Tuple[float, float, bool]]) -> tuple[np.ndarray, tuple[float, float]]:
    """Convert absolute stroke positions into centred deltas."""

    if not points:
        raise ValueError("No points to convert")

    coords = np.array([(x, y) for x, y, _ in points], dtype=np.float32)
    pen = np.array([float(pen_up) for _, _, pen_up in points], dtype=np.float32)

    # Recentre the coordinates to avoid very large values during training.
    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    centroid = (min_xy + max_xy) / 2.0
    centred = coords - centroid

    deltas = np.zeros((len(points), 3), dtype=np.float32)
    prev = np.zeros(2, dtype=np.float32)

    for idx, xy in enumerate(centred):
        deltas[idx, :2] = xy - prev
        deltas[idx, 2] = pen[idx]
        prev = xy

    return deltas, (float(centroid[0]), float(centroid[1]))


def normalise_deltas(deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalisation for dx/dy, pen state untouched."""

    mu = deltas[:, :2].mean(axis=0)
    sigma = deltas[:, :2].std(axis=0) + 1e-6
    normalised = deltas.copy()
    normalised[:, :2] = (deltas[:, :2] - mu) / sigma
    return normalised, mu.astype(np.float32), sigma.astype(np.float32)


def load_example_stroke(text: str = "hello") -> StrokeDataset:
    """Prepare the stroke dataset for training/visualisation."""

    raw_points = parse_example_stroke_xml()
    deltas, centroid = absolute_to_deltas(raw_points)
    norm_deltas, mu, sigma = normalise_deltas(deltas)

    return StrokeDataset(
        text=text,
        deltas=deltas,
        norm_deltas=norm_deltas,
        norm_mu=mu,
        norm_sigma=sigma,
        stroke_centroid=centroid,
    )


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------


def build_prediction_model(num_components: int = 20) -> DeepHandwritingPredictionModel:
    model = DeepHandwritingPredictionModel(
        units=256,
        num_layers=3,
        num_mixture_components=num_components,
        name="overfit_prediction",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=None, run_eagerly=False)
    return model


def prepare_training_tensors(dataset: StrokeDataset, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create repeated training samples and valid lengths for masking."""

    seq = dataset.norm_deltas
    seq_len = len(seq)

    inputs = np.tile(seq[None, :, :], (batch_size, 1, 1)).astype(np.float32)
    targets = np.zeros_like(inputs)
    targets[:, :-1, :] = inputs[:, 1:, :]
    targets[:, -1, :] = 0.0  # final step is dummy and will be masked

    lengths = np.full((batch_size,), seq_len - 1, dtype=np.int32)

    return inputs, targets, lengths


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _extract_mdn_params(mdn_output: np.ndarray, num_components: int) -> dict[str, np.ndarray]:
    pi = mdn_output[:num_components]
    mu_x = mdn_output[num_components : 2 * num_components]
    mu_y = mdn_output[2 * num_components : 3 * num_components]
    sigma_x = mdn_output[3 * num_components : 4 * num_components]
    sigma_y = mdn_output[4 * num_components : 5 * num_components]
    rho = mdn_output[5 * num_components : 6 * num_components]
    eos_logit = mdn_output[6 * num_components : 6 * num_components + 1]

    return {
        "pi": pi,
        "mu_x": mu_x,
        "mu_y": mu_y,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "rho": rho,
        "eos": eos_logit,
    }


def greedy_autoregressive_sample(
    model: DeepHandwritingPredictionModel,
    dataset: StrokeDataset,
    start_step: np.ndarray | None = None,
    num_components: int = 20,
) -> GeneratedSample:
    """Generate a sequence by repeatedly picking the dominant mixture component."""

    norm_seq = dataset.norm_deltas
    seq_len = len(norm_seq)

    if start_step is None:
        # Use the ground-truth first step to bootstrap. This mirrors training.
        start_step = norm_seq[0]

    generated = [start_step.astype(np.float32)]

    for _ in range(seq_len - 1):
        context = np.array(generated, dtype=np.float32)[None, :, :]
        mdn_raw = model(context, training=False)[0, -1].numpy()
        params = _extract_mdn_params(mdn_raw, num_components)

        dominant = int(np.argmax(params["pi"]))
        dx = params["mu_x"][dominant]
        dy = params["mu_y"][dominant]
        eos_prob = 1.0 / (1.0 + math.exp(-float(params["eos"][0])))
        # Use stochastic sampling instead of deterministic threshold
        if eos_prob < 0.01:
            pen_up = 0.0
        else:
            pen_up = 1.0 if np.random.random() < eos_prob else 0.0

        generated.append(np.array([dx, dy, pen_up], dtype=np.float32))

    generated_arr = np.vstack(generated)

    # De-normalise the deltas back to original space
    deltas_denorm = generated_arr.copy()
    deltas_denorm[:, :2] = deltas_denorm[:, :2] * dataset.norm_sigma + dataset.norm_mu

    # Integrate to absolute coordinates within the centred frame
    coords = np.zeros_like(deltas_denorm)
    current = np.zeros(2, dtype=np.float32)
    for idx, (dx, dy, pen_state) in enumerate(deltas_denorm):
        current = current + np.array([dx, dy], dtype=np.float32)
        coords[idx, 0:2] = current
        coords[idx, 2] = pen_state

    # Shift coordinates back to the original centroid position
    coords[:, 0] += dataset.stroke_centroid[0]
    coords[:, 1] += dataset.stroke_centroid[1]

    return GeneratedSample(deltas=generated_arr, coords=coords)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def plot_strokes(coords: np.ndarray, title: str, outfile: str) -> None:
    """Render a stroke sequence as a PNG."""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_aspect("equal")

    # Split strokes whenever pen lifts.
    start = 0
    for idx, pen_up in enumerate(coords[:, 2]):
        if pen_up > 0.5:
            segment = coords[start : idx + 1]
            if len(segment) > 1:
                ax.plot(segment[:, 0], -segment[:, 1], "k-", linewidth=2)
            start = idx + 1

    if start < len(coords):
        segment = coords[start:]
        if len(segment) > 1:
            ax.plot(segment[:, 0], -segment[:, 1], "k-", linewidth=2)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Overfit DeepHandwritingPredictionModel (example-stroke.xml) ===")

    dataset = load_example_stroke()
    print(f"Loaded {len(dataset.deltas)} stroke points for text '{dataset.text}'.")

    inputs, targets, lengths = prepare_training_tensors(dataset, batch_size=64)
    print(f"Training tensor shape: {inputs.shape}")

    num_components = 1

    model = build_prediction_model(num_components=num_components)

    train_batch_size = 16
    train_ds = (
        tf.data.Dataset.from_tensor_slices((inputs, targets, lengths))
        .shuffle(inputs.shape[0])
        .batch(train_batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    history = model.fit(
        train_ds,
        epochs=400,
    )

    print(f"Initial loss: {history.history['loss'][0]:.6f}")
    print(f"Final loss:   {history.history['loss'][-1]:.6f}")

    # Save a model copy for later reuse
    model_path = "overfit_prediction_example.keras"
    model.save(model_path)
    print(f"Saved overfitted model to {model_path}")

    sample = greedy_autoregressive_sample(model, dataset, num_components=num_components)

    # Visualise targets vs generated output
    target_coords = np.zeros_like(dataset.deltas)
    current = np.zeros(2, dtype=np.float32)
    for idx, (dx, dy, pen) in enumerate(dataset.deltas):
        current = current + np.array([dx, dy], dtype=np.float32)
        target_coords[idx, :2] = current + np.array(dataset.stroke_centroid)
        target_coords[idx, 2] = pen

    os.makedirs("prediction_visualizations", exist_ok=True)
    target_path = os.path.join("prediction_visualizations", "prediction_target.png")
    generated_path = os.path.join("prediction_visualizations", "prediction_generated.png")

    plot_strokes(target_coords, "Target sequence", target_path)
    plot_strokes(sample.coords, "Model reconstruction", generated_path)

    print("Output files:")
    print(f"  - {target_path}")
    print(f"  - {generated_path}")


if __name__ == "__main__":
    main()
