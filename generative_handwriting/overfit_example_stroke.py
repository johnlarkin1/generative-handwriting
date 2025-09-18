#!/usr/bin/env python3
"""
Overfitting test using the example-stroke.xml data.

This script uses the specific stroke data from example-stroke.xml for a focused
overfitting test to verify the model can learn from real handwriting data.
"""

import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from alphabet import ALPHABET_SIZE, encode_ascii
from model.handwriting_models import DeepHandwritingSynthesisModel
from writer import Calligrapher

# Force CPU to avoid GPU memory issues during debugging
tf.config.set_visible_devices([], "GPU")
tf.keras.mixed_precision.set_global_policy("float32")


def parse_example_stroke_xml(xml_file: str = "example-stroke.xml") -> List[Tuple[float, float, bool]]:
    """Parse the example stroke XML file and return list of (x, y, pen_up) tuples."""

    if not os.path.exists(xml_file):
        print(f"Example stroke file not found: {xml_file}")
        return []

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        strokes = []

        # Find all stroke elements
        for stroke in root.findall(".//Stroke"):
            stroke_points = []

            # Get all points in this stroke
            for point in stroke.findall("Point"):
                x = float(point.get("x"))
                y = float(point.get("y"))
                stroke_points.append((x, y))

            # Add stroke points with pen_up=False except for the last point
            for i, (x, y) in enumerate(stroke_points):
                pen_up = i == len(stroke_points) - 1  # Pen up at end of stroke
                strokes.append((x, y, pen_up))

        print(f"Loaded {len(strokes)} stroke points from {xml_file}")
        print(f"{strokes=}")
        return strokes

    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return []


def convert_to_deltas(absolute_points: List[Tuple[float, float, bool]]) -> np.ndarray:
    """Convert absolute coordinates to delta movements."""

    if not absolute_points:
        return np.array([])

    # CRITICAL FIX: Center the coordinates first to avoid huge absolute values
    # Find the bounding box and center the coordinates
    coords = np.array([(x, y) for x, y, _ in absolute_points])
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    print(f"Original coordinates range: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
    print(f"Centering at: ({x_center:.1f}, {y_center:.1f})")

    deltas = []
    prev_x, prev_y = 0.0, 0.0  # Start from origin in centered space

    for x, y, pen_up in absolute_points:
        # Center the coordinates
        x_centered = x - x_center
        y_centered = y - y_center

        dx = x_centered - prev_x
        dy = y_centered - prev_y
        deltas.append([dx, dy, float(pen_up)])
        prev_x, prev_y = x_centered, y_centered

    return np.array(deltas, dtype=np.float32)


def normalize_strokes(strokes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize stroke coordinates using dataset-level statistics.

    Returns:
        normalized_strokes: Normalized stroke data
        mu: Mean of deltas (x, y)
        sigma: Standard deviation of deltas (x, y)
    """

    if len(strokes) == 0:
        return strokes, np.zeros(2), np.ones(2)

    # Create copy to avoid modifying original
    normalized = strokes.copy()

    # Compute dataset-level statistics for x, y deltas (not pen state)
    deltas = strokes[:, :2]  # Only x, y coordinates
    mu = deltas.mean(axis=0)  # Mean of deltas
    sigma = deltas.std(axis=0) + 1e-6  # Std of deltas + epsilon

    # Normalize only x and y coordinates, keep pen state as-is
    normalized[:, :2] = (deltas - mu) / sigma

    return normalized, mu, sigma


def visualize_target_sequence(stroke_seq, text, filename="target_example.png"):
    """Create a matplotlib visualization of the target handwriting sequence."""

    print(f"\nCreating target visualization: {filename}")

    # Convert deltas to absolute coordinates
    coords = []
    pen_states = []
    x, y = 0, 0

    for dx, dy, pen_up in stroke_seq:
        x += dx
        y += dy
        coords.append((x, y))
        pen_states.append(pen_up)

    if not coords:
        print("No coordinates to visualize")
        return

    coords = np.array(coords)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_aspect("equal")

    # Draw strokes
    if len(coords) > 0:
        # Group coordinates by stroke (pen up/down segments)
        current_stroke = []
        stroke_colors = ["blue", "green", "red", "purple", "orange"]  # Different colors for each stroke
        stroke_idx = 0

        for i, (x, y) in enumerate(coords):
            current_stroke.append((x, y))

            # If pen is up or this is the last point, draw the stroke
            if pen_states[i] > 0.5 or i == len(coords) - 1:
                if len(current_stroke) > 1:
                    stroke_coords = np.array(current_stroke)
                    color = stroke_colors[stroke_idx % len(stroke_colors)]
                    ax.plot(
                        stroke_coords[:, 0],
                        -stroke_coords[:, 1],
                        "-",
                        color=color,
                        linewidth=2,
                        label=f"Stroke {stroke_idx + 1}",
                    )
                    stroke_idx += 1

                # Mark pen-up points
                if len(current_stroke) > 0:
                    ax.plot(current_stroke[-1][0], -current_stroke[-1][1], "ro", markersize=4, alpha=0.7)

                current_stroke = []

    ax.set_title(f'Example Handwriting: "{text}"', fontsize=16, fontweight="bold")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add stats as text
    ax.text(
        0.02,
        0.98,
        f"{len(stroke_seq)} stroke points, {stroke_idx} strokes",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Target sequence saved as '{filename}'")
    return filename


def load_example_stroke_data():
    """Load and process the example stroke data."""

    print("Loading example stroke data...")

    # Parse the XML file
    absolute_strokes = parse_example_stroke_xml()
    if not absolute_strokes:
        return None, None

    # Convert to deltas
    delta_strokes = convert_to_deltas(absolute_strokes)
    print(f"{delta_strokes=}")

    # Normalize using dataset-level statistics
    normalized_strokes, mu, sigma = normalize_strokes(delta_strokes)

    # The strokes represent "hello" (5 letters)
    text = "hello"

    print(f"  ✓ Processed {len(normalized_strokes)} stroke points")
    print(
        f"  Stroke range: x=[{normalized_strokes[:, 0].min():.3f}, {normalized_strokes[:, 0].max():.3f}], "
        f"y=[{normalized_strokes[:, 1].min():.3f}, {normalized_strokes[:, 1].max():.3f}]"
    )

    return {
        "normalized_strokes": normalized_strokes,
        "text": text,
        "norm_mu": mu,
        "norm_sigma": sigma,
        "original_strokes": delta_strokes,
    }


def create_minimal_dataset(data_dict):
    """Create a minimal dataset with the example sequence repeated multiple times."""

    # Extract data from dict
    stroke_seq = data_dict["normalized_strokes"]
    text = data_dict["text"]

    # Encode text
    char_seq = encode_ascii(text.lower())

    print("\nCreated dataset:")
    print(f"  Text: '{text}' -> {char_seq}")
    print(f"  Strokes: {stroke_seq.shape} points")

    # Repeat the same sequence multiple times for batch training
    batch_size = 8
    seq_len = len(stroke_seq)
    char_len = len(char_seq)

    # Pad sequences to accommodate the data
    max_stroke_len = max(300, seq_len + 50)  # Ensure it fits with margin
    max_char_len = max(30, char_len + 10)

    # Create batched data
    x_strokes = np.zeros((batch_size, max_stroke_len, 3), dtype=np.float32)
    x_chars = np.zeros((batch_size, max_char_len), dtype=np.int32)
    x_stroke_lens = np.full(batch_size, seq_len, dtype=np.int32)
    x_char_lens = np.full(batch_size, char_len, dtype=np.int32)

    # Fill with repeated data (with slight variations)
    for i in range(batch_size):
        # Add very slight variation to each copy
        variation = np.random.normal(0, 0.001, stroke_seq.shape)  # Tiny variation
        variation[:, 2] = 0  # Keep pen state exact
        varied_strokes = stroke_seq + variation

        x_strokes[i, :seq_len] = varied_strokes
        x_chars[i, :char_len] = char_seq

    # Create target data (shifted by one timestep)
    y_strokes = np.zeros_like(x_strokes)
    y_strokes[:, :-1] = x_strokes[:, 1:]  # Shift left
    y_stroke_lens = x_stroke_lens - 1

    return {
        "x_strokes": x_strokes,
        "x_stroke_lens": x_stroke_lens,
        "x_chars": x_chars,
        "x_char_lens": x_char_lens,
        "y_strokes": y_strokes,
        "y_stroke_lens": y_stroke_lens,
        "original_text": text,
        "original_strokes": data_dict["original_strokes"],  # Keep original deltas
        "normalized_strokes": stroke_seq,  # This is the normalized version for visualization
        "norm_mu": data_dict["norm_mu"],
        "norm_sigma": data_dict["norm_sigma"],
    }


def train_overfit_model(data, epochs=75):
    """Train model to overfit on the example sequence."""

    print("\nCreating model...")

    # Enable TensorFlow debugging output
    tf.debugging.set_log_device_placement(False)  # Set to True to see device placement
    tf.config.optimizer.set_jit(False)  # Disable XLA for better debugging

    # Create model for example stroke training with increased attention scale for short text
    model = DeepHandwritingSynthesisModel(
        units=128,  # Reasonable size for real handwriting
        num_layers=3,  # Must be 3 (hardcoded in the model architecture)
        num_mixture_components=10,  # More mixtures for complex handwriting
        num_chars=ALPHABET_SIZE,
        num_attention_gaussians=5,  # More attention components
        gradient_clip_value=1.0,
        enable_mdn_regularization=False,  # Disable regularization for overfitting
        attention_kappa_scale=1 / 10.0,  # Increased from default 1/25 for short text like "hello"
    )

    # Use moderate learning rate with gradient clipping at optimizer level
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3, global_clipnorm=1.0),  # Add gradient clipping
        loss=None,
        run_eagerly=False,
    )

    # Create dataset
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                {
                    "input_strokes": data["x_strokes"],
                    "input_stroke_lens": data["x_stroke_lens"],
                    "target_stroke_lens": data["y_stroke_lens"],
                    "input_chars": data["x_chars"],
                    "input_char_lens": data["x_char_lens"],
                },
                data["y_strokes"],
            )
        )
        .batch(8)
        .repeat()
    )

    print(f"Training for {epochs} epochs...")
    print("Expecting loss to decrease as model learns the example handwriting...")

    # Create TensorBoard log directory
    log_dir = f"logs/overfit_example/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print("Run 'tensorboard --logdir logs/overfit_example' to view training progress")

    # Custom callback to monitor loss and model internals
    class DetailedMonitor(tf.keras.callbacks.Callback):
        def __init__(self):
            self.losses = []
            self.batch_losses = []
            self.gradients = []

        def on_batch_end(self, batch, logs=None):
            if logs:
                self.batch_losses.append(logs.get("loss", 0))
                # Log every 5 batches for detailed monitoring
                if batch % 5 == 0:
                    print(f"    Batch {batch}: loss = {logs.get('loss', 0):.6f}")

        def on_epoch_end(self, epoch, logs=None):
            current_loss = logs.get("loss", 0)
            self.losses.append(current_loss)

            # More detailed logging for debugging
            if epoch % 10 == 0 or epoch < 5:
                nll = logs.get("nll", 0)
                print(f"\n  Epoch {epoch:3d}: loss = {current_loss:.6f}, nll = {nll:.6f}")

                # Debug: Check if loss is actually zero
                if current_loss == 0.0:
                    print(f"    WARNING: Loss is exactly zero at epoch {epoch}!")

                # Check model weights for extreme values
                for layer in self.model.layers:
                    if hasattr(layer, "weights") and layer.weights:
                        for weight in layer.weights:
                            values = weight.numpy()
                            max_val = np.abs(values).max()
                            mean_val = np.abs(values).mean()
                            if max_val > 100 or np.isnan(max_val):
                                print(
                                    f"    WARNING: Layer {layer.name}, weight {weight.name}: max={max_val:.2f}, mean={mean_val:.2f}"
                                )
                            elif epoch == 0:
                                print(
                                    f"    Layer {layer.name}, weight {weight.name}: max={max_val:.2f}, mean={mean_val:.2f}"
                                )

                # Log attention and MDN statistics if available
                if epoch % 20 == 0 and epoch > 0:
                    print(f"    Average batch loss (last 10): {np.mean(self.batch_losses[-10:]):.6f}")
                    print(
                        f"    Loss trend: {self.losses[0]:.4f} -> {self.losses[-1]:.4f} (reduction: {(1 - self.losses[-1] / self.losses[0]) * 100:.1f}%)"
                    )

    monitor = DetailedMonitor()

    # Add TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=10,  # Log weight histograms every 10 epochs
        write_graph=True,
        write_images=False,
        update_freq="epoch",
        profile_batch=0,  # Disable profiling for simplicity
    )

    # Early stopping callback to prevent overfitting too much
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=50,
        restore_best_weights=False,  # We want to overfit, so don't restore
        verbose=1,
        min_delta=1e-6,
    )

    # Model checkpoint callback - save every 25 epochs
    class PeriodicCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 25 == 0:
                filepath = f"overfit_checkpoints/epoch_{epoch + 1:03d}_loss_{logs.get('loss', 0):.4f}.keras"
                self.model.save(filepath)
                print(f"    Checkpoint saved: {filepath}")

    checkpoint_callback = PeriodicCheckpoint()

    # Create checkpoints directory if it doesn't exist
    os.makedirs("overfit_checkpoints", exist_ok=True)
    os.makedirs("logs/overfit_example", exist_ok=True)

    # Train with all callbacks
    history = model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=15,  # More steps for real data
        callbacks=[monitor, tensorboard_callback, early_stopping, checkpoint_callback],
        verbose=0,
    )

    print("\nTraining complete!")
    print(f"  Initial loss: {monitor.losses[0]:.4f}")
    print(f"  Final loss:   {monitor.losses[-1]:.4f}")
    print(f"  Loss reduction: {monitor.losses[0] - monitor.losses[-1]:.4f}")

    if monitor.losses[-1] < monitor.losses[0] * 0.3:  # Reasonable for real data
        print("✓ Good! Loss decreased significantly (model is learning)")
    else:
        print("⚠ Warning: Loss didn't decrease much (potential issue)")

    return model, monitor.losses


def visualize_generated_sequence(generated_seq, text, filename="generated_example.png"):
    """Create a matplotlib visualization of the generated handwriting sequence."""

    print(f"Creating generated visualization: {filename}")

    if not generated_seq or len(generated_seq) == 0:
        print("No generated sequence to visualize")
        return

    # Extract coordinates and pen states from generated sequence
    coords = np.array([(x, y) for x, y, _ in generated_seq])
    pen_states = [pen_up for _, _, pen_up in generated_seq]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_aspect("equal")

    # Draw strokes
    if len(coords) > 0:
        # Group coordinates by stroke (pen up/down segments)
        current_stroke = []
        stroke_colors = ["blue", "green", "red", "purple", "orange"]
        stroke_idx = 0

        for i, (x, y) in enumerate(coords):
            current_stroke.append((x, y))

            # If pen is up or this is the last point, draw the stroke
            if pen_states[i] > 0.5 or i == len(coords) - 1:
                if len(current_stroke) > 1:
                    stroke_coords = np.array(current_stroke)
                    color = stroke_colors[stroke_idx % len(stroke_colors)]
                    ax.plot(
                        stroke_coords[:, 0],
                        -stroke_coords[:, 1],
                        "-",
                        color=color,
                        linewidth=2,
                        label=f"Stroke {stroke_idx + 1}",
                    )
                    stroke_idx += 1

                # Mark pen-up points
                if len(current_stroke) > 0:
                    ax.plot(current_stroke[-1][0], -current_stroke[-1][1], "ro", markersize=4, alpha=0.7)

                current_stroke = []

    ax.set_title(f'Generated Handwriting: "{text}"', fontsize=16, fontweight="bold")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add stats as text
    ax.text(
        0.02,
        0.98,
        f"{len(generated_seq)} generated points, {stroke_idx} strokes",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Generated sequence saved as '{filename}'")
    return filename


def test_generation(model, original_text, original_strokes, norm_mu=None, norm_sigma=None):
    """Test generation with the overfitted example model."""

    print("\nTesting generation...")

    # Debug: Check what the model expects
    print(f"Debug - Text to generate: '{original_text}'")
    print(f"Debug - Text length: {len(original_text)} characters")
    print(f"Debug - Original strokes had {len(original_strokes)} points")
    print(f"Debug - Original strokes had {int(original_strokes[:, 2].sum())} pen lifts")

    # Save model temporarily
    temp_model_path = "temp_example_model.keras"
    model.save(temp_model_path)

    try:
        # Create calligrapher with normalization stats
        norm_stats_path = "overfit_example_norm_stats.json"
        print(f"Debug - Looking for norm stats at: {norm_stats_path}")
        print(f"Debug - Norm stats file exists: {os.path.exists(norm_stats_path)}")

        # Print the actual normalization stats used in training
        if norm_mu is not None and norm_sigma is not None:
            print(f"Debug - Training μ: {norm_mu}")
            print(f"Debug - Training σ: {norm_sigma}")
        else:
            print("Debug - No normalization stats provided")

        calligrapher = Calligrapher(temp_model_path, num_output_mixtures=10, norm_stats_path=norm_stats_path)

        # DEBUGGING CONFIGURATIONS - systematic testing
        test_settings = [
            # Test 1: FORCE pen lifts every 30 steps (manual stroke separation)
            ("manual_pen_lifts", 0.1, 0.8, False, 20, True, 0.1, 0.05),
            # Test 2: Very aggressive thresholds (almost any EOS activity = pen lift)
            ("ultra_aggressive_eos", 0.4, 0.0, False, 20, False, 0.001, 0.001),
            # Test 3: Original for comparison
            ("original_bernoulli", 0.4, 0.0, True, 20, False, 0.5, 0.5),
        ]

        best_sequence = None
        best_style = None

        for style_name, temp, bias, use_bernoulli, burn_in, greedy, eos_up, eos_down in test_settings:
            print(f"\n  Testing {style_name}:")
            print(f"    temp={temp}, bias={bias}, greedy={greedy}")
            print(f"    bernoulli_eos={use_bernoulli}, eos_thresholds={eos_up}/{eos_down}, burn_in={burn_in}")

            try:
                sequences, eos_probs = calligrapher.sample(
                    [original_text.lower()],
                    temperature=temp,
                    bias=bias,
                    max_steps=min(300, len(original_strokes) * 3),
                    step_scale=1.0,  # Use our fixed scaling
                    burn_in_steps=burn_in,  # Variable burn-in for testing
                    use_bernoulli_eos=use_bernoulli,  # Test both Bernoulli and thresholds
                    use_stop_condition=False,  # Disable early stopping for debugging
                    greedy=greedy,  # Test greedy vs sampling
                    eos_threshold_up=eos_up,  # Custom EOS thresholds
                    eos_threshold_down=eos_down,
                )

                if sequences and len(sequences[0]) > 0:
                    seq = sequences[0]
                    print(f"    ✓ Generated {len(seq)} points")

                    # EOS PROBABILITY ANALYSIS
                    if len(eos_probs) > 0 and len(eos_probs[0]) > 0:
                        eos_vals = eos_probs[0]
                        print(
                            f"    EOS probs: min={eos_vals.min():.3f}, max={eos_vals.max():.3f}, mean={eos_vals.mean():.3f}"
                        )
                        print(f"    EOS >0.5: {(eos_vals > 0.5).sum()}/{len(eos_vals)} steps")
                        print(f"    First 10 EOS probs: {eos_vals[:10]}")

                    # Check if generated points have reasonable scale
                    if len(seq) > 0:
                        coords = np.array([(x, y) for x, y, _ in seq])
                        pen_states = np.array([pen for _, _, pen in seq])
                        x_range = coords[:, 0].max() - coords[:, 0].min()
                        y_range = coords[:, 1].max() - coords[:, 1].min()
                        num_strokes = int(pen_states.sum()) + 1  # Count pen lifts + 1

                        print(f"    Range: x={x_range:.2f}, y={y_range:.2f}")
                        print(f"    X bounds: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
                        print(f"    Y bounds: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
                        print(f"    Number of strokes (pen lifts): {num_strokes}")

                        # Print first few and last few points to debug
                        print(f"    First 3 points: {seq[:3]}")
                        print(f"    Last 3 points: {seq[-3:]}")

                        if x_range > 0.5 and y_range > 0.5:
                            print("    ✓ Reasonable scale (good handwriting size)")
                        else:
                            print("    ⚠ Small range - check if scaling is correct")

                        # Save the greedy result as our best example for analysis
                        if "greedy" in style_name:
                            best_sequence = seq
                            best_style = style_name
                else:
                    print("    ✗ No points generated")

            except Exception as e:
                print(f"    ✗ Generation failed: {e}")

        # Create matplotlib visualization of the generated sequence
        if best_sequence:
            print(f"\n  Creating matplotlib visualization of {best_style} style...")
            visualize_generated_sequence(best_sequence, original_text, "generated_example.png")

        # Create SVG for final comparison with better parameters
        try:
            calligrapher.write(
                [original_text.lower()],
                "example_generated.svg",
                temperature=0.3,
                bias=0.2,
                step_scale=1.0,
                burn_in_steps=20,  # Use moderate burn-in for SVG output
            )
            print("\n  ✓ Generated SVG saved as 'example_generated.svg'")
        except Exception as e:
            print(f"\n  ✗ SVG generation failed: {e}")

    finally:
        # Clean up
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)


def load_or_train_model(data, epochs=150):
    """Load existing overfitted model or train a new one."""

    model_path = "overfit_example_model.keras"

    if os.path.exists(model_path):
        print(f"\n✓ Found existing overfitted model: {model_path}")
        print("Loading saved model...")

        try:
            from model.attention_mechanism import AttentionMechanism
            from model.attention_rnn_cell import AttentionRNNCell
            from model.handwriting_models import DeepHandwritingSynthesisModel
            from model.mixture_density_network import MixtureDensityLayer, mdn_loss
            from model_io import load_model_if_exists

            model, loaded = load_model_if_exists(
                model_path,
                custom_objects={
                    "mdn_loss": mdn_loss,
                    "AttentionMechanism": AttentionMechanism,
                    "AttentionRNNCell": AttentionRNNCell,
                    "MixtureDensityLayer": MixtureDensityLayer,
                    "DeepHandwritingSynthesisModel": DeepHandwritingSynthesisModel,
                },
            )

            if loaded:
                print("✓ Successfully loaded existing model!")
                return model, [0.0]  # Dummy losses since we didn't train
            else:
                print("✗ Failed to load existing model, will train new one...")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Will train a new model...")

    print(f"\nNo existing model found. Training new model for {epochs} epochs...")
    model, losses = train_overfit_model(data, epochs)

    # Save the trained model
    print(f"\nSaving overfitted model to {model_path}...")
    try:
        model.save(model_path)
        print("✓ Model saved successfully!")

        # Save normalization stats separately
        norm_stats = {
            "norm_mu": data["norm_mu"].tolist(),
            "norm_sigma": data["norm_sigma"].tolist(),
        }
        import json

        with open("overfit_example_norm_stats.json", "w") as f:
            json.dump(norm_stats, f)
        print("✓ Normalization stats saved!")

    except Exception as e:
        print(f"✗ Error saving model: {e}")

    return model, losses


def main():
    """Run the example stroke overfitting test."""

    print("=== Example Stroke Overfitting Test ===")
    print("Testing model on example-stroke.xml handwriting data")
    print()

    # Load example stroke data
    stroke_data = load_example_stroke_data()
    if stroke_data is None:
        return

    # Create minimal dataset
    data = create_minimal_dataset(stroke_data)

    # Visualize the target sequence before training (use original delta strokes for readable display)
    visualize_target_sequence(data["original_strokes"], data["original_text"])

    # Load existing model or train new one
    model, losses = load_or_train_model(data, epochs=300)

    # Test generation
    test_generation(model, data["original_text"], data["original_strokes"], data["norm_mu"], data["norm_sigma"])

    print("\n=== Test Summary ===")
    print(f"Original text: '{data['original_text']}'")
    print(f"Training loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    if losses[-1] < losses[0] * 0.3:
        print("✓ Model successfully learned the example handwriting!")
        print("✓ Architecture and training loop work with real stroke data")
        print("✓ Your bug fixes are working correctly")
    else:
        print("⚠ Model learning was limited")
        print("⚠ May need more training time or architecture adjustments")

    print("\nFiles created:")
    print("  - target_example.png (target handwriting visualization)")
    print("  - generated_example.png (generated handwriting visualization)")
    print("  - example_generated.svg (generated result in SVG format)")
    print("\nCompare target_example.png vs generated_example.png to see how well the model learned!")


if __name__ == "__main__":
    main()
