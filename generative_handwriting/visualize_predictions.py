import os
from datetime import datetime
from typing import Tuple

import imageio.v2 as imageio  # Fix deprecation warning
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from loader import HandwritingDataLoader
from matplotlib.patches import Ellipse
from model.handwriting_models import DeepHandwritingPredictionModel
from model.mixture_density_network import MixtureDensityLayer, mdn_loss
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import multivariate_normal


def offsets_to_coords(offsets):
    """Convert from offsets to absolute coordinates."""
    coords = np.copy(offsets)
    coords[:, :2] = np.cumsum(offsets[:, :2], axis=0)
    return coords


def coords_to_offsets(coords):
    """Convert from coordinates to offsets."""
    offsets = np.copy(coords)
    offsets[1:, :2] = coords[1:, :2] - coords[:-1, :2]
    offsets[0, :2] = coords[0, :2]  # First point stays as-is
    return offsets


def sample_from_mdn_output(
    pi: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    rho: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[float, float]:
    """Sample a point from the MDN output distribution."""
    # Apply temperature to mixture weights
    pi = pi ** (1 / temperature)
    pi = pi / np.sum(pi)

    # Sample component
    component = np.random.choice(len(pi), p=pi)

    # Get parameters for selected component
    mx = mu_x[component]
    my = mu_y[component]
    sx = sigma_x[component] * temperature
    sy = sigma_y[component] * temperature
    r = rho[component]

    # Sample from 2D Gaussian
    mean = [mx, my]
    cov = [[sx**2, r * sx * sy], [r * sx * sy, sy**2]]
    sample = np.random.multivariate_normal(mean, cov)

    return sample[0], sample[1]


def compute_probability_grid(
    pi: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    rho: np.ndarray,
    grid_size: int = 200,
    grid_range: float = 10.0,
    use_weights: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute probability density on a 2D grid from MDN parameters.

    Args:
        use_weights: If True, weight each component by pi. If False, show unweighted distributions.
    """
    # Create grid
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Initialize probability grid
    prob_grid = np.zeros((grid_size, grid_size))

    # Sum contributions from all mixture components
    for i in range(len(pi)):
        if use_weights and pi[i] < 1e-5:  # Skip components with very low weight only if weighting
            continue

        mean = [mu_x[i], mu_y[i]]
        sx, sy, r = sigma_x[i], sigma_y[i], rho[i]

        # Construct covariance matrix
        cov = [[sx**2, r * sx * sy], [r * sx * sy, sy**2]]

        try:
            rv = multivariate_normal(mean, cov)
            # Apply weighting or not based on parameter
            weight = pi[i] if use_weights else (1.0 if pi[i] > 0.01 else 0.0)  # Only show components with some minimal activation
            prob_grid += weight * rv.pdf(pos)
        except:
            # Handle singular matrix cases
            continue

    return X, Y, prob_grid


def extract_mdn_parameters(mdn_output: tf.Tensor, num_components: int = 20) -> dict:
    """Extract MDN parameters from model output.

    IMPORTANT: The MDN layer already applies transformations to its outputs:
    - pi is already softmaxed
    - sigma values are already exponentiated
    - rho is already tanh'd
    - eos is raw logits (needs sigmoid)
    """
    # The model outputs are already transformed except for EOS
    pi = mdn_output[:num_components]  # Already softmaxed
    mu_x = mdn_output[num_components : 2 * num_components]
    mu_y = mdn_output[2 * num_components : 3 * num_components]
    sigma_x = mdn_output[3 * num_components : 4 * num_components]  # Already exp'd
    sigma_y = mdn_output[4 * num_components : 5 * num_components]  # Already exp'd
    rho = mdn_output[5 * num_components : 6 * num_components]  # Already tanh'd
    eos = tf.sigmoid(mdn_output[6 * num_components : 6 * num_components + 1])  # Apply sigmoid to logits

    return {
        "pi": pi.numpy(),
        "mu_x": mu_x.numpy(),
        "mu_y": mu_y.numpy(),
        "sigma_x": sigma_x.numpy(),
        "sigma_y": sigma_y.numpy(),
        "rho": rho.numpy(),
        "eos": eos.numpy(),
    }


def create_cumulative_heatmap(
    prob_grids: list,
    positions: list,
    stroke_sequence: np.ndarray,
    save_path: str,
    grid_range: float = 15.0,
):
    """Create a cumulative heatmap showing overlaid probability distributions like Figure 10."""
    print(f"Creating cumulative heatmap with {len(prob_grids)} probability distributions...")

    # Determine the global bounds for the heatmap
    all_x = [pos[0] for pos in positions]
    all_y = [pos[1] for pos in positions]

    # Add some padding around the trajectory
    min_x, max_x = min(all_x) - grid_range, max(all_x) + grid_range
    min_y, max_y = min(all_y) - grid_range, max(all_y) + grid_range

    # Create a high-resolution global grid
    grid_size = 300
    x_global = np.linspace(min_x, max_x, grid_size)
    y_global = np.linspace(min_y, max_y, grid_size)
    X_global, Y_global = np.meshgrid(x_global, y_global)

    # Initialize cumulative probability grid
    cumulative_grid = np.zeros((grid_size, grid_size))

    # For each timestep, add its probability distribution to the cumulative grid
    local_grid_size = len(prob_grids[0])  # Assuming all grids have same size
    local_range = grid_range

    for i, (prob_grid, pos) in enumerate(zip(prob_grids, positions)):
        # Create local coordinate grid for this timestep
        x_local = np.linspace(-local_range, local_range, local_grid_size)
        y_local = np.linspace(-local_range, local_range, local_grid_size)

        # Convert local coordinates to global coordinates
        x_local_global = x_local + pos[0]
        y_local_global = y_local + pos[1]

        # Interpolate the local probability grid onto the global grid

        # Create interpolator for the local grid
        interpolator = RegularGridInterpolator(
            (y_local_global, x_local_global),
            prob_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Sample the local grid at global grid points
        points = np.column_stack([Y_global.ravel(), X_global.ravel()])
        interpolated_values = interpolator(points).reshape(grid_size, grid_size)

        # Add to cumulative grid
        cumulative_grid += interpolated_values

    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Top plot: Cumulative probability heatmap
    im1 = ax1.imshow(
        cumulative_grid,
        extent=[min_x, max_x, min_y, max_y],
        origin='lower',
        cmap='hot',
        alpha=0.8,
        aspect='equal'
    )

    # Overlay the actual handwriting trajectory
    # Handle pen-up/pen-down by splitting strokes
    stroke_ends = np.where(stroke_sequence[:, 2] == 1)[0]
    start_idx = 0

    for end_idx in stroke_ends:
        if end_idx >= start_idx and end_idx < len(stroke_sequence):
            stroke_segment = stroke_sequence[start_idx:end_idx + 1]
            if len(stroke_segment) > 1:
                ax1.plot(stroke_segment[:, 0], stroke_segment[:, 1], 'cyan', linewidth=2, alpha=0.9)
            start_idx = end_idx + 1

    # Handle final segment if no pen-up at end
    if start_idx < len(stroke_sequence):
        stroke_segment = stroke_sequence[start_idx:]
        if len(stroke_segment) > 1:
            ax1.plot(stroke_segment[:, 0], stroke_segment[:, 1], 'cyan', linewidth=2, alpha=0.9)

    # Don't show prediction position markers - just show pure probability distributions
    # pred_x = [pos[0] for pos in positions]
    # pred_y = [pos[1] for pos in positions]
    # ax1.scatter(pred_x, pred_y, c='white', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)

    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_title('Cumulative Probability Density Heatmap (Figure 10 Style)')
    plt.colorbar(im1, ax=ax1, label='Cumulative Probability Density', shrink=0.8)

    # Bottom plot: Component activations over time (like the original bottom plot)
    # This recreates the mixture component timeline from our existing data
    # We'll need to track component activations during the main loop for this

    # For now, create a placeholder showing the trajectory timeline
    time_steps = list(range(len(positions)))
    trajectory_x = [pos[0] for pos in positions]
    trajectory_y = [pos[1] for pos in positions]

    # Create a 2D representation showing x and y over time
    ax2.plot(time_steps, trajectory_x, 'b-', label='X coordinate', alpha=0.7)
    ax2.plot(time_steps, trajectory_y, 'r-', label='Y coordinate', alpha=0.7)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Coordinate Value')
    ax2.set_title('Trajectory Coordinates Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Handwriting Prediction: Cumulative Probability Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Cumulative heatmap saved to {save_path}")


def visualize_prediction_sequence(
    model: tf.keras.Model,
    stroke_sequence: np.ndarray,
    save_path: str = "prediction_sequence.gif",
    num_components: int = 20,
    grid_size: int = 100,
    fps: int = 2,
    generate_cumulative_heatmap: bool = True,
):
    """Create a GIF showing prediction evolution as more context is provided."""
    figures = []

    # Prepare for saving individual frames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_frame_dir = "prediction_frames"
    frame_dir = os.path.join(base_frame_dir, f"sequence_{save_path.split('_')[-1].split('.')[0]}_{timestamp}")
    os.makedirs(frame_dir, exist_ok=True)
    print(f"📁 Saving frames to: {frame_dir}")

    # Track EOS predictions and analysis
    eos_log = []

    # Track mixture component activations over time for visualization
    component_activations = []  # Will store pi values for each timestep

    # Track cumulative probability distributions for Figure 10-style visualization
    cumulative_prob_grids = []  # Will store probability grids for each timestep
    cumulative_positions = []   # Will store the positions where predictions were made

    # Get sequence length
    seq_len = min(len(stroke_sequence), 400)  # Limit to 400 points for visualization

    print(f"Generating prediction visualizations for {seq_len} steps...")

    for t in range(2, seq_len):
        # Get context up to time t
        # The stroke_sequence is already in coordinates, convert to offsets for model
        coords_context = stroke_sequence[:t]
        offsets_context = coords_to_offsets(coords_context)
        context = offsets_context.reshape(1, t, 3)

        # Debug: Check for unreasonable jumps
        if t == 11:  # Debug the problematic frame
            print(f"Debug frame {t}:")
            print(f"  Coords context shape: {coords_context.shape}")
            print(f"  Last 3 coords: {coords_context[-3:, :2]}")
            print(f"  Last 3 offsets: {offsets_context[-3:, :2]}")
            print(f"  Max offset magnitude: {np.max(np.abs(offsets_context[:, :2]))}")

        # Get prediction for next step
        prediction = model(context, training=False)

        # Extract MDN parameters from last timestep
        last_pred = prediction[0, -1, :]
        params = extract_mdn_parameters(last_pred, num_components)

        # Store component activations for time plot
        component_activations.append(params["pi"].copy())

        # Store cumulative probability information for Figure 10-style visualization
        if generate_cumulative_heatmap:
            # Compute probability grid for this timestep (relative to current position)
            # USE UNWEIGHTED distributions to show all active components including high-variance ones
            X_rel, Y_rel, prob_grid_rel = compute_probability_grid(
                params["pi"],
                params["mu_x"],
                params["mu_y"],
                params["sigma_x"],
                params["sigma_y"],
                params["rho"],
                grid_size=150,
                grid_range=15.0,  # Wider range for cumulative view
                use_weights=False,  # UNWEIGHTED to see all components including end-of-stroke jumps
            )

            # Store the relative grid and current position
            cumulative_prob_grids.append(prob_grid_rel.copy())
            cumulative_positions.append([stroke_sequence[t - 1, 0], stroke_sequence[t - 1, 1]])

        # Create figure with 6 subplots in 2x3 grid
        fig = plt.figure(figsize=(24, 12))
        ax1 = plt.subplot(2, 3, 1)  # Trajectory with components (zoomed in)
        ax2 = plt.subplot(2, 3, 2)  # Probability heatmap (zoomed in)
        ax3 = plt.subplot(2, 3, 3)  # Full word/sequence view
        ax4 = plt.subplot(2, 3, 4)  # Trajectory with components (zoomed out)
        ax5 = plt.subplot(2, 3, 5)  # Probability heatmap (zoomed out)
        ax6 = plt.subplot(2, 3, 6)  # Mixture component activations over time

        # Plot 1: Trajectory with prediction ellipses
        # Handle pen-up/pen-down by splitting strokes
        past_coords = stroke_sequence[:t]

        # Split trajectory at pen-up points (where column 2 == 1)
        stroke_ends = np.where(past_coords[:, 2] == 1)[0]
        start_idx = 0

        for end_idx in stroke_ends:
            if end_idx >= start_idx:
                stroke_segment = past_coords[start_idx : end_idx + 1]
                if len(stroke_segment) > 1:
                    ax1.plot(stroke_segment[:, 0], stroke_segment[:, 1], "b-", alpha=0.5)
                start_idx = end_idx + 1

        # Plot remaining segment if any
        if start_idx < len(past_coords):
            stroke_segment = past_coords[start_idx:]
            if len(stroke_segment) > 1:
                ax1.plot(stroke_segment[:, 0], stroke_segment[:, 1], "b-", alpha=0.5)

        # Plot future ground truth
        if t + 5 < len(stroke_sequence):
            future_coords = stroke_sequence[t : t + 5]
            ax1.plot(future_coords[:, 0], future_coords[:, 1], "g--", alpha=0.3, label="Future (ground truth)")

        # Current position
        ax1.scatter(
            stroke_sequence[t - 1, 0], stroke_sequence[t - 1, 1], color="red", s=100, zorder=5, label="Current position"
        )

        # Draw ellipses for ALL mixture components with significant weight
        active_components = np.where(params["pi"] > 0.005)[0]  # Show all components > 0.5%
        colors = plt.cm.viridis(np.linspace(0, 1, len(active_components)))

        for i, idx in enumerate(active_components):
            # Calculate ellipse parameters
            sx = params["sigma_x"][idx]
            sy = params["sigma_y"][idx]
            r = params["rho"][idx]

            # Eigenvalues and rotation angle for ellipse
            cov = np.array([[sx**2, r * sx * sy], [r * sx * sy, sy**2]])
            try:
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

                # Create ellipse (1 standard deviation for better visibility)
                ellipse = Ellipse(
                    (params["mu_x"][idx] + stroke_sequence[t - 1, 0], params["mu_y"][idx] + stroke_sequence[t - 1, 1]),
                    2 * np.sqrt(eigenvalues[0]),  # 1 std dev
                    2 * np.sqrt(eigenvalues[1]),
                    angle=angle,
                    alpha=min(params["pi"][idx] * 5, 0.7),  # Scale alpha for visibility
                    facecolor=colors[i],
                    edgecolor=colors[i],
                    linewidth=1,
                )
                ax1.add_patch(ellipse)

                # Add component index label
                ax1.text(
                    params["mu_x"][idx] + stroke_sequence[t - 1, 0],
                    params["mu_y"][idx] + stroke_sequence[t - 1, 1],
                    f"{idx}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                )
            except:
                continue

        ax1.set_xlabel("X coordinate")
        ax1.set_ylabel("Y coordinate")
        ax1.set_title(f"Step {t}: Trajectory and Prediction Components")
        ax1.legend()

        # Set reasonable axis limits based on trajectory - ZOOMED IN for better visibility
        trajectory_range = 5  # Show 5 units around current position for tighter zoom
        curr_x, curr_y = stroke_sequence[t - 1, 0], stroke_sequence[t - 1, 1]
        ax1.set_xlim(curr_x - trajectory_range, curr_x + trajectory_range)
        ax1.set_ylim(curr_y - trajectory_range, curr_y + trajectory_range)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Enhanced probability heatmap with ZOOMED IN view for spike visibility
        X, Y, prob_grid = compute_probability_grid(
            params["pi"],
            params["mu_x"],
            params["mu_y"],
            params["sigma_x"],
            params["sigma_y"],
            params["rho"],
            grid_size=200,  # Higher resolution for better detail
            grid_range=3.0,  # Much tighter view to see probability peaks clearly
        )

        # Shift grid to current position
        X += stroke_sequence[t - 1, 0]
        Y += stroke_sequence[t - 1, 1]

        # Plot heatmap with more levels for better detail
        im = ax2.contourf(X, Y, prob_grid, levels=50, cmap="hot", alpha=0.8)

        # Plot trajectory with pen-up/pen-down handling
        past_coords = stroke_sequence[:t]
        stroke_ends = np.where(past_coords[:, 2] == 1)[0]
        start_idx = 0

        for end_idx in stroke_ends:
            if end_idx >= start_idx:
                stroke_segment = past_coords[start_idx : end_idx + 1]
                if len(stroke_segment) > 1:
                    ax2.plot(stroke_segment[:, 0], stroke_segment[:, 1], "c-", linewidth=2, alpha=0.9)
                start_idx = end_idx + 1

        if start_idx < len(past_coords):
            stroke_segment = past_coords[start_idx:]
            if len(stroke_segment) > 1:
                ax2.plot(stroke_segment[:, 0], stroke_segment[:, 1], "c-", linewidth=2, alpha=0.9)

        # Mark current position
        ax2.scatter(
            stroke_sequence[t - 1, 0],
            stroke_sequence[t - 1, 1],
            color="cyan",
            s=150,
            zorder=5,
            edgecolor="black",
            linewidth=2,
        )

        # Mark next true position if available
        if t < len(stroke_sequence) - 1:
            ax2.scatter(
                stroke_sequence[t, 0], stroke_sequence[t, 1], color="lime", s=100, marker="x", zorder=5, linewidth=3
            )

        # Add centers of top Gaussian components
        top_5 = np.argsort(params["pi"])[-5:][::-1]
        for i, idx in enumerate(top_5):
            if params["pi"][idx] > 0.01:
                center_x = params["mu_x"][idx] + stroke_sequence[t - 1, 0]
                center_y = params["mu_y"][idx] + stroke_sequence[t - 1, 1]
                ax2.scatter(
                    center_x,
                    center_y,
                    color="white",
                    s=100,
                    marker="o",
                    zorder=6,
                    edgecolor="black",
                    linewidth=2,
                    alpha=0.9,
                )
                ax2.text(
                    center_x,
                    center_y,
                    f"{idx}",
                    fontsize=10,
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        ax2.set_xlabel("X coordinate")
        ax2.set_ylabel("Y coordinate")
        ax2.set_title(f"Probability Density Heatmap (EOS: {params['eos'][0]:.3f})")

        # Set matching axis limits for consistency - using tighter zoom for heatmap
        heatmap_range = 3  # Match the grid_range for proper zooming
        ax2.set_xlim(curr_x - heatmap_range, curr_x + heatmap_range)
        ax2.set_ylim(curr_y - heatmap_range, curr_y + heatmap_range)
        plt.colorbar(im, ax=ax2, label="Probability Density", shrink=0.8)

        # Check if this is near an end-of-stroke (move this up before using it)
        is_eos_prediction = params["eos"][0] > 0.08  # High EOS probability (90th percentile)
        actual_eos = False
        if t < len(stroke_sequence) - 1:
            actual_eos = stroke_sequence[t, 2] == 1  # Next point is end-of-stroke

        # Plot 3: Clean full sequence overview
        full_coords = stroke_sequence  # Complete sequence

        # Plot the complete handwriting with pen-up/pen-down handling
        stroke_ends = np.where(full_coords[:, 2] == 1)[0]
        start_idx = 0

        # Use a single color for cleaner appearance
        for end_idx in stroke_ends:
            if end_idx >= start_idx and end_idx < len(full_coords):
                stroke_segment = full_coords[start_idx : end_idx + 1]
                if len(stroke_segment) > 1:
                    # Split into past and future based on current position
                    if start_idx <= t <= end_idx:
                        # Current stroke - show past and future differently
                        past_segment = stroke_segment[: min(t - start_idx + 1, len(stroke_segment))]
                        future_segment = stroke_segment[min(t - start_idx, len(stroke_segment)) :]

                        if len(past_segment) > 1:
                            ax3.plot(past_segment[:, 0], past_segment[:, 1], "black", linewidth=2, alpha=0.8)
                        if len(future_segment) > 1:
                            ax3.plot(future_segment[:, 0], future_segment[:, 1], "gray", linewidth=1, alpha=0.4)
                    else:
                        # Complete stroke - solid if past, light if future
                        alpha = 0.8 if end_idx < t else 0.4
                        linewidth = 2 if end_idx < t else 1
                        color = "black" if end_idx < t else "gray"
                        ax3.plot(
                            stroke_segment[:, 0], stroke_segment[:, 1], color=color, linewidth=linewidth, alpha=alpha
                        )

                start_idx = end_idx + 1

        # Handle final segment if no pen-up at end
        if start_idx < len(full_coords):
            stroke_segment = full_coords[start_idx:]
            if len(stroke_segment) > 1:
                if start_idx <= t:
                    past_segment = (
                        stroke_segment[: t - start_idx + 1]
                        if t - start_idx + 1 < len(stroke_segment)
                        else stroke_segment
                    )
                    future_segment = stroke_segment[t - start_idx :] if t - start_idx < len(stroke_segment) else []

                    if len(past_segment) > 1:
                        ax3.plot(past_segment[:, 0], past_segment[:, 1], "black", linewidth=2, alpha=0.8)
                    if len(future_segment) > 1:
                        ax3.plot(future_segment[:, 0], future_segment[:, 1], "gray", linewidth=1, alpha=0.4)
                else:
                    ax3.plot(stroke_segment[:, 0], stroke_segment[:, 1], "gray", linewidth=1, alpha=0.4)

        # Mark only the current position with a simple red dot
        ax3.scatter(stroke_sequence[t - 1, 0], stroke_sequence[t - 1, 1], color="red", s=80, zorder=10)

        ax3.set_xlabel("X coordinate")
        ax3.set_ylabel("Y coordinate")
        ax3.set_title(f"Complete Handwriting Sequence (Step {t}/{len(stroke_sequence)})")
        ax3.axis("equal")

        # Add simple progress indicator
        progress = t / len(stroke_sequence) * 100
        ax3.text(
            0.02,
            0.98,
            f"Progress: {progress:.1f}%",
            transform=ax3.transAxes,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            fontsize=12,
            fontweight="bold",
            verticalalignment="top",
        )

        # ======= Plot 4: Zoomed-out Trajectory with Components =======
        # Similar to plot 1 but with larger view
        zoomed_out_range = 20  # Larger view area

        # Plot past trajectory with pen handling
        stroke_ends = np.where(past_coords[:, 2] == 1)[0]
        start_idx = 0
        for end_idx in stroke_ends:
            if end_idx >= start_idx:
                stroke_segment = past_coords[start_idx : end_idx + 1]
                if len(stroke_segment) > 1:
                    ax4.plot(stroke_segment[:, 0], stroke_segment[:, 1], "b-", alpha=0.5)
                start_idx = end_idx + 1
        if start_idx < len(past_coords):
            stroke_segment = past_coords[start_idx:]
            if len(stroke_segment) > 1:
                ax4.plot(stroke_segment[:, 0], stroke_segment[:, 1], "b-", alpha=0.5)

        # Plot future ground truth
        if t + 5 < len(stroke_sequence):
            future_coords = stroke_sequence[t : t + 5]
            ax4.plot(future_coords[:, 0], future_coords[:, 1], "g--", alpha=0.3, label="Future (ground truth)")

        # Current position
        ax4.scatter(
            stroke_sequence[t - 1, 0], stroke_sequence[t - 1, 1], color="red", s=100, zorder=5, label="Current position"
        )

        # Draw ellipses for mixture components (same as plot 1)
        for i, idx in enumerate(active_components):
            sx = params["sigma_x"][idx]
            sy = params["sigma_y"][idx]
            r = params["rho"][idx]
            cov = np.array([[sx**2, r * sx * sy], [r * sx * sy, sy**2]])
            try:
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                ellipse = Ellipse(
                    (params["mu_x"][idx] + stroke_sequence[t - 1, 0], params["mu_y"][idx] + stroke_sequence[t - 1, 1]),
                    2 * np.sqrt(eigenvalues[0]),
                    2 * np.sqrt(eigenvalues[1]),
                    angle=angle,
                    alpha=min(params["pi"][idx] * 5, 0.7),
                    facecolor=colors[i],
                    edgecolor=colors[i],
                    linewidth=1,
                )
                ax4.add_patch(ellipse)
            except:
                continue

        ax4.set_xlabel("X coordinate")
        ax4.set_ylabel("Y coordinate")
        ax4.set_title("Trajectory (Zoomed Out)")
        ax4.set_xlim(curr_x - zoomed_out_range, curr_x + zoomed_out_range)
        ax4.set_ylim(curr_y - zoomed_out_range, curr_y + zoomed_out_range)
        ax4.grid(True, alpha=0.3)

        # ======= Plot 5: Zoomed-out Probability Heatmap =======
        X_out, Y_out, prob_grid_out = compute_probability_grid(
            params["pi"],
            params["mu_x"],
            params["mu_y"],
            params["sigma_x"],
            params["sigma_y"],
            params["rho"],
            grid_size=150,
            grid_range=10.0,  # Larger view
        )
        X_out += stroke_sequence[t - 1, 0]
        Y_out += stroke_sequence[t - 1, 1]

        # Plot heatmap
        im_out = ax5.contourf(X_out, Y_out, prob_grid_out, levels=30, cmap="hot", alpha=0.8)

        # Plot trajectory
        start_idx = 0
        for end_idx in stroke_ends:
            if end_idx >= start_idx:
                stroke_segment = past_coords[start_idx : end_idx + 1]
                if len(stroke_segment) > 1:
                    ax5.plot(stroke_segment[:, 0], stroke_segment[:, 1], "c-", linewidth=2, alpha=0.9)
                start_idx = end_idx + 1
        if start_idx < len(past_coords):
            stroke_segment = past_coords[start_idx:]
            if len(stroke_segment) > 1:
                ax5.plot(stroke_segment[:, 0], stroke_segment[:, 1], "c-", linewidth=2, alpha=0.9)

        ax5.scatter(
            stroke_sequence[t - 1, 0],
            stroke_sequence[t - 1, 1],
            color="cyan",
            s=150,
            zorder=5,
            edgecolor="black",
            linewidth=2,
        )

        if t < len(stroke_sequence) - 1:
            ax5.scatter(
                stroke_sequence[t, 0], stroke_sequence[t, 1], color="lime", s=100, marker="x", zorder=5, linewidth=3
            )

        ax5.set_xlabel("X coordinate")
        ax5.set_ylabel("Y coordinate")
        ax5.set_title("Probability Heatmap (Zoomed Out)")
        ax5.set_xlim(curr_x - zoomed_out_range, curr_x + zoomed_out_range)
        ax5.set_ylim(curr_y - zoomed_out_range, curr_y + zoomed_out_range)
        plt.colorbar(im_out, ax=ax5, label="Probability Density", shrink=0.8)

        # ======= Plot 6: Mixture Component Activations Over Time =======
        # Create activation heatmap from accumulated data
        if len(component_activations) > 1:
            activation_array = np.array(component_activations).T  # Components x Time

            # Plot as heatmap
            im_activation = ax6.imshow(
                activation_array, aspect="auto", cmap="viridis", interpolation="nearest", origin="lower"
            )

            ax6.set_xlabel("Time Step")
            ax6.set_ylabel("Component Index")
            ax6.set_title(f"Mixture Component Activations (Step 2-{t})")

            # Add colorbar
            plt.colorbar(im_activation, ax=ax6, label="Activation Weight", shrink=0.8)

            # Set x-axis to show actual step numbers
            num_steps = len(component_activations)
            if num_steps > 10:
                step_indices = np.linspace(0, num_steps - 1, 10, dtype=int)
                ax6.set_xticks(step_indices)
                ax6.set_xticklabels([str(i + 2) for i in step_indices])

            # Highlight most active components
            mean_activations = np.mean(activation_array, axis=1)
            top_3_components = np.argsort(mean_activations)[-3:][::-1]
            for rank, comp in enumerate(top_3_components):
                ax6.axhline(y=comp, color="red", linestyle="--", alpha=0.5, linewidth=1)
                ax6.text(num_steps * 0.98, comp, f"#{rank + 1}", color="red", fontsize=8, va="center", ha="right")
        else:
            ax6.text(
                0.5, 0.5, "Waiting for more data...", transform=ax6.transAxes, ha="center", va="center", fontsize=12
            )
            ax6.set_title("Mixture Component Activations")
            ax6.axis("off")

        # Log EOS information for analysis
        avg_variance = np.mean(params["sigma_x"] ** 2 + params["sigma_y"] ** 2)
        eos_log.append(
            {
                "step": int(t),
                "eos_pred": float(params["eos"][0]),
                "actual_eos": bool(actual_eos),
                "avg_variance": float(avg_variance),
                "max_component_weight": float(params["pi"].max()),
                "entropy": float(-np.sum(params["pi"] * np.log(params["pi"] + 1e-10))),
            }
        )

        # Add enhanced info text with EOS information
        info_text = f"Context: {t} points\n"
        info_text += f"EOS Prediction: {params['eos'][0]:.3f} {'HIGH!' if is_eos_prediction else 'Normal'}\n"
        info_text += f"Actual EOS: {'YES' if actual_eos else 'NO'}\n"
        info_text += f"Top component weight: {params['pi'].max():.3f}\n"
        info_text += f"Entropy: {-np.sum(params['pi'] * np.log(params['pi'] + 1e-10)):.3f}\n"

        info_text += f"Avg variance: {avg_variance:.2f}"

        # Color the info box differently for EOS predictions
        box_color = "lightcoral" if is_eos_prediction else "wheat"
        fig.text(0.02, 0.02, info_text, fontsize=10, bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8))

        plt.suptitle(f"Handwriting Prediction Visualization - Step {t}/{seq_len}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save frame with fixed dimensions to ensure consistent size
        frame_path = os.path.join(frame_dir, f"frame_{t:03d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches=None, pad_inches=0.2)

        # Load image
        img = imageio.imread(frame_path)

        # Crop/pad to consistent size if needed
        if len(figures) == 0:
            target_shape = img.shape
            print(f"Target shape: {target_shape}")
        elif img.shape != target_shape:
            # Resize by cropping/padding to match target
            h_diff = img.shape[0] - target_shape[0]
            w_diff = img.shape[1] - target_shape[1]

            if h_diff > 0:
                img = img[h_diff // 2 : h_diff // 2 + target_shape[0]]
            elif h_diff < 0:
                pad_h = (-h_diff // 2, -h_diff - (-h_diff // 2))
                img = np.pad(img, ((pad_h[0], pad_h[1]), (0, 0), (0, 0)), mode="constant", constant_values=255)

            if w_diff > 0:
                img = img[:, w_diff // 2 : w_diff // 2 + target_shape[1]]
            elif w_diff < 0:
                pad_w = (-w_diff // 2, -w_diff - (-w_diff // 2))
                img = np.pad(img, ((0, 0), (pad_w[0], pad_w[1]), (0, 0)), mode="constant", constant_values=255)

            # Ensure exact shape match
            img = img[: target_shape[0], : target_shape[1]]

        figures.append(img)
        plt.close()

        if t % 10 == 0:
            print(f"  Generated frame {t}/{seq_len}")

    # Generate the cumulative probability heatmap (Figure 10 style)
    if generate_cumulative_heatmap and len(cumulative_prob_grids) > 0:
        print("Generating cumulative probability heatmap...")
        create_cumulative_heatmap(
            cumulative_prob_grids,
            cumulative_positions,
            stroke_sequence,
            save_path.replace(".gif", "_cumulative_heatmap.png")
        )

    # Create GIF
    print(f"Creating GIF: {save_path}")
    if len(figures) > 5:  # Only create GIF if we have enough frames
        try:
            imageio.mimsave(save_path, figures, fps=fps)
            print(f"✅ GIF saved to {save_path}")
        except Exception as e:
            print(f"❌ GIF creation failed: {e}")
            print("Creating static image instead...")
            figures[0].save(save_path.replace(".gif", "_static.png"))
    else:
        print(f"⚠️ Not enough valid frames ({len(figures)}), skipping GIF creation")

    # Keep frame files - don't clean up!
    print(f"🖼️ Individual frames preserved in: {frame_dir}")
    print(f"📊 Total frames saved: {len(figures)}")

    # Save EOS analysis
    eos_analysis_path = save_path.replace(".gif", "_eos_analysis.json")
    import json

    with open(eos_analysis_path, "w") as f:
        json.dump(eos_log, f, indent=2)

    # Print EOS summary
    eos_predictions = [x for x in eos_log if x["eos_pred"] > 0.08]
    actual_eos_points = [x for x in eos_log if x["actual_eos"]]
    high_variance_points = [x for x in eos_log if x["avg_variance"] > 10.0]

    print("\n📊 EOS ANALYSIS SUMMARY:")
    print(f"   High EOS predictions (>0.08): {len(eos_predictions)}")
    print(f"   Actual EOS points: {len(actual_eos_points)}")
    print(f"   High variance predictions (>10): {len(high_variance_points)}")
    print(f"   EOS analysis saved to: {eos_analysis_path}")

    if len(eos_predictions) > 0:
        avg_variance_at_eos_pred = np.mean([x["avg_variance"] for x in eos_predictions])
        print(f"   Average variance at high EOS predictions: {avg_variance_at_eos_pred:.2f}")

    if len(actual_eos_points) > 0:
        avg_eos_pred_at_actual = np.mean([x["eos_pred"] for x in actual_eos_points])
        print(f"   Average EOS prediction at actual EOS: {avg_eos_pred_at_actual:.3f}")

    print(f"✅ GIF saved to {save_path}")
    return save_path


def load_model_for_visualization(model_path: str, num_components: int = 20):
    """Load the trained model for visualization."""
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "model_mdn_loss": lambda actual, outputs, combined_train_lengths, num_mixture_components: mdn_loss(
                    actual, outputs, combined_train_lengths, num_mixture_components
                ),
                "MixtureDensityLayer": MixtureDensityLayer,
                "DeepHandwritingPredictionModel": DeepHandwritingPredictionModel,
            },
        )
        print(f"✅ Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def main():
    """Main function to run the visualization."""
    # Configuration
    curr_directory = os.path.dirname(os.path.realpath(__file__))
    model_path = f"{curr_directory}/saved_models/full_handwriting_prediction/best_model.keras"
    output_dir = f"{curr_directory}/prediction_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_model_for_visualization(model_path)
    if model is None:
        print("⚠️  No model found. Please train a model first.")
        return

    # Load data
    print("Loading handwriting data...")
    data_loader = HandwritingDataLoader()
    data_loader.prepare_data()

    # Get a sample stroke sequence from training data
    sample_idx = np.random.randint(0, len(data_loader.combined_train_strokes))
    stroke_sequence_offsets = data_loader.combined_train_strokes[sample_idx]

    # Convert offsets to absolute coordinates for visualization
    stroke_sequence = offsets_to_coords(stroke_sequence_offsets)

    print(f"Selected stroke sequence with {len(stroke_sequence)} points")

    # Generate visualization
    gif_path = os.path.join(output_dir, f"prediction_sequence_{sample_idx}.gif")
    visualize_prediction_sequence(
        model,
        stroke_sequence,
        save_path=gif_path,
        num_components=20,
        grid_size=80,
        fps=2,
        generate_cumulative_heatmap=True
    )

    # Also create a static visualization of a single prediction
    print("\nGenerating static prediction visualization...")

    # Take first 20 points as context
    context_length = min(20, len(stroke_sequence) - 1)
    context = stroke_sequence[:context_length].reshape(1, context_length, 3)

    # Get prediction
    prediction = model(context, training=False)
    last_pred = prediction[0, -1, :]
    params = extract_mdn_parameters(last_pred, 20)

    # Create detailed static plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Trajectory
    ax = axes[0, 0]
    ax.plot(stroke_sequence[:context_length, 0], stroke_sequence[:context_length, 1], "b-", label="Context")
    ax.plot(
        stroke_sequence[context_length : context_length + 10, 0],
        stroke_sequence[context_length : context_length + 10, 1],
        "g--",
        alpha=0.5,
        label="Ground truth",
    )
    ax.scatter(
        stroke_sequence[context_length - 1, 0],
        stroke_sequence[context_length - 1, 1],
        color="red",
        s=100,
        zorder=5,
        label="Current",
    )
    ax.set_title("Handwriting Trajectory")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    # Plot 2: Mixture weights
    ax = axes[0, 1]
    ax.bar(range(len(params["pi"])), params["pi"])
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Weight")
    ax.set_title("Mixture Component Weights")
    ax.grid(True, alpha=0.3)

    # Plot 3: Probability heatmap
    ax = axes[1, 0]
    X, Y, prob_grid = compute_probability_grid(
        params["pi"], params["mu_x"], params["mu_y"], params["sigma_x"], params["sigma_y"], params["rho"], grid_size=100
    )
    X += stroke_sequence[context_length - 1, 0]
    Y += stroke_sequence[context_length - 1, 1]
    im = ax.contourf(X, Y, prob_grid, levels=30, cmap="hot")
    ax.plot(stroke_sequence[:context_length, 0], stroke_sequence[:context_length, 1], "c-", linewidth=2)
    ax.scatter(
        stroke_sequence[context_length - 1, 0], stroke_sequence[context_length - 1, 1], color="cyan", s=100, zorder=5
    )
    ax.set_title("Next Position Probability Density")
    ax.axis("equal")
    plt.colorbar(im, ax=ax)

    # Plot 4: Component parameters
    ax = axes[1, 1]
    ax.text(0.1, 0.9, f"End-of-stroke prob: {params['eos'][0]:.4f}", transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.8, f"Max component weight: {params['pi'].max():.4f}", transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.7, f"Active components (>0.01): {np.sum(params['pi'] > 0.01)}", transform=ax.transAxes, fontsize=12)
    ax.text(
        0.1,
        0.6,
        f"Entropy: {-np.sum(params['pi'] * np.log(params['pi'] + 1e-10)):.4f}",
        transform=ax.transAxes,
        fontsize=12,
    )

    # Show top 5 components
    top_5 = np.argsort(params["pi"])[-5:][::-1]
    ax.text(0.1, 0.4, "Top 5 components:", transform=ax.transAxes, fontsize=12, fontweight="bold")
    for i, idx in enumerate(top_5):
        ax.text(
            0.1,
            0.3 - i * 0.05,
            f"  {i + 1}. π={params['pi'][idx]:.3f}, μ=({params['mu_x'][idx]:.2f}, {params['mu_y'][idx]:.2f}), "
            f"σ=({params['sigma_x'][idx]:.2f}, {params['sigma_y'][idx]:.2f}), ρ={params['rho'][idx]:.2f}",
            transform=ax.transAxes,
            fontsize=10,
        )

    ax.set_title("MDN Parameters")
    ax.axis("off")

    plt.suptitle("Handwriting Prediction Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()

    static_path = os.path.join(output_dir, f"prediction_analysis_{sample_idx}.png")
    plt.savefig(static_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"✅ Static visualization saved to {static_path}")
    print(f"\n📊 Model outputs {20} mixture components")
    print("📍 Prediction is relative to current position")
    print("🎯 Green crosses show ground truth next positions")


if __name__ == "__main__":
    main()
