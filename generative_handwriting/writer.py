import os
from typing import List, Optional, Tuple

import numpy as np
import svgwrite
import tensorflow as tf
from alphabet import encode_ascii
from constants import MAX_CHAR_LEN
from model.attention_mechanism import AttentionMechanism
from model.attention_rnn_cell import AttentionRNNCell
from model.handwriting_models import DeepHandwritingSynthesisModel
from model.mixture_density_network import MixtureDensityLayer, mdn_loss
from model_io import load_model_if_exists


class Calligrapher:
    """A class for generating handwritten text using a trained handwriting synthesis model.

    This class handles the generation of handwritten text from input strings, including
    sampling from the model's probability distributions and rendering the output as SVG.

    Attributes:
        model_path: Path to the trained model
        num_output_mixtures: Number of mixture components in the MDN
        model: The loaded handwriting model
        sigma_min: Minimum value for standard deviations (numerical stability)
        sigma_max: Maximum value for standard deviations
        rho_max: Maximum absolute correlation coefficient
    """

    def __init__(
        self,
        model_path: str,
        num_output_mixtures: int,
        sigma_min: float = 2e-3,
        sigma_max: float = 3.0,
        rho_max: float = 0.95,
    ) -> None:
        """Initialize the Calligrapher.

        Args:
            model_path: Path to the saved model file
            num_output_mixtures: Number of Gaussian mixture components
            sigma_min: Minimum standard deviation for numerical stability
            sigma_max: Maximum standard deviation
            rho_max: Maximum absolute correlation coefficient

        Raises:
            ValueError: If the model cannot be loaded from the given path
        """
        self.model_path = model_path
        self.num_output_mixtures = num_output_mixtures
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho_max = rho_max

        # Load the model with custom objects
        self.model, self.loaded = load_model_if_exists(
            model_path,
            custom_objects={
                "mdn_loss": mdn_loss,
                "AttentionMechanism": AttentionMechanism,
                "AttentionRNNCell": AttentionRNNCell,
                "MixtureDensityLayer": MixtureDensityLayer,
                "DeepHandwritingSynthesisModel": DeepHandwritingSynthesisModel,
            },
        )
        if not self.loaded:
            raise ValueError(f"Model could not be loaded from {model_path}")

    def sample_gaussian_2d(self, mu1: float, mu2: float, s1: float, s2: float, rho: float) -> Tuple[float, float]:
        """Sample a point from a 2D Gaussian using the reparameterization trick.

        Args:
            mu1: Mean of the first dimension
            mu2: Mean of the second dimension
            s1: Standard deviation of the first dimension
            s2: Standard deviation of the second dimension
            rho: Correlation coefficient between dimensions

        Returns:
            Tuple containing sampled x and y coordinates
        """
        # Use reparameterization trick for better numerical stability
        epsilon1 = np.random.normal()
        epsilon2 = np.random.normal()

        # Ensure parameters are within bounds
        s1 = np.clip(s1, self.sigma_min, self.sigma_max)
        s2 = np.clip(s2, self.sigma_min, self.sigma_max)
        rho = np.clip(rho, -self.rho_max, self.rho_max)

        # Sample using reparameterization
        x = mu1 + s1 * epsilon1
        y = mu2 + s2 * (rho * epsilon1 + np.sqrt(1 - rho**2) * epsilon2)

        return x, y

    def adjust_parameters(
        self,
        pi: np.ndarray,
        mu1: np.ndarray,
        mu2: np.ndarray,
        sigma1: np.ndarray,
        sigma2: np.ndarray,
        rho: np.ndarray,
        bias: float = 0.0,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, ...]:
        """Adjust the MDN parameters for biased sampling.

        Args:
            pi: Mixture weights (already softmaxed probabilities)
            mu1, mu2: Means for both dimensions
            sigma1, sigma2: Standard deviations
            rho: Correlation coefficients
            bias: Controls diversity vs quality (0.0-1.0)
            temperature: Controls randomness (0.1-1.0)

        Returns:
            Tuple of adjusted parameters
        """
        # Clamp sigmas before adjustment to prevent explosion
        sigma1 = np.clip(sigma1, self.sigma_min, self.sigma_max)
        sigma2 = np.clip(sigma2, self.sigma_min, self.sigma_max)

        # Adjust standard deviations with stronger reduction
        sigma1_adj = np.exp(np.log(sigma1) - bias * 2) * np.sqrt(temperature)
        sigma2_adj = np.exp(np.log(sigma2) - bias * 2) * np.sqrt(temperature)

        # Ensure adjusted sigmas stay within bounds
        sigma1_adj = np.clip(sigma1_adj, self.sigma_min, self.sigma_max)
        sigma2_adj = np.clip(sigma2_adj, self.sigma_min, self.sigma_max)

        # Adjust mixture weights (pi is already probabilities, not logits)
        pi_adj = np.power(pi, (1 + bias) / temperature)
        pi_adj /= np.sum(pi_adj, axis=-1, keepdims=True)

        return pi_adj, mu1, mu2, sigma1_adj, sigma2_adj, rho

    def sample(
        self,
        lines: List[str],
        max_char_len: int = MAX_CHAR_LEN,
        bias: float = 0.0,
        temperature: float = 1.0,
        max_steps: Optional[int] = None,
        eos_threshold_up: float = 0.7,
        eos_threshold_down: float = 0.3,
        burn_in_steps: int = 20,
        greedy: bool = False,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Sample handwriting sequences for given lines of text using autoregressive generation.

        Args:
            lines: List of text strings to render
            max_char_len: Maximum number of characters per line
            bias: Controls the trade-off between quality and diversity (0.0-1.0)
                Higher values produce cleaner but less diverse samples
            temperature: Controls randomness in sampling (0.1-1.0)
                Lower values produce more consistent writing
                Higher values produce more varied writing
            max_steps: Maximum number of generation steps (auto-determined if None)
            eos_threshold_up: Threshold for pen up when currently pen down (hysteresis)
            eos_threshold_down: Threshold for pen down when currently pen up (hysteresis)
            burn_in_steps: Number of initial steps to run without recording
            greedy: Whether to use greedy component selection instead of sampling

        Returns:
            Tuple of (sampled_points, end_of_stroke_probs)

        Raises:
            ValueError: If lines is empty or contains invalid characters
        """
        if not lines:
            raise ValueError("Must provide at least one line of text")

        # ---- Encode characters (per batch) ----
        enc = [encode_ascii(line)[:max_char_len] for line in lines]
        B = len(enc)
        char_seq = np.zeros((B, max_char_len), dtype=np.int32)
        char_len = np.array([len(e) for e in enc], dtype=np.int32)
        for i, e in enumerate(enc):
            char_seq[i, : len(e)] = e

        # Heuristic: strokes are longer than chars; pick a sane horizon if not provided
        if max_steps is None:
            max_steps = max(60, int(18 * np.max(char_len)))

        # ---- Bind character sequence to the attention cell ----
        cell = self.model.attention_rnn_cell
        cell.char_seq_one_hot = tf.one_hot(char_seq, depth=self.model.num_chars)
        cell.char_seq_len = tf.convert_to_tensor(char_len, dtype=tf.int32)

        # ---- Initial recurrent state & first input (pen up) ----
        init = cell.get_initial_state(batch_size=B, dtype=tf.float32)
        state_list = []
        for i in range(self.model.num_layers):
            state_list.extend([init[f"lstm_{i}_h"], init[f"lstm_{i}_c"]])
        state_list.extend([init["kappa"], init["w"]])

        x_prev = tf.constant(np.tile([[0.0, 0.0, 0.0]], (B, 1)), dtype=tf.float32)  # (dx, dy, pen_up=0 to start writing)

        K = self.num_output_mixtures
        sequences: List[List[Tuple[float, float, float]]] = [[] for _ in range(B)]
        eos_probs_log: List[List[float]] = [[] for _ in range(B)]

        # Track pen state with hysteresis
        prev_up_bin = np.zeros(B, dtype=np.int32)  # Start with pen down to encourage writing

        for t in range(max_steps + burn_in_steps):
            # One recurrent step through the cell
            h_t, state_list = cell(x_prev, state_list)             # h_t: [B, units]
            mdn_t = self.model.mdn_layer(h_t[:, None, :])[:, 0, :] # [B, 6K+1]

            # Split parameters
            pi, mu1, mu2, s1, s2, rho, eos_logit = tf.split(mdn_t, [K, K, K, K, K, K, 1], axis=-1)
            pi, mu1, mu2, s1, s2, rho = [
                a.numpy() for a in (pi, mu1, mu2, s1, s2, rho)
            ]
            eos_prob = 1.0 / (1.0 + np.exp(-eos_logit.numpy().reshape(-1) / max(1e-6, temperature)))

            # Temperature/bias adjustment (pi already prob., sigmas positive)
            pi, mu1, mu2, s1, s2, rho = self.adjust_parameters(
                pi, mu1, mu2, s1, s2, rho, bias=bias, temperature=temperature
            )

            # Sample each batch item and feed back as next input
            x_prev_np = np.zeros((B, 3), dtype=np.float32)

            for b in range(B):
                p = pi[b] / np.clip(pi[b].sum(), 1e-8, None)

                # Greedy or probabilistic component selection
                if greedy:
                    k = np.argmax(p)
                else:
                    k = np.random.choice(K, p=p)

                # For very stable generation, optionally use means directly
                if bias > 0.7:  # High bias = use means directly for stability
                    dx = mu1[b, k] * 0.1  # Scale down means significantly
                    dy = mu2[b, k] * 0.1
                else:
                    dx, dy = self.sample_gaussian_2d(
                        mu1[b, k], mu2[b, k], s1[b, k], s2[b, k], rho[b, k]
                    )
                    # Scale down the sampled values
                    dx *= 0.1
                    dy *= 0.1

                # Additional clamping for safety
                max_delta = 1.0  # Much smaller maximum movement
                dx = np.clip(dx, -max_delta, max_delta)
                dy = np.clip(dy, -max_delta, max_delta)

                # Hysteresis thresholding for pen state
                if prev_up_bin[b]:
                    pen_up_bin = float(eos_prob[b] > eos_threshold_down)
                else:
                    pen_up_bin = float(eos_prob[b] > eos_threshold_up)
                prev_up_bin[b] = int(pen_up_bin)

                # Only record points after burn-in period
                if t >= burn_in_steps:
                    # Accumulate deltas -> absolute coordinates for drawing
                    if len(sequences[b]) == 0:
                        abs_x, abs_y = dx, dy
                    else:
                        prev_x, prev_y, _ = sequences[b][-1]
                        abs_x, abs_y = prev_x + dx, prev_y + dy

                    sequences[b].append((abs_x, abs_y, pen_up_bin))
                    eos_probs_log[b].append(eos_prob[b])

                x_prev_np[b] = (dx, dy, pen_up_bin)

            x_prev = tf.convert_to_tensor(x_prev_np, dtype=tf.float32)

            # Stop condition: attention has passed the end of text and pen is mostly up
            if t >= burn_in_steps + 20:  # Give some time after burn-in
                kappa = state_list[-2].numpy()  # Attention position
                attention_done = (kappa.mean(axis=1) > (char_len + 1.5)).all()
                pen_mostly_up = np.mean(eos_prob) > 0.6

                if attention_done and pen_mostly_up:
                    break

        # Convert to arrays and apply scaling for SVG display
        sequences_arrays = []
        for seq in sequences:
            if len(seq) == 0:
                sequences_arrays.append([])
                continue

            # Extract coordinates and pen states
            coords = np.array([(x, y) for x, y, _ in seq])
            pen_states = [pen_up for _, _, pen_up in seq]

            # Only scale if coordinates are reasonable
            if len(coords) > 0:
                # Shift to origin
                coords -= coords.min(axis=0)

                # Scale more conservatively
                x_range = coords[:, 0].max() - coords[:, 0].min()
                y_range = coords[:, 1].max() - coords[:, 1].min()

                if x_range > 0:
                    # Scale to fit width, but not too aggressively
                    scale = min(50.0, 800.0 / x_range)  # Cap scaling factor
                    coords *= scale

                # Center vertically
                coords[:, 1] += 30  # Offset from top

            # Reconstruct with scaled coordinates and preserved pen states
            scaled_seq = [(float(coords[j][0]), float(coords[j][1]), pen_states[j]) for j in range(len(coords))]
            sequences_arrays.append(scaled_seq)

        return sequences_arrays, np.array([np.array(e) for e in eos_probs_log], dtype=np.float32)

    def write(
        self,
        lines: List[str],
        filename: str = "output.svg",
        bias: float = 0.0,
        temperature: float = 1.0,
        stroke_colors: Optional[List[str]] = None,
        stroke_widths: Optional[List[float]] = None,
        show_grid: bool = False,
        show_endpoints: bool = False,
        line_height: int = 60,
        greedy: bool = False,
    ) -> None:
        """Generate and save handwritten text as an SVG file.

        Args:
            lines: List of text strings to render
            filename: Output SVG filename
            bias: Quality vs diversity trade-off
            temperature: Sampling temperature
            stroke_colors: Colors for each line
            stroke_widths: Stroke widths for each line
            show_grid: Whether to show background grid
            show_endpoints: Whether to mark stroke endpoints
            line_height: Vertical spacing between lines

        Raises:
            ValueError: If inputs are invalid
        """
        if not lines:
            raise ValueError("Must provide at least one line of text")

        stroke_colors = stroke_colors or ["black"] * len(lines)
        stroke_widths = stroke_widths or [2.0] * len(lines)

        if len(stroke_colors) != len(lines) or len(stroke_widths) != len(lines):
            raise ValueError("Number of colors/widths must match number of lines")

        # Generate samples with optimized parameters
        strokes, _ = self.sample(
            lines,
            bias=bias,
            temperature=temperature,
            greedy=greedy,
            burn_in_steps=15,
            eos_threshold_up=0.8,
            eos_threshold_down=0.2,
            max_steps=max(100, 20 * max(len(line) for line in lines))
        )

        # Setup SVG canvas
        view_width = 1000
        view_height = line_height * (len(lines) + 1)
        dwg = svgwrite.Drawing(filename=filename, size=(view_width, view_height))

        # Add background
        dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill="white"))

        # Add grid if requested
        if show_grid:
            self._add_grid(dwg, view_width, view_height, line_height)

        # Draw strokes
        initial_y_offset = line_height
        for stroke, color, width in zip(strokes, stroke_colors, stroke_widths, strict=False):
            # The stroke array now contains (x, y, eos) tuples already
            if len(stroke) > 0:
                stroke_data = [(x, y, eos) for x, y, eos in stroke]
                self._draw_stroke(dwg, stroke_data, initial_y_offset, color, width, show_endpoints)
            initial_y_offset += line_height

        dwg.save()

    def _add_grid(self, dwg: svgwrite.Drawing, width: int, height: int, spacing: int) -> None:
        """Add background grid to SVG."""
        for y in range(0, height, spacing):
            dwg.add(dwg.line(start=(0, y), end=(width, y), stroke="#EEEEEE", stroke_width=1))

    def _draw_stroke(
        self,
        dwg: svgwrite.Drawing,
        stroke_data: List[Tuple[float, float, float]],
        y_offset: float,
        color: str,
        width: float,
        show_endpoints: bool,
    ) -> None:
        """Draw a single stroke in the SVG using consistent binary pen state."""
        prev_up = 1  # Start with pen up
        path_data = f"M 0,{y_offset}"

        for x, y, pen_up in stroke_data:
            # Use the same binary pen state that was fed back to the model
            command = "M" if prev_up else "L"
            path_data += f" {command} {x},{-y + y_offset}"
            prev_up = int(pen_up)

            if show_endpoints and pen_up > 0.5:
                dwg.add(dwg.circle(center=(x, -y + y_offset), r=2, fill="red"))

        path = svgwrite.path.Path(path_data)
        path = path.stroke(color=color, width=width, linecap="round").fill("none")
        dwg.add(path)


if __name__ == "__main__":
    curr_directory = os.path.dirname(os.path.realpath(__file__))
    model_save_dir = f"{curr_directory}/saved_models/handwriting_synthesis_single_batch_subset/"
    model_load_path = os.path.join(model_save_dir, "best_model.keras")

    # Example usage
    texts = ["Better to have loved", "and lost", "than never loved at all"]

    # Create calligrapher with default settings
    calligrapher = Calligrapher(
        model_load_path,
        num_output_mixtures=20,  # Increased from 1 to match paper
    )

    # Generate samples with different styles
    calligrapher.write(texts, "output_default.svg", temperature=1.0, bias=0.0)

    # Generate cleaner, more consistent writing
    calligrapher.write(texts, "output_clean.svg", temperature=0.5, bias=0.2)

    # Generate more creative, varied writing
    calligrapher.write(texts, "output_creative.svg", temperature=1.2, bias=0.0)
