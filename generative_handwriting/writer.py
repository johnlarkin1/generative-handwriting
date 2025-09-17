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
        sigma_min: float = 1e-4,
        sigma_max: float = 1e4,
        rho_max: float = 0.99,
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
        # Adjust standard deviations
        sigma1_adj = np.exp(np.log(sigma1) - bias) * np.sqrt(temperature)
        sigma2_adj = np.exp(np.log(sigma2) - bias) * np.sqrt(temperature)

        # Adjust mixture weights (pi is already probabilities, not logits)
        pi_adj = np.power(pi, (1 + bias) / temperature)
        pi_adj /= np.sum(pi_adj, axis=-1, keepdims=True)

        return pi_adj, mu1, mu2, sigma1_adj, sigma2_adj, rho

    def sample(
        self, lines: List[str], max_char_len: int = MAX_CHAR_LEN, bias: float = 0.0, temperature: float = 1.0
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Sample handwriting sequences for given lines of text.

        Args:
            lines: List of text strings to render
            max_char_len: Maximum number of characters per line
            bias: Controls the trade-off between quality and diversity (0.0-1.0)
                Higher values produce cleaner but less diverse samples
            temperature: Controls randomness in sampling (0.1-1.0)
                Lower values produce more consistent writing
                Higher values produce more varied writing

        Returns:
            Tuple of (sampled_points, end_of_stroke_probs)

        Raises:
            ValueError: If lines is empty or contains invalid characters
        """
        if not lines:
            raise ValueError("Must provide at least one line of text")

        # Encode input text
        encoded_lines = np.array([encode_ascii(line) for line in lines])
        batch_size = len(encoded_lines)

        # Prepare model inputs
        x_in = np.zeros((batch_size, max_char_len, 3), dtype=np.float32)
        char_seq_len = np.array([len(line) for line in lines])
        char_seq = np.zeros((len(lines), max_char_len), dtype=int)

        for i, line in enumerate(encoded_lines):
            char_seq[i, : len(line)] = line

        # Get model predictions
        mdn_outputs = self.model({"input_strokes": x_in, "input_chars": char_seq, "input_char_lens": char_seq_len})

        # Split outputs into components
        # NOTE: MDN layer outputs already transformed values (pi is softmaxed, sigma is exp'd, rho is tanh'd)
        pi, mu1, mu2, sigma1, sigma2, rho, eos_logits = tf.split(
            mdn_outputs, [self.num_output_mixtures] * 6 + [1], axis=-1
        )

        # Apply temperature and bias adjustments
        if bias != 0.0 or temperature != 1.0:
            pi, mu1, mu2, sigma1, sigma2, rho = self.adjust_parameters(
                pi.numpy(),  # Already softmaxed
                mu1.numpy(),
                mu2.numpy(),
                sigma1.numpy(),
                sigma2.numpy(),
                rho.numpy(),
                bias,
                temperature,
            )
        else:
            pi = pi.numpy()  # Already softmaxed, no need to apply softmax again

        # Sample from mixture
        indices = [np.random.choice(self.num_output_mixtures, p=pi[i]) for i in range(pi.shape[0])]
        indices = np.array(indices)[:, np.newaxis]

        # Select components based on sampled indices
        selected = {
            "mu1": np.take_along_axis(mu1, indices, axis=-1),
            "mu2": np.take_along_axis(mu2, indices, axis=-1),
            "sigma1": np.take_along_axis(sigma1, indices, axis=-1),
            "sigma2": np.take_along_axis(sigma2, indices, axis=-1),
            "rho": np.take_along_axis(rho, indices, axis=-1),
        }

        # Generate points
        sampled_points = [
            self.sample_gaussian_2d(
                selected["mu1"][i, 0],
                selected["mu2"][i, 0],
                selected["sigma1"][i, 0],
                selected["sigma2"][i, 0],
                selected["rho"][i, 0],
            )
            for i in range(pi.shape[0])
        ]

        # Sample end-of-stroke probabilities from logits with temperature
        eos_prob = tf.sigmoid(eos_logits / temperature).numpy()

        return sampled_points, eos_prob

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

        # Generate samples
        strokes, eos = self.sample(lines, bias=bias, temperature=temperature)

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
            self._draw_stroke(dwg, stroke, initial_y_offset, color, width, show_endpoints)
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
        """Draw a single stroke in the SVG."""
        prev_eos = 1.0
        path_data = f"M 0,{y_offset}"

        for x, y, eos in stroke_data:
            command = "M" if prev_eos == 1.0 else "L"
            path_data += f" {command} {x},{-y + y_offset}"
            prev_eos = eos

            if show_endpoints and eos > 0.5:
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
