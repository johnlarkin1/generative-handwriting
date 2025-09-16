"""
Debugging utilities for detecting and diagnosing NaN issues during training.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import datetime
import json
import os


class NaNMonitor:
    """Monitor for detecting and logging NaN occurrences during training."""

    def __init__(self, checkpoint_dir: str = None):
        """
        Initialize the NaN monitor.

        Args:
            checkpoint_dir: Directory to save emergency checkpoints when NaN detected
        """
        self.checkpoint_dir = checkpoint_dir
        self.nan_history: List[Dict[str, Any]] = []
        self.last_good_weights = None
        self.nan_count = 0

    def check_tensors(self, tensors: Dict[str, tf.Tensor], step: int, prefix: str = "") -> bool:
        """
        Check multiple tensors for NaN/Inf values.

        Args:
            tensors: Dictionary of tensor name to tensor
            step: Current training step
            prefix: Prefix for logging

        Returns:
            True if NaN/Inf detected, False otherwise
        """
        has_nan = False
        for name, tensor in tensors.items():
            if tensor is None:
                continue

            # Only check NaN/Inf for floating-point tensors
            if tf.dtypes.as_dtype(tensor.dtype).is_floating:
                has_nan_val = tf.reduce_any(tf.math.is_nan(tensor))
                has_inf_val = tf.reduce_any(tf.math.is_inf(tensor))
            else:
                has_nan_val = False
                has_inf_val = False

            if has_nan_val or has_inf_val:
                has_nan = True
                self.nan_count += 1

                # Log detailed information
                nan_info = {
                    "step": step,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "tensor_name": f"{prefix}/{name}",
                    "has_nan": bool(has_nan_val.numpy()),
                    "has_inf": bool(has_inf_val.numpy()),
                    "shape": tensor.shape.as_list(),
                    "dtype": str(tensor.dtype),
                }

                # Try to get statistics before NaN (only for floating-point tensors)
                try:
                    if tf.dtypes.as_dtype(tensor.dtype).is_floating:
                        nan_mask = tf.math.is_nan(tensor)
                        inf_mask = tf.math.is_inf(tensor)
                        valid_mask = tf.logical_not(tf.logical_or(nan_mask, inf_mask))
                    else:
                        # For non-floating point tensors, all values are valid
                        valid_mask = tf.ones_like(tensor, dtype=tf.bool)

                    if tf.reduce_any(valid_mask):
                        valid_values = tf.boolean_mask(tensor, valid_mask)
                        nan_info["valid_mean"] = float(tf.reduce_mean(valid_values).numpy())
                        nan_info["valid_std"] = float(tf.math.reduce_std(valid_values).numpy())
                        nan_info["valid_min"] = float(tf.reduce_min(valid_values).numpy())
                        nan_info["valid_max"] = float(tf.reduce_max(valid_values).numpy())
                except:
                    pass

                self.nan_history.append(nan_info)
                print(f"\n⚠️  NaN/Inf detected at step {step} in {prefix}/{name}")
                print(f"   Shape: {tensor.shape}, Has NaN: {has_nan_val.numpy()}, Has Inf: {has_inf_val.numpy()}")

        return has_nan

    def check_model_weights(self, model: tf.keras.Model, step: int) -> bool:
        """
        Check all model weights for NaN values.

        Args:
            model: Keras model to check
            step: Current training step

        Returns:
            True if NaN detected, False otherwise
        """
        weights_dict = {}
        for var in model.trainable_variables:
            weights_dict[var.name] = var

        return self.check_tensors(weights_dict, step, prefix="weights")

    def check_gradients(self, gradients: List[tf.Tensor], variables: List[tf.Variable], step: int) -> bool:
        """
        Check gradients for NaN values.

        Args:
            gradients: List of gradient tensors
            variables: List of corresponding variables
            step: Current training step

        Returns:
            True if NaN detected, False otherwise
        """
        grad_dict = {}
        for grad, var in zip(gradients, variables):
            if grad is not None:
                grad_dict[var.name] = grad

        return self.check_tensors(grad_dict, step, prefix="gradients")

    def save_emergency_checkpoint(self, model: tf.keras.Model, step: int):
        """Save model checkpoint when NaN detected."""
        if self.checkpoint_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"emergency_checkpoint_step_{step}_{timestamp}.keras"
            )
            try:
                model.save(checkpoint_path)
                print(f"💾 Emergency checkpoint saved to {checkpoint_path}")
            except Exception as e:
                print(f"❌ Failed to save emergency checkpoint: {e}")

    def save_nan_report(self, filepath: str = "nan_report.json"):
        """Save NaN detection history to a JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.nan_history, f, indent=2)
            print(f"📊 NaN report saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save NaN report: {e}")


def add_gradient_noise(gradients: List[tf.Tensor], noise_scale: float = 1e-8) -> List[tf.Tensor]:
    """
    Add small noise to gradients to prevent exact zero gradients.

    Args:
        gradients: List of gradient tensors
        noise_scale: Scale of noise to add

    Returns:
        List of gradients with noise added
    """
    noisy_gradients = []
    for grad in gradients:
        if grad is not None:
            noise = tf.random.normal(grad.shape, mean=0.0, stddev=noise_scale)
            grad = grad + noise
        noisy_gradients.append(grad)
    return noisy_gradients


def clip_gradient_norms(gradients: List[tf.Tensor], max_norm: float) -> Tuple[List[tf.Tensor], float]:
    """
    Clip gradients by global norm.

    Args:
        gradients: List of gradient tensors
        max_norm: Maximum gradient norm

    Returns:
        Tuple of (clipped gradients, global norm before clipping)
    """
    global_norm = tf.linalg.global_norm(gradients)
    clipped_grads, _ = tf.clip_by_global_norm(gradients, max_norm)
    return clipped_grads, global_norm


def log_tensor_statistics(tensor: tf.Tensor, name: str, step: int = None):
    """
    Log statistics for a tensor.

    Args:
        tensor: Tensor to analyze
        name: Name for logging
        step: Optional step number
    """
    if tensor is None:
        print(f"{name}: None")
        return

    try:
        mean = tf.reduce_mean(tensor)
        std = tf.math.reduce_std(tensor)
        min_val = tf.reduce_min(tensor)
        max_val = tf.reduce_max(tensor)

        step_str = f"[Step {step}]" if step is not None else ""
        print(f"{step_str} {name}: mean={mean:.6f}, std={std:.6f}, min={min_val:.6f}, max={max_val:.6f}")
    except Exception as e:
        print(f"Failed to compute statistics for {name}: {e}")


def safe_log(x: tf.Tensor, epsilon: float = 1e-8) -> tf.Tensor:
    """
    Compute log with numerical stability.

    Args:
        x: Input tensor
        epsilon: Small value for numerical stability

    Returns:
        log(x + epsilon)
    """
    return tf.math.log(x + epsilon)


def safe_divide(numerator: tf.Tensor, denominator: tf.Tensor, epsilon: float = 1e-8) -> tf.Tensor:
    """
    Perform division with protection against division by zero.

    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        epsilon: Small value to prevent division by zero

    Returns:
        numerator / (denominator + epsilon)
    """
    return numerator / (denominator + epsilon)


def stabilize_variance(x: tf.Tensor, min_std: float = 1e-3, max_std: float = 100.0) -> tf.Tensor:
    """
    Stabilize variance values to prevent numerical issues.

    Args:
        x: Input tensor (typically standard deviations)
        min_std: Minimum allowed standard deviation
        max_std: Maximum allowed standard deviation

    Returns:
        Clipped tensor
    """
    return tf.clip_by_value(x, min_std, max_std)


def check_data_batch(batch_data: Dict[str, tf.Tensor], batch_idx: int) -> bool:
    """
    Check a data batch for invalid values.

    Args:
        batch_data: Dictionary of batch tensors
        batch_idx: Batch index

    Returns:
        True if batch contains NaN/Inf, False otherwise
    """
    has_invalid = False
    for key, tensor in batch_data.items():
        if tensor is None:
            continue

        # Only check NaN/Inf for floating-point tensors
        if tf.dtypes.as_dtype(tensor.dtype).is_floating:
            has_nan = tf.reduce_any(tf.math.is_nan(tensor))
            has_inf = tf.reduce_any(tf.math.is_inf(tensor))

            if has_nan or has_inf:
                has_invalid = True
                print(f"⚠️  Invalid data in batch {batch_idx}, key '{key}':")
                print(f"   Has NaN: {has_nan.numpy()}, Has Inf: {has_inf.numpy()}")
                print(f"   Shape: {tensor.shape}")

    return has_invalid