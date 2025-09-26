import json
import os
from typing import Any, Tuple

import tensorflow as tf

from .constants import MAX_STROKE_LEN, MAX_CHAR_LEN


def load_model_if_exists(model_save_path, custom_objects: dict[str, Any]):
    try:
        model = tf.keras.models.load_model(
            model_save_path,
            custom_objects=custom_objects,
        )
        print("successfully loaded model")
        return (
            model,
            True,
        )
    except Exception as e:
        print(f"Issue loading! {e}")
        return None, False


def save_epochs_info(epoch, epochs_info_path):
    info = {"last_epoch": epoch}
    with open(epochs_info_path, "w") as file:
        json.dump(info, file)


def load_epochs_info(epochs_info_path):
    if os.path.exists(epochs_info_path):
        with open(epochs_info_path, "r") as file:
            info = json.load(file)
        return info["last_epoch"]
    return 0


def materialize_model(model: tf.keras.Model) -> None:
    """Force materialization of all model variables by doing a dummy forward pass.

    This ensures that all layers (especially RNN/attention layers) have their
    variables created before saving, preventing Keras lazy materialization issues.

    Args:
        model: The Keras model to materialize

    Raises:
        Exception: If materialization fails
    """
    try:
        print("🔧 Materializing model variables...")

        # Create dummy inputs matching expected model signature
        dummy_inputs = {
            "input_strokes": tf.zeros([1, MAX_STROKE_LEN, 3], tf.float32),
            "input_stroke_lens": tf.constant([MAX_STROKE_LEN], tf.int32),
            "target_stroke_lens": tf.constant([MAX_STROKE_LEN-1], tf.int32),
            "input_chars": tf.zeros([1, MAX_CHAR_LEN], tf.int32),
            "input_char_lens": tf.constant([MAX_CHAR_LEN], tf.int32),
        }

        # Force forward pass to create all variables
        _ = model(dummy_inputs, training=False)

        print(f"✅ Model materialized with {len(model.trainable_variables)} trainable variables")

    except Exception as e:
        print(f"❌ Model materialization failed: {e}")
        raise


def validate_model_parameters(model: tf.keras.Model) -> Tuple[bool, bool, int]:
    """Validate that the model has expected LSTM/attention parameters.

    Args:
        model: The Keras model to validate

    Returns:
        Tuple of (has_lstm_params, has_attention_params, total_param_count)
    """
    variable_names = [v.name.lower() for v in model.trainable_variables]

    # Check for LSTM/RNN parameters
    has_lstm = any(keyword in name for name in variable_names
                   for keyword in ['lstm', 'rnn', 'peephole', 'cell'])

    # Check for attention parameters
    # Look for explicit attention keywords OR the characteristic attention Dense layer shape
    has_attention_keywords = any(keyword in name for name in variable_names
                                for keyword in ['attention', 'kappa', 'alpha', 'beta'])

    # Look for attention Dense layer by shape: (479, 30) kernel and (30,) bias
    # 479 = 3(stroke) + 400(LSTM) + 76(chars), 30 = 3*10(Gaussians)
    has_attention_dense = any(
        (v.shape == (479, 30) or v.shape == (30,)) and 'mdn' not in v.name.lower()
        for v in model.trainable_variables
    )

    has_attention = has_attention_keywords or has_attention_dense

    # Count total parameters
    total_params = sum(tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables)

    return has_lstm, has_attention, total_params


def save_model_robustly(model: tf.keras.Model, model_path: str,
                       require_lstm: bool = True, require_attention: bool = True) -> bool:
    """Robustly save a model with materialization and validation.

    Args:
        model: The Keras model to save
        model_path: Path where to save the model
        require_lstm: Whether to require LSTM parameters (default True)
        require_attention: Whether to require attention parameters (default True)

    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Materialize all variables
        materialize_model(model)

        # Validate parameters
        has_lstm, has_attention, total_params = validate_model_parameters(model)

        print(f"📊 Model validation:")
        print(f"   LSTM parameters: {'✅' if has_lstm else '❌'}")
        print(f"   Attention parameters: {'✅' if has_attention else '❌'}")
        print(f"   Total parameters: {total_params:,}")

        # Check requirements
        if require_lstm and not has_lstm:
            raise ValueError("Model missing required LSTM parameters")
        if require_attention and not has_attention:
            raise ValueError("Model missing required attention parameters")

        # Expected parameter count for synthesis model should be much higher than MDN-only
        if total_params < 100000:  # Less than 100K suggests MDN-only
            print(f"⚠️ Warning: Low parameter count ({total_params:,}) suggests incomplete model")

        # Save the full model
        model.save(model_path)
        print(f"✅ Model saved successfully to: {model_path}")

        # Save weights as backup
        weights_path = model_path.replace('.keras', '_weights.keras')
        model.save_weights(weights_path)
        print(f"💾 Backup weights saved to: {weights_path}")

        return True

    except Exception as e:
        print(f"❌ Model save failed: {e}")
        return False


def load_model_robustly(model_path: str, custom_objects: dict[str, Any],
                       require_lstm: bool = True, require_attention: bool = True) -> Tuple[tf.keras.Model, bool]:
    """Robustly load a model with materialization and validation.

    Args:
        model_path: Path to the saved model
        custom_objects: Custom objects dict for loading
        require_lstm: Whether to require LSTM parameters (default True)
        require_attention: Whether to require attention parameters (default True)

    Returns:
        Tuple of (model, success_flag)
    """
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("📥 Model loaded successfully")

        # Materialize all variables
        materialize_model(model)

        # Validate parameters
        has_lstm, has_attention, total_params = validate_model_parameters(model)

        print(f"📊 Loaded model validation:")
        print(f"   LSTM parameters: {'✅' if has_lstm else '❌'}")
        print(f"   Attention parameters: {'✅' if has_attention else '❌'}")
        print(f"   Total parameters: {total_params:,}")

        # Check requirements
        if require_lstm and not has_lstm:
            raise ValueError(
                "Loaded model missing LSTM parameters! "
                "This checkpoint appears to be MDN-only. "
                "Please retrain or use a complete checkpoint."
            )
        if require_attention and not has_attention:
            raise ValueError(
                "Loaded model missing attention parameters! "
                "This checkpoint appears incomplete for text synthesis. "
                "Please retrain or use a complete checkpoint."
            )

        if total_params < 100000:
            print(f"⚠️ Warning: Suspiciously low parameter count ({total_params:,})")

        return model, True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, False
