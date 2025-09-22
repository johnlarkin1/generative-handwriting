# ruff: noqa: E402
import os

import numpy as np
import tensorflow as tf

tf.config.optimizer.set_jit(True)

from constants import (
    BATCH_SIZE,
    GRADIENT_CLIP_VALUE,
    LEARNING_RATE,
    NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
    NUM_EPOCH,
)

from generative_handwriting.common import print_model_parameters

# Debug utilities removed - using simpler error handling
from generative_handwriting.loader import HandwritingDataLoader
from generative_handwriting.model.handwriting_models import DeepHandwritingPredictionModel
from generative_handwriting.model.mixture_density_network import MixtureDensityLayer, mdn_loss
from generative_handwriting.model_io import load_epochs_info, save_epochs_info

curr_directory = os.path.dirname(os.path.realpath(__file__))
model_save_dir = f"{curr_directory}/saved_models/full_handwriting_prediction/"
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, "best_model.keras")
epochs_info_path = os.path.join(model_save_dir, "epochs_info.json")
num_mixture_components = NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS


@tf.keras.utils.register_keras_serializable()
def model_mdn_loss(actual, outputs, combined_train_lengths, num_mixture_components):
    return mdn_loss(actual, outputs, combined_train_lengths, num_mixture_components)


def compute_graves_log_loss(model, x_data, y_data, seq_lengths, num_mixture_components, batch_size=32):
    """Compute Graves log-loss metric (mean NLL per timestep in nats).

    This matches the metric reported in Table 3 of Graves' paper.
    """
    total_nll = 0.0
    total_timesteps = 0
    total_sse = 0.0  # Sum squared error

    num_samples = len(x_data)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_x = x_data[start_idx:end_idx]
        batch_y = y_data[start_idx:end_idx]
        batch_len = seq_lengths[start_idx:end_idx]

        # Get predictions
        predictions = model(batch_x, training=False)

        # Compute NLL for this batch
        batch_nll = mdn_loss(batch_y, predictions, batch_len, num_mixture_components)

        # The mdn_loss returns average loss per valid timestep
        # We need to accumulate the total
        batch_timesteps = np.sum(batch_len)
        total_nll += batch_nll.numpy() * batch_timesteps
        total_timesteps += batch_timesteps

        # Compute SSE (sum squared error) for comparison
        # Extract means from predictions (indices num_components:3*num_components)
        mu_x = predictions[:, :, num_mixture_components : 2 * num_mixture_components]
        mu_y = predictions[:, :, 2 * num_mixture_components : 3 * num_mixture_components]
        pi = predictions[:, :, :num_mixture_components]

        # Get most likely component for each timestep
        best_components = tf.argmax(pi, axis=-1)
        batch_size_curr = tf.shape(mu_x)[0]
        seq_len = tf.shape(mu_x)[1]

        # Gather the means for the most likely components
        indices = tf.stack(
            [
                tf.range(batch_size_curr)[:, None] * tf.ones([1, seq_len], dtype=tf.int32),
                tf.range(seq_len)[None, :] * tf.ones([batch_size_curr, 1], dtype=tf.int32),
                best_components,
            ],
            axis=-1,
        )

        pred_x = tf.gather_nd(mu_x, indices)
        pred_y = tf.gather_nd(mu_y, indices)

        # Compute SSE
        actual_x = batch_y[:, :, 0]
        actual_y = batch_y[:, :, 1]

        # Apply mask for valid timesteps
        mask = tf.sequence_mask(batch_len, maxlen=tf.shape(batch_y)[1], dtype=tf.float32)
        squared_error = ((pred_x - actual_x) ** 2 + (pred_y - actual_y) ** 2) * mask
        total_sse += tf.reduce_sum(squared_error).numpy()

    # Compute averages
    graves_log_loss = -total_nll / max(total_timesteps, 1)  # Negative log-likelihood in nats
    mean_sse = total_sse / max(total_timesteps, 1)

    return graves_log_loss, mean_sse


def load_model_if_exists(model_save_path):
    try:
        return (
            tf.keras.models.load_model(
                model_save_path,
                custom_objects={
                    "model_mdn_loss": model_mdn_loss,
                    "MixtureDensityLayer": MixtureDensityLayer,
                    "DeepHandwritingPredictionModel": DeepHandwritingPredictionModel,
                },
            ),
            True,
        )
    except Exception as e:
        print(f"Issue loading! {e}")
        return None, False


if __name__ == "__main__":
    # Load and prepare data only when script is run directly
    data_loader = HandwritingDataLoader()
    data_loader.prepare_data()
    combined_train_strokes, combined_train_lengths = (
        data_loader.combined_train_strokes,
        data_loader.combined_train_stroke_lengths,
    )

    # Get validation data for Graves metric
    validation_strokes, validation_lengths = (
        data_loader.validation_strokes,
        data_loader.validation_stroke_lengths,
    )

    stroke_model, model_loaded = load_model_if_exists(model_save_path)
    desired_epochs = NUM_EPOCH

    x_train = combined_train_strokes
    x_train_len = combined_train_lengths
    y_train = np.zeros_like(x_train)
    y_train[:, :-1, :] = x_train[:, 1:, :]
    # Graves: labels are x_{t+1}, so there are (len-1) valid targets per sequence.
    y_train_len = np.maximum(x_train_len - 1, 1)

    # EOS Data Verification - analyze distribution of end-of-stroke signals
    print("🔍 EOS Data Verification:")
    eos_values = combined_train_strokes[:, :, 2]  # EOS is the 3rd column (index 2)
    total_timesteps = np.sum(combined_train_lengths)
    eos_count = np.sum(eos_values)
    eos_ratio = eos_count / total_timesteps
    sequences_with_eos = np.mean(np.any(eos_values == 1, axis=1))

    print(f"  • Total timesteps: {total_timesteps}")
    print(f"  • EOS count: {eos_count}")
    print(f"  • EOS ratio: {eos_ratio:.4f} ({eos_ratio * 100:.2f}%)")
    print(f"  • Sequences with EOS: {sequences_with_eos:.2f}")
    print(f"  • Expected strokes per sequence: {eos_count / len(combined_train_strokes):.1f}")
    print(f"  • EOS value range: [{np.min(eos_values):.1f}, {np.max(eos_values):.1f}]")
    print("✅ EOS data verification complete\n")

    # Prepare validation data
    x_val = validation_strokes
    x_val_len = validation_lengths
    y_val = np.zeros_like(x_val)
    y_val[:, :-1, :] = x_val[:, 1:, :]
    y_val_len = np.maximum(x_val_len - 1, 1)

    best_loss = float("inf")

    batch_size = BATCH_SIZE
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, y_train_len))
    # Optimize dataset pipeline for GPU
    dataset = (
        dataset.batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)  # Prefetch data to overlap computation
        .cache()
    )  # Cache dataset in memory for faster access

    # Use Keras 3 compatible optimizer with mixed precision
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Configure GPU memory growth to avoid OOM
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("🎯 GPU memory growth enabled")
        except RuntimeError as e:
            print(f"⚠️ GPU memory configuration failed: {e}")

    # Simplified error handling without debug utilities
    print("🔍 Training with basic error monitoring")

    def simple_check_data_batch(batch_data, step):
        """Simple NaN check for batch data."""
        for key, tensor in batch_data.items():
            if tf.reduce_any(tf.math.is_nan(tensor)) or tf.reduce_any(tf.math.is_inf(tensor)):
                print(f"⚠️ NaN/Inf found in {key} at step {step}")
                return True
        return False

    if not model_loaded:
        print("No suitable saved model found or model has changed, initializing a new one...")
        stroke_model = DeepHandwritingPredictionModel(num_mixture_components=num_mixture_components)

    assert stroke_model is not None, "Model not loaded or initialized."

    last_trained_epoch = load_epochs_info(epochs_info_path)
    total_step = 0
    nan_detected = False  # Track if NaN is detected to break training

    for epoch in range(last_trained_epoch, desired_epochs):
        print(f"Epoch {epoch + 1}/{desired_epochs}")
        epoch_losses = []  # Track losses for each epoch

        for step, (batch_x, batch_y, batch_len) in enumerate(dataset):
            total_step += 1

            # Check input data for NaN/Inf
            batch_data = {"batch_x": batch_x, "batch_y": batch_y, "batch_len": batch_len}
            if simple_check_data_batch(batch_data, step):
                print(f"⚠️  Skipping batch {step} due to invalid input data")
                continue

            with tf.GradientTape() as tape:
                # Use tf.function for XLA compilation of forward pass
                predictions = stroke_model(batch_x, training=True)

                # Check predictions for NaN before loss calculation
                if tf.reduce_any(tf.math.is_nan(predictions)) or tf.reduce_any(tf.math.is_inf(predictions)):
                    print(f"🚨 NaN/Inf detected in model predictions at step {total_step}")
                    nan_detected = True
                    break

                nll = model_mdn_loss(batch_y, predictions, batch_len, num_mixture_components)
                # Add layer regularizers (e.g., from MixtureDensityLayer)
                reg_loss = tf.add_n(stroke_model.losses) if stroke_model.losses else 0.0
                total_loss = nll + reg_loss
                loss = total_loss

                # Check loss for NaN
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print(f"🚨 NaN/Inf detected in loss at step {total_step}: {loss.numpy()}")
                    nan_detected = True
                    break
            gradients = tape.gradient(loss, stroke_model.trainable_variables)

            # Check gradients for NaN
            has_nan_grad = any(tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None)
            if has_nan_grad:
                print(f"🚨 NaN detected in gradients at step {total_step}")
                nan_detected = True
                break

            clipped = [
                (tf.clip_by_value(g, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE), v_)
                for g, v_ in zip(gradients, stroke_model.trainable_variables, strict=False)
            ]
            optimizer.apply_gradients(clipped)

            # Check model weights after update
            has_nan_weights = any(tf.reduce_any(tf.math.is_nan(w)) for w in stroke_model.trainable_variables)
            if has_nan_weights:
                print(f"🚨 NaN detected in model weights after update at step {total_step}")
                nan_detected = True
                break
            epoch_losses.append(loss.numpy())

            if epoch == 0 and step == 0:
                print_model_parameters(stroke_model)

            if step == 0:
                # Calculate EOS monitoring metrics
                eos_logits = predictions[:, :, -1]  # Last column contains EOS logits
                eos_probs = tf.sigmoid(eos_logits)
                eos_predictions = tf.round(eos_probs)
                eos_targets = batch_y[:, :, 2]  # EOS is the 3rd column (index 2)

                # Apply sequence mask for accurate metrics
                mask = tf.sequence_mask(batch_len, maxlen=tf.shape(batch_y)[1], dtype=tf.float32)
                masked_correct = tf.cast(tf.equal(eos_predictions, eos_targets), tf.float32) * mask
                eos_accuracy = tf.reduce_sum(masked_correct) / tf.maximum(tf.reduce_sum(mask), 1.0)
                mean_eos_prob = tf.reduce_sum(eos_probs * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
                mean_eos_target = tf.reduce_sum(eos_targets * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)

                print(
                    f"Step {step}, Loss: {loss.numpy():.4f}, EOS Acc: {eos_accuracy.numpy():.3f}, "
                    f"EOS Prob: {mean_eos_prob.numpy():.3f}, EOS Target: {mean_eos_target.numpy():.3f}"
                )
                # Simple loss statistics
                loss_min = tf.reduce_min(loss)
                loss_max = tf.reduce_max(loss)
                loss_mean = tf.reduce_mean(loss)
                print(f"  Loss stats: min={loss_min:.6f}, max={loss_max:.6f}, mean={loss_mean:.6f}")

            # Log simple statistics every 50 steps
            if step % 50 == 0:
                pred_min = tf.reduce_min(predictions)
                pred_max = tf.reduce_max(predictions)
                pred_mean = tf.reduce_mean(predictions)
                print(f"  Predictions stats: min={pred_min:.6f}, max={pred_max:.6f}, mean={pred_mean:.6f}")

        # Break if NaN detected
        if nan_detected:
            print("💔 Training stopped due to NaN detection")
            break

        # After each epoch, check if the average loss is the best so far
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} Average Training Loss: {epoch_loss}")

        # Compute validation metrics (Graves log-loss and SSE)
        print("Computing validation metrics...")
        val_log_loss, val_sse = compute_graves_log_loss(stroke_model, x_val, y_val, y_val_len, num_mixture_components)
        print(f"Validation Graves Log-Loss: {val_log_loss:.1f} nats")
        print(f"Validation SSE: {val_sse:.4f}")

        # Also compute training set Graves metric for comparison
        train_log_loss, train_sse = compute_graves_log_loss(
            stroke_model, x_train[:1000], y_train[:1000], y_train_len[:1000], num_mixture_components
        )
        print(f"Training Graves Log-Loss (first 1000): {train_log_loss:.1f} nats")
        print(f"Training SSE (first 1000): {train_sse:.4f}")

        # Use validation log-loss for model selection
        if val_log_loss < best_loss:
            print(f"New best validation log-loss {val_log_loss:.1f}, saving model.")
            best_loss = val_log_loss
            stroke_model.save(model_save_path)  # Save the model
        save_epochs_info(epoch + 1, epochs_info_path)

    print("✅ Training completed successfully")

    print("Training completed.")
