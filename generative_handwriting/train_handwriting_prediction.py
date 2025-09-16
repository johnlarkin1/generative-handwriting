import os

import numpy as np
import tensorflow as tf

# Enable XLA compilation for better GPU utilization
tf.config.optimizer.set_jit(True)
print("🚀 XLA JIT compilation enabled")
from common import (
    print_model_parameters,
)
from constants import (
    BATCH_SIZE,
    GRADIENT_CLIP_VALUE,
    LEARNING_RATE,
    NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
    NUM_EPOCH,
)
from debug_utils import NaNMonitor, log_tensor_statistics, check_data_batch
from loader import HandwritingDataLoader
from model.handwriting_models import DeepHandwritingPredictionModel
from model.mixture_density_network import MixtureDensityLayer, mdn_loss
from model_io import load_epochs_info, save_epochs_info

curr_directory = os.path.dirname(os.path.realpath(__file__))
model_save_dir = f"{curr_directory}/saved_models/full_handwriting_prediction/"
os.makedirs(model_save_dir, exist_ok=True)  # Create directory if it doesn't exist
model_save_path = os.path.join(model_save_dir, "best_model.keras")
epochs_info_path = os.path.join(model_save_dir, "epochs_info.json")
num_mixture_components = NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS


@tf.keras.utils.register_keras_serializable()
def model_mdn_loss(actual, outputs, combined_train_lengths, num_mixture_components):
    return mdn_loss(actual, outputs, combined_train_lengths, num_mixture_components)


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

    stroke_model, model_loaded = load_model_if_exists(model_save_path)
    desired_epochs = NUM_EPOCH

    x_train = combined_train_strokes
    x_train_len = combined_train_lengths
    y_train = np.zeros_like(x_train)
    y_train[:, :-1, :] = x_train[:, 1:, :]
    y_train_len = np.zeros_like(x_train_len)
    y_train_len = x_train_len[1:]

    best_loss = float("inf")

    batch_size = BATCH_SIZE
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, x_train_len))
    # Optimize dataset pipeline for GPU
    dataset = (dataset
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE)  # Prefetch data to overlap computation
               .cache())  # Cache dataset in memory for faster access

    # Use Keras 3 compatible optimizer with mixed precision
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Configure GPU memory growth to avoid OOM
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("🎯 GPU memory growth enabled")
        except RuntimeError as e:
            print(f"⚠️ GPU memory configuration failed: {e}")

    # Initialize NaN monitor
    nan_monitor = NaNMonitor(checkpoint_dir=model_save_dir)
    print("🔍 NaN monitoring enabled")

    if not model_loaded:
        print("No suitable saved model found or model has changed, initializing a new one...")
        stroke_model = DeepHandwritingPredictionModel(num_mixture_components=num_mixture_components)

    assert stroke_model is not None, "Model not loaded or initialized."

    last_trained_epoch = load_epochs_info(epochs_info_path)
    total_step = 0

    for epoch in range(last_trained_epoch, desired_epochs):
        print(f"Epoch {epoch + 1}/{desired_epochs}")
        epoch_losses = []  # Track losses for each epoch

        for step, (batch_x, batch_y, batch_len) in enumerate(dataset):
            total_step += 1

            # Check input data for NaN/Inf
            batch_data = {"batch_x": batch_x, "batch_y": batch_y, "batch_len": batch_len}
            if check_data_batch(batch_data, step):
                print(f"⚠️  Skipping batch {step} due to invalid input data")
                continue

            with tf.GradientTape() as tape:
                # Use tf.function for XLA compilation of forward pass
                predictions = stroke_model(batch_x, training=True)

                # Check predictions for NaN before loss calculation
                if nan_monitor.check_tensors({"predictions": predictions}, total_step, "model_output"):
                    print(f"🚨 NaN detected in model predictions at step {total_step}")
                    nan_monitor.save_emergency_checkpoint(stroke_model, total_step)
                    break

                nll = model_mdn_loss(batch_y, predictions, batch_len, num_mixture_components)
                # Add layer regularizers (e.g., from MixtureDensityLayer)
                total_loss = nll + tf.add_n(stroke_model.losses)
                loss = total_loss

                # Check loss for NaN
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    print(f"🚨 NaN/Inf detected in loss at step {total_step}: {loss.numpy()}")
                    nan_monitor.save_emergency_checkpoint(stroke_model, total_step)
                    break
            gradients = tape.gradient(loss, stroke_model.trainable_variables)

            # Check gradients for NaN
            if nan_monitor.check_gradients(gradients, stroke_model.trainable_variables, total_step):
                print(f"🚨 NaN detected in gradients at step {total_step}")
                nan_monitor.save_emergency_checkpoint(stroke_model, total_step)
                break

            clipped = [
                (tf.clip_by_value(g, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE), v_)
                for g, v_ in zip(gradients, stroke_model.trainable_variables, strict=False)
            ]
            optimizer.apply_gradients(clipped)

            # Check model weights after update
            if nan_monitor.check_model_weights(stroke_model, total_step):
                print(f"🚨 NaN detected in model weights after update at step {total_step}")
                nan_monitor.save_emergency_checkpoint(stroke_model, total_step)
                break
            epoch_losses.append(loss.numpy())

            if epoch == 0 and step == 0:
                print_model_parameters(stroke_model)

            if step % 5 == 0:
                print(f"Step {step}, Loss: {loss.numpy()}")
                log_tensor_statistics(loss, "loss", total_step)

            # Log detailed statistics every 50 steps
            if step % 50 == 0:
                log_tensor_statistics(predictions, "predictions", total_step)

        # Break if NaN detected
        if nan_monitor.nan_count > 0:
            print(f"💔 Training stopped due to {nan_monitor.nan_count} NaN detections")
            nan_monitor.save_nan_report(os.path.join(model_save_dir, "nan_report.json"))
            break

        # After each epoch, check if the average loss is the best so far
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} Average Loss: {epoch_loss}")

        # Optionally, evaluate on a validation dataset here and use that loss instead
        if epoch_loss < best_loss:
            print(f"New best loss {epoch_loss}, saving model.")
            best_loss = epoch_loss
            stroke_model.save(model_save_path)  # Save the model
        save_epochs_info(epoch + 1, epochs_info_path)

    # Save final NaN report
    if nan_monitor.nan_count > 0:
        nan_monitor.save_nan_report(os.path.join(model_save_dir, "nan_report.json"))
        print(f"📊 Final NaN report: {nan_monitor.nan_count} NaN occurrences detected")
    else:
        print("✅ Training completed successfully with no NaN detections")

    print("Training completed.")
