import datetime
import os

import numpy as np
import tensorflow as tf
from constants import (
    BATCH_SIZE,
    GRADIENT_CLIP_VALUE,
    LEARNING_RATE,
    NUM_ATTENTION_GAUSSIAN_COMPONENTS,
    NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
)

from generative_handwriting.alphabet import ALPHABET_SIZE
from generative_handwriting.callbacks import ExtendedModelCheckpoint, PrintModelParametersCallback
from generative_handwriting.loader import HandwritingDataLoader
from generative_handwriting.model.handwriting_models import DeepHandwritingSynthesisModel
from generative_handwriting.model.mixture_density_network import mdn_loss

# We want this to ensure CPU <-> GPU compatibility
tf.keras.mixed_precision.set_global_policy("float32")

# Path logic for saved files
curr_directory = os.path.dirname(os.path.realpath(__file__))
model_name = "full_handwriting_synthesis"
model_save_dir = f"{curr_directory}/saved_models/{model_name}/"
start_day = datetime.datetime.now().strftime("%Y%m%d")
log_dir = f"{curr_directory}/saved/logs/{model_name}/profile/{start_day}"
debugging_dir = f"{curr_directory}/saved/logs/{model_name}/debug/{start_day}"
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(debugging_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, "best_model.keras")
checkpoint_model_filepath = os.path.join(model_save_dir, "model_{epoch:02d}_{val_loss:.2f}.keras")
model_save_path_final = os.path.join(model_save_dir, "best_model_final.keras")
epochs_info_path = os.path.join(model_save_dir, "epochs_info.json")

# Hyper parameters
num_mixture_components = NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS
num_attention_gaussians = NUM_ATTENTION_GAUSSIAN_COMPONENTS

# Training parameters
desired_epochs = 10_000
model_chkpt_period = 200


# Get the data
data_loader = HandwritingDataLoader()
data_loader.prepare_data()
combined_train_strokes, combined_train_lengths = (
    data_loader.combined_train_strokes,
    data_loader.combined_train_stroke_lengths,
)
combined_train_trans, combined_trans_lengths = (
    data_loader.combined_train_transcriptions,
    data_loader.combined_train_transcription_lengths,
)

# Get validation data for Graves metric
validation_strokes, validation_lengths = (
    data_loader.validation_strokes,
    data_loader.validation_stroke_lengths,
)


def compute_graves_log_loss_synthesis(model, stroke_data, stroke_lengths, char_data, char_lengths, num_mixture_components, batch_size=32):
    """Compute mean negative log-likelihood (NLL) per timestep in nats.

    This aligns with the training objective and avoids confusing sign conventions.
    """
    total_nll = 0.0
    total_timesteps = 0
    total_sse = 0.0  # Sum squared error

    num_samples = len(stroke_data)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_strokes = stroke_data[start_idx:end_idx]
        batch_stroke_lens = stroke_lengths[start_idx:end_idx]
        batch_chars = char_data[start_idx:end_idx]
        batch_char_lens = char_lengths[start_idx:end_idx]

        # Prepare targets (next stroke predictions)
        batch_targets = np.zeros_like(batch_strokes)
        batch_targets[:, :-1, :] = batch_strokes[:, 1:, :]
        batch_target_lens = np.maximum(batch_stroke_lens - 1, 1)

        # Prepare model inputs
        inputs = {
            "input_strokes": batch_strokes,
            "input_stroke_lens": batch_stroke_lens,
            "target_stroke_lens": batch_target_lens,
            "input_chars": batch_chars,
            "input_char_lens": batch_char_lens,
        }

        # Get predictions
        predictions = model(inputs, training=False)

        # Compute NLL for this batch
        batch_nll = mdn_loss(batch_targets, predictions, batch_target_lens, num_mixture_components)

        # Accumulate totals
        batch_timesteps = np.sum(batch_target_lens)
        total_nll += batch_nll.numpy() * batch_timesteps
        total_timesteps += batch_timesteps

        # Compute SSE (sum squared error) for comparison
        # Extract means from predictions
        mu_x = predictions[:, :, num_mixture_components:2*num_mixture_components]
        mu_y = predictions[:, :, 2*num_mixture_components:3*num_mixture_components]
        pi = predictions[:, :, :num_mixture_components]

        # Get most likely component for each timestep
        best_components = tf.cast(tf.argmax(pi, axis=-1), tf.int32)
        batch_size_curr = tf.shape(mu_x)[0]
        seq_len = tf.shape(mu_x)[1]

        # Gather the means for the most likely components
        indices = tf.stack([tf.range(batch_size_curr)[:, None] * tf.ones([1, seq_len], dtype=tf.int32),
                           tf.range(seq_len)[None, :] * tf.ones([batch_size_curr, 1], dtype=tf.int32),
                           best_components], axis=-1)

        pred_x = tf.gather_nd(mu_x, indices)
        pred_y = tf.gather_nd(mu_y, indices)

        # Compute SSE
        actual_x = batch_targets[:, :, 0]
        actual_y = batch_targets[:, :, 1]

        # Apply mask for valid timesteps
        mask = tf.sequence_mask(batch_target_lens, maxlen=tf.shape(batch_targets)[1], dtype=tf.float32)
        # Guard against NaNs in predictions (should be rare after stability fixes)
        pred_x = tf.where(tf.math.is_finite(pred_x), pred_x, tf.zeros_like(pred_x))
        pred_y = tf.where(tf.math.is_finite(pred_y), pred_y, tf.zeros_like(pred_y))
        squared_error = ((pred_x - actual_x) ** 2 + (pred_y - actual_y) ** 2) * mask
        total_sse += tf.reduce_sum(squared_error).numpy()

    # Compute averages
    graves_log_loss = total_nll / max(total_timesteps, 1)  # Mean NLL in nats
    mean_sse = total_sse / max(total_timesteps, 1)

    return graves_log_loss, mean_sse

if __name__ == "__main__":
    # Preparing the input and target data for training
    x_train = combined_train_strokes
    x_train_len = combined_train_lengths
    y_train = np.zeros_like(x_train)
    y_train[:, :-1, :] = x_train[:, 1:, :]
    y_train_len = x_train_len - 1

    # dbeugging check
    print(f"Original input lengths - min: {x_train_len.min()}, max: {x_train_len.max()}")
    print(f"Target lengths - min: {y_train_len.min()}, max: {y_train_len.max()}")
    print(f"Number of sequences with target_len <= 0: {(y_train_len <= 0).sum()}")
    y_train_len = np.maximum(y_train_len, 1)

    char_seq = combined_train_trans
    char_seq_len = combined_trans_lengths

    best_loss = float("inf")
    batch_size = BATCH_SIZE
    print(f"Debugging info: {debugging_dir}")
    # tf.debugging.experimental.enable_dump_debug_info(  # Disabled for XLA compatibility
    #     debugging_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1
    # )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "input_strokes": x_train,
                "input_stroke_lens": x_train_len,
                "target_stroke_lens": y_train_len,
                "input_chars": char_seq,
                "input_char_lens": char_seq_len,
            },
            y_train,
        )
    ).batch(BATCH_SIZE)

    stroke_model = DeepHandwritingSynthesisModel(
        units=400,
        num_layers=3,
        num_mixture_components=num_mixture_components,
        num_chars=ALPHABET_SIZE,
        num_attention_gaussians=num_attention_gaussians,
        gradient_clip_value=GRADIENT_CLIP_VALUE,
    )
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=50_000,
        decay_rate=0.96,
        staircase=True,
    )
    stroke_model.compile(
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate_schedule,
            global_clipnorm=GRADIENT_CLIP_VALUE,  # Use optimizer's global clipping instead of manual
        ),
        loss=None,  # Correct when overriding train_step
        run_eagerly=False,  # Faster + works better with XLA; set True only when debugging
        jit_compile=True,  # Enable XLA JIT compilation for better performance
    )

    # Add custom callback for validation metrics
    class GravesMetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_strokes, val_stroke_lens, val_chars, val_char_lens, num_components):
            super().__init__()
            self.val_strokes = val_strokes
            self.val_stroke_lens = val_stroke_lens
            self.val_chars = val_chars
            self.val_char_lens = val_char_lens
            self.num_components = num_components
            self.best_val_loss = float('inf')

        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:  # Evaluate every 5 epochs to save time
                print(f"\nComputing validation metrics for epoch {epoch + 1}...")
                val_log_loss, val_sse = compute_graves_log_loss_synthesis(
                    self.model, self.val_strokes, self.val_stroke_lens,
                    self.val_chars, self.val_char_lens, self.num_components
                )
                print(f"Validation NLL: {val_log_loss:.1f} nats")
                print(f"Validation SSE: {val_sse:.4f}")

                # Save model if validation loss improved
                if val_log_loss < self.best_val_loss:
                    print(f"New best validation NLL {val_log_loss:.1f}, saving model.")
                    self.best_val_loss = val_log_loss
                    self.model.save(model_save_path)

    # Prepare validation data for synthesis model
    val_chars = data_loader.validation_transcriptions
    val_char_lens = data_loader.validation_transcription_lengths

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.8, patience=10, verbose=1, min_lr=1e-6),
        ExtendedModelCheckpoint(model_name),
        PrintModelParametersCallback(),
        GravesMetricsCallback(validation_strokes, validation_lengths, val_chars, val_char_lens, num_mixture_components),
    ]

    history = stroke_model.fit(dataset, epochs=desired_epochs, callbacks=callbacks)
    stroke_model.save(model_save_path_final)
    print("Training completed.")
