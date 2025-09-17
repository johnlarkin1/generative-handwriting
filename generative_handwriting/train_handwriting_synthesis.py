import datetime
import os

import numpy as np
import tensorflow as tf
from alphabet import ALPHABET_SIZE
from callbacks import ExtendedModelCheckpoint, ModelCheckpointWithPeriod, PrintModelParametersCallback
from constants import (
    BATCH_SIZE,
    GRADIENT_CLIP_VALUE,
    LEARNING_RATE,
    NUM_ATTENTION_GAUSSIAN_COMPONENTS,
    NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
)
from loader import HandwritingDataLoader
from model.handwriting_models import DeepHandwritingSynthesisModel

# We want this to ensure CPU <-> GPU compatibility
tf.keras.mixed_precision.set_global_policy("float32")

# Path logic for saved files
curr_directory = os.path.dirname(os.path.realpath(__file__))
model_name = "handwriting_synthesis"
model_save_dir = f"{curr_directory}/saved/models/{model_name}/"
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
            global_clipnorm=GRADIENT_CLIP_VALUE,
        ),
        loss=None,  # i don't think we need this because we have a custom loss function
        run_eagerly=True,
    )

    callbacks = [
        # Disabled for XLA compatibility
        # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch="500,520"),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0),  # Basic logging only for XLA compatibility
        ModelCheckpointWithPeriod(model_name, period=200),
        ExtendedModelCheckpoint(model_name),
        PrintModelParametersCallback(),
    ]

    history = stroke_model.fit(dataset, epochs=desired_epochs, callbacks=callbacks)
    stroke_model.save(model_save_path)
    print("Training completed.")
