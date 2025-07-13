import datetime
import os

import numpy as np
import tensorflow as tf
from alphabet import ALPHABET_SIZE
from callbacks import ExtendedModelCheckpoint
from common import (
    plot_strokes_from_dx_dy,
    prepare_data_for_sequential_prediction,
)
from constants import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
)
from loader import HandwritingDataLoader
from model.handwriting_models import (
    DeepHandwritingSynthesisModel,
)
from model.mixture_density_network import mdn_loss

model_save_dir = (
    "/Users/johnlarkin/Documents/coding/generative-handwriting/src/saved_models/"
    "handwriting_synthesis_single_batch_subset/"
)
model_pred_dir = (
    "/Users/johnlarkin/Documents/coding/generative-handwriting/"
    "handwriting_visualizations/single_stroke_synth_single_batch_subset/"
)
model_save_path = os.path.join(model_save_dir, "best_model.keras")
epochs_info_path = os.path.join(model_save_dir, "epochs_info.json")


@tf.keras.utils.register_keras_serializable()
def model_mdn_loss(actual, outputs, stroke_lengths, num_mixture_components):
    return mdn_loss(actual, outputs, stroke_lengths, num_mixture_components)


desired_epochs = 10_000
(
    strokes,
    stroke_lengths,
    chars,
    char_len,
) = HandwritingDataLoader().load_individual_stroke_and_c_data("a01/a01-000/a01-000u-01.xml")
print("strokes shape:", strokes.shape)
print("stroke_lengths:", stroke_lengths)
print("chars:", chars)

print("char_len:", char_len)

# plot_original_strokes_from_xml("a01/a01-000/a01-000u-01.xml")
reconstructed_data = plot_strokes_from_dx_dy(strokes, show_image=False)
print("reconstructed_data.shape", reconstructed_data.shape)
stroke_lengths = np.array([stroke_lengths])
char_len = np.array([char_len])

x_stroke, y_stroke = prepare_data_for_sequential_prediction(strokes)
batch_size = BATCH_SIZE

x_stroke = np.reshape(x_stroke, (1, x_stroke.shape[0], x_stroke.shape[1]))
y_stroke = np.reshape(y_stroke, (1, y_stroke.shape[0], y_stroke.shape[1]))
stroke_lengths = np.reshape(stroke_lengths, (1, stroke_lengths.shape[0]))
chars = np.reshape(chars, (1, chars.shape[0]))
chars = tf.cast(chars, tf.int8)

dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"input_strokes": x_stroke, "input_chars": chars, "input_char_lens": char_len},
        y_stroke,
    )
).batch(BATCH_SIZE)

x_batches, y_batches = x_stroke[0, 0:100, :], y_stroke[0, 0:100, :]
num_mixture_components = 1
initial_learning_rate = LEARNING_RATE

log_dir = "logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch="500,520"),
]

stroke_model = DeepHandwritingSynthesisModel(
    units=400,
    num_layers=3,
    num_mixture_components=NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
    num_chars=ALPHABET_SIZE,
    num_attention_gaussians=10,
)

stroke_model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
    loss=lambda y_true, y_pred: model_mdn_loss(
        y_true, y_pred, stroke_lengths, NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS
    ),
)


# Callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch="500,520"),
    ExtendedModelCheckpoint("single_batch_synthesis_opt"),
]

# Fit the model
history = stroke_model.fit(dataset, epochs=desired_epochs, callbacks=callbacks)

# Save the final model
stroke_model.save(model_save_path)
