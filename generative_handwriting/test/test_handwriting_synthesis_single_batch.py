import datetime
import os

import numpy as np
import tensorflow as tf
from alphabet import ALPHABET_SIZE
from common import (
    plot_strokes_from_dx_dy,
    prepare_data_for_sequential_prediction,
    print_model_parameters,
)
from constants import (
    BATCH_SIZE,
    GRADIENT_CLIP_VALUE,
    LEARNING_RATE,
)
from loader import HandwritingDataLoader
from model.attention_mechanism import AttentionMechanism
from model.attention_rnn_cell import AttentionRNNCell
from model.handwriting_models import (
    DeepHandwritingSynthesisModel,
)
from model.lstm_peephole_cell import LSTMPeepholeCell
from model.mixture_density_network import MixtureDensityLayer, mdn_loss
from model_io import load_epochs_info, load_model_if_exists, save_epochs_info
from plotting import generate_and_plot_heatmap_full

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

dataset = tf.data.Dataset.from_tensor_slices((x_stroke, y_stroke, stroke_lengths, chars, char_len))

x_batches, y_batches = x_stroke[0, 0:100, :], y_stroke[0, 0:100, :]
num_mixture_components = 1
initial_learning_rate = LEARNING_RATE

callbacks = []

log_dir = "logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch="500,520")
callbacks.append(tensorboard_callback)

stroke_model, is_success = load_model_if_exists(
    model_save_path,
    custom_objects={
        "model_mdn_loss": model_mdn_loss,
        "DeepHandwritingSynthesisModel": DeepHandwritingSynthesisModel,
        "AttentionLayer": LSTMPeepholeCell,
        "AttentionMechanism": AttentionMechanism,
        "AttentionRNNCell": AttentionRNNCell,
        "MixtureDensityLayer": MixtureDensityLayer,
    },
)
last_trained_epoch = load_epochs_info(epochs_info_path)
best_loss = np.inf
if not is_success or last_trained_epoch < desired_epochs:
    if not is_success:
        # If model load was not successful, define the model
        stroke_model = DeepHandwritingSynthesisModel(
            units=400,
            num_layers=3,
            num_mixture_components=num_mixture_components,
            num_chars=ALPHABET_SIZE,
            num_attention_gaussians=10,
        )
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    y_stroke = tf.cast(y_stroke, tf.float32)

    for epoch in range(last_trained_epoch, desired_epochs):
        for callback in callbacks:
            callback.on_epoch_begin(epoch)

        epoch_losses = []
        chars = tf.cast(chars, tf.int8)
        with tf.GradientTape() as tape:
            predictions = stroke_model(
                x_stroke,
                chars,
                char_len,
                training=True,
            )
            predictions = tf.cast(predictions, tf.float32)
            loss = model_mdn_loss(y_stroke, predictions, stroke_lengths, num_mixture_components)
        gradients = tape.gradient(loss, stroke_model.trainable_variables)
        clipped_gradients = [
            (tf.clip_by_value(grad, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE), var)
            for grad, var in zip(gradients, stroke_model.trainable_variables, strict=False)
        ]
        optimizer.apply_gradients(clipped_gradients)

        if epoch == 0:
            print_model_parameters(stroke_model)

        epoch_losses.append(loss.numpy())
        epoch_loss = np.mean(epoch_losses)
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1} Average Loss: {epoch_loss}")

        # If this epoch's loss is better (lower) than the best seen so far, update the best loss and save the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"[[ Epoch {epoch + 1} ]]: New best model found! Loss {best_loss}. Saving model...")
            stroke_model.save(model_save_path)
            save_epochs_info(desired_epochs, epochs_info_path)

        for callback in callbacks:
            callback.on_epoch_end(epoch)

else:
    print("Model already exists")

generate_and_plot_heatmap_full(
    stroke_model,
    x_stroke,
    chars,
    char_len,
    reconstructed_data,
    num_mixture_components,
)
