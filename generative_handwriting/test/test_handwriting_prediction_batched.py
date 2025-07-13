import hashlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from common import (
    create_gif,
    create_subsequence_batches,
    mdn_to_heatmap,
    plot_strokes_from_dx_dy,
    prepare_data_for_sequential_prediction,
)
from constants import LEARNING_RATE, TEST_BATCH_SIZE, TEST_NUM_MIXTURES
from loader import HandwritingDataLoader
from model.handwriting_models import DeepHandwritingPredictionModel
from model.mixture_density_network import MixtureDensityLayer, mdn_loss
from tensorflow.keras.callbacks import ModelCheckpoint

model_save_dir = (
    "/Users/johnlarkin/Documents/coding/generative-handwriting/src/saved_models/"
    "handwriting_prediction_simple_single_stroke/"
)
model_save_path = os.path.join(model_save_dir, "best_model.keras")
epochs_info_path = os.path.join(model_save_dir, "epochs_info.json")


@tf.keras.utils.register_keras_serializable()
def model_mdn_loss(actual, outputs):
    return mdn_loss(actual, outputs, None, num_mixture_components)


def get_model_hash(num_mixture_components, learning_rate):
    hash_str = f"DeepHandwritingPredictionModel_mixtures={num_mixture_components}_lr={learning_rate}"
    return hashlib.sha256(hash_str.encode()).hexdigest()


def load_model_if_exists(model_save_path, expected_hash):
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


sequence_length = 50
desired_epochs = 1000
strokes, stroke_lengths = HandwritingDataLoader().load_individual_stroke_data("a01/a01-000/a01-000u-01.xml")
# plot_original_strokes_from_xml("a01/a01-000/a01-000u-01.xml")
reconstructed_data = plot_strokes_from_dx_dy(strokes, show_image=False)

x_stroke, y_stroke = prepare_data_for_sequential_prediction(strokes)
x_train_stroke, y_train_stroke = create_subsequence_batches(x_stroke, y_stroke, sequence_length)
x_batches, y_batches = x_train_stroke, y_train_stroke

num_mixture_components = TEST_NUM_MIXTURES
learning_rate = LEARNING_RATE

model_hash = get_model_hash(num_mixture_components, learning_rate)
stroke_model, model_loaded = load_model_if_exists(model_save_path, model_hash)

if not model_loaded:
    print("No suitable saved model found or model has changed, initializing a new one...")
    stroke_model = DeepHandwritingPredictionModel(num_mixture_components=num_mixture_components, stateful=True)
    stroke_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=model_mdn_loss,
    )

last_trained_epoch = load_epochs_info(epochs_info_path)
if last_trained_epoch < desired_epochs or True:
    checkpoint_callback = ModelCheckpoint(model_save_path, save_best_only=True, monitor="loss", mode="min")
    stroke_model.fit(
        x_batches,
        y_batches,
        epochs=desired_epochs,
        initial_epoch=last_trained_epoch,
        batch_size=TEST_BATCH_SIZE,
        verbose=1,
        callbacks=[checkpoint_callback],
    )
    save_epochs_info(desired_epochs, epochs_info_path)

else:
    print("Model already trained to the desired epochs. Skipping training.")


def plot_predictions_through_sequence(
    model,
    model_name,
    training_data,
    full_reconstructed_data,
    num_components,
    sequence_length,
    save_dir,
):
    predicted_output = model.predict(training_data)

    mu1 = predicted_output[:, :, num_components : num_components * 2]  # Means for x
    mu2 = predicted_output[:, :, num_components * 2 : num_components * 3]  # Means for y
    max_mu1 = np.max(mu1)
    max_mu2 = np.max(mu2)
    print("max_mu1:", max_mu1)
    print("max_mu2:", max_mu2)
    for batch_idx in range(training_data.shape[0]):
        # Basically iterating from 0 to 1150
        # get our input data in terms of fully reconstructed data
        input_sequence_full = full_reconstructed_data[batch_idx : batch_idx + sequence_length, :]
        last_known_point = input_sequence_full[-1]
        last_step_output = predicted_output[batch_idx, -1, :]
        last_step_output = np.expand_dims(last_step_output, axis=0)
        actual_next_point = full_reconstructed_data[batch_idx + sequence_length, :2]
        x_min, x_max = (
            input_sequence_full[:, 0].min() - 5,
            max(
                input_sequence_full[:, 0].max() + 5,
                actual_next_point[0] + 5,
            ),
        )
        y_min, y_max = (
            input_sequence_full[:, 1].min() - 5,
            max(
                input_sequence_full[:, 1].max() + 5,
                actual_next_point[1] + 5,
            ),
        )
        grid_x = np.linspace(x_min, x_max, 50)
        grid_y = np.linspace(y_min, y_max, 50)

        pdf_total = mdn_to_heatmap(last_step_output, num_components, grid_x, grid_y, last_known_point)

        plt.figure(figsize=(10, 6))
        plt.axis("equal")
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.scatter(
            full_reconstructed_data[:, 0],
            full_reconstructed_data[:, 1],
            alpha=0.2,
            label="Full Dataset",
            color="grey",
        )
        plt.scatter(
            input_sequence_full[:, 0],
            input_sequence_full[:, 1],
            color="blue",
            label="Subsequence",
        )
        plt.scatter(
            last_known_point[0],
            last_known_point[1],
            color="green",
            label="Last Subsequence Point",
            zorder=50,
        )
        plt.scatter(
            actual_next_point[0],
            actual_next_point[1],
            color="red",
            label="Actual Next Point",
            zorder=100,
        )
        plt.contourf(grid_x, grid_y, pdf_total, levels=10, cmap="viridis", alpha=0.5)
        plt.colorbar(label="Probability Density")
        plt.legend()
        plt.title(f"Predictive Heatmap for Batch Index {batch_idx}")
        filename = f"{model_name}_heatmap_seq_{batch_idx}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()


print("x_batches", x_batches.shape)
plot_predictions_through_sequence(
    stroke_model,
    "single_stroke_pass_through",
    x_batches,
    reconstructed_data,
    TEST_NUM_MIXTURES,
    sequence_length,
    "handwriting_visualizations/single_stroke_batched/",
)

create_gif(
    "handwriting_visualizations/single_stroke_batched/",
    "single_stroke_pass_through_heatmap_seq_*",
    "single_stroke_pass_through",
)
