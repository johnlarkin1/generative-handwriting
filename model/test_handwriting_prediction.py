import os
from common import (
    create_gif,
    create_sequences,
    create_subsequence_batches,
    generate_loop_da_loop_data,
    generate_zig_zag_data,
    mdn_to_heatmap,
    plot_original_strokes_from_xml,
    plot_strokes_from_dx_dy,
    prepare_data_for_sequential_prediction,
    prepare_data_for_sequential_prediction_with_eos,
    show_initial_data,
)
import imageio
from model.lstm_peephole_cell import LSTMCellWithPeepholes
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from loader import HandwritingDataLoader
from model.handwriting_mdn import MDNLayer, mdn_loss
from model.handwriting_prediction import DeepHandwritingPredictionModel
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import json
import hashlib

from constants import LEARNING_RATE, TEST_NUM_MIXTURES, TEST_NUM_EPOCHS, TEST_BATCH_SIZE, TEST_SEQUENCE_LENGTH

model_save_dir = (
    "/Users/johnlarkin/Documents/coding/generative-handwriting/src/saved_models/handwriting_prediction_simple/"
)
model_save_path = os.path.join(model_save_dir, "best_model.keras")
epochs_info_path = os.path.join(model_save_dir, "epochs_info.json")


def plot_predictions_through_sequence(
    model, model_name, training_data, full_reconstructed_data, num_components, sequence_length
):
    for seq_index in range(len(training_data)):
        sub_sequence = training_data[seq_index, :, :]
        predicted_output = model.predict(np.expand_dims(sub_sequence, axis=0))
        last_step_output = predicted_output[:, -1, :]

        start_idx = seq_index
        end_idx = start_idx + sequence_length
        absolute_subsequence = full_reconstructed_data[start_idx:end_idx, :2]
        last_known_point = absolute_subsequence[-1, :]

        # Setting up the grid for the heatmap based on the last known point
        grid_x = np.linspace(np.min(absolute_subsequence[:, 0]), np.max(absolute_subsequence[:, 0]), 100)
        grid_y = np.linspace(np.min(absolute_subsequence[:, 1]), np.max(absolute_subsequence[:, 1]), 100)

        # Generate the heatmap data from the model's predicted output
        pdf_total = mdn_to_heatmap(last_step_output, num_components, grid_x, grid_y, last_known_point)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(
            full_reconstructed_data[:, 0], full_reconstructed_data[:, 1], alpha=0.2, label="Full Dataset", color="grey"
        )
        plt.scatter(absolute_subsequence[:, 0], absolute_subsequence[:, 1], color="blue", label="Subsequence")
        plt.contourf(grid_x, grid_y, pdf_total, levels=50, cmap="viridis", alpha=0.5)
        plt.colorbar(label="Probability Density")
        plt.legend()

        plt.xlim([np.min(absolute_subsequence[:, 0]), np.max(absolute_subsequence[:, 0])])
        plt.ylim([np.min(absolute_subsequence[:, 1]), np.max(absolute_subsequence[:, 1])])
        plt.axis("equal")
        plt.title(f"Predictive Heatmap for Subsequence {start_idx} to {end_idx}")
        plt.savefig(f"{model_name}_heatmap_seq_{seq_index}.png")
        plt.close()


class HandwritingVisualizeCallback(Callback):
    def __init__(self, input_sequence, real_sequence, full_dataset, model_name, frequency=10):
        super(HandwritingVisualizeCallback, self).__init__()
        self.input_sequence = input_sequence
        self.real_sequence = real_sequence
        self.full_dataset = full_dataset
        self.model_name = model_name
        self.frequency = frequency
        # Prepare a directory to save visualizations
        self.image_dir = f"handwriting_visualizations/{self.model_name}"
        os.makedirs(self.image_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            start_idx = 4
            end_idx = start_idx + sequence_length
            sub_sequence = self.input_sequence[0, start_idx:end_idx, :]
            input_reshaped = np.expand_dims(sub_sequence, axis=0)
            predicted_output = self.model.predict(input_reshaped)
            print(f"Input shape: {input_reshaped.shape}")
            print(f"Predicted output shape: {predicted_output.shape}")

            last_step_output = predicted_output[:, end_idx, :]

            # Determine the range of the subsequence to establish plotting limits

            absolute_subsequence = self.full_dataset[start_idx:end_idx, :2]
            last_known_point = absolute_subsequence[-1, :]
            x_min = np.min(self.full_dataset[max(start_idx - 50, 0) :, 0])
            y_min, y_max = np.min(self.full_dataset[:, 1]), np.max(self.full_dataset[:, 1]) + 10

            # Update x_max to the latest point in the subsequence
            if end_idx < self.full_dataset.shape[0]:
                actual_next_point = self.full_dataset[end_idx, :2]
                x_max = actual_next_point[0] + 10
            else:  # If at the end of the dataset, use the last known point
                x_max = last_known_point[0] + 10

            # Set the grid for the heatmap
            grid_x = np.linspace(x_min, x_max, 100)
            grid_y = np.linspace(y_min, y_max, 100)

            # Generate the heatmap data from the model's predicted output
            pdf_total = mdn_to_heatmap(last_step_output, TEST_NUM_MIXTURES, grid_x, grid_y, last_known_point)

            # Plotting
            print(f"Plotting for sequence {start_idx}")
            plt.figure(figsize=(10, 6))
            plt.axis("equal")
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            plt.scatter(
                self.full_dataset[:, 0],
                self.full_dataset[:, 1],
                alpha=0.2,
                label="Full Dataset",
                color="grey",
            )
            plt.scatter(absolute_subsequence[:, 0], absolute_subsequence[:, 1], color="blue", label="Subsequence")
            if end_idx < self.full_dataset.shape[0]:
                plt.scatter(actual_next_point[0], actual_next_point[1], color="red", label="Actual Next Point")
            plt.contourf(grid_x, grid_y, pdf_total, levels=50, cmap="viridis", alpha=0.5)
            plt.colorbar(label="Probability Density")
            plt.legend()
            plt.title(f"Predictive Heatmap for Subsequence {start_idx} to {end_idx}")
            filename = f"{self.model_name}_heatmap_epoch_{epoch}.png"
            filepath = os.path.join(self.image_dir, filename)
            plt.savefig(filepath)
            plt.close()

    # def on_epoch_end(self, epoch, logs=None):
    #     if epoch % self.frequency == 0:
    #         target_begin_sequence = 4
    #         sub_sequence = self.input_sequence[target_begin_sequence, :, :]
    #         predicted_output = self.model.predict(np.expand_dims(sub_sequence, axis=0))
    #         last_step_output = predicted_output[:, -1, :]

    #         # Determine the range of the subsequence to establish plotting limits
    #         start_idx = target_begin_sequence
    #         end_idx = start_idx + sequence_length
    #         absolute_subsequence = self.full_dataset[start_idx:end_idx, :2]
    #         last_known_point = absolute_subsequence[-1, :]
    #         x_min = np.min(self.full_dataset[max(start_idx - 50, 0) :, 0])
    #         y_min, y_max = np.min(self.full_dataset[:, 1]), np.max(self.full_dataset[:, 1]) + 10

    #         # Update x_max to the latest point in the subsequence
    #         if end_idx < self.full_dataset.shape[0]:
    #             actual_next_point = self.full_dataset[end_idx, :2]
    #             x_max = actual_next_point[0] + 10
    #         else:  # If at the end of the dataset, use the last known point
    #             x_max = last_known_point[0] + 10

    #         # Set the grid for the heatmap
    #         grid_x = np.linspace(x_min, x_max, 100)
    #         grid_y = np.linspace(y_min, y_max, 100)

    #         # Generate the heatmap data from the model's predicted output
    #         pdf_total = mdn_to_heatmap(last_step_output, TEST_NUM_MIXTURES, grid_x, grid_y, last_known_point)

    #         # Plotting
    #         print(f"Plotting for sequence {target_begin_sequence}")
    #         plt.figure(figsize=(10, 6))
    #         plt.axis("equal")
    #         plt.xlim([x_min, x_max])
    #         plt.ylim([y_min, y_max])
    #         plt.scatter(
    #             self.full_dataset[:, 0],
    #             self.full_dataset[:, 1],
    #             alpha=0.2,
    #             label="Full Dataset",
    #             color="grey",
    #         )
    #         plt.scatter(absolute_subsequence[:, 0], absolute_subsequence[:, 1], color="blue", label="Subsequence")
    #         if end_idx < self.full_dataset.shape[0]:
    #             plt.scatter(actual_next_point[0], actual_next_point[1], color="red", label="Actual Next Point")
    #         plt.contourf(grid_x, grid_y, pdf_total, levels=50, cmap="viridis", alpha=0.5)
    #         plt.colorbar(label="Probability Density")
    #         plt.legend()
    #         plt.title(f"Predictive Heatmap for Subsequence {start_idx} to {end_idx}")
    #         filename = f"{self.model_name}_heatmap_epoch_{epoch}.png"
    #         filepath = os.path.join(self.image_dir, filename)
    #         plt.savefig(filepath)
    #         plt.close()


@tf.keras.utils.register_keras_serializable()
def model_mdn_loss(actual, outputs):
    return mdn_loss(actual, outputs, num_mixture_components)


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
                    "MDNLayer": MDNLayer,
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
desired_epochs = 2000
strokes, stroke_lengths = HandwritingDataLoader().load_individual_stroke_data("a01/a01-000/a01-000u-01.xml")
# plot_original_strokes_from_xml("a01/a01-000/a01-000u-01.xml")
reconstructed_data = plot_strokes_from_dx_dy(strokes, show_image=False)

x_stroke, y_stroke = prepare_data_for_sequential_prediction(strokes)
x_train_stroke = np.reshape(x_stroke, (1, x_stroke.shape[0], x_stroke.shape[1]))
y_train_stroke = np.reshape(y_stroke, (1, y_stroke.shape[0], y_stroke.shape[1]))
x_train_stroke = x_train_stroke[:,]
# x_train_stroke, y_train_stroke = create_subsequence_batches(x_stroke, y_stroke, sequence_length)
# x_train_stroke, y_train_stroke = x_train_stroke[4, :, :], y_train_stroke[4, :, :]
# x_train_stroke = np.reshape(x_train_stroke, (1, x_train_stroke.shape[0], x_train_stroke.shape[1]))
# y_train_stroke = np.reshape(y_train_stroke, (1, y_train_stroke.shape[0], y_train_stroke.shape[1]))


# x_batches, y_batches = create_subsequence_batches(x_stroke, y_stroke, sequence_length)
x_batches, y_batches = x_train_stroke, y_train_stroke
num_mixture_components = 20
learning_rate = LEARNING_RATE

model_hash = get_model_hash(num_mixture_components, learning_rate)
stroke_model, model_loaded = load_model_if_exists(model_save_path, model_hash)

if not model_loaded:
    print("No suitable saved model found or model has changed, initializing a new one...")
    stroke_model = DeepHandwritingPredictionModel(num_mixture_components=num_mixture_components)
    stroke_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=model_mdn_loss,
    )

last_trained_epoch = load_epochs_info(epochs_info_path)
if last_trained_epoch < desired_epochs:
    checkpoint_callback = ModelCheckpoint(model_save_path, save_best_only=True, monitor="loss", mode="min")
    visualize_callback = HandwritingVisualizeCallback(
        input_sequence=x_batches,
        real_sequence=y_batches,
        full_dataset=reconstructed_data,
        model_name="handwriting_simple",
        frequency=5,
    )

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
    model, model_name, training_data, full_reconstructed_data, num_components, sequence_length, save_dir
):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Perform a single prediction for the entire input sequence
    full_sequence_prediction = model.predict(training_data)

    # Set a buffer for the y-axis
    buffer = 5

    # Iterate through the desired sequence indices
    for seq_index in range(sequence_length, len(full_reconstructed_data) - sequence_length):
        # Extract the MDN parameters for the current subsequence index
        last_step_output = full_sequence_prediction[:, seq_index - 1, :]  # seq_index-1 because it's zero-indexed

        # Establish the range of x and y to plot, including a buffer
        x_min, x_max = (
            full_reconstructed_data[seq_index - sequence_length : seq_index, 0].min() - buffer,
            full_reconstructed_data[seq_index - sequence_length : seq_index + 10, 0].max() + buffer,
        )
        y_min, y_max = (
            full_reconstructed_data[seq_index - sequence_length : seq_index, 1].min() - buffer,
            full_reconstructed_data[seq_index - sequence_length : seq_index, 1].max() + buffer,
        )
        last_known_point = full_reconstructed_data[seq_index - 1, :2]

        grid_x = np.linspace(x_min, x_max, 50)
        grid_y = np.linspace(y_min, y_max, 50)

        # Generate the heatmap data from the model's predicted output
        pdf_total = mdn_to_heatmap(last_step_output, num_components, grid_x, grid_y, last_known_point)

        # Plotting
        print(f"Plotting sequence index {seq_index}")
        plt.figure(figsize=(10, 6))
        plt.axis("equal")
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.scatter(
            full_reconstructed_data[:, 0], full_reconstructed_data[:, 1], alpha=0.2, label="Full Dataset", color="grey"
        )
        plt.scatter(
            full_reconstructed_data[seq_index - sequence_length : seq_index, 0],
            full_reconstructed_data[seq_index - sequence_length : seq_index, 1],
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
            full_reconstructed_data[seq_index, 0],
            full_reconstructed_data[seq_index, 1],
            color="red",
            label="Actual Next Point",
            zorder=100,
        )
        plt.contourf(grid_x, grid_y, pdf_total, levels=10, cmap="viridis", alpha=0.5)
        plt.colorbar(label="Probability Density")
        plt.legend()
        plt.title(f"Predictive Heatmap for Sequence Index {seq_index}")
        filename = f"{model_name}_heatmap_seq_{seq_index}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()


# # Call the function with appropriate parameters
plot_predictions_through_sequence(
    stroke_model,
    "single_stroke_pass_through",
    x_batches,
    reconstructed_data,
    TEST_NUM_MIXTURES,
    sequence_length,
    "handwriting_visualizations/single_stroke/",
)

create_gif(
    "handwriting_visualizations/single_stroke/",
    "single_stroke_pass_through_heatmap_seq_*",
    "single_stroke_pass_through",
)
