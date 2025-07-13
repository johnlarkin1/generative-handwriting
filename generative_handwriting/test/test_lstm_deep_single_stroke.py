import os
from common import (
    create_gif,
    create_sequences,
    create_subsequence_batches,
    generate_loop_da_loop_data,
    generate_zig_zag_data,
    mdn_to_heatmap,
    plot_strokes_from_dx_dy,
    prepare_data_for_sequential_prediction,
    prepare_data_for_sequential_prediction_with_eos,
)
from loader import HandwritingDataLoader
from model.mixture_density_network import mdn_loss
from model.handwriting_models import (
    DeepHandwritingPredictionModel,
    SimpleHandwritingPredictionModel,
)
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

from constants import (
    ADAM_CLIP_NORM,
    LEARNING_RATE,
    NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
    NUM_LSTM_HIDDEN_LAYERS,
    TEST_NUM_EPOCHS,
    TEST_BATCH_SIZE,
    TEST_NUM_MIXTURES,
    TEST_SEQUENCE_LENGTH,
)


# We don't care about end of stroke data
def custom_loss(y_true, y_pred):
    mse_xy = tf.keras.losses.mean_squared_error(y_true[:, :, :2], y_pred[:, :, :2])
    mse_xy_reduced = tf.reduce_mean(mse_xy, axis=1)
    return tf.reduce_mean(mse_xy_reduced)


@tf.keras.utils.register_keras_serializable()
def model_mdn_loss(actual, outputs):
    return mdn_loss(actual, outputs, num_mixture_components)


zigzag_data = generate_zig_zag_data()
loop_data = generate_loop_da_loop_data()

# Prepare the data for sequential prediction
x_zigzag, y_zigzag = prepare_data_for_sequential_prediction_with_eos(zigzag_data)
x_loop, y_loop = prepare_data_for_sequential_prediction_with_eos(loop_data)

# Create sequences
x_train_zigzag, y_train_zigzag = create_sequences(x_zigzag, TEST_SEQUENCE_LENGTH)
x_train_loop, y_train_loop = create_sequences(x_loop, TEST_SEQUENCE_LENGTH)

strokes, stroke_lengths = HandwritingDataLoader().load_individual_stroke_data(
    "a01/a01-000/a01-000u-01.xml"
)
# plot_original_strokes_from_xml("a01/a01-000/a01-000u-01.xml")
reconstructed_data = plot_strokes_from_dx_dy(strokes, show_image=False)

x_stroke, y_stroke = prepare_data_for_sequential_prediction(strokes)
# x_stroke = np.reshape(x_stroke, (1, x_stroke.shape[0], x_stroke.shape[1]))
# y_stroke = np.reshape(y_stroke, (1, y_stroke.shape[0], y_stroke.shape[1]))
sequence_length = 50
desired_epochs = 200

x_train_stroke, y_train_stroke = create_subsequence_batches(
    x_stroke, y_stroke, sequence_length
)
x_train_stroke, y_train_stroke = x_train_stroke[4, :, :], y_train_stroke[4, :, :]
x_train_stroke = np.reshape(
    x_train_stroke, (1, x_train_stroke.shape[0], x_train_stroke.shape[1])
)
y_train_stroke = np.reshape(
    y_train_stroke, (1, y_train_stroke.shape[0], y_train_stroke.shape[1])
)
# x_train_stroke, y_train_stroke = x_stroke, y_stroke
num_mixture_components = TEST_NUM_MIXTURES
learning_rate = LEARNING_RATE * 2

# Define the models
zigzag_model = SimpleHandwritingPredictionModel(
    units=NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
    num_layers=NUM_LSTM_HIDDEN_LAYERS,
    feature_size=3,
)
loop_model = SimpleHandwritingPredictionModel(
    units=NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
    num_layers=NUM_LSTM_HIDDEN_LAYERS,
    feature_size=3,
)
stroke_model = SimpleHandwritingPredictionModel(
    units=NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
    num_layers=NUM_LSTM_HIDDEN_LAYERS,
    feature_size=3,
)
deep_stroke_model = DeepHandwritingPredictionModel(num_mixture_components=5)

# Compile the models
zigzag_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, clipnorm=ADAM_CLIP_NORM
    ),
    loss=custom_loss,
)
loop_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, clipnorm=ADAM_CLIP_NORM
    ),
    loss=custom_loss,
)
stroke_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, clipnorm=ADAM_CLIP_NORM
    ),
    loss=custom_loss,
)
deep_stroke_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, clipnorm=ADAM_CLIP_NORM
    ),
    loss=model_mdn_loss,
)


class SimpleHandwritingVisualizeCallback(Callback):
    def __init__(
        self,
        input_sequence,
        real_sequence,
        model_name,
        frequency=10,
        selected_sequence=60,
    ):
        super(SimpleHandwritingVisualizeCallback, self).__init__()
        self.input_sequence = input_sequence  # Use all sequences for the background
        self.selected_input_sequence = input_sequence[
            selected_sequence : selected_sequence + 1
        ]  # Selected sequence for prediction
        self.real_sequence = real_sequence[selected_sequence : selected_sequence + 1]
        self.model_name = model_name
        self.frequency = frequency
        self.image_dir = f"handwriting_visualizations/{self.model_name}"
        os.makedirs(self.image_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            predicted_sequence = self.model.predict(self.selected_input_sequence)
            last_predicted_point = predicted_sequence[
                0, -1, :
            ]  # Last point in the predicted sequence
            length_of_pred = len(self.selected_input_sequence)

            plt.figure(figsize=(10, 6))
            for seq in self.input_sequence:
                plt.plot(
                    seq[:, 0], seq[:, 1], color="lightgray", linewidth=0.5, alpha=0.5
                )

            plt.plot(
                self.real_sequence[0, :, 0],
                self.real_sequence[0, :, 1],
                "r-",
                label="Actual Sequence",
                linewidth=2,
            )
            plt.plot(
                predicted_sequence[0, :, 0],
                predicted_sequence[0, :, 1],
                "b--",
                label="Predicted Sequence",
                linewidth=2,
            )
            plt.scatter(
                last_predicted_point[0],
                last_predicted_point[1],
                color="blue",
                s=50,
                zorder=5,
                label="Predicted Next Point",
            )

            plt.title(
                f"Model: {self.model_name} - Epoch: {epoch} (Pred Length: {length_of_pred})"
            )
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.legend()

            plt.savefig(
                os.path.join(
                    self.image_dir,
                    f"{self.model_name}_epoch_{epoch}_len{length_of_pred}.png",
                )
            )
            plt.close()  # Close the figure to free memory


class StrokeVisualizeCallback(Callback):
    def __init__(
        self,
        input_sequence,
        real_sequence,
        reconstructed_data,
        model_name,
        frequency=10,
        selected_sequence=4,
    ):
        super(StrokeVisualizeCallback, self).__init__()
        self.input_sequence = input_sequence  # Use all sequences for the background
        self.selected_input_sequence = input_sequence[
            :, selected_sequence : selected_sequence + 50, :
        ]  # Selected sequence for prediction
        self.real_sequence = real_sequence[
            :, selected_sequence : selected_sequence + 50, :
        ]
        self.reconstructed_data = reconstructed_data
        self.model_name = model_name
        self.frequency = frequency
        self.image_dir = f"handwriting_visualizations/{self.model_name}"
        os.makedirs(self.image_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            predicted_sequence = self.model.predict(self.selected_input_sequence)
            last_predicted_point = predicted_sequence[:, -1, :]
            start_idx = 4
            end_idx = start_idx + sequence_length
            absolute_subsequence = self.reconstructed_data[start_idx:end_idx, :2]
            last_known_point = absolute_subsequence[-1, :]
            x_min = np.min(self.reconstructed_data[max(start_idx - 50, 0) :, 0])
            y_min, y_max = (
                np.min(self.reconstructed_data[:, 1]),
                np.max(self.reconstructed_data[:, 1]) + 10,
            )

            # Update x_max to the latest point in the subsequence
            if end_idx < self.reconstructed_data.shape[0]:
                actual_next_point = self.reconstructed_data[end_idx, :2]
                x_max = actual_next_point[0] + 10
            else:  # If at the end of the dataset, use the last known point
                x_max = last_known_point[0] + 10

            grid_x = np.linspace(
                np.min(absolute_subsequence[:, 0]),
                np.max(absolute_subsequence[:, 0]),
                100,
            )
            grid_y = np.linspace(
                np.min(absolute_subsequence[:, 1]),
                np.max(absolute_subsequence[:, 1]),
                100,
            )

            pdf_total = mdn_to_heatmap(
                last_predicted_point, 5, grid_x, grid_y, last_known_point
            )

            # Plotting
            print(f"Plotting for sequence {start_idx}")
            plt.figure(figsize=(10, 6))
            plt.axis("equal")
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            plt.scatter(
                self.reconstructed_data[:, 0],
                self.reconstructed_data[:, 1],
                alpha=0.2,
                label="Full Dataset",
                color="grey",
            )
            plt.contourf(
                grid_x, grid_y, pdf_total, levels=50, cmap="viridis", alpha=0.5
            )
            plt.colorbar(label="Probability Density")
            plt.scatter(
                absolute_subsequence[:, 0],
                absolute_subsequence[:, 1],
                color="blue",
                label="Subsequence",
            )
            plt.scatter(
                last_known_point[0], last_known_point[1], color="c", label="Last Point"
            )
            plt.scatter(
                last_predicted_point[0, 0] + last_known_point[0],
                last_predicted_point[0, 1] + last_known_point[1],
                color="red",
                alpha=0.5,
                s=50,
                zorder=5,
                label="Predicted Next Point",
            )
            plt.scatter(
                self.reconstructed_data[end_idx, 0],
                self.reconstructed_data[end_idx, 1],
                color="purple",
                alpha=0.5,
                s=50,
                zorder=5,
                label="Actual Next Point",
            )
            plt.legend()
            plt.title(f"Predictive Point for Subsequence {start_idx} to {end_idx}")
            filename = f"{self.model_name}_epoch_{epoch}"
            filepath = os.path.join(self.image_dir, filename)
            plt.savefig(filepath)
            plt.close()


class StrokeVisualizeCallback2(Callback):
    def __init__(
        self,
        input_sequence,
        real_sequence,
        reconstructed_data,
        model_name,
        frequency=10,
        selected_sequence=4,
    ):
        super(StrokeVisualizeCallback2, self).__init__()
        self.input_sequence = input_sequence  # Use all sequences for the background
        self.real_sequence = real_sequence
        self.reconstructed_data = reconstructed_data
        self.model_name = model_name
        self.frequency = frequency
        self.image_dir = f"handwriting_visualizations/{self.model_name}"
        os.makedirs(self.image_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            predicted_sequence = self.model.predict(self.input_sequence)
            last_predicted_point = predicted_sequence[:, -1, :]
            print("last_predicted_point", last_predicted_point.shape)
            start_idx = 4
            end_idx = start_idx + sequence_length
            absolute_subsequence = self.reconstructed_data[start_idx:end_idx, :2]
            last_known_point = absolute_subsequence[-1, :]
            x_min = np.min(self.reconstructed_data[max(start_idx - 50, 0) :, 0])
            y_min, y_max = (
                np.min(self.reconstructed_data[:, 1]),
                np.max(self.reconstructed_data[:, 1]) + 10,
            )

            # Update x_max to the latest point in the subsequence
            if end_idx < self.reconstructed_data.shape[0]:
                actual_next_point = self.reconstructed_data[end_idx, :2]
                x_max = actual_next_point[0] + 10
            else:  # If at the end of the dataset, use the last known point
                x_max = last_known_point[0] + 10

            grid_x = np.linspace(
                np.min(absolute_subsequence[:, 0]),
                np.max(absolute_subsequence[:, 0]),
                100,
            )
            grid_y = np.linspace(
                np.min(absolute_subsequence[:, 1]),
                np.max(absolute_subsequence[:, 1]),
                100,
            )

            pdf_total = mdn_to_heatmap(
                last_predicted_point, 5, grid_x, grid_y, last_known_point
            )

            # Plotting
            print(f"Plotting for sequence {start_idx}")
            plt.figure(figsize=(10, 6))
            plt.axis("equal")
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            plt.scatter(
                self.reconstructed_data[:, 0],
                self.reconstructed_data[:, 1],
                alpha=0.2,
                label="Full Dataset",
                color="grey",
            )
            plt.contourf(
                grid_x, grid_y, pdf_total, levels=50, cmap="viridis", alpha=0.5
            )
            plt.colorbar(label="Probability Density")
            plt.scatter(
                absolute_subsequence[:, 0],
                absolute_subsequence[:, 1],
                color="blue",
                label="Subsequence",
            )
            plt.scatter(
                last_known_point[0], last_known_point[1], color="c", label="Last Point"
            )
            # plt.scatter(
            #     last_predicted_point[0, 0] + last_known_point[0],
            #     last_predicted_point[0, 1] + last_known_point[1],
            #     color="red",
            #     alpha=0.5,
            #     s=50,
            #     zorder=5,
            #     label="Predicted Next Point",
            # )
            plt.scatter(
                self.reconstructed_data[end_idx, 0],
                self.reconstructed_data[end_idx, 1],
                color="purple",
                alpha=0.5,
                s=50,
                zorder=5,
                label="Actual Next Point",
            )
            plt.legend()
            plt.title(f"Predictive Point for Subsequence {start_idx} to {end_idx}")
            filename = f"{self.model_name}_epoch_{epoch}"
            filepath = os.path.join(self.image_dir, filename)
            plt.savefig(filepath)
            plt.close()


# Train the Zig Zag model with visualization
# zigzag_model.fit(
#     x_train_zigzag,
#     y_train_zigzag,
#     epochs=TEST_NUM_EPOCHS,
#     batch_size=TEST_BATCH_SIZE,
#     verbose=1,
#     callbacks=[
#         SimpleHandwritingVisualizeCallback(
#             input_sequence=x_train_zigzag,  # Starting point for predictions
#             real_sequence=y_train_zigzag,  # Full actual sequence for comparison
#             model_name="handwriting_zigzag_simplified",
#         )
#     ],
# )

# # Train the Loop Da Loop model with visualization
# loop_model.fit(
#     x_train_loop,
#     y_train_loop,
#     epochs=TEST_NUM_EPOCHS,
#     batch_size=TEST_BATCH_SIZE,
#     verbose=1,
#     callbacks=[
#         SimpleHandwritingVisualizeCallback(
#             input_sequence=x_train_loop,  # Starting point for predictions
#             real_sequence=y_train_loop,  # Full actual sequence for comparison
#             model_name="handwriting_loop_simplified",
#         )
#     ],
# )

deep_stroke_model.fit(
    x_train_stroke,
    y_train_stroke,
    epochs=TEST_NUM_EPOCHS * 5,
    batch_size=TEST_BATCH_SIZE,
    verbose=1,
    callbacks=[
        StrokeVisualizeCallback2(
            input_sequence=x_train_stroke,
            real_sequence=y_train_stroke,
            reconstructed_data=reconstructed_data,
            model_name="handwriting_stroke_simplified",
        )
    ],
)

# stroke_model.fit(
#     x_train_stroke,
#     y_train_stroke,
#     epochs=TEST_NUM_EPOCHS * 5,
#     batch_size=TEST_BATCH_SIZE,
#     verbose=1,
#     callbacks=[
#         StrokeVisualizeCallback2(
#             input_sequence=x_train_stroke,
#             real_sequence=y_train_stroke,
#             reconstructed_data=reconstructed_data,
#             model_name="handwriting_stroke_simplified",
#         )
#     ],
# )

create_gif(
    "handwriting_visualizations/handwriting_zigzag_simplified",
    "handwriting_zigzag_simplified*",
    "test_zigzag_simplified",
)
create_gif(
    "handwriting_visualizations/handwriting_loop_simplified",
    "handwriting_loop_simplified*",
    "test_zigzag_simplified",
)
