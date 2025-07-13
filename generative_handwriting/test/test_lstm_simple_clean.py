import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from common import (
    create_gif,
    create_subsequence_batches,
    plot_strokes_from_dx_dy,
    prepare_data_for_sequential_prediction,
)
from constants import (
    ADAM_CLIP_NORM,
    LEARNING_RATE,
    NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
    NUM_LSTM_HIDDEN_LAYERS,
    TEST_BATCH_SIZE,
    TEST_NUM_EPOCHS,
    TEST_NUM_MIXTURES,
)
from loader import HandwritingDataLoader
from model.handwriting_models import (
    SimpleHandwritingPredictionModel,
)
from tensorflow.keras.callbacks import Callback


def custom_loss(y_true, y_pred):
    mse_xy = tf.keras.losses.mean_squared_error(y_true[:, :, :2], y_pred[:, :, :2])
    mse_xy_reduced = tf.reduce_mean(mse_xy, axis=1)
    return tf.reduce_mean(mse_xy_reduced)


strokes, stroke_lengths = HandwritingDataLoader().load_individual_stroke_data("a01/a01-000/a01-000u-01.xml")
# plot_original_strokes_from_xml("a01/a01-000/a01-000u-01.xml")
reconstructed_data = plot_strokes_from_dx_dy(strokes, show_image=False)
x_stroke, y_stroke = prepare_data_for_sequential_prediction(strokes)
# x_stroke = np.reshape(x_stroke, (1, x_stroke.shape[0], x_stroke.shape[1]))
# y_stroke = np.reshape(y_stroke, (1, y_stroke.shape[0], y_stroke.shape[1]))
sequence_length = 50
desired_epochs = 200

x_train_stroke, y_train_stroke = create_subsequence_batches(x_stroke, y_stroke, sequence_length)
x_train_stroke, y_train_stroke = x_train_stroke[4, :, :], y_train_stroke[4, :, :]
x_train_stroke = np.reshape(x_train_stroke, (1, x_train_stroke.shape[0], x_train_stroke.shape[1]))
y_train_stroke = np.reshape(y_train_stroke, (1, y_train_stroke.shape[0], y_train_stroke.shape[1]))
# x_train_stroke, y_train_stroke = x_stroke, y_stroke
num_mixture_components = TEST_NUM_MIXTURES
learning_rate = LEARNING_RATE

stroke_model = SimpleHandwritingPredictionModel(
    units=NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
    num_layers=NUM_LSTM_HIDDEN_LAYERS,
    feature_size=3,
)

stroke_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 2, clipnorm=ADAM_CLIP_NORM),
    loss="mean_squared_error",
)


class PostTrainingVisualizeCallback(Callback):
    def __init__(self, test_data, plot_frequency=1):
        super().__init__()
        self.test_data = test_data
        self.plot_frequency = plot_frequency

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.plot_frequency == 0:
            x_test, y_test = self.test_data
            y_pred = self.model.predict(x_test)

            # Plotting just one example for simplicity
            index = np.random.choice(len(y_test))  # Pick a sample to plot
            real_data = plot_strokes_from_dx_dy(y_test[index], show_image=False, title="Real Stroke")
            predicted_data = plot_strokes_from_dx_dy(y_pred[index], show_image=False, title="Predicted Stroke")
            plt.figure(figsize=(10, 6))
            plt.axis("equal")
            plt.plot(real_data[:, 0], real_data[:, 1], color="blue", label="Real Stroke")
            plt.plot(
                predicted_data[:, 0],
                predicted_data[:, 1],
                color="red",
                label="Predicted Stroke",
            )
            plt.title("Real vs Predicted")
            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")
            plt.legend()
            filename = f"simple_stroke_epoch_{epoch}"
            filepath = os.path.join("handwriting_visualizations/single_char", filename)
            plt.savefig(filepath)
            plt.close()


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
        self.real_sequence = real_sequence[:, selected_sequence : selected_sequence + 50, :]
        self.reconstructed_data = reconstructed_data
        self.model_name = model_name
        self.frequency = frequency
        self.image_dir = f"handwriting_visualizations/{self.model_name}"
        os.makedirs(self.image_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            predicted_sequence = self.model.predict(self.selected_input_sequence)
            last_predicted_point = predicted_sequence[0, -1, :]
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
            plt.scatter(
                absolute_subsequence[:, 0],
                absolute_subsequence[:, 1],
                color="blue",
                label="Subsequence",
            )
            plt.scatter(last_known_point[0], last_known_point[1], color="c", label="Last Point")
            plt.scatter(
                last_predicted_point[0] + last_known_point[0],
                last_predicted_point[1] + last_known_point[1],
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


stroke_model.fit(
    x_train_stroke,
    y_train_stroke,
    epochs=TEST_NUM_EPOCHS * 5,
    batch_size=TEST_BATCH_SIZE,
    verbose=1,
    callbacks=[
        StrokeVisualizeCallback(
            input_sequence=x_train_stroke,
            real_sequence=y_train_stroke,
            reconstructed_data=reconstructed_data,
            model_name="handwriting_stroke_simplified",
        ),
        PostTrainingVisualizeCallback(test_data=(x_train_stroke, y_train_stroke), plot_frequency=10),
    ],
)

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
