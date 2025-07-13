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
    DeepHandwritingPredictionModel,
)
from model.mixture_density_network import mdn_loss
from tensorflow.keras.callbacks import Callback

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

# Plot inputs vs targets
plt.figure(figsize=(12, 8))
plt.axis("equal")
plt.plot(
    np.cumsum(x_train_stroke[0, :, 0]),
    np.cumsum(x_train_stroke[0, :, 1]),
    "bo",
    label="Inputs",
)
plt.plot(
    np.cumsum(y_train_stroke[0, :, 0]),
    np.cumsum(y_train_stroke[0, :, 1]),
    "ro",
    label="Targets",
)
plt.title("Inputs vs Targets")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.show()
# x_train_stroke, y_train_stroke = x_stroke, y_stroke
num_mixture_components = TEST_NUM_MIXTURES
learning_rate = LEARNING_RATE


# zigzag_data = generate_zig_zag_data()
# x_zigzag, y_zigzag = prepare_data_for_sequential_prediction_with_eos(zigzag_data)
# print
# x_zigzag = np.reshape(x_zigzag, (1, x_zigzag.shape[0], x_zigzag.shape[1]))
# y_zigzag = np.reshape(y_zigzag, (1, y_zigzag.shape[0], y_zigzag.shape[1]))
# x_train_stroke = x_zigzag
# y_train_stroke = y_zigzag

stroke_model = DeepHandwritingPredictionModel(
    units=NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
    num_layers=NUM_LSTM_HIDDEN_LAYERS,
    num_mixture_components=TEST_NUM_MIXTURES,
)

stroke_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE * 2, clipnorm=ADAM_CLIP_NORM),
    loss=lambda x, y: mdn_loss(x, y, None, num_mixture_components),
)


class PostTrainingVisualizeCallback(Callback):
    def __init__(self, test_data, plot_frequency=1, num_mixtures=5):
        super().__init__()
        self.test_data = test_data
        self.plot_frequency = plot_frequency
        self.num_mixtures = num_mixtures
        self.save_dir = "handwriting_visualizations/test_mdn_simple_clean/"

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.plot_frequency == 0:
            x_test, y_test = self.test_data
            y_pred = self.model.predict(x_test)

            for i in range(len(x_test)):
                plt.figure(figsize=(12, 8))

                # Convert dx, dy in x_test to absolute coordinates
                dx_dy_real = x_test[i, :, :2]  # Ignoring eos
                abs_coords_real = np.cumsum(dx_dy_real, axis=0)
                abs_coords_real = np.vstack(([0, 0], abs_coords_real))  # Assuming start at (0, 0)

                # Plot real data
                plt.plot(
                    abs_coords_real[:, 0],
                    abs_coords_real[:, 1],
                    "ko-",
                    label="Real Path",
                    zorder=2,
                )
                print("dx_dy_real", dx_dy_real.shape)

                for t in range(1, len(dx_dy_real)):  # Start at 1 because we need a 'next' point
                    # The last known point at time t
                    last_known_point = abs_coords_real[t - 1]
                    # The actual next point
                    next_point = abs_coords_real[t]

                    # Plot the actual next point
                    plt.scatter(
                        next_point[0],
                        next_point[1],
                        color="red",
                        label="Actual Next Point",
                        zorder=3 if t == 1 else 2,
                    )

                    for j in range(self.num_mixtures):
                        # Extract mus for each mixture component at time t-1
                        mu1 = y_pred[i, t - 1, j + self.num_mixtures]
                        mu2 = y_pred[i, t - 1, j + 2 * self.num_mixtures]

                        # Plot each mu as a point relative to the last known point
                        plt.scatter(
                            last_known_point[0] + mu1,
                            last_known_point[1] + mu2,
                            label=(f"Mixture {j + 1} Mean at Timestep {t - 1}" if t == 1 else ""),
                            alpha=0.5,
                            zorder=4,
                        )  # Adjust zorder for visibility, alpha for transparency

                plt.title(f"Sequence {i + 1} Prediction vs Real Path")
                plt.xlabel("X Coordinate")
                plt.ylabel("Y Coordinate")
                # Only show the legend once per mixture component
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles, strict=False))
                plt.legend(by_label.values(), by_label.keys())
                filename = f"mdn_prediction_{epoch}.png"
                filepath = os.path.join(self.save_dir, filename)
                plt.savefig(filepath)


stroke_model.fit(
    x_train_stroke,
    y_train_stroke,
    epochs=TEST_NUM_EPOCHS * 50,
    batch_size=TEST_BATCH_SIZE,
    verbose=1,
    callbacks=[
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
