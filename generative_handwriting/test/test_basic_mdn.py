import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from common import (
    create_gif,
    generate_loop_da_loop_data,
    generate_zig_zag_data,
    prepare_data_for_sequential_prediction,
    show_initial_data,
)
from constants import (
    EPSILON,
    NUM_LSTM_CELLS_PER_HIDDEN_LAYER,
    TEST_BATCH_SIZE,
    TEST_NUM_EPOCHS,
    TEST_NUM_MIXTURES,
)
from model.basic_mdn import BasicMixtureDensityNetwork, mdn_loss_function
from tensorflow.keras.callbacks import Callback


class VisualizeCallback(Callback):
    def __init__(self, data, model_name, frequency=10):
        super(VisualizeCallback, self).__init__()
        self.data = data  # Use the entire dataset for context
        self.target = data[5:6]  # Select the sixth point (ensure it's in a batched shape)
        self.model_name = model_name
        self.frequency = frequency

    def gaussian_pdf(self, y, mean, std):
        norm = tf.reduce_sum((y - mean) ** 2 / (std**2 + EPSILON), axis=-1)
        return tf.exp(-0.5 * norm) / (2.0 * np.pi * tf.reduce_prod(std, axis=-1) + EPSILON)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            predictions = self.model.predict(self.target)

            # Assuming the first two are mean for x, y and the next two are log standard deviations
            mean = predictions[:, :2]
            std = tf.exp(predictions[:, 2:4])

            # Generate a grid over which to compute the PDF
            x_min, x_max = np.min(self.data[:, 0]) - 1, np.max(self.data[:, 0]) + 1
            y_min, y_max = np.min(self.data[:, 1]) - 1, np.max(self.data[:, 1]) + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            grid = np.column_stack((xx.ravel(), yy.ravel()))

            grid = tf.constant(grid, dtype=tf.float32)
            mean = tf.constant(mean, dtype=tf.float32)
            std = tf.constant(std, dtype=tf.float32)
            pdf_values = self.gaussian_pdf(tf.expand_dims(grid, 1), mean, std)
            pdf_values = tf.reduce_sum(pdf_values, axis=1).numpy().reshape(xx.shape)

            plt.figure(figsize=(10, 6))
            plt.contourf(xx, yy, pdf_values, levels=50, cmap="viridis", alpha=0.5)
            plt.scatter(self.data[:, 0], self.data[:, 1], c="grey", label="Data Points")
            plt.scatter(self.data[:5, 0], self.data[:5, 1], c="red", label="First 5 Points")
            plt.scatter(
                self.data[5, 0],
                self.data[5, 1],
                c="green",
                marker="X",
                label="Sixth Point",
            )
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"Model Predictions for Sixth Point and Heatmap at Epoch {epoch}")
            plt.legend()
            plt.savefig(f"mdn_visualizations/{self.model_name}_heatmap_epoch_{epoch}.png")
            plt.close()


# Load your data
zigzag_data = generate_zig_zag_data()
loop_data = generate_loop_da_loop_data()
show_initial_data(zigzag_data, loop_data)

# Prepare the data for sequential prediction
x_train_zigzag, y_train_zigzag = prepare_data_for_sequential_prediction(zigzag_data)
x_train_loop, y_train_loop = prepare_data_for_sequential_prediction(loop_data)

# Define the models
zigzag_model = BasicMixtureDensityNetwork(num_mixtures=TEST_NUM_MIXTURES, hidden_units=NUM_LSTM_CELLS_PER_HIDDEN_LAYER)
loop_model = BasicMixtureDensityNetwork(num_mixtures=TEST_NUM_MIXTURES, hidden_units=NUM_LSTM_CELLS_PER_HIDDEN_LAYER)

# Compile the models
zigzag_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=lambda actual, outputs: mdn_loss_function(actual, outputs, TEST_NUM_MIXTURES),
)
loop_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=lambda actual, outputs: mdn_loss_function(actual, outputs, TEST_NUM_MIXTURES),
)

# Train the Zig Zag model with visualization
zigzag_model.fit(
    x_train_zigzag,
    y_train_zigzag,
    epochs=TEST_NUM_EPOCHS,
    batch_size=TEST_BATCH_SIZE,
    verbose=1,
    callbacks=[VisualizeCallback(x_train_zigzag, "zigzag", frequency=10)],
)

# Train the Loop Da Loop model with visualization
loop_model.fit(
    x_train_loop,
    y_train_loop,
    epochs=TEST_NUM_EPOCHS,
    batch_size=TEST_BATCH_SIZE,
    verbose=1,
    callbacks=[VisualizeCallback(x_train_loop, "loop", frequency=10)],
)

create_gif("mdn_visualizations", "zigzag")
create_gif("mdn_visualizations", "loop")
create_gif("mdn_visualizations", "zigzag_heatmap")
create_gif("mdn_visualizations", "loop_heatmap")
create_gif("mdn_visualizations", "zigzag_prediction")
create_gif("mdn_visualizations", "loop_prediction")
