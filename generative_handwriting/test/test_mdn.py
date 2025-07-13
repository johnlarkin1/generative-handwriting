import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model.mixture_density_network import MDNLayer, mdn_loss
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras.models import Sequential

SEQUENCE_LENGTH = 5


class MDNHeatmapCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test, num_points=100, num_components=5):
        """
        Args:
        - x_test: A fixed range of x values to predict y for.
        - num_points: Number of points to sample for the heatmap.
        - num_components: Number of mixture components in the MDN.
        """
        self.x_test = x_test
        self.y_test = y_test
        self.num_points = num_points
        self.num_components = num_components

    def on_epoch_end(self, epoch, logs=None):
        # Predict using the current state of the model
        preds = self.model.predict(self.x_test)

        # Assuming preds.shape is (num_samples, seq_length, num_components * 6 + 1)
        # and your MDN is outputting pi, mu1, mu2, sigma1, sigma2, rho, eos in that order
        x_values = self.x_test[0, :, 0]  # Get the x values of the first sequence
        y_values = self.x_test[0, :, 1]  # Get the y values of the first sequence
        actual_x_values = self.y_test[0, :, 0]
        actual_y_values = self.y_test[0, :, 1]

        pi, mu1, mu2, sigma1, sigma2, rho, eos = np.split(
            preds,
            [
                self.num_components,
                2 * self.num_components,
                3 * self.num_components,
                4 * self.num_components,
                5 * self.num_components,
                6 * self.num_components,
            ],
            axis=-1,
        )
        pi = pi[..., 0]  # Assuming we're just visualizing the first time step for simplicity

        # We will simplify by visualizing the distribution as if each component is independent
        # This is a simplification and does not take into account the correlations
        y = np.linspace(-1.5, 1.5, self.num_points)
        heatmap = np.zeros((self.num_points, len(self.x_test)))

        for i, (pi_i, mu1_i, sigma1_i) in enumerate(zip(pi, mu1, sigma1, strict=False)):
            for j, y_val in enumerate(y):
                # For each component, calculate the probability density
                pdf_components = [
                    pi_ij * (1 / (np.sqrt(2 * np.pi) * sigma1_ij)) * np.exp(-0.5 * ((y_val - mu1_ij) / sigma1_ij) ** 2)
                    for pi_ij, mu1_ij, sigma1_ij in zip(pi_i, mu1_i, sigma1_i, strict=False)
                ]

                # Sum the contributions from all mixture components to get the total probability density at this point
                pdf_total = np.sum(pdf_components)
                heatmap[j, i] = pdf_total

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        plt.scatter(x_values, y_values, color="red", label="Sequence Points")
        plt.scatter(
            actual_x_values,
            actual_y_values,
            color="blue",
            label="Actual Sequence Points",
        )
        plt.imshow(
            heatmap,
            extent=[self.x_test.min(), self.x_test.max(), y.min(), y.max()],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        plt.colorbar(label="Probability Density")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title(f"Epoch: {epoch + 1}")
        plt.show()


def generate_synthetic_data(num_sequences=1000, seq_length=SEQUENCE_LENGTH, batch_size=32):
    sequences = []
    for _ in range(num_sequences):
        start_x = np.random.uniform(-10, 10)
        x_vals = np.linspace(start_x, start_x + 2 * np.pi, seq_length)
        y_vals = np.sin(x_vals) + np.random.normal(0, 0.1, size=x_vals.shape)
        eos_vals = np.ones_like(x_vals)  # End of stroke set to 1
        sequence = np.stack([x_vals, y_vals, eos_vals], axis=1)
        sequences.append(sequence)
    sequences = np.array(sequences)
    return sequences


def basic_network():
    model = Sequential(
        [
            Input(shape=(SEQUENCE_LENGTH, 3)),
            LSTM(64, return_sequences=True),
            LSTM(64, return_sequences=True),
            MDNLayer(num_components=5),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=lambda y_true, y_pred: mdn_loss(y_true, y_pred, num_components=5),
    )
    return model


def main() -> None:
    data = generate_synthetic_data()
    inputs, targets = data[:, :-1], data[:, 1:]
    model = basic_network()
    heatmap_callback = MDNHeatmapCallback(inputs[:1], targets[:1])
    # model.fit(inputs, targets, epochs=100)
    model.fit(inputs, targets, epochs=100, callbacks=[heatmap_callback])


if __name__ == "__main__":
    main()
