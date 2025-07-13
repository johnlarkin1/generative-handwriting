from model.handwriting_models import SimpleLSTMModel
import numpy as np
import tensorflow as tf

# A small synthetic sequential dataset
data_size, sequence_length, feature_size = 100, 10, 3

# Create a dataset where the target is the same as the input, shifted by one timestep
X = np.random.rand(data_size, sequence_length, feature_size).astype(np.float32)
Y = np.roll(X, -1, axis=1)

if __name__ == "__main__":
    # Instantiate the model
    model = SimpleLSTMModel(units=50, output_size=feature_size)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    # Fit the model to the synthetic data
    history = model.fit(X, Y, epochs=200)

    # Check if the loss has decreased to a very low value
    final_loss = history.history["loss"][-1]
    print(f"Final training loss: {final_loss}")

    # Predict on the training data
    predictions = model.predict(X)

    # Check the predictions against the targets
    print("Predictions: ", predictions[0])
    print("Targets: ", Y[0])

    import matplotlib.pyplot as plt

    # Assuming `predictions` and `Y` are your predictions and targets respectively
    # and you're interested in the first example in the batch

    # Select which feature to plot, e.g., 0 for the first feature
    feature_to_plot = 0

    # Select the first example in the batch
    true_sequence = Y[0, :, feature_to_plot]
    predicted_sequence = predictions[0, :, feature_to_plot]

    # Generate a time axis based on the sequence length
    time_steps = np.arange(sequence_length)

    # Create a plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, true_sequence, "o-", label="Target")
    plt.plot(time_steps, predicted_sequence, "x-", label="Prediction")
    plt.title("Prediction vs Target")
    plt.xlabel("Time Step")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.show()
