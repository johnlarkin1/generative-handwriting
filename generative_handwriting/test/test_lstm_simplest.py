from common import create_gif


USE_DUMMY_LSTM = False


# def generate_cosine_data_with_noise(sequence_length, total_sequences, noise_factor=0.01):
#     x_values = np.linspace(0, 5 * np.pi, sequence_length * total_sequences)
#     cosine_values = np.cos(x_values) + np.random.normal(0, noise_factor, size=x_values.shape)
#     additional_feature = np.ones_like(x_values)
#     X = np.stack([x_values, cosine_values, additional_feature], axis=-1).reshape(total_sequences, sequence_length, 3)
#     Y = np.roll(X, -1, axis=1)
#     return X, Y


# sequence_length = 50
# total_sequences = 500
# X, Y = generate_cosine_data_with_noise(sequence_length, total_sequences)


# class SimpleLSTMModel(models.Model):
#     def __init__(self, units, num_layers, feature_size):
#         super(SimpleLSTMModel, self).__init__()
#         self.lstm_layers = [layers.LSTM(units, return_sequences=True) for _ in range(num_layers)]
#         self.dense = layers.Dense(feature_size)

#     def call(self, inputs):
#         x = inputs
#         for lstm_layer in self.lstm_layers:
#             x = lstm_layer(x)
#         return self.dense(x)


# if USE_DUMMY_LSTM:
#     model = SimpleLSTMModel(
#         units=50,  # Adjust according to your needs
#         num_layers=2,  # Adjust according to your needs
#         feature_size=3,  # Match the output feature size
#     )
# else:
#     model = SimplestHandwritingPredictionModel(units=NUM_LSTM_CELLS_PER_HIDDEN_LAYER, feature_size=3)

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mean_squared_error")


# class VisualizeCallback(tf.keras.callbacks.Callback):
#     def __init__(self, input_sequence, real_sequence, model_name, full_cosine_data, frequency=1):
#         super(VisualizeCallback, self).__init__()
#         self.input_sequence = input_sequence
#         self.real_sequence = real_sequence
#         self.full_cosine_data = full_cosine_data
#         self.model_name = model_name
#         self.frequency = frequency
#         self.image_dir = f"lstm_cosine/{self.model_name}"
#         os.makedirs(self.image_dir, exist_ok=True)
#         self.colors = ["r", "g", "b", "c", "m", "y", "k"]
#         self.marker_size = 30

#     def on_epoch_end(self, epoch, logs=None):
#         if epoch % self.frequency == 0:
#             example_indexes = [(30, 35), (80, 120), (250, 251), (320, 340)]
#             plt.figure(figsize=(10, 6))

#             # Get the x and y coordinates for the full dataset (noisy data)
#             full_x = self.full_cosine_data[:, 0]
#             full_y = self.full_cosine_data[:, 1]
#             plt.plot(full_x, full_y, "k-", alpha=0.3, label="Full Noisy Cosine Data")

#             for idx, (start_idx, end_idx) in enumerate(example_indexes):
#                 color = self.colors[idx % len(self.colors)]

#                 # Extract sequences based on the current index range
#                 input_sequence = self.input_sequence[start_idx:end_idx]
#                 actual_sequence = self.real_sequence[start_idx:end_idx]

#                 # Concatenate and find unique values (if necessary) for input and actual sequences
#                 sequence_input = np.concatenate(input_sequence, axis=0)
#                 sequence_actual = np.concatenate(actual_sequence, axis=0)
#                 sequence_input = np.unique(sequence_input, axis=0)
#                 sequence_actual = np.unique(sequence_actual, axis=0)

#                 # Reshape for prediction
#                 sequence_input_reshaped = sequence_input.reshape(1, -1, sequence_input.shape[-1])
#                 prediction = self.model.predict(sequence_input_reshaped, batch_size=1)

#                 # Extract x and y coordinates for actual and predicted sequences
#                 actual_x = sequence_actual[:, 0]
#                 actual_y = sequence_actual[:, 1]
#                 pred_x = prediction[0, :, 0]
#                 pred_y = prediction[0, :, 1]

#                 # Plot actual values
#                 plt.scatter(
#                     actual_x,
#                     actual_y,
#                     marker="x",
#                     color=color,
#                     s=self.marker_size,
#                     alpha=0.3,
#                     label=f"Actual idx={start_idx}-{end_idx}",
#                 )
#                 # Plot predicted values
#                 plt.scatter(
#                     pred_x,
#                     pred_y,
#                     color=color,
#                     s=self.marker_size,
#                     alpha=0.3,
#                     label=f"Predicted idx={start_idx}-{end_idx}",
#                 )

#             plt.title(f"Model: {self.model_name} - Epoch: {epoch}")
#             plt.xlabel("X Value")
#             plt.ylabel("Cosine Value")
#             plt.legend(loc="upper right")
#             plt.savefig(f"{self.image_dir}/epoch_{epoch}_{'simple' if USE_DUMMY_LSTM else 'real'}.png")
#             plt.close()

#     # def on_epoch_end(self, epoch, logs=None):
#     #     if epoch % self.frequency == 0:
#     #         predictions = self.model.predict(self.input_sequence, verbose=0)
#     #         example_indexes = [30, 230, 300]

#     #         sequences = self.input_sequence[80:120]
#     #         long_actual = self.real_sequence[80:120]
#     #         long_sequence_input = np.concatenate(sequences, axis=0)
#     #         long_sequence_actual = np.concatenate(long_actual, axis=0)
#     #         long_sequence_input = np.unique(long_sequence_input, axis=0)
#     #         long_sequence_actual = np.unique(long_sequence_actual, axis=0)
#     #         long_sequence_input = long_sequence_input.reshape(1, -1, long_sequence_input.shape[-1])
#     #         long_sequence_actual = long_sequence_actual.reshape(1, -1, long_sequence_actual.shape[-1])
#     #         long_prediction = model.predict(long_sequence_input, batch_size=1)

#     #         plt.figure(figsize=(10, 6))
#     #         long_x = long_sequence_actual[0, :, 0]
#     #         long_y = long_sequence_actual[0, :, 1]
#     #         plt.scatter(
#     #             long_x, long_y, color="m", marker="x", s=self.marker_size, alpha=0.3, label="Long Sequence Actual"
#     #         )
#     #         # Plot the prediction for the long sequence
#     #         long_pred_x = long_prediction[0, :, 0]
#     #         long_pred_y = long_prediction[0, :, 1]
#     #         plt.scatter(
#     #             long_pred_x,
#     #             long_pred_y,
#     #             color="m",
#     #             s=self.marker_size,
#     #             alpha=0.3,
#     #             label="Long Sequence Prediction",
#     #         )

#     #         # Get the x and y coordinates for the full dataset (noisy data)
#     #         full_x = self.full_cosine_data[:, 0]
#     #         full_y = self.full_cosine_data[:, 1]

#     #         plt.plot(full_x, full_y, "k-", alpha=0.3, label="Full Noisy Cosine Data")

#     #         for idx, example_idx in enumerate(example_indexes):
#     #             color = self.colors[idx % len(self.colors)]

#     #             # Get the x and y coordinates for the prediction
#     #             pred_x = predictions[example_idx, :, 0]
#     #             pred_y = predictions[example_idx, :, 1]
#     #             print("pred_x:", pred_x.shape)
#     #             actual_x = self.real_sequence[example_idx, :, 0]
#     #             actual_y = self.real_sequence[example_idx, :, 1]

#     #             # Plot hollow circles for actual values
#     #             plt.scatter(
#     #                 actual_x,
#     #                 actual_y,
#     #                 marker="x",
#     #                 color=color,
#     #                 s=self.marker_size,
#     #                 alpha=0.3,
#     #                 label=f"Actual idx={example_idx}",
#     #             )
#     #             # Plot filled circles for predicted values, with stronger alpha
#     #             plt.scatter(
#     #                 pred_x,
#     #                 pred_y,
#     #                 color=color,
#     #                 s=self.marker_size,
#     #                 alpha=0.3,
#     #                 label=f"Predicted idx={example_idx}",
#     #             )

#     #         plt.title(f"Model: {self.model_name} - Epoch: {epoch}")
#     #         plt.xlabel("X Value")
#     #         plt.ylabel("Cosine Value")
#     #         plt.legend(loc="upper right")
#     #         plt.savefig(f"{self.image_dir}/epoch_{epoch}_{'simple' if USE_DUMMY_LSTM else 'real'}.png")
#     #         plt.close()


# # When creating the callback, pass the entire dataset's cosine data as an argument
# full_cosine_data = np.concatenate(X, axis=0)  # Assuming X is your input data array

# history = model.fit(
#     X,
#     Y,
#     epochs=250,
#     batch_size=64,
#     callbacks=[
#         VisualizeCallback(
#             input_sequence=X,  # Starting point for predictions
#             real_sequence=Y,  # Full actual sequence for comparison
#             model_name="cosine_wave_prediction",
#             full_cosine_data=full_cosine_data,  # Pass the full noisy cosine data here
#             frequency=10,  # Adjust visualization frequency as needed
#         )
#     ],
# )

# final_loss = history.history["loss"][-1]
# print(f"Final training loss: {final_loss}")

# # Predict on the first sequence
# predictions = model.predict(X[:1])


create_gif(
    "lstm_cosine/cosine_wave_prediction",
    "epoch_*_real",
    "variable_length_peephole_lstm",
    fps=5,
)
create_gif(
    "lstm_cosine/cosine_wave_prediction/saved",
    "epoch_*_real",
    "same_length_peephole_lstm",
    fps=5,
)
