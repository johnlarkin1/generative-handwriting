import os

import numpy as np
import tensorflow as tf
from common import (
    print_model_parameters,
)
from constants import (
    BATCH_SIZE,
    GRADIENT_CLIP_VALUE,
    LEARNING_RATE,
    NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
    NUM_EPOCH,
)
from loader import HandwritingDataLoader
from model.handwriting_models import DeepHandwritingPredictionModel
from model.mixture_density_network import MixtureDensityLayer, mdn_loss
from model_io import load_epochs_info, save_epochs_info

curr_directory = os.path.dirname(os.path.realpath(__file__))
model_save_dir = f"{curr_directory}/saved_models/full_handwriting_prediction/"
model_save_path = os.path.join(model_save_dir, "best_model.keras")
epochs_info_path = os.path.join(model_save_dir, "epochs_info.json")
num_mixture_components = NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS
data_loader = HandwritingDataLoader()
data_loader.prepare_data()
combined_train_strokes, combined_train_lengths = (
    data_loader.combined_train_strokes,
    data_loader.combined_train_stroke_lengths,
)


@tf.keras.utils.register_keras_serializable()
def model_mdn_loss(actual, outputs, combined_train_lengths, num_mixture_components):
    return mdn_loss(actual, outputs, combined_train_lengths, num_mixture_components)


def load_model_if_exists(model_save_path):
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


if __name__ == "__main__":
    stroke_model, model_loaded = load_model_if_exists(model_save_path)
    desired_epochs = NUM_EPOCH

    x_train = combined_train_strokes
    x_train_len = combined_train_lengths
    y_train = np.zeros_like(x_train)
    y_train[:, :-1, :] = x_train[:, 1:, :]
    y_train_len = np.zeros_like(x_train_len)
    y_train_len = x_train_len[1:]

    best_loss = float("inf")

    batch_size = BATCH_SIZE
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, x_train_len))
    dataset = dataset.batch(batch_size)
    # need this for m4 instance
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)

    if not model_loaded:
        print("No suitable saved model found or model has changed, initializing a new one...")
        stroke_model = DeepHandwritingPredictionModel(num_mixture_components=num_mixture_components)

    assert stroke_model is not None, "Model not loaded or initialized."

    last_trained_epoch = load_epochs_info(epochs_info_path)
    for epoch in range(last_trained_epoch, desired_epochs):
        print(f"Epoch {epoch + 1}/{desired_epochs}")
        epoch_losses = []  # Track losses for each epoch

        for step, (batch_x, batch_y, batch_len) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = stroke_model(batch_x, training=True)
                loss = model_mdn_loss(batch_y, predictions, batch_len, num_mixture_components)
            gradients = tape.gradient(loss, stroke_model.trainable_variables)
            clipped = [
                (tf.clip_by_value(g, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE), v_)
                for g, v_ in zip(gradients, stroke_model.trainable_variables)
            ]
            optimizer.apply_gradients(zip(gradients, stroke_model.trainable_variables))
            epoch_losses.append(loss.numpy())

            if epoch == 0 and step == 0:
                print_model_parameters(stroke_model)

            if step % 5 == 0:
                print(f"Step {step}, Loss: {loss.numpy()}")

        # After each epoch, check if the average loss is the best so far
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} Average Loss: {epoch_loss}")

        # Optionally, evaluate on a validation dataset here and use that loss instead
        if epoch_loss < best_loss:
            print(f"New best loss {epoch_loss}, saving model.")
            best_loss = epoch_loss
            stroke_model.save(model_save_path)  # Save the model
        save_epochs_info(epoch + 1, epochs_info_path)
    print("Training completed.")
