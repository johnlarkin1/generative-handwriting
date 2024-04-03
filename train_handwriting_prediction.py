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

from constants import (
    LEARNING_RATE,
    NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS,
    NUM_EPOCH,
    TEST_NUM_MIXTURES,
    TEST_NUM_EPOCHS,
    TEST_BATCH_SIZE,
    TEST_SEQUENCE_LENGTH,
)

curr_directory = os.path.dirname(os.path.realpath(__file__))
model_save_dir = f"{curr_directory}/saved_models/full_handwriting_prediction/"
model_save_path = os.path.join(model_save_dir, "best_model.keras")
epochs_info_path = os.path.join(model_save_dir, "epochs_info.json")


@tf.keras.utils.register_keras_serializable()
def model_mdn_loss(actual, outputs):
    return mdn_loss(actual, outputs, num_mixture_components)


def load_model_if_exists(model_save_path):
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


if __name__ == "__main__":
    num_mixture_components = NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS
    stroke_model, model_loaded = load_model_if_exists(model_save_path)
    desired_epochs = NUM_EPOCH

    if not model_loaded:
        print("No suitable saved model found or model has changed, initializing a new one...")
        stroke_model = DeepHandwritingPredictionModel(num_mixture_components=num_mixture_components)
        stroke_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=model_mdn_loss,
        )

    last_trained_epoch = load_epochs_info(epochs_info_path)
    if last_trained_epoch < desired_epochs:
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
