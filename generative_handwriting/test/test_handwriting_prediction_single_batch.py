import os
from common import (
    plot_strokes_from_dx_dy,
    prepare_data_for_sequential_prediction,
)
from model_io import load_model_if_exists
from plotting import generate_and_plot_heatmap
from loader import HandwritingDataLoader
from model.mixture_density_network import mdn_loss
from model.handwriting_models import DeepHandwritingPredictionModel

import tensorflow as tf
import numpy as np

from constants import (
    LEARNING_RATE,
)
from callbacks import ExtendedModelCheckpoint

model_save_dir = "/Users/johnlarkin/Documents/coding/generative-handwriting/src/saved_models/handwriting_prediction_single_batch_subset/"
model_pred_dir = "/Users/johnlarkin/Documents/coding/generative-handwriting/handwriting_visualizations/single_stroke_single_batch_subset/"
model_save_path = os.path.join(model_save_dir, "best_model.keras")
epochs_info_path = os.path.join(model_save_dir, "epochs_info.json")


@tf.keras.utils.register_keras_serializable()
def model_mdn_loss(actual, outputs):
    return mdn_loss(actual, outputs, None, num_mixture_components)


desired_epochs = 1000
strokes, stroke_lengths = HandwritingDataLoader().load_individual_stroke_data(
    "a01/a01-000/a01-000u-01.xml"
)
# plot_original_strokes_from_xml("a01/a01-000/a01-000u-01.xml")
reconstructed_data = plot_strokes_from_dx_dy(strokes, show_image=False)

x_stroke, y_stroke = prepare_data_for_sequential_prediction(strokes)
x_train_stroke = np.reshape(x_stroke, (1, x_stroke.shape[0], x_stroke.shape[1]))
y_train_stroke = np.reshape(y_stroke, (1, y_stroke.shape[0], y_stroke.shape[1]))

x_batches, y_batches = x_train_stroke[:, 0:100, :], y_train_stroke[:, 0:100, :]
num_mixture_components = 1
initial_learning_rate = LEARNING_RATE

stroke_model, is_success = load_model_if_exists(
    model_save_path,
    custom_objects={
        "model_mdn_loss": model_mdn_loss,
        "DeepHandwritingPredictionModel": DeepHandwritingPredictionModel,
    },
)

if not is_success:
    stroke_model = DeepHandwritingPredictionModel(
        num_mixture_components=num_mixture_components
    )
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    stroke_model.compile(
        optimizer=optimizer,
        loss=model_mdn_loss,
    )
    stroke_model.fit(
        x_batches,
        y_batches,
        epochs=desired_epochs,
        batch_size=1,
        callbacks=[
            ExtendedModelCheckpoint(
                model_name="handwriting_prediction_single_batch_subset",
            )
        ],
    )
else:
    print("Model already exists")

generate_and_plot_heatmap(
    stroke_model,
    "handwriting_prediction_single_batch",
    x_batches,
    reconstructed_data,
    num_mixture_components,
    model_pred_dir,
)
