import json
import os
from typing import Any

import tensorflow as tf


def load_model_if_exists(model_save_path, custom_objects: dict[str, Any]):
    try:
        model = tf.keras.models.load_model(
            model_save_path,
            custom_objects=custom_objects,
        )
        print("successfully loaded model")
        return (
            model,
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
