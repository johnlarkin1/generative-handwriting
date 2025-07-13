import os

import numpy as np
from common import print_model_parameters
from tensorflow.keras.callbacks import Callback, ModelCheckpoint


class PrintModelParametersCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            print("Model parameters after the 1st epoch:")
            self.model.summary()
            print_model_parameters(self.model)


class ExtendedModelCheckpoint(ModelCheckpoint):
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(self.base_dir, "saved_models", model_name)
        os.makedirs(model_dir, exist_ok=True)
        filepath = os.path.join(model_dir, "best_model.keras")
        super().__init__(filepath, save_best_only=True, monitor="loss", mode="min", **kwargs)
        self.last_best = None

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        current_loss = logs.get("loss")
        if self.last_best is None or current_loss < self.last_best:
            print(f"\nEpoch {epoch + 1}: Loss improved from {self.last_best} to {current_loss}, saving model.")
            self.last_best = current_loss
        super().on_epoch_end(epoch, logs)


class ModelCheckpointWithPeriod(ModelCheckpoint):
    def __init__(self, model_name: str, period: int, **kwargs) -> None:
        self.model_name = model_name
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.period = period
        model_dir = os.path.join(self.base_dir, "saved_models", model_name)
        os.makedirs(model_dir, exist_ok=True)
        filepath = os.path.join(model_dir, "model_{epoch:02d}_{loss:.2f}.keras")
        super().__init__(filepath, save_best_only=False, **kwargs)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        # Adjusting to ensure it saves on epoch 1 then every 'period' epochs thereafter
        if (epoch + 1) % self.period == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}: Saving model.")
            super().on_epoch_end(epoch, logs)


class HandwritingVisualizeCallback(Callback):
    """
    This callback is meant for data that has not been batched.
    It's meant to be called as a callback, so that we can visualize for the entire
    predicted, how the heatmap is changing across all points. So if you have a seq length
    of 1000, you should see roughly 1000 different distributions.
    """

    def __init__(
        self,
        input_data: np.ndarray,
        num_mixtures: int,
        **kwargs,
    ) -> None:
        super(HandwritingVisualizeCallback, self).__init__(**kwargs)
        self.input_data = input_data
        self.num_mixtures = num_mixtures

        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.image_dir = os.path.join(self.base_dir, "handwriting_visualizations")
        assert self.input_data.shape[1] > 0, "Input data must have at least 1 sequence"
        assert self.input_data.shape[2] == 3, "Input data must have 3 features"

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        """
        This callback is going to be called at the end of each epoch.
        It's going to generate a plot of the input data and the predicted data.

        The output data is going to be a mixture density network, so we'll have to
        convert the MDN parameters to bivariate Gaussian distributions.

        We'll grab a single sequence and plot the input data and the predicted data as a heatmap for each point.
        """
        pass
        # print(f"Epoch {epoch}")
        # predictions = self.model.predict(self.input_data)
        # input_batch = self.input_data[0]
        # predictions = predictions[0]
        # input_sequence = input_batch[0]
        # predicted_sequence = predictions[0]

        # Convert the MDN parameters to bivariate Gaussian distributions
        # We'll have to grab the parameters for each point in the sequence
        # The parameters are ordered as follows:
        # - Mixture weights
        # - Means for x
        # - Means for y
        # - Standard deviations for x
        # - Standard deviations for y
        # - Correlation
        # - End of stroke probability
        # We'll have to grab the parameters for each point in the sequence.
        # predicted_sequence will be [sequence_length, num_mixtures * 6 + 1]
