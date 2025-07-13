import numpy as np
from common import (
    create_subsequence_batches,
    plot_strokes_from_dx_dy,
    prepare_data_for_sequential_prediction,
)
from constants import LEARNING_RATE, TEST_NUM_MIXTURES
from loader import HandwritingDataLoader

strokes, stroke_lengths = HandwritingDataLoader().load_individual_stroke_data("a01/a01-000/a01-000u-01.xml")
# plot_original_strokes_from_xml("a01/a01-000/a01-000u-01.xml")
reconstructed_data = plot_strokes_from_dx_dy(strokes, show_image=False)

x_stroke, y_stroke = prepare_data_for_sequential_prediction(strokes)
sequence_length = 50
desired_epochs = 5

# Prepare your data for training
x_batches, y_batches = create_subsequence_batches(x_stroke, y_stroke, sequence_length)
num_mixture_components = TEST_NUM_MIXTURES
learning_rate = LEARNING_RATE
sequence_of_interest = 4

x_train_stroke = np.reshape(x_stroke, (1, x_stroke.shape[0], x_stroke.shape[1]))
y_train_stroke = np.reshape(y_stroke, (1, y_stroke.shape[0], y_stroke.shape[1]))

print("Shapes")
print(x_batches.shape, y_batches.shape)

print(f"Sequence of interest: {sequence_of_interest}")
print(np.array2string(x_batches[sequence_of_interest], separator=", "))
print(np.array2string(y_batches[sequence_of_interest], separator=", "))

# print(f"Sequence of interest: {sequence_of_interest}")
# print(x_batches[55, :, :])
# x_max_index = np.unravel_index(np.argmax(x_batches[:, :, 0]), x_batches[:, :, 0].shape)
# print(
#     f"Max value of x_batches[:, :, 0] is {x_batches[:, :, 0].max()} at index {x_max_index}"
# )
# # print(y_batches[55, :, :])
