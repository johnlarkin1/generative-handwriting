"""
This module provides various hyper parameters and also 
architecture constraints similar to Graves's paper.
"""

# Preprocessing Data Threshold

STROKE_SPACE_THRESHOLD = 500

STROKE_SCALE_FACTOR = 10

# Hyperparameters

EPSILON = 1e-6

MAX_STROKE_LEN = 1200

NUM_LSTM_HIDDEN_LAYERS = 3

NUM_LSTM_CELLS_PER_HIDDEN_LAYER = 400

NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS = 20

NUM_MDN_COMPONENTS = 2

DROPOUT_RATE = 0.2

INITIAL_STANDARD_DEVIATION = 0.075

# We have 2 dimensions for each bivariate Gaussian... hence bi
NUM_BIVARIATE_OUTPUT_DIMENSION = 2

LSTM_OUTPUT_DIMS = 3

NUM_MIXTURE_COMPONENTS_PER_COMPONENT = 6

# Training Parameters

NUM_EPOCH = 1000

BATCH_SIZE = 64

ADAM_CLIP_NORM = 1.0

LEARNING_RATE = 5e-5


# Test Parameters

TEST_NUM_MIXTURES = 5
TEST_SEQUENCE_LENGTH = 50
TEST_NUM_POINTS = 1000
TEST_NUM_EPOCHS = 200
TEST_BATCH_SIZE = 128
