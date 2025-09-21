# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a handwriting synthesis project that implements Alex Graves' paper "Generating Sequences With Recurrent Neural Networks". The model uses LSTM networks with attention mechanisms to generate realistic handwriting from text input.

## Commands

### Dependency Management

```bash
poetry install                        # Install dependencies
poetry install --with dev             # Install with dev dependencies
poetry run <command>                  # Run command in virtual environment
poetry shell                          # Activate virtual environment
```

### Code Quality

```bash
poetry run ruff check generative_handwriting          # Linting
poetry run ruff check --fix generative_handwriting    # Linting with auto-fix
poetry run ruff format generative_handwriting         # Format code
poetry run mypy generative_handwriting                # Type checking
```

### Testing

```bash
poetry run pytest                                 # Run all tests
poetry run pytest -m unit_test                    # Run unit tests only
poetry run pytest -m integration_test             # Run integration tests only
poetry run pytest -m end_to_end_test              # Run end-to-end tests only
poetry run pytest <test_file>::<test_function>    # Run single test
```

### Training

```bash
poetry run python generative_handwriting/train_handwriting_prediction.py    # Train prediction model
poetry run python generative_handwriting/train_handwriting_synthesis.py     # Train synthesis model
```

### Monitoring & Visualization

```bash
poetry run python generative_handwriting/monitor_training.py              # Check training status
poetry run python generative_handwriting/monitor_training.py --watch      # Live monitoring (every 60s)
poetry run python generative_handwriting/visualize_predictions.py         # Generate prediction heatmaps & GIFs
```

## Architecture

### Core Model Components (`generative_handwriting/model/`)

- **handwriting_models.py**: Main model classes

  - `SimpleLSTMModel`: Basic LSTM model for sequence prediction
  - `SimpleHandwritingPredictionModel`: Single-layer LSTM with peephole connections
  - `DeepHandwritingPredictionModel`: Multi-layer LSTM with MDN output
  - `HandwritingSynthesisModel`: Full synthesis model with attention
  - `DeepHandwritingSynthesisModel`: Advanced synthesis with attention mechanisms

- **attention_mechanism.py**: Character-level attention for text-to-handwriting alignment
- **attention_rnn_cell.py**: Custom RNN cell that integrates attention
- **lstm_peephole_cell.py**: LSTM cell with peephole connections for improved gradient flow
- **mixture_density_network.py**: MDN layer for probabilistic coordinate generation
- **basic_mdn.py**: Simplified MDN implementation for testing

### Data Processing & Training

- **loader.py**: Loads and preprocesses IAM handwriting dataset
- **alphabet.py**: Handles text encoding/decoding (ASCII alphabet support)
- **constants.py**: Global constants (model dimensions, batch size, etc.)
- **config.py**: `HandwritingConfig` dataclass for architecture parameters
- **model_io.py**: Model serialization and checkpoint management

### Inference & Visualization

- **writer.py**: `Calligrapher` class for generating handwritten text from strings
- **visualize.py**: Visualization utilities for model outputs
- **visualize_predictions.py**: Generate heatmaps and GIFs of prediction distributions
- **monitor_training.py**: Monitor training progress and test model predictions
- **drawing.py**: SVG generation for vector handwriting output

### Key Design Patterns

1. **Mixture Density Networks (MDN)**: Models the distribution of next pen position as a mixture of 2D Gaussians with correlation
2. **Attention Mechanism**: Learns alignment between text characters and handwriting strokes
3. **Peephole LSTM**: Direct connections from cell state to gates for better long-term dependencies
4. **Teacher Forcing**: During training, uses ground truth for next step prediction
5. **Biased Sampling**: During inference, adjusts sampling temperature for stroke generation

### Test Structure

Tests in `tests/` are organized by complexity:

- Please use pytest for our tests

## Data

Training data is located in `generative_handwriting/data/` with train/validation/test splits in text files. The project expects IAM handwriting dataset format.

## Reference Implementation

This repository includes `handwriting-synthesis/`, a **TensorFlow 1.x reference implementation** of the same paper by Sean Vasquez. This serves as a working baseline for comparison and debugging.

### Project Structure

```
handwriting-synthesis/                    # Reference TensorFlow implementation
├── demo.py                               # Main interface (Hand class)
├── rnn.py                                # Core LSTM+attention model
├── tf_base_model.py                      # Training infrastructure
├── rnn_cell.py                           # Custom LSTM cell with attention
├── drawing.py                            # Stroke processing & SVG generation
├── checkpoints/                          # Pre-trained model weights
└── styles/                               # Pre-computed style vectors (0-12)
```

### Key Differences from Our Implementation

- **Framework**: TensorFlow 1.x vs. our modern architecture
- **Pre-trained**: Includes working model checkpoints and style vectors (0-12)
- **Interface**: `Hand` class provides simple text-to-SVG generation
- **Limitations**: 75 chars/line, missing some punctuation, TF 1.x dependency

### Usage for Comparison

```python
# Generate samples using reference implementation
from handwriting_synthesis.demo import Hand

hand = Hand()
hand.write(
    filename='reference_output.svg',
    lines=['Your text here'],
    biases=[0.75],    # 0-1 (neatness)
    styles=[9],       # 0-12 (writer style)
)
```

### Value for Development

- **Debugging**: Compare outputs between implementations
- **Hyperparameters**: Reference working settings
- **Architecture validation**: Cross-check model topology
- **Quality baseline**: Expected output quality benchmarks
