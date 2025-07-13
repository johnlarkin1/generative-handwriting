# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a handwriting synthesis project that implements Alex Graves' paper "Generating Sequences With Recurrent Neural Networks". The model uses LSTM networks with attention mechanisms to generate realistic handwriting from text input.

## Commands

### Dependency Management
```bash
uv sync                        # Install dependencies
uv sync --extra dev            # Install with dev dependencies
uv run <command>               # Run command in virtual environment
```

### Code Quality
```bash
ruff check                     # Run linting
ruff format                    # Format code
mypy generative_handwriting    # Type checking
```

### Testing
```bash
pytest                         # Run all tests
pytest -m unit_test           # Run unit tests only
pytest -m integration_test    # Run integration tests only
pytest -m end_to_end_test     # Run end-to-end tests only
```

### Training
```bash
python generative_handwriting/train_handwriting_prediction.py    # Train prediction model
python generative_handwriting/train_handwriting_synthesis.py     # Train synthesis model
```

## Architecture

### Core Components

- **Model Architecture** (`generative_handwriting/model/`):
  - `handwriting_models.py`: Main model classes (SimpleLSTMModel, HandwritingSynthesisModel, etc.)
  - `attention_mechanism.py`: Attention mechanism implementation
  - `attention_rnn_cell.py`: Custom RNN cell with attention
  - `lstm_peephole_cell.py`: LSTM cell with peephole connections
  - `mixture_density_network.py`: Mixture Density Network for coordinate prediction
  - `basic_mdn.py`: Basic MDN implementation

- **Data Processing**:
  - `loader.py`: Data loading and preprocessing
  - `alphabet.py`: Text encoding and alphabet handling
  - `constants.py`: Model hyperparameters and constants

- **Training & Inference**:
  - `writer.py`: Main `Calligrapher` class for generating handwritten text
  - `train_handwriting_*.py`: Training scripts
  - `model_io.py`: Model saving/loading utilities

- **Visualization**:
  - `plotting.py`: Training visualization utilities
  - `visualize.py`: Model output visualization
  - `drawing.py`: Drawing utilities for SVG output

### Configuration

- `config.py`: `HandwritingConfig` dataclass for model architecture parameters (LSTM units, mixture components, etc.)
- `training_config.py`: `TrainingConfig` dataclass for training hyperparameters (batch size, learning rate, etc.)

### Key Design Patterns

1. **Mixture Density Networks**: Used for modeling stroke coordinate distributions
2. **Attention Mechanism**: Character-level attention during synthesis
3. **Custom LSTM Cells**: Peephole connections for improved gradient flow
4. **SVG Rendering**: High-quality vector output for generated handwriting

### Test Structure

Tests are located in `generative_handwriting/test/` and are organized by markers:
- `unit_test`: Tests individual functions in isolation
- `integration_test`: Tests with external infrastructure
- `end_to_end_test`: Full application tests

## Development Notes

- The project uses uv for dependency management
- PyTorch is the primary ML framework
- Code style enforced by Ruff (line length 120)
- Type checking with MyPy
- The `archived/` directory contains previous experiments and visualizations

## Data

Training data is located in `generative_handwriting/data/` with train/validation/test splits in text files.