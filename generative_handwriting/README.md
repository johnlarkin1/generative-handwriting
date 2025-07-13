# Handwriting Synthesis with LSTM Networks

This project implements the handwriting synthesis model described in Alex Graves' paper ["Generating Sequences With Recurrent Neural Networks"](https://arxiv.org/abs/1308.0850). The model can generate realistic handwriting in various styles and convert text into handwritten form.

## Overview

This implementation uses LSTM networks with attention mechanisms to generate handwriting that mimics human writing patterns. The model can:

- Generate handwriting synthesis from text input
- Learn and reproduce different handwriting styles
- Generate realistic-looking handwritten text with natural variations

## Project Structure

```
.
├── alphabet.py           # Alphabet encoding and processing
├── callbacks.py         # Custom training callbacks
├── common.py           # Common utility functions
├── config.py           # Configuration settings
├── constants.py        # Project constants
├── drawing.py          # Drawing utilities
├── loader.py           # Data loading and preprocessing
├── model/              # Model architecture components
├── model_io.py         # Model saving and loading utilities
├── plotting.py         # Visualization utilities
├── train_handwriting_prediction.py  # Training script for prediction
├── train_handwriting_synthesis.py   # Training script for synthesis
├── visualize.py        # Visualization tools
└── writer.py           # Main handwriting generation class
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- svgwrite (for rendering output)

## Key Components

### Calligrapher Class

The `Calligrapher` class in `writer.py` is the main interface for generating handwritten text. It handles:

- Loading trained models
- Text to handwriting conversion
- SVG rendering of generated handwriting

### Model Architecture

The implementation includes:

- LSTM-based sequence generation
- Attention mechanism for character-level focus
- Mixture Density Network (MDN) for stroke prediction
- Bivariate Gaussian mixtures for coordinate generation

## Usage

### Training

To train the handwriting synthesis model:

```python
python train_handwriting_synthesis.py
```

To train the handwriting prediction model:

```python
python train_handwriting_prediction.py
```

### Generating Handwriting

```python
from writer import Calligrapher

# Initialize the calligrapher with a trained model
calligrapher = Calligrapher(
    model_path="path/to/model",
    num_output_mixtures=20
)

# Generate handwriting
calligrapher.write(
    lines=["Hello, World!"],
    filename="output.svg",
    temperature=0.3,  # Controls randomness
    show_grid=True    # Optional grid display
)
```

## Model Parameters

- `NUM_ATTENTION_GAUSSIAN_COMPONENTS`: Number of Gaussian components in the attention mechanism
- `NUM_BIVARIATE_GAUSSIAN_MIXTURE_COMPONENTS`: Number of mixture components for coordinate prediction
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Initial learning rate
- `GRADIENT_CLIP_VALUE`: Maximum gradient norm for stability

## Implementation Details

The model uses:

- Mixture Density Networks for modeling stroke distributions
- Attention mechanism for character-level focus during synthesis
- Custom training callbacks for model checkpointing and parameter tracking
- SVG-based rendering for high-quality output

## Output Customization

The handwriting generation can be customized with parameters like:

- `temperature`: Controls randomness in generation
- `bias`: Adjusts the bias in mixture component selection
- `stroke_colors`: Customizes stroke colors
- `stroke_widths`: Adjusts stroke thickness
- `show_grid`: Displays reference grid
- `show_endpoints`: Shows stroke endpoints

## References

1. Graves, A. (2013). ["Generating Sequences With Recurrent Neural Networks"](https://arxiv.org/abs/1308.0850)
2. Implementation based on the architecture described in the paper, with modifications for improved stability and performance

## License

This project is open-source and available for research and development purposes.
