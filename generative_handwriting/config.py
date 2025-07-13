from dataclasses import dataclass


@dataclass
class HandwritingConfig:
    num_lstm_layers: int = 3
    lstm_units: int = 400
    num_mixture_components: int = 20
    num_attention_components: int = 10
    style_dimension: int = 256
    dropout_rate: float = 0.2
