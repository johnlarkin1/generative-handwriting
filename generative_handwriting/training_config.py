from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.0005
    warmup_steps: int = 1000
    gradient_clip_norm: float = 5.0
    min_learning_rate: float = 1e-5
