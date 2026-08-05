from dataclasses import dataclass
import torch

def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
@dataclass
class BaseConfig:
    DEVICE: str = get_default_device()
    max_norm: float = 1.0
@dataclass
class TrainingConfig:

    TIMESTEPS: int = 1000
    IMG_SHAPE: tuple[int, int, int] = (2, 16, 32)
    NUM_EPOCHS: int = 1600
    BATCH_SIZE: int = 128
    LR: float = 1e-4
    NUM_WORKERS: int = 0
    Print_step: int = 10

@dataclass
class ModelConfig:
    BASE_CH: int = 64
    BASE_CH_MULT: tuple[int, ...] = (1, 2, 4, 4)
    APPLY_ATTENTION: tuple[bool, ...] = (True, False, False, True)
    DROPOUT_RATE: float = 0.1
    TIME_EMB_MULT: int = 4
