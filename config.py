import torch
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    # Training/validation split ratio
    training_split: float
    # Dimension of embedding vectors
    n_embed: int
    # Number of attention heads in each layer
    n_head: int
    # Number of transformer layers
    n_layer: int
    # Maximum context length (sequence length)
    block_size: int
    # Number of samples processed in parallel
    batch_size: int
    # Dimension of each attention head (n_embed // n_head)
    head_size: int
    # Learning rate for optimizer
    learning_rate: float
    # Dropout probability for regularization
    dropout: float
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        assert self.n_head * self.head_size == self.n_embed, (
            "n_head * head_size must equal n_embed"
        )


@dataclass
class GPT10MConfig(GPTConfig):
    """Configuration for 10M parameter GPT model"""

    vocab_size: int = 65
    training_split: float = 0.9
    n_embed: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    batch_size: int = 64
    head_size: int = 64
    learning_rate: float = 3e-4
    dropout: float = 0.2


@dataclass
class GPTSmallConfig(GPTConfig):
    """Configuration for small GPT model"""

    vocab_size: int = 65
    training_split: float = 0.9
    n_embed: int = 32
    n_head: int = 4
    n_layer: int = 6
    block_size: int = 8
    batch_size: int = 32
    head_size: int = 8
    learning_rate: float = 1e-3
    dropout: float = 0.2
