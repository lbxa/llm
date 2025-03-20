# training data
# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import GPTConfig


class Head(nn.Module):
    """
    Attention head
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embed is not None
        assert config.head_size is not None
        assert config.block_size is not None
        assert config.dropout is not None

        self.key = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.query = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k: torch.Tensor = self.key(x)
        q: torch.Tensor = self.query(x)
        wei: torch.Tensor = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, 16) @ (B, 16, T) => (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v: torch.Tensor = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of attention running in parallel

    One is not enough
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embed is not None
        assert config.head_size is not None
        assert config.n_head is not None
        assert config.dropout is not None

        self.heads = nn.ModuleList([Head(config=config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embed is not None
        assert config.dropout is not None

        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    """
    Self-attention + feed forward blocks are stacked for more housepower
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embed is not None
        assert config.n_head is not None
        assert config.dropout is not None

        self.sa = MultiHeadAttention(config=config)
        self.ffw = FeedForward(config=config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # addition -> residual pathways (DL optimisation technique)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.n_embed is not None
        assert config.block_size is not None

        self.config = config

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(
            *[Block(config=config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        self.device = config.device

    def get_param_count(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding_table.weight.numel()
            n_params -= self.position_embedding_table.weight.numel()
        return n_params

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B,T,n_embd)
        pos = torch.arange(0, T, device=self.device)  # (T)
        position_embeddings = self.position_embedding_table(pos)  # (T,n_embd)
        x = token_embeddings + position_embeddings  # (B,T,n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        idx = idx.to(self.device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # becomes (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T + 1)

        return idx
