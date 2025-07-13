"""A tiny decoder-only transformer: the world model.

Nothing here is specific to RLVR -- this is the autoregressive model that the
paper trains with maximum likelihood and then post-trains with RLVR.  A
question ``q(s, a)`` (the rendered state followed by the action) is the prompt;
the response ``o(s')`` is the rendered next state followed by ``<eos>``.

Attention is written out by hand rather than using a fused kernel, because the
tests need to poke at the causal mask directly.  At the default configuration
the model has ~0.3M parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

NEG_INF = -1e9


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 96
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 256
    max_len: int = 64

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


@lru_cache(maxsize=32)
def causal_mask(length: int) -> mx.array:
    """Additive ``(1, 1, T, T)`` mask: position t may attend to positions <= t."""
    rows = mx.arange(length)[:, None]
    cols = mx.arange(length)[None, :]
    mask = mx.where(cols <= rows, mx.array(0.0), mx.array(NEG_INF))
    return mask.reshape(1, 1, length, length)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        q = mx.transpose(qkv[:, :, 0], (0, 2, 1, 3))
        k = mx.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        v = mx.transpose(qkv[:, :, 2], (0, 2, 1, 3))
        scores = (q * self.scale) @ mx.swapaxes(k, -1, -2) + mask
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v  # (B, H, T, head_dim)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(batch, length, -1)
        return self.out(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


class WorldModel(nn.Module):
    """Maps token ids ``(B, T)`` to next-token logits ``(B, T, vocab)``."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def __call__(self, ids: mx.array) -> mx.array:
        if ids.ndim != 2:
            raise ValueError("expected token ids of shape (batch, length)")
        _, length = ids.shape
        if length > self.cfg.max_len:
            raise ValueError(
                f"sequence of length {length} exceeds max_len {self.cfg.max_len}"
            )
        positions = mx.arange(length)
        x = self.tok_emb(ids) + self.pos_emb(positions)[None, :, :]
        mask = causal_mask(length)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))

    # ----------------------------------------------------------------- helpers
    def logprobs(self, ids: mx.array) -> mx.array:
        """Log-softmax of the next-token distribution, ``(B, T, vocab)``."""
        return nn.log_softmax(self(ids), axis=-1)

    def num_params(self) -> int:
        from mlx.utils import tree_flatten

        return int(sum(v.size for _, v in tree_flatten(self.parameters())))


def token_accuracy(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    """Fraction of masked positions where ``argmax(logits) == target``."""
    predicted = mx.argmax(logits, axis=-1)
    hits = (predicted == targets).astype(mx.float32) * mask
    return hits.sum() / mx.maximum(mask.sum(), mx.array(1.0))
