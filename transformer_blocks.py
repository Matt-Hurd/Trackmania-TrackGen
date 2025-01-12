# transformer_blocks.py

from typing import Any
import flax.linen as nn
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    num_heads: int
    d_model: int
    mlp_dim: int
    dropout_rate: float
    attention_dropout_rate: float
    dtype: Any
    deterministic: bool
class TransformerEncoderBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, train=True):
        x = nn.LayerNorm(dtype=self.config.dtype)(inputs)
        x = nn.SelfAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.d_model,
            out_features=self.config.d_model,
            dropout_rate=self.config.attention_dropout_rate,
            deterministic=self.config.deterministic,
            dtype=self.config.dtype,
        )(x, training=train)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=self.config.deterministic)
        x = x + inputs

        y = nn.LayerNorm(dtype=self.config.dtype)(x)
        y = nn.Dense(features=self.config.mlp_dim, dtype=self.config.dtype)(y)
        y = nn.relu(y)
        y = nn.Dropout(rate=self.config.dropout_rate)(y, deterministic=self.config.deterministic)
        y = nn.Dense(features=self.config.d_model, dtype=self.config.dtype)(y)
        y = nn.Dropout(rate=self.config.dropout_rate)(y, deterministic=self.config.deterministic)

        return x + y