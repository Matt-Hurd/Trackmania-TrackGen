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

    def setup(self):
        self.layer_norm1 = nn.LayerNorm(dtype=self.config.dtype)
        self.self_attention = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            dtype=self.config.dtype,
            qkv_features=self.config.d_model,
            dropout_rate=self.config.attention_dropout_rate,
            deterministic=self.config.deterministic,
        )
        self.dropout1 = nn.Dropout(rate=self.config.dropout_rate)

        self.layer_norm2 = nn.LayerNorm(dtype=self.config.dtype)
        self.ffn = nn.Sequential([
            nn.Dense(self.config.mlp_dim),
            nn.relu,
            nn.Dense(self.config.d_model)
        ])
        self.dropout2 = nn.Dropout(rate=self.config.dropout_rate)

    def __call__(self, x, train: bool = True):
        # Self-Attention Block
        norm_x = self.layer_norm1(x)
        attn_output = self.self_attention(norm_x, norm_x, norm_x)
        attn_output = self.dropout1(attn_output, deterministic=not train)
        x = x + attn_output

        # Feed-Forward Network (FFN) Block
        norm_x = self.layer_norm2(x)
        ffn_output = self.ffn(norm_x)
        ffn_output = self.dropout2(ffn_output, deterministic=not train)
        x = x + ffn_output

        return x
