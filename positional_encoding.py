import jax.numpy as jnp
import flax.linen as nn

class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    def setup(self):
        # Compute the positional encodings once in log space.
        pe = jnp.zeros((self.max_len, self.d_model), dtype=jnp.float32)
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe[None, :, :]

    def __call__(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return x