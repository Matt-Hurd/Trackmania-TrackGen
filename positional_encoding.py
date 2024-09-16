import jax.numpy as jnp
import flax.linen as nn

class PositionalEncoding(nn.Module):
    d_model: int  # Dimension of the embeddings
    max_len: int = 5000  # Maximum length of the sequence

    def setup(self):
        # Create a matrix of shape (max_len, d_model) with positional encodings
        position = jnp.arange(self.max_len)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * -(jnp.log(10000.0) / self.d_model))
        pe = jnp.zeros((self.max_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe  # Shape: (max_len, d_model)

    def __call__(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, timesteps, d_model)
        Returns:
            Tensor with positional encoding added
        """
        timesteps = x.shape[1]
        return x + self.pe[:timesteps]
