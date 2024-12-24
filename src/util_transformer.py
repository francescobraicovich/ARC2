# util_transformer.py

# Import necessary libraries
import torch
from dataclasses import dataclass, field
from typing import Optional, Any

# Configuration for a single Transformer layer
@dataclass
class TransformerLayerConfig:
    """Configuration for a single Transformer layer."""
    num_heads: int = 8  # Number of attention heads
    emb_dim_per_head: int = 16  # Embedding dimension per head
    mlp_dim_factor: float = 4.0  # Factor to determine MLP dimension
    dropout_rate: float = 0.0  # Dropout rate for the layer
    attention_dropout_rate: float = 0.0  # Dropout rate for attention
    use_bias: bool = False  # Whether to use bias in linear layers
    activation: str = "relu"  # Activation function ("relu" or "silu")
    dtype: Any = torch.float32  # Data type for tensors

    emb_dim: int = field(init=False)  # Total embedding dimension

    def __post_init__(self):
        # Calculate total embedding dimension
        object.__setattr__(self, "emb_dim", self.num_heads * self.emb_dim_per_head)

# Configuration for the entire Transformer Encoder
@dataclass
class EncoderTransformerConfig:
    """
    Configuration for the entire Transformer Encoder.
    
    - You can pass `emb_dim` to override the default (128).
    - `transformer_layer` is optional; if None, a default is created.
    """
    vocab_size: int = 11  # Size of the vocabulary (10 colors + 1 for padding)
    #output_vocab_size: int = 10  # Output vocabulary size (unused in encoder)
    num_layers: int = 2  # Number of Transformer layers
    latent_dim: int = 256  # Dimension of the latent space
    variational: bool = False  # Whether to use a variational approach
    max_rows: int = 30  # Maximum number of rows
    max_cols: int = 30  # Maximum number of columns
    latent_projection_bias: bool = False  # Bias in latent projection
    scaled_position_embeddings: bool = False  # Use scaled position embeddings
    transformer_dropout: float = 0.0  # Dropout rate for the Transformer
    emb_dim: int = 96  # Embedding dimension

    transformer_layer: Optional[TransformerLayerConfig] = field(
        default_factory=lambda: TransformerLayerConfig()
    )  # Configuration for Transformer layers

    max_len: int = field(init=False)  # Maximum sequence length
    dtype: Any = field(init=False)  # Data type for tensors

    def __post_init__(self):
        # Update embedding dimensions based on transformer layer config
        if self.transformer_layer is not None:
            emb_dim_per_head = max(1, self.emb_dim // self.transformer_layer.num_heads)
            object.__setattr__(self.transformer_layer, 'emb_dim_per_head', emb_dim_per_head)
            object.__setattr__(
                self.transformer_layer,
                'emb_dim',
                self.transformer_layer.num_heads * emb_dim_per_head
            )
        # Set the data type
        object.__setattr__(self, "dtype", self.transformer_layer.dtype)
        # Calculate the maximum sequence length
        object.__setattr__(self, "max_len", self.max_rows * self.max_cols)