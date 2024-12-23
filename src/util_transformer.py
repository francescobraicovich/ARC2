# util_transformer.py

import torch
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class TransformerLayerConfig:
    """Configuration for a single Transformer layer."""
    num_heads: int = 8
    emb_dim_per_head: int = 16
    mlp_dim_factor: float = 4.0
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    use_bias: bool = False
    activation: str = "relu"  # or "silu"
    dtype: Any = torch.float32  # Unused in PyTorch, but kept for consistency

    emb_dim: int = field(init=False)

    def __post_init__(self):
        # emb_dim = num_heads * emb_dim_per_head
        object.__setattr__(self, "emb_dim", self.num_heads * self.emb_dim_per_head)


@dataclass
class EncoderTransformerConfig:
    """
    Configuration for the entire Transformer Encoder.
    
    - You can pass `emb_dim` to override the default (128).
    - `transformer_layer` is optional; if None, a default is created.
    """
    vocab_size: int = 10
    output_vocab_size: int = 10  # not used in the encoder, but kept for parity
    num_layers: int = 2
    latent_dim: int = 32
    variational: bool = False
    max_rows: int = 30
    max_cols: int = 30
    latent_projection_bias: bool = False
    scaled_position_embeddings: bool = False
    transformer_dropout: float = 0.0

    # Let the user specify an emb_dim. If they don't, default = 128
    emb_dim: int = 128

    # Provide a default factory for the layer config,
    # so we don't run into mutable default errors
    transformer_layer: Optional[TransformerLayerConfig] = field(
        default_factory=lambda: TransformerLayerConfig()
    )

    # We'll compute max_len in __post_init__. We'll also sync
    # emb_dim to the sub-layer if we want them to match.
    max_len: int = field(init=False)
    dtype: Any = field(init=False)  # store the final dtype if needed

    def __post_init__(self):
        # If user didn't pass a custom transformer_layer, we have the default one from default_factory

        # If you want to keep your `emb_dim` in sync with the transformer's config:
        # compute emb_dim_per_head or just do a direct override. Example:
        if self.transformer_layer is not None:
            # We'll set emb_dim_per_head = emb_dim // num_heads
            # so that emb_dim = num_heads * emb_dim_per_head
            emb_dim_per_head = max(1, self.emb_dim // self.transformer_layer.num_heads)
            object.__setattr__(self.transformer_layer, 'emb_dim_per_head', emb_dim_per_head)
            object.__setattr__(
                self.transformer_layer,
                'emb_dim',
                self.transformer_layer.num_heads * emb_dim_per_head
            )

        # mirror dtype
        object.__setattr__(self, "dtype", self.transformer_layer.dtype)

        # compute max_len
        object.__setattr__(self, "max_len", self.max_rows * self.max_cols)