import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from typing import Optional, Tuple

# Go up one directory to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.util import set_device

DEVICE = set_device('transformer.py')

class EncoderTransformerConfig:
    """
    Configuration object for the Encoder Transformer.
    Defines hyperparameters and settings used throughout the model.
    """
    def __init__(
        self,
        vocab_size: int = 11,                       # Size of the input vocabulary (e.g., number of token types) + 1 for padding
        max_rows: int = 30,                         # Max number of rows (e.g., height in a grid input)
        max_cols: int = 30,                         # Max number of columns (e.g., width in a grid input)
        emb_dim: int = 32,                          # Dimensionality of token embeddings and model hidden states
        latent_dim: int = 128,                      # Dimensionality of latent space (used in VAE settings)
        num_layers: int = 2,                        # Number of transformer encoder layers
        num_heads: int = 8,                         # Number of attention heads in multi-head self-attention
        attention_dropout_rate: float = 0.1,        # Dropout probability applied within the attention layer
        dropout_rate: float = 0.1,                  # Dropout probability for all other dropout layers
        mlp_dim_factor: int = 3,                    # Expansion factor for the MLP hidden layer (usually 4× emb_dim)
        use_bias: bool = True,                      # Whether to use bias terms in linear layers
        activation: str = "silu",                   # Activation function used in MLP block: 'relu', 'gelu', or 'silu'
        scaled_position_embeddings: bool = False,   # Whether to scale positional encodings (for stability)
        variational: bool = False,                  # Whether to use variational components (for VAE-style training)
        latent_projection_bias: bool = False,       # Whether to include a bias in the latent projection layer
        dtype: torch.dtype = torch.float32,         # Default data type for model tensors
    ):
        self.vocab_size = vocab_size
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_dropout_rate = attention_dropout_rate
        self.dropout_rate = dropout_rate
        self.mlp_dim_factor = mlp_dim_factor
        self.use_bias = use_bias
        self.activation = activation
        self.scaled_position_embeddings = scaled_position_embeddings
        self.variational = variational
        self.latent_projection_bias = latent_projection_bias
        self.dtype = dtype

        # Derived property: total sequence length (flattened grid)
        self.max_len = max_rows * max_cols

class MlpBlock(nn.Module):
    """
    Position-wise feed-forward network used in transformers.
    """
    def __init__(self, config: EncoderTransformerConfig):
        super().__init__()
        hidden_dim = int(config.mlp_dim_factor * config.emb_dim)

        self.fc1 = nn.Linear(config.emb_dim, hidden_dim, bias=config.use_bias)
        self.fc2 = nn.Linear(hidden_dim, config.emb_dim, bias=config.use_bias)

        if config.activation == "relu":
            self.activation_fn = F.relu
        elif config.activation == "gelu":
            self.activation_fn = F.gelu
        elif config.activation == "silu":
            self.activation_fn = F.silu
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")

        self.dropout_rate = config.dropout_rate

    def forward(self, x: torch.Tensor, dropout_eval: bool = False) -> torch.Tensor:
        dropout_p = 0.0 if dropout_eval else self.dropout_rate
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=dropout_p, training=not dropout_eval)
        x = self.fc2(x)
        return x

class TransformerLayer(nn.Module):
    """
    A single transformer encoder block with improved layer normalization and residual scaling.
    """
    def __init__(self, config: EncoderTransformerConfig):
        super().__init__()
        self.config = config

        # Use learnable layer norms (elementwise_affine=True)
        self.ln1 = nn.LayerNorm(config.emb_dim, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(config.emb_dim, elementwise_affine=True)

        self.mha = nn.MultiheadAttention(
            embed_dim=config.emb_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout_rate,
            bias=config.use_bias,
            batch_first=True,
        )

        self.mlp_block = MlpBlock(config)
        self.resid_dropout = nn.Dropout(p=config.dropout_rate)

        # Learnable scaling factors for the residual connections
        self.res_scale_attn = nn.Parameter(torch.ones(1))
        self.res_scale_mlp = nn.Parameter(torch.ones(1))

    def forward(
        self,
        embeddings: torch.Tensor,
        dropout_eval: bool = False,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask = None
        key_padding_mask = None

        if pad_mask is not None:
            if pad_mask.dim() == 3:
                attn_mask = pad_mask
            elif pad_mask.dim() == 2:
                key_padding_mask = pad_mask

        # Pre-normalization before attention
        normed_embeddings = self.ln1(embeddings)
        attn_output, _ = self.mha(
            normed_embeddings, normed_embeddings, normed_embeddings,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        dropout_p = 0.0 if dropout_eval else self.config.dropout_rate
        attn_output = F.dropout(attn_output, p=dropout_p, training=not dropout_eval)
        # Residual connection with learnable scaling
        embeddings = embeddings + self.res_scale_attn * attn_output

        # Pre-normalization before MLP
        normed_embeddings = self.ln2(embeddings)
        mlp_out = self.mlp_block(normed_embeddings, dropout_eval=dropout_eval)
        mlp_out = F.dropout(mlp_out, p=dropout_p, training=not dropout_eval)
        # Residual connection with learnable scaling
        embeddings = embeddings + self.res_scale_mlp * mlp_out

        return embeddings