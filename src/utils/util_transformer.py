import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TransformerLayerConfig:
    """
    Global hyperparameters used to minimize obnoxious kwarg plumbing.
    """
    num_heads: int = 2
    emb_dim_per_head: int = 8
    mlp_dim_factor: float = 1.0
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    use_bias: bool = False
    activation: str = "silu"
    dtype: Any = torch.float32
    emb_dim: int = field(default=None)

    def __post_init__(self):
        # Total embedding dimension is num_heads * emb_dim_per_head
        object.__setattr__(self, "emb_dim", self.num_heads * self.emb_dim_per_head)


@dataclass
class EncoderTransformerConfig:
    """
    Global hyperparameters used to minimize obnoxious kwarg plumbing.
    """
    transformer_layer: TransformerLayerConfig = field(default_factory=TransformerLayerConfig)
    vocab_size: int = 10
    output_vocab_size: int = 10
    num_layers: int = 1
    latent_dim: int = 96
    variational: bool = False
    max_rows: int = 30
    max_cols: int = 30
    latent_projection_bias: bool = False
    scaled_position_embeddings: bool = False
    dtype: Any = field(default=None)
    emb_dim: int = field(default=None)
    max_len: int = field(default=None)

    def __post_init__(self):
        # Set defaults from the transformer layer config
        object.__setattr__(self, "dtype", self.transformer_layer.dtype)
        object.__setattr__(self, "emb_dim", self.transformer_layer.emb_dim)
        object.__setattr__(self, "max_len", self.max_rows * self.max_cols)


class MlpBlock(nn.Module):
    """
    Transformer MLP / feed-forward block.
    """

    def __init__(self, config: TransformerLayerConfig):
        super().__init__()
        self.config = config

        # Determine activation
        if config.activation == "relu":
            self.activation_fn = F.relu
        elif config.activation == "silu":
            # In PyTorch, silu is equivalent to the SiLU activation
            self.activation_fn = F.silu
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")

        # First linear layer: emb_dim -> (mlp_dim_factor * emb_dim)
        self.fc1 = nn.Linear(
            in_features=config.emb_dim,
            out_features=int(config.mlp_dim_factor * config.emb_dim),
            bias=config.use_bias,
        )

        # Second linear layer: (mlp_dim_factor * emb_dim) -> emb_dim
        self.fc2 = nn.Linear(
            in_features=int(config.mlp_dim_factor * config.emb_dim),
            out_features=config.emb_dim,
            bias=config.use_bias,
        )

        # Dropout is often set in forward pass if we want to control rate dynamically.
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(self, inputs: torch.Tensor, dropout_eval: bool) -> torch.Tensor:
        """
        Args:
            inputs: shape (..., emb_dim)
            dropout_eval: if True, disables dropout (as an alternative to model.eval())
        """
        return inputs
        if dropout_eval:
            dropout_p = 0.0
        else:
            dropout_p = self.config.dropout_rate

        x = self.fc1(inputs)
        x = self.activation_fn(x)
        x = self.fc2(x)
        # We can use nn.functional.dropout here to set p dynamically
        x = F.dropout(x, p=dropout_p, training=not dropout_eval)
        return x


class TransformerLayer(nn.Module):
    """
    Transformer encoder layer.
    """

    def __init__(self, config: TransformerLayerConfig):
        super().__init__()
        self.config = config

        # LayerNorm without learnable affine params if we want use_scale=False
        self.ln1 = nn.LayerNorm(config.emb_dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(config.emb_dim, elementwise_affine=False)

        # MultiheadAttention:
        #   - embed_dim = config.emb_dim
        #   - num_heads = config.num_heads
        #   - dropout = attention_dropout_rate
        #   - bias = config.use_bias
        #   - batch_first = True to handle shape (batch, seq_len, emb_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=config.emb_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout_rate,
            bias=config.use_bias,
            batch_first=True,
        )

        # MLP block
        self.mlp_block = MlpBlock(config)

        # Dropout to match the config
        self.resid_dropout = nn.Dropout(p=config.dropout_rate)

    def forward(
        self,
        embeddings: torch.Tensor,
        dropout_eval: bool,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: shape (batch, seq_len, emb_dim)
            dropout_eval: if True, disables dropout
            pad_mask: Optional mask to avoid attending to certain tokens.
                      Should be broadcastable or shaped appropriately for PyTorch MHA.
                      In PyTorch, the MHA can take either an 'attn_mask' of shape (batch_size, seq_len)
                      or (batch_size, seq_len, seq_len), or a 'key_padding_mask' of shape (batch_size, seq_len).
        """
        # If dropout_eval is True, we disable dropout (set p=0.0).
        if dropout_eval:
            attn_dropout_p = 0.0
            resid_dropout_p = 0.0
        else:
            attn_dropout_p = self.config.attention_dropout_rate
            resid_dropout_p = self.config.dropout_rate

        # 1) Pre-LN
        x = self.ln1(embeddings)

        # 2) Multihead Attention
        # For MHA in PyTorch, query/key/value typically come from the same input in an encoder.
        # We pass the same x three times (self-attention).
        # We'll interpret 'pad_mask' as 'key_padding_mask' if it is of shape (batch, seq_len).
        # or 'attn_mask' if it is (batch, seq_len, seq_len).

        # key_padding_mask is used to exclude certain positions from the entire sequence.
        # attn_mask is more general (for custom attention patterns).

        attn_mask = None
        key_padding_mask = None

        if pad_mask is not None:
            # Example:
            # If pad_mask is shape (batch, seq_len), we can treat it as a key_padding_mask (True=pad).
            # If pad_mask is shape (batch, seq_len, seq_len), we can pass it as attn_mask.
            if pad_mask.dim() == 2:
                # We assume shape [batch, seq_len], True means ignore
                key_padding_mask = pad_mask
            elif pad_mask.dim() == 3:
                # We assume shape [batch, seq_len, seq_len]
                # PyTorch MHA expects (seq_len, seq_len) or (batch*num_heads, seq_len, seq_len).
                # We'll interpret this as an attn_mask. For batch_first=True, pass it directly.
                attn_mask = pad_mask
            else:
                raise ValueError("Unsupported shape for pad_mask")

        # We'll temporarily override MHA's dropout with attn_dropout_p by functional call if needed
        # but typically setting 'dropout=self.config.attention_dropout_rate' in the constructor is enough.

        # MHA returns:
        #   attn_output of shape (batch, seq_len, emb_dim)
        #   attn_output_weights of shape (batch, seq_len, seq_len) if batch_first=True
        attn_output, _ = self.mha(
            x,  # query
            x,  # key
            x,  # value
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # 3) Residual connection + dropout
        attn_output = F.dropout(attn_output, p=resid_dropout_p, training=not dropout_eval)
        embeddings = embeddings + attn_output

        # 4) Second LN
        x = self.ln2(embeddings)

        # 5) MLP block
        mlp_out = self.mlp_block(x, dropout_eval=dropout_eval)

        # 6) Residual connection
        mlp_out = F.dropout(mlp_out, p=resid_dropout_p, training=not dropout_eval)
        embeddings = embeddings + mlp_out

        return embeddings
