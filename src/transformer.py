import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TransformerLayerConfig:
    emb_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    use_bias: bool = True

@dataclass
class EncoderTransformerConfig:
    vocab_size: int
    max_rows: int
    max_cols: int
    emb_dim: int = 128
    latent_dim: int = 64
    num_layers: int = 2
    transformer_layer: TransformerLayerConfig = TransformerLayerConfig(128, 8)
    variational: bool = True
    latent_projection_bias: bool = True
    dtype: torch.dtype = torch.float32
    scaled_position_embeddings: bool = False
    max_len: int = 30  # R*C
    transformer_dropout: float = 0.1

class TransformerLayer(nn.Module):
    def __init__(self, config: TransformerLayerConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.emb_dim,
            num_heads=config.num_heads,
            dropout=config.dropout_rate,
            bias=config.use_bias,
            batch_first=True
        )
        self.ff1 = nn.Linear(config.emb_dim, 4 * config.emb_dim, bias=config.use_bias)
        self.ff2 = nn.Linear(4 * config.emb_dim, config.emb_dim, bias=config.use_bias)
        self.norm1 = nn.LayerNorm(config.emb_dim)
        self.norm2 = nn.LayerNorm(config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, pad_mask=None, dropout_eval=False):
        mask = None
        if pad_mask is not None:
            mask = (~pad_mask).float() * float('-inf')
        attn_output, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        ff_out = self.ff2(F.relu(self.ff1(x)))
        x = self.norm2(x + self.dropout(ff_out))
        return x

class EncoderTransformer(nn.Module):
    def __init__(self, config: EncoderTransformerConfig):
        super().__init__()
        self.config = config
        # Embeddings
        self.colors_embed = nn.Embedding(config.vocab_size, config.emb_dim)
        self.channels_embed = nn.Embedding(2, config.emb_dim)
        self.pos_row_embed = nn.Embedding(1 if config.scaled_position_embeddings else config.max_rows, config.emb_dim)
        self.pos_col_embed = nn.Embedding(1 if config.scaled_position_embeddings else config.max_cols, config.emb_dim)
        self.grid_shapes_row_embed = nn.Embedding(config.max_rows, config.emb_dim)
        self.grid_shapes_col_embed = nn.Embedding(config.max_cols, config.emb_dim)
        self.cls_token = nn.Embedding(1, config.emb_dim)

        self.layers = nn.ModuleList(
            [TransformerLayer(config.transformer_layer) for _ in range(config.num_layers)]
        )
        self.cls_layer_norm = nn.LayerNorm(config.emb_dim, elementwise_affine=config.transformer_layer.use_bias)
        self.latent_mu = nn.Linear(config.emb_dim, config.latent_dim, bias=config.latent_projection_bias)
        self.latent_logvar = nn.Linear(config.emb_dim, config.latent_dim, bias=config.latent_projection_bias) \
            if config.variational else None
        self.dropout = nn.Dropout(config.transformer_dropout)

    def forward(self, pairs: torch.Tensor, grid_shapes: torch.Tensor, dropout_eval: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embed_grids(pairs, grid_shapes, dropout_eval)
        pad_mask = self.make_pad_mask(grid_shapes)
        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask, dropout_eval=dropout_eval)

        cls_embed = x[..., 0, :]
        cls_embed = self.cls_layer_norm(cls_embed)
        latent_mu = self.latent_mu(cls_embed).float()
        latent_logvar = self.latent_logvar(cls_embed).float() if self.latent_logvar else None
        return latent_mu, latent_logvar

    def embed_grids(self, pairs, grid_shapes, dropout_eval):
        config = self.config
        # Position embeddings
        if config.scaled_position_embeddings:
            r_emb = self.pos_row_embed(torch.zeros(config.max_rows, dtype=torch.long))
            c_emb = self.pos_col_embed(torch.zeros(config.max_cols, dtype=torch.long))
            row_factors = torch.arange(1, config.max_rows + 1).unsqueeze(-1) * r_emb
            col_factors = torch.arange(1, config.max_cols + 1).unsqueeze(-1) * c_emb
            pos_embed = row_factors[:, None, None, :] + col_factors[None, :, None, :]
        else:
            r_emb = self.pos_row_embed(torch.arange(config.max_rows, dtype=torch.long))
            c_emb = self.pos_col_embed(torch.arange(config.max_cols, dtype=torch.long))
            pos_embed = r_emb[:, None, None, :] + c_emb[None, :, None, :]

        colors = self.colors_embed(pairs)
        channels = self.channels_embed(torch.arange(2, dtype=torch.long))
        x = colors + pos_embed + channels
        x = x.view(*x.shape[:-4], -1, x.shape[-1])  # flatten

        row_embed = self.grid_shapes_row_embed(grid_shapes[..., 0, :] - 1) + channels
        col_embed = self.grid_shapes_col_embed(grid_shapes[..., 1, :] - 1) + channels
        shape_embed = torch.cat([row_embed, col_embed], dim=-2)
        x = torch.cat([shape_embed, x], dim=-2)

        cls_tok = self.cls_token(torch.zeros_like(x[..., :1, 0], dtype=torch.long))
        x = torch.cat([cls_tok, x], dim=-2)
        x = self.dropout(x) if not dropout_eval else x
        return x

    def make_pad_mask(self, grid_shapes):
        bshape = grid_shapes.shape[:-2]
        row_arange = torch.arange(self.config.max_rows).view(*(1,)*len(bshape), self.config.max_rows, 1)
        col_arange = torch.arange(self.config.max_cols).view(*(1,)*len(bshape), self.config.max_cols, 1)

        row_mask = row_arange < grid_shapes[..., 0:1, :]
        col_mask = col_arange < grid_shapes[..., 1:2, :]
        pad_mask = (row_mask & col_mask).view(*row_mask.shape[:-3], 1, -1)

        token_mask = torch.ones((*pad_mask.shape[:-1], 1+4), dtype=torch.bool)
        pad_mask = torch.cat([token_mask, pad_mask], dim=-1)
        pad_mask = pad_mask.unsqueeze(-2) & pad_mask.unsqueeze(-1)
        return pad_mask

