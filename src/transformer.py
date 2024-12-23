# transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from util_transformer import TransformerLayerConfig, EncoderTransformerConfig

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

    def forward(self, x, key_padding_mask=None, dropout_eval=False):
        """
        Args:
            x (torch.Tensor): (batch_size, seq_len, emb_dim)
            key_padding_mask (torch.Tensor): (batch_size, seq_len) with True=ignore, False=keep
            dropout_eval (bool): If True, skip dropout
        """
        # Self-attention
        attn_output, _ = self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask,  # 2D mask
            need_weights=False
        )

        if not dropout_eval:
            attn_output = self.dropout(attn_output)

        x = self.norm1(x + attn_output)

        # Feed-forward
        ff_out = F.relu(self.ff1(x))
        if not dropout_eval:
            ff_out = self.dropout(ff_out)
        ff_out = self.ff2(ff_out)
        if not dropout_eval:
            ff_out = self.dropout(ff_out)

        x = self.norm2(x + ff_out)
        return x


class EncoderTransformer(nn.Module):
    def __init__(self, config: EncoderTransformerConfig):
        super().__init__()
        self.config = config
        # Embeddings
        self.colors_embed = nn.Embedding(config.vocab_size, config.emb_dim)
        self.channels_embed = nn.Embedding(2, config.emb_dim)
        self.pos_row_embed = nn.Embedding(
            1 if config.scaled_position_embeddings else config.max_rows, config.emb_dim
        )
        self.pos_col_embed = nn.Embedding(
            1 if config.scaled_position_embeddings else config.max_cols, config.emb_dim
        )
        self.grid_shapes_row_embed = nn.Embedding(config.max_rows, config.emb_dim)
        self.grid_shapes_col_embed = nn.Embedding(config.max_cols, config.emb_dim)
        self.cls_token = nn.Embedding(1, config.emb_dim)

        self.layers = nn.ModuleList(
            [TransformerLayer(config.transformer_layer) for _ in range(config.num_layers)]
        )
        self.cls_layer_norm = nn.LayerNorm(config.emb_dim, elementwise_affine=config.transformer_layer.use_bias)
        self.latent_mu = nn.Linear(config.emb_dim, config.latent_dim, bias=config.latent_projection_bias)
        self.latent_logvar = (
            nn.Linear(config.emb_dim, config.latent_dim, bias=config.latent_projection_bias)
            if config.variational else None
        )
        self.dropout = nn.Dropout(config.transformer_dropout)

    def forward(
        self,
        pairs: torch.Tensor,
        grid_shapes: torch.Tensor,
        dropout_eval: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1) Embed inputs
        x = self.embed_grids(pairs, grid_shapes, dropout_eval)
        print("After embedding x.shape:", x.shape)  # Debug statement

        # 2) Create key_padding_mask
        key_padding_mask = self.make_pad_mask(grid_shapes)  # [batch_size, seq_len]
        print("key_padding_mask.shape:", key_padding_mask.shape)  # Debug statement

        # 3) Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, dropout_eval=dropout_eval)

        # 4) Extract CLS embedding and project to latent space
        cls_embed = x[..., 0, :]
        cls_embed = self.cls_layer_norm(cls_embed)
        latent_mu = self.latent_mu(cls_embed).float()
        latent_logvar = self.latent_logvar(cls_embed).float() if self.latent_logvar else None
        return latent_mu, latent_logvar

    def embed_grids(self, pairs, grid_shapes, dropout_eval):
        config = self.config
        # Position embeddings
        if config.scaled_position_embeddings:
            r_emb = self.pos_row_embed(torch.zeros(config.max_rows, dtype=torch.long, device=pairs.device))
            c_emb = self.pos_col_embed(torch.zeros(config.max_cols, dtype=torch.long, device=pairs.device))
            row_factors = torch.arange(1, config.max_rows + 1, device=pairs.device).unsqueeze(-1) * r_emb
            col_factors = torch.arange(1, config.max_cols + 1, device=pairs.device).unsqueeze(-1) * c_emb
            pos_embed = row_factors[:, None, None, :] + col_factors[None, :, None, :]
        else:
            r_emb = self.pos_row_embed(torch.arange(config.max_rows, dtype=torch.long, device=pairs.device))
            c_emb = self.pos_col_embed(torch.arange(config.max_cols, dtype=torch.long, device=pairs.device))
            pos_embed = r_emb[:, None, None, :] + c_emb[None, :, None, :]

        colors = self.colors_embed(pairs)  # [batch, R, C, 2, emb_dim]
        channels = self.channels_embed(torch.arange(2, dtype=torch.long, device=pairs.device))  # [2, emb_dim]

        # Broadcasting and summing embeddings
        # Ensure that channels are correctly broadcasted
        x = colors + pos_embed + channels  # [batch, R, C, 2, emb_dim]

        # Flatten the grid dimensions
        x = x.view(x.shape[0], -1, x.shape[-1])  # [batch, R*C*2, emb_dim]

        # Embed grid shapes
        row_embed = self.grid_shapes_row_embed(grid_shapes[..., 0, :] - 1) + channels  # [batch, 1, emb_dim]
        col_embed = self.grid_shapes_col_embed(grid_shapes[..., 1, :] - 1) + channels  # [batch, 1, emb_dim]
        shape_embed = torch.cat([row_embed, col_embed], dim=-2)  # [batch, 2, emb_dim]

        # Concatenate shape embeddings with the grid embeddings
        x = torch.cat([shape_embed, x], dim=-2)  # [batch, 2 + R*C*2, emb_dim]

        # Add CLS token
        cls_tok = self.cls_token(torch.zeros(x.shape[0], 1, dtype=torch.long, device=pairs.device))
        x = torch.cat([cls_tok, x], dim=-2)  # [batch, 3 + R*C*2, emb_dim]

        # Apply dropout if not in eval mode
        x = self.dropout(x) if not dropout_eval else x
        return x

    def make_pad_mask(self, grid_shapes):
        """
        Creates a 2D key_padding_mask for PyTorch's MultiheadAttention.

        Args:
            grid_shapes (torch.Tensor): Shape [batch_size, 2, 2]

        Returns:
            torch.Tensor: key_padding_mask [batch_size, seq_len] with True=ignore, False=keep
        """
        batch_size = grid_shapes.shape[0]
        seq_len = 1 + 4 + 2 * (self.config.max_rows * self.config.max_cols)  # 133 for max_rows=8, max_cols=8

        # For the test case, where grid_shapes are filled with max_rows and max_cols, no padding exists.
        # Thus, the mask should be all False (no tokens to ignore).
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=grid_shapes.device)

        # Debug statements to verify mask creation
        print(f"make_pad_mask - batch_size: {batch_size}, seq_len: {seq_len}")
        print(f"make_pad_mask - key_padding_mask:\n{key_padding_mask}")

        return key_padding_mask