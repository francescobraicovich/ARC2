import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# go up one directory to import utils
from utils.util import set_device

# Determine the device: CUDA -> MPS -> CPU
DEVICE = set_device('action_space_embed.py')

from transformer import TransformerLayer

class EncoderTransformer(nn.Module):
    def __init__(self, config):
        """
        config should have at least:
          - max_rows, max_cols: maximum grid dimensions.
          - emb_dim: embedding dimension.
          - vocab_size: size of the color vocabulary.
          - max_len: usually R * C (so that 2*max_len equals the flattened spatial tokens).
          - scaled_position_embeddings: bool, whether to use scaled positional embeddings.
          - latent_dim: dimension of the latent projection.
          - latent_projection_bias: bool, whether to use bias in the latent projection.
          - variational: bool, whether to produce a log-variance.
          - num_layers: number of transformer layers.
          - transformer_layer: a sub-config with fields:
              * dropout_rate
              * num_heads
              * use_bias
              * ffn_dim
        """
        super(EncoderTransformer, self).__init__()
        self.config = config

        # Position embeddings.
        if config.scaled_position_embeddings:
            self.pos_row_embed = nn.Embedding(1, config.emb_dim).to(DEVICE)
            self.pos_col_embed = nn.Embedding(1, config.emb_dim).to(DEVICE)
        else:
            self.pos_row_embed = nn.Embedding(config.max_rows, config.emb_dim).to(DEVICE)
            self.pos_col_embed = nn.Embedding(config.max_cols, config.emb_dim).to(DEVICE)

        # Colors and channels embeddings.
        self.colors_embed = nn.Embedding(config.vocab_size, config.emb_dim).to(DEVICE)

        # Grid shapes embeddings.
        self.grid_shapes_row_embed = nn.Embedding(config.max_rows, config.emb_dim).to(DEVICE)
        self.grid_shapes_col_embed = nn.Embedding(config.max_cols, config.emb_dim).to(DEVICE)

        # CLS token embedding.
        self.cls_token = nn.Embedding(1, config.emb_dim).to(DEVICE)

        # Dropout for embedding.
        self.embed_dropout = nn.Dropout(config.dropout_rate)

        # Transformer layers.
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ]).to(DEVICE)

        # CLS layer normalization.
        self.cls_layer_norm = nn.LayerNorm(config.emb_dim, elementwise_affine=True).to(DEVICE)
        # Fix the scaling: set weight to 1 and freeze it.
        with torch.no_grad():
            self.cls_layer_norm.weight.fill_(1.0)
        self.cls_layer_norm.weight.requires_grad = False
        if not config.use_bias:
            if self.cls_layer_norm.bias is not None:
                self.cls_layer_norm.bias.data.zero_().to(DEVICE)
                self.cls_layer_norm.bias.requires_grad = False

        # Latent projection.
        self.latent_mu = nn.Linear(config.emb_dim, config.latent_dim, bias=config.latent_projection_bias).to(DEVICE)
        if config.variational:
            self.latent_logvar = nn.Linear(config.emb_dim, config.latent_dim, bias=config.latent_projection_bias)
        else:
            self.latent_logvar = None

    def forward(self, state, shape, dropout_eval):
        """
        Args:
            state: Tensor of token indices with shape (B, R, C)
            grid_shapes: Tensor with shape (B, 2) containing grid sizes (for rows and columns)
            dropout_eval: bool; if True, dropout is skipped.
        Returns:
            latent_mu: Tensor of shape (B, latent_dim)
            latent_logvar: Tensor of shape (B, latent_dim) or None.
        """
        x = self.embed_grids(state, shape, dropout_eval)

        pad_mask = self.make_pad_mask(shape)
        for layer in self.transformer_layers:
            x = layer(embeddings=x, dropout_eval=dropout_eval, pad_mask=pad_mask)
            # TODO: Check the pad mask is used correctly in the layer (not inverted).
        # Extract the CLS token (first token).
        print('x.shape before CLS token', x.shape)
        cls_embed = x[:, 0, :]
        print('cls_embed.shape', cls_embed.shape)
        cls_embed = self.cls_layer_norm(cls_embed)
        print('cls_embed.shape after layer norm', cls_embed.shape)
        latent_mu = self.latent_mu(cls_embed).to(torch.float32)
        print('latent_mu.shape', latent_mu.shape)
        if self.config.variational:
            latent_logvar = self.latent_logvar(cls_embed).to(torch.float32)
        else:
            latent_logvar = None
        return latent_mu, latent_logvar

    def embed_grids(self, state, shape, dropout_eval):
        """
        Build embeddings from the input tokens and grid shape tokens.
        """
        config = self.config
        device = DEVICE

        # Position embedding block.
        if config.scaled_position_embeddings:
            pos_row_indices = torch.zeros(config.max_rows, dtype=torch.long, device=DEVICE)
            pos_row_embed = self.pos_row_embed(pos_row_indices)  # (max_rows, emb_dim)
            pos_col_indices = torch.zeros(config.max_cols, dtype=torch.long, device=DEVICE)
            pos_col_embed = self.pos_col_embed(pos_col_indices)  # (max_cols, emb_dim)
            pos_row_factors = torch.arange(1, config.max_rows + 1, device=DEVICE).unsqueeze(1).type_as(pos_row_embed)
            pos_row_embeds = pos_row_factors * pos_row_embed  # (max_rows, emb_dim)
            pos_col_factors = torch.arange(1, config.max_cols + 1, device=DEVICE).unsqueeze(1).type_as(pos_col_embed)
            pos_col_embeds = pos_col_factors * pos_col_embed  # (max_cols, emb_dim)
            # Resulting pos_embed: shape (max_rows, max_cols, emb_dim)
            pos_embed = pos_row_embeds.unsqueeze(1)+ pos_col_embeds.unsqueeze(0)
        else:
            pos_row_indices = torch.arange(config.max_rows, dtype=torch.long, device=DEVICE)
            pos_row_embed = self.pos_row_embed(pos_row_indices)  # (max_rows, emb_dim)
            pos_col_indices = torch.arange(config.max_cols, dtype=torch.long, device=DEVICE)
            pos_col_embed = self.pos_col_embed(pos_col_indices)  # (max_cols, emb_dim)
            pos_embed = pos_row_embed.unsqueeze(1) + pos_col_embed.unsqueeze(0)

        print('pos_embed.shape', pos_embed.shape)

        # Colors embedding block.
        # pairs: (B, R, C) -> colors_embed: (B, R, C, emb_dim)
        colors_embed = self.colors_embed(state)
        print('colors_embed.shape', colors_embed.shape)

        # Combine all embeddings.
        # Broadcasting: pos_embed (max_rows, max_cols, 1, emb_dim) will broadcast to (R, C, 2, emb_dim) if R==max_rows and C==max_cols.
        x = colors_embed + pos_embed  # (B, R, C, emb_dim)
        print('x.shape', x.shape)

        # Flatten the spatial and channel dimensions.
        B = x.shape[0]
        x = x.view(B, -1, x.shape[-1])  # (B, R*C, emb_dim)
        print('x.shape after flattening', x.shape)

        # Embed grid shape tokens.
        # grid_shapes: (B, 2, 2)
        grid_shapes_row = shape[:, 0].long() - 1  # (B)
        print('grid_shapes_row.shape', grid_shapes_row.shape)
        grid_shapes_row_embed = self.grid_shapes_row_embed(grid_shapes_row)  # (B, emb_dim)
        print('grid_shapes_row_embed.shape', grid_shapes_row_embed.shape)
        grid_shapes_row_embed = grid_shapes_row_embed.unsqueeze(1)  # (B, 1, emb_dim)

        grid_shapes_col = shape[:, 1].long() - 1  # (B)
        grid_shapes_col_embed = self.grid_shapes_col_embed(grid_shapes_col)  # (B, emb_dim)
        grid_shapes_col_embed = grid_shapes_col_embed.unsqueeze(1)  # (B, 1, emb_dim) 

        # Concatenate row and column grid tokens.
        grid_shapes_embed = torch.cat([grid_shapes_row_embed, grid_shapes_col_embed], dim=1)  # (B, 2, emb_dim)
        print('grid_shapes_embed.shape', grid_shapes_embed.shape)
        x = torch.cat([grid_shapes_embed, x], dim=1)  # (B, 2 + R*C, emb_dim)
        print('x.shape after grid shapes', x.shape)

        # Add the CLS token.
        cls_token = self.cls_token(torch.zeros(x.shape[0], 1, dtype=torch.long, device=DEVICE))  # (B, 1, emb_dim)
        print('cls_token.shape', cls_token.shape)
        x = torch.cat([cls_token, x], dim=1)  # (B, 1 + 2 + 2*R*C, emb_dim)
        print('x.shape after CLS token', x.shape)

        expected_seq_len = 1 + 2 + config.max_len
        assert x.shape[1] == expected_seq_len, f"Expected sequence length {expected_seq_len}, got {x.shape[1]}"

        if not dropout_eval:
            x = self.embed_dropout(x)
        return x

    def make_pad_mask(self, shape):
        """
        Create an attention pad mask that is True for valid tokens.
        Args:
            shape: Tensor with shape (B, 2)
        Returns:
            A boolean mask of shape (B, 1, T, T) where T = 1+2+max_rows*max_cols.
        """
        print('\n\nmake_pad_mask')
        B = shape.shape[0]
        # Create a row mask of shape (B, max_rows)
        row_arange = torch.arange(self.config.max_rows, device=DEVICE).view(1, self.config.max_rows) # (1, max_rows)
        row_mask = row_arange < shape[:, 0].view(B, 1)  # (B, max_rows)
        print('row_mask.shape', row_mask.shape)

        # Create a column mask of shape (B, max_cols)
        col_arange = torch.arange(self.config.max_cols, device=DEVICE).view(1, self.config.max_cols)  # (1, max_cols)
        col_mask = col_arange < shape[:, 1].view(B, 1)  # (B, max_cols)
        # Combine to get a spatial mask.
        row_mask = row_mask.unsqueeze(2)  # (B, max_rows, 1)
        col_mask = col_mask.unsqueeze(1)  # (B, 1, max_cols)
        print('row_mask.shape', row_mask.shape)
        print('col_mask.shape', col_mask.shape)
        pad_mask = row_mask & col_mask  # (B, max_rows, max_cols)
        pad_mask = pad_mask.view(B, 1, -1)  # (B, 1, max_rows*max_cols)
        print('pad_mask.shape', pad_mask.shape)
        # Prepend ones for the CLS token and grid shape tokens (1+4 tokens).
        ones_mask = torch.ones(B, 1, 1 + 2, dtype=torch.bool, device=DEVICE)
        pad_mask = torch.cat([ones_mask, pad_mask], dim=-1)  # (B, 1, 1+2+max_rows*max_cols)
        print('pad_mask.shape after adding CLS and grid shapes', pad_mask.shape)

        # Outer product to build a full attention mask.
        pad_mask = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(3)  # (B, 1, T, T)
        print('pad_mask.shape after outer product', pad_mask.shape)
        return pad_mask
    
    def embed_state(self, state, new_episode=False):
        #NOTE: This method is just a sketch
        current_state = state[:, :self.config.max_cols]
        target_state = state[:, self.config.max_cols:]

        if torch.equal(target_state, self.target_state):
            embedded_target_state = self.target_state_embed
        else:
            self.target_state = target_state
            self.target_state_embed = self.forward(target_state)

        current_state_embed = self.forward(current_state)
        x = torch.cat([current_state_embed, self.target_state_embed], dim=1)
        return x