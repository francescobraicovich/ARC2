import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import set_device

# Determine the device: CUDA -> MPS -> CPU
DEVICE = set_device('action_space_embed.py')

from transformer import EncoderTransformerConfig, TransformerLayer
from typing import Tuple, Optional

class EncoderTransformer(nn.Module):
    """
    PyTorch re-implementation of the Flax-based EncoderTransformer.
    """

    def __init__(self, config: EncoderTransformerConfig):
        super().__init__()
        self.config = config

        # -----------------------------------------------------------------
        # Position embeddings
        # For scaled_position_embeddings=True, we embed only index=0 and
        # multiply by torch.arange(1, ...). Otherwise, we embed each row/col index.
        # -----------------------------------------------------------------
        if self.config.scaled_position_embeddings:
            self.pos_row_embed = nn.Embedding(1, self.config.emb_dim).to(DEVICE)
            self.pos_col_embed = nn.Embedding(1, self.config.emb_dim).to(DEVICE)
        else:
            self.pos_row_embed = nn.Embedding(self.config.max_rows, self.config.emb_dim).to(DEVICE)
            self.pos_col_embed = nn.Embedding(self.config.max_cols, self.config.emb_dim).to(DEVICE)

        # Embeddings for colors, channels, grid shapes, and CLS token
        self.colors_embed = nn.Embedding(self.config.vocab_size, self.config.emb_dim).to(DEVICE)
        #self.channels_embed = nn.Embedding(2, self.config.emb_dim).to(DEVICE)

        self.grid_shapes_row_embed = nn.Embedding(self.config.max_rows, self.config.emb_dim).to(DEVICE)
        self.grid_shapes_col_embed = nn.Embedding(self.config.max_cols, self.config.emb_dim).to(DEVICE)

        self.cls_token = nn.Embedding(1, self.config.emb_dim).to(DEVICE)

        # Dropout for the embedded sequence
        self.embed_dropout = nn.Dropout(self.config.transformer_layer.dropout_rate).to(DEVICE)

        # Layers: stack of TransformerLayer
        self.layers = nn.ModuleList([
            TransformerLayer(self.config.transformer_layer).to(DEVICE) for _ in range(self.config.num_layers)
        ])

        # Final layer norm of CLS token
        self.cls_layer_norm = nn.LayerNorm(
            self.config.emb_dim,
            elementwise_affine=False  # mimic use_scale=False, use_bias=False
        ).to(DEVICE)

        # Projections for latent mean/logvar
        self.latent_mu = nn.Linear(
            self.config.emb_dim,
            self.config.latent_dim,
            bias=self.config.latent_projection_bias
        ).to(DEVICE)

        if self.config.variational:
            self.latent_logvar = nn.Linear(
                self.config.emb_dim,
                self.config.latent_dim,
                bias=self.config.latent_projection_bias
            ).to(DEVICE)
        else:
            self.latent_logvar = None

    def forward(
        self,
        pairs: torch.Tensor,
        grid_shapes: torch.Tensor,
        dropout_eval: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            pairs: shape (*B, R, C, 2), with token IDs in [0, vocab_size).
            grid_shapes: shape (*B, 2, 2). The last two dims = [[R_in, R_out], [C_in, C_out]].
            dropout_eval: if True, disables dropout.

        Returns:
            (latent_mu, latent_logvar) or (latent_mu, None) if not variational.
        """
        # 1) Embed the input grids
        x = self.embed_grids(pairs, grid_shapes, dropout_eval=dropout_eval)

        # 2) Create key_padding_mask
        key_padding_mask = self.make_pad_mask(grid_shapes)  # shape (B, T)

        # 3) Apply stacked Transformer layers
        for layer in self.layers:
            x = layer(x, dropout_eval=dropout_eval, pad_mask=key_padding_mask)

        # 4) Extract CLS token (batch_first => x[:, 0, :])
        cls_embed = x[:, 0, :]

        # 5) LayerNorm on CLS
        cls_embed = self.cls_layer_norm(cls_embed)

        # 6) Project to latent space
        latent_mu = self.latent_mu(cls_embed).to(torch.float32)
        if self.config.variational:
            latent_logvar = self.latent_logvar(cls_embed).to(torch.float32)
        else:
            latent_logvar = None
        return latent_mu#, latent_logvar

    def embed_grids(
        self,
        pairs: torch.Tensor,
        grid_shapes: torch.Tensor,
        dropout_eval: bool = False
    ) -> torch.Tensor:
        """
        Creates the token embeddings for:
          - Colors
          - Positions (row/col)
          - Channels
          - Grid shapes
          - CLS
        Returns:
          x of shape (batch, 1 + 4 + 2 * R * C, emb_dim)
        """
        batch_size = pairs.shape[0]
        R = pairs.shape[1]
        C = pairs.shape[2]

        # Embedding for color tokens: shape (B, R, C, 2) -> (B, R, C, 2, emb_dim)
        colors_embed = self.colors_embed(pairs.long())

        # Position embeddings
        if self.config.scaled_position_embeddings:
            # pos_row_embed is nn.Embedding(1, emb_dim). We'll "lookup" zeros, shape -> [R, emb_dim].
            # Then multiply by torch.arange(1, R+1).
            row_vec = torch.zeros((R,), dtype=torch.long, device=DEVICE)
            col_vec = torch.zeros((C,), dtype=torch.long, device=DEVICE)

            row_embed = self.pos_row_embed(row_vec)  # shape [R, emb_dim]
            col_embed = self.pos_col_embed(col_vec)  # shape [C, emb_dim]

            # scale them by the row/col indices
            row_scales = torch.arange(1, R + 1, device=DEVICE, dtype=row_embed.dtype).unsqueeze(-1)
            col_scales = torch.arange(1, C + 1, device=DEVICE, dtype=col_embed.dtype).unsqueeze(-1)

            pos_row_embeds = row_scales * row_embed  # [R, emb_dim]
            pos_col_embeds = col_scales * col_embed  # [C, emb_dim]

            # shape [R, C, 1, emb_dim] for broadcast
            pos_row_embeds = pos_row_embeds.unsqueeze(1).unsqueeze(2)
            pos_col_embeds = pos_col_embeds.unsqueeze(0).unsqueeze(2)
            pos_embed = pos_row_embeds + pos_col_embeds  # [R, C, 1, emb_dim]
        else:
            # Normal indexing (0..R-1), (0..C-1)
            row_ids = torch.arange(R, device=DEVICE, dtype=torch.long)
            col_ids = torch.arange(C, device=DEVICE, dtype=torch.long)
            row_embed = self.pos_row_embed(row_ids)  # [R, emb_dim]
            col_embed = self.pos_col_embed(col_ids)  # [C, emb_dim]
            row_embed = row_embed.unsqueeze(1).unsqueeze(2)  # [R, 1, 1, emb_dim]
            col_embed = col_embed.unsqueeze(0).unsqueeze(2)  # [1, C, 1, emb_dim]
            pos_embed = row_embed + col_embed  # [R, C, 1, emb_dim]

        # Channels embedding
        # shape [2, emb_dim], then broadcast to [R, C, 2, emb_dim].
        channel_ids = torch.arange(2, device=DEVICE, dtype=torch.long)
        channel_emb = self.channels_embed(channel_ids)  # [2, emb_dim]
        # reshape to [1, 1, 2, emb_dim] for broadcast
        channel_emb = channel_emb.unsqueeze(0).unsqueeze(0)

        # Combine: [B, R, C, 2, emb_dim] + [R, C, 1, emb_dim] + [R, C, 2, emb_dim]
        # We can reshape pos_embed to [R, C, 1, emb_dim], it will broadcast across batch & channel.
        # channel_emb is [1, 1, 2, emb_dim], it will broadcast across R, C, batch.
        x = colors_embed + pos_embed + channel_emb  # [B, R, C, 2, emb_dim]

        # Flatten [R, C, 2] -> single sequence dim
        x = x.view(batch_size, R * C * 2, self.config.emb_dim)  # (B, 2*R*C, emb_dim)

        # --------------------------------------------------------------------
        # Embed the grid shape tokens
        # grid_shapes has shape (B, 2, 2): e.g. [[R_in,R_out],[C_in,C_out]].
        # We'll embed each row/col count. Then add channel_emb again for clarity.
        # --------------------------------------------------------------------
        # grid_shapes[..., 0, :] => shape (B, 2) for row [R_in, R_out].
        # grid_shapes[..., 1, :] => shape (B, 2) for col [C_in, C_out].
        # They are in [1..max_rows] or [1..max_cols], but the embedding is 0-based -> subtract 1.
        grid_shapes = grid_shapes.long()
        row_part = self.grid_shapes_row_embed(grid_shapes[:, 0, :] - 1)  # [B, 2, emb_dim]
        col_part = self.grid_shapes_col_embed(grid_shapes[:, 1, :] - 1)  # [B, 2, emb_dim]

        # Add channel embedding to them as well:
        # channel_emb for shape tokens => shape [2, emb_dim].
        # We want to add this per 'input' or 'output' channel, so broadcast:
        row_part = row_part + channel_emb.squeeze(0)  # channel_emb.squeeze(0) => shape [1, 2, emb_dim]
        col_part = col_part + channel_emb.squeeze(0)  # same

        # Concat along "sequence" dimension => [B, 4, emb_dim]
        grid_shapes_embed = torch.cat([row_part, col_part], dim=1)  # [B, 4, emb_dim]

        # Now we prepend these 4 tokens to x => final shape [B, 4 + 2*R*C, emb_dim]
        x = torch.cat([grid_shapes_embed, x], dim=1)

        # --------------------------------------------------------------------
        # Add the CLS token => shape [B, 1, emb_dim] to the front
        # --------------------------------------------------------------------
        cls_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=DEVICE)
        cls_token = self.cls_token(cls_ids)  # [B, 1, emb_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, 1 + 4 + 2*R*C, emb_dim]

        # Check final size is 1 + 4 + 2*max_len
        # NOTE: we rely on R = max_rows, C = max_cols for the shape to match exactly.
        # If R or C < max_rows/cols for a given example, the mask will handle it.
        # But the length is still allocated for the full max. That's consistent
        # with the original approach in JAX.

        # Apply dropout
        if dropout_eval:
            embed_dropout_p = 0.0
        else:
            embed_dropout_p = self.config.transformer_layer.dropout_rate

        x = F.dropout(x, p=embed_dropout_p, training=not dropout_eval)
        return x
    
    def make_pad_mask(self, grid_shapes: torch.Tensor) -> torch.Tensor:

        B = grid_shapes.shape[0]
        T = 1 + 4 + 2 * (self.config.max_rows * self.config.max_cols)

        # Compute used tokens: 1 CLS + 4 grid shapes + 2*(R*C)
        rows_used = torch.max(grid_shapes[:, 0, :], dim=-1)[0]
        cols_used = torch.max(grid_shapes[:, 1, :], dim=-1)[0]

        used_tokens = 1 + 4 + 2 * (rows_used * cols_used)

        # Initialize mask with all False (no padding)
        key_padding_mask = torch.zeros((B, T), dtype=torch.bool, device=DEVICE)

        for b in range(B):
            n = used_tokens[b].long().item()

            if n < 0 or n > T:
                print(f"Invalid token count at batch {b}: n={n}, T={T}")
                continue  # Skip this batch to avoid errors

            if n < T:
                key_padding_mask[b, n:] = True  # Mask out padding tokens

        return key_padding_mask
    

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
            self.pos_row_embed = nn.Embedding(1, config.emb_dim)
            self.pos_col_embed = nn.Embedding(1, config.emb_dim)
        else:
            self.pos_row_embed = nn.Embedding(config.max_rows, config.emb_dim)
            self.pos_col_embed = nn.Embedding(config.max_cols, config.emb_dim)

        # Colors and channels embeddings.
        self.colors_embed = nn.Embedding(config.vocab_size, config.emb_dim)

        # Grid shapes embeddings.
        self.grid_shapes_row_embed = nn.Embedding(config.max_rows, config.emb_dim)
        self.grid_shapes_col_embed = nn.Embedding(config.max_cols, config.emb_dim)

        # CLS token embedding.
        self.cls_token = nn.Embedding(1, config.emb_dim)

        # Dropout for embedding.
        self.embed_dropout = nn.Dropout(config.transformer_layer.dropout_rate)

        # Transformer layers.
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config.transformer_layer) for _ in range(config.num_layers)
        ])

        # CLS layer normalization.
        self.cls_layer_norm = nn.LayerNorm(config.emb_dim, elementwise_affine=True)
        # Fix the scaling: set weight to 1 and freeze it.
        with torch.no_grad():
            self.cls_layer_norm.weight.fill_(1.0)
        self.cls_layer_norm.weight.requires_grad = False
        if not config.transformer_layer.use_bias:
            if self.cls_layer_norm.bias is not None:
                self.cls_layer_norm.bias.data.zero_()
                self.cls_layer_norm.bias.requires_grad = False

        # Latent projection.
        self.latent_mu = nn.Linear(config.emb_dim, config.latent_dim, bias=config.latent_projection_bias)
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

        pad_mask = self.make_pad_mask(grid_shapes)
        for layer in self.transformer_layers:
            x = layer(embeddings=x, dropout_eval=dropout_eval, pad_mask=pad_mask)
        # Extract the CLS token (first token).
        cls_embed = x[:, 0, :]
        cls_embed = self.cls_layer_norm(cls_embed)
        latent_mu = self.latent_mu(cls_embed).to(torch.float32)
        if self.config.variational:
            latent_logvar = self.latent_logvar(cls_embed).to(torch.float32)
        else:
            latent_logvar = None
        return latent_mu, latent_logvar

    def embed_grids(self, pairs, grid_shapes, dropout_eval):
        """
        Build embeddings from the input tokens and grid shape tokens.
        """
        config = self.config
        device = pairs.device

        # Position embedding block.
        if config.scaled_position_embeddings:
            pos_row_indices = torch.zeros(config.max_rows, dtype=torch.long, device=device)
            pos_row_embed = self.pos_row_embed(pos_row_indices)  # (max_rows, emb_dim)
            pos_col_indices = torch.zeros(config.max_cols, dtype=torch.long, device=device)
            pos_col_embed = self.pos_col_embed(pos_col_indices)  # (max_cols, emb_dim)
            pos_row_factors = torch.arange(1, config.max_rows + 1, device=device).unsqueeze(1).type_as(pos_row_embed)
            pos_row_embeds = pos_row_factors * pos_row_embed  # (max_rows, emb_dim)
            pos_col_factors = torch.arange(1, config.max_cols + 1, device=device).unsqueeze(1).type_as(pos_col_embed)
            pos_col_embeds = pos_col_factors * pos_col_embed  # (max_cols, emb_dim)
            # Resulting pos_embed: shape (max_rows, max_cols, 1, emb_dim)
            pos_embed = pos_row_embeds.unsqueeze(1).unsqueeze(2) + pos_col_embeds.unsqueeze(0).unsqueeze(2)
        else:

            pos_row_indices = torch.arange(config.max_rows, dtype=torch.long, device=device)
            pos_row_embed = self.pos_row_embed(pos_row_indices)  # (max_rows, emb_dim)
            pos_col_indices = torch.arange(config.max_cols, dtype=torch.long, device=device)
            pos_col_embed = self.pos_col_embed(pos_col_indices)  # (max_cols, emb_dim)
            pos_embed = pos_row_embed.unsqueeze(1).unsqueeze(2) + pos_col_embed.unsqueeze(0).unsqueeze(2)

        # Colors embedding block.
        # pairs: (B, R, C, 2) -> colors_embed: (B, R, C, 2, emb_dim)
        colors_embed = self.colors_embed(pairs)

        # Channels embedding block.
        channels_indices = torch.arange(2, dtype=torch.long, device=device)
        channels_embed = self.channels_embed(channels_indices)  # (2, emb_dim)

        # Combine all embeddings.
        # Broadcasting: pos_embed (max_rows, max_cols, 1, emb_dim) will broadcast to (R, C, 2, emb_dim) if R==max_rows and C==max_cols.
        x = colors_embed + pos_embed + channels_embed  # (B, R, C, 2, emb_dim)

        # Flatten the spatial and channel dimensions.
        B = x.shape[0]
        x = x.view(B, -1, x.shape[-1])  # (B, 2*R*C, emb_dim)

        # Embed grid shape tokens.
        # grid_shapes: (B, 2, 2)
        grid_shapes_row = grid_shapes[:, 0, :].long() - 1  # (B, 2)
        grid_shapes_row_embed = self.grid_shapes_row_embed(grid_shapes_row)  # (B, 2, emb_dim)
        grid_shapes_row_embed = grid_shapes_row_embed + channels_embed  # broadcast addition

        grid_shapes_col = grid_shapes[:, 1, :].long() - 1  # (B, 2)
        grid_shapes_col_embed = self.grid_shapes_col_embed(grid_shapes_col)  # (B, 2, emb_dim)
        grid_shapes_col_embed = grid_shapes_col_embed + channels_embed

        # Concatenate row and column grid tokens.
        grid_shapes_embed = torch.cat([grid_shapes_row_embed, grid_shapes_col_embed], dim=1)  # (B, 4, emb_dim)
        x = torch.cat([grid_shapes_embed, x], dim=1)  # (B, 4 + 2*R*C, emb_dim)

        # Add the CLS token.
        cls_token = self.cls_token(torch.zeros(x.shape[0], 1, dtype=torch.long, device=device))  # (B, 1, emb_dim)
        x = torch.cat([cls_token, x], dim=1)  # (B, 1 + 4 + 2*R*C, emb_dim)

        expected_seq_len = 1 + 4 + 2 * config.max_len
        assert x.shape[1] == expected_seq_len, f"Expected sequence length {expected_seq_len}, got {x.shape[1]}"

        if not dropout_eval:
            x = self.embed_dropout(x)
        return x

    def make_pad_mask(self, grid_shapes):
        """
        Create an attention pad mask that is True for valid tokens.
        Args:
            grid_shapes: Tensor with shape (B, 2, 2)
        Returns:
            A boolean mask of shape (B, 1, T, T) where T = 1+4+2*max_rows*max_cols.
        """
        B = grid_shapes.shape[0]
        device = grid_shapes.device
        # Create a row mask of shape (B, max_rows, 2)
        row_arange = torch.arange(self.config.max_rows, device=device).view(1, self.config.max_rows, 1)
        row_mask = row_arange < grid_shapes[:, 0:1, :].long()  # (B, max_rows, 2)
        # Create a column mask of shape (B, max_cols, 2)
        col_arange = torch.arange(self.config.max_cols, device=device).view(1, self.config.max_cols, 1)
        col_mask = col_arange < grid_shapes[:, 1:2, :].long()  # (B, max_cols, 2)
        # Combine to get a spatial mask.
        row_mask = row_mask.unsqueeze(2)  # (B, max_rows, 1, 2)
        col_mask = col_mask.unsqueeze(1)  # (B, 1, max_cols, 2)
        pad_mask = row_mask & col_mask  # (B, max_rows, max_cols, 2)
        pad_mask = pad_mask.view(B, 1, -1)  # (B, 1, max_rows*max_cols*2)
        # Prepend ones for the CLS token and grid shape tokens (1+4 tokens).
        ones_mask = torch.ones(B, 1, 1 + 4, dtype=torch.bool, device=device)
        pad_mask = torch.cat([ones_mask, pad_mask], dim=-1)  # (B, 1, 1+4+max_rows*max_cols*2)
        # Outer product to build a full attention mask.
        pad_mask = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(1)  # (B, 1, T, T)
        return pad_mask

config = EncoderTransformerConfig()
model = EncoderTransformer(config)
# Dummy inputs:
pairs = torch.randint(0, config.vocab_size, (2, config.max_rows, config.max_cols, 2))
grid_shapes = torch.tensor([[[30, 30], [30, 30]], [[30, 30], [30, 30]]])
latent_mu, latent_logvar = model(pairs, grid_shapes, dropout_eval=False)
print(latent_mu.shape, latent_logvar.shape)