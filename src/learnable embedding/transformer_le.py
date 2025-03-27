import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from utils.util import set_device

DEVICE = set_device()
print("Using device for EncoderTransformer:", DEVICE)
# ---------------------------------------------------------------------
# Example placeholders for configs and the PyTorch version of TransformerLayer.
# Adjust these imports or definitions as needed to match your project.
# ---------------------------------------------------------------------

# from our current transformer we import

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.emb_dim, elementwise_affine=False).to(DEVICE)
        self.ln2 = nn.LayerNorm(config.emb_dim, elementwise_affine=False).to(DEVICE)
        self.mha = nn.MultiheadAttention(
            embed_dim=config.emb_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout_rate,
            bias=config.use_bias,
            batch_first=True,
        )
        self.mlp_block = MlpBlock(config).to(DEVICE)  # MlpBlock from your code
        self.resid_dropout = nn.Dropout(p=config.dropout_rate).to(DEVICE)

    def forward(
        self,
        embeddings: torch.Tensor,
        dropout_eval: bool,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if dropout_eval:
            attn_dropout_p = 0.0
            resid_dropout_p = 0.0
        else:
            attn_dropout_p = self.config.attention_dropout_rate
            resid_dropout_p = self.config.dropout_rate

        x = self.ln1(embeddings)
        attn_mask = None
        key_padding_mask = None
        if pad_mask is not None:
            # If pad_mask is shape [batch, seq_len, seq_len], treat it as attn_mask (boolean).
            # If pad_mask is shape [batch, seq_len], treat it as key_padding_mask.
            # Adjust logic as needed for your application.
            if pad_mask.dim() == 3:
                attn_mask = ~pad_mask  # PyTorch expects "True=do not attend", hence invert if needed
            elif pad_mask.dim() == 2:
                key_padding_mask = ~pad_mask

        attn_output, _ = self.mha(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        attn_output = F.dropout(attn_output, p=resid_dropout_p, training=not dropout_eval)
        embeddings = embeddings + attn_output

        x = self.ln2(embeddings)
        mlp_out = self.mlp_block(x, dropout_eval=dropout_eval)
        mlp_out = F.dropout(mlp_out, p=resid_dropout_p, training=not dropout_eval)
        embeddings = embeddings + mlp_out
        return embeddings


class MlpBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.activation == "relu":
            self.activation_fn = F.relu
        elif config.activation == "silu":
            self.activation_fn = F.silu
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")
        hidden_dim = int(config.mlp_dim_factor * config.emb_dim)
        self.fc1 = nn.Linear(config.emb_dim, hidden_dim, bias=config.use_bias)
        self.fc2 = nn.Linear(hidden_dim, config.emb_dim, bias=config.use_bias)

    def forward(self, x: torch.Tensor, dropout_eval: bool) -> torch.Tensor:
        dropout_p = 0.0 if dropout_eval else self.config.dropout_rate
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=dropout_p, training=not dropout_eval)
        x = self.fc2(x)
        return x


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
        self.channels_embed = nn.Embedding(2, self.config.emb_dim).to(DEVICE)

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
        """
        Creates a key_padding_mask of shape (B, T), where True indicates padding tokens.
        
        Args:
            grid_shapes: shape (B, 2, 2), representing [[R_in, R_out], [C_in, C_out]].
        
        Returns:
            key_padding_mask: shape (B, T), where T = 1 + 4 + 2 * max_rows * max_cols.
                            True indicates positions that are padding and should be masked.
        """
        B = grid_shapes.shape[0]
        T = 1 + 4 + 2 * (self.config.max_rows * self.config.max_cols)

        # Compute used tokens: 1 CLS + 4 grid shapes + 2*(R*C)
        rows_used = torch.max(grid_shapes[:, 0, :], dim=-1)[0]  # shape (B,)
        cols_used = torch.max(grid_shapes[:, 1, :], dim=-1)[0]  # shape (B,)
        used_tokens = 1 + 4 + 2 * (rows_used * cols_used)  # shape (B,)

        # Initialize mask with all False (no padding)
        key_padding_mask = torch.zeros((B, T), dtype=torch.bool, device=DEVICE)

        # Set True for padding positions
        for b in range(B):
            n = used_tokens[b].long().item()
            if n < T:
                key_padding_mask[b, n:] = True  # Mask out padding tokens

        return key_padding_mask
    
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

class DecoderTransformerTorch(nn.Module):
    '''

    PyTorch re-implementation of the Flax-based DecoderTransformer.
    
    '''
    def __init__(self, config: DecoderTransformerConfig):
        super().__init__()
        self.config = config
    
        # Embeddings
        self.context_embed = nn.Linear(config.latent_dim, config.emb_dim).to(DEVICE)
    
        # Position embeddings
        self.pos_row_embed = nn.Embedding(config.max_rows, config.emb_dim).to(DEVICE)
        self.pos_col_embed = nn.Embedding(config.max_cols, config.emb_dim).to(DEVICE)
        self.grid_shapes_row_embed = nn.Embedding(config.max_rows, config.emb_dim).to(DEVICE)
        self.grid_shapes_col_embed = nn.Embedding(config.max_cols, config.emb_dim).to(DEVICE)
        self.colors_embed = nn.Embedding(config.vocab_size, config.emb_dim).to(DEVICE)
        self.input_output_embed = nn.Embedding(2, config.emb_dim).to(DEVICE)

        # Create transformer layers
        self.layers = nn.ModuleList([TransformerLayer(config.transformer_layer).to(DEVICE) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.emb_dim).to(DEVICE)
    
        # Projections to logits
        self.shape_row_proj = nn.Linear(config.emb_dim, config.max_rows).to(DEVICE)
        self.shape_col_proj = nn.Linear(config.emb_dim, config.max_cols).to(DEVICE)
        self.grid_proj = nn.Linear(config.emb_dim, config.vocab_size).to(DEVICE)

    def forward(self, embedded_action, embedded_state, input_seq, output_seq, context, dropout_eval: bool):
       
        '''
        Args:
             input_seq: shape (B, T_in), with token IDs in [0, vocab_size).
             output_seq: shape (B, T_out), with token IDs in [0, vocab_size).
             NO context: shape (B, latent_dim), the latent representation from the encoder.
             dropout_eval: if True, disables dropout.
         
         Returns:
             shape_row_logits: shape (B, max_rows), the logits for grid shape row.
             shape_col_logits: shape (B, max_cols), the logits for grid shape col.
             grid_logits: shape (B, T_out-3, vocab_size), the logits for grid tokens.
        
        
        '''
        # Concatenate embedded action and embedded state
        x = torch.cat([embedded_action, embedded_state], dim=1)
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, dropout_eval=dropout_eval)
        # Apply layer normalization
        x = self.layer_norm(x) 
        # Project to get logits for shape row, shape col, and grid
        shape_row_logits = self.shape_row_proj(x[:, input_seq.size(1)+1, :])
        shape_col_logits = self.shape_col_proj(x[:, input_seq.size(1)+2, :])
        grid_logits = self.grid_proj(x[:, input_seq.size(1)+3:, :])
        return shape_row_logits, shape_col_logits, grid_logits
