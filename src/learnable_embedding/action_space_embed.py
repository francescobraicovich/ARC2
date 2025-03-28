import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import set_device

# Determine the device: CUDA -> MPS -> CPU
DEVICE = set_device('action_space_embed.py')

from transformer import EncoderTransformerConfig, TransformerLayer
from typing import Tuple, Optional

#NOTE: EncoderTransformerConfig to set
#NOTE: EncoderTransformerConfig set by Filo

class ActionEmbedding(nn.Module):
    def __init__(self, num_actions, embed_dim):
        """
        num_actions: Total number of discrete actions (e.g., 50,000)
        embed_dim: Dimensionality of the action embedding vector.
        """
        super(ActionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_actions, embed_dim)
        
    def forward(self, action):
        """
        action: single action index (int) or a scalar tensor.
        Returns:
            A tensor of shape [embed_dim] representing the embedded action.
        """
        # Convert to tensor if needed.
        if not torch.is_tensor(action):
            action = torch.tensor([action], dtype=torch.long, device=next(self.embedding.parameters()).device)
        else:
            # Ensure the action has a batch dimension.
            if action.dim() == 0:
                action = action.unsqueeze(0)
        
        # Get embedding; output shape is [1, embed_dim].
        embedded = self.embedding(action)
        # Remove the batch dimension to return shape [embed_dim].
        return embedded.squeeze(0)

class EncoderTransformer(nn.Module):
    """
    PyTorch re-implementation of the Flax-based EncoderTransformer.
    """
    #NOTE per ora non l'ho toccato l'init
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
        self.channels_embed = nn.Embedding(2, self.config.emb_dim).to(DEVICE) # 2 channels: 0 current, 1 target

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

    def forward(
        self,
        state: torch.Tensor,
        grid_shape: torch.Tensor,
        dropout_eval: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            state: shape (*B, R'*C'). input is the padded state with token IDs in [-1, vocab_size). 
            grid_shapes: shape (*B, R, C)
            dropout_eval: if True, disables dropout.

        Returns:
            Encoded sequence of shape (B, emb_dim) 
        """

        # 1) Embed the input grid - pass grid_shape parameter
        x = self.embed_grid(state, grid_shape=grid_shape, dropout_eval=dropout_eval)

        # 2) Create key_padding_mask
        key_padding_mask = self.make_pad_mask(grid_shape)  # shape (B, T)

        # 3) Apply stacked Transformer layers
        for layer in self.layers:
            x = layer(x, dropout_eval=dropout_eval, pad_mask=key_padding_mask)

        # 4) Extract and normalize the CLS token (first position)
        cls_output = self.cls_layer_norm(x[:, 0])
    
        # Return the CLS token as the state embedding
        return cls_output


    def embed_grid(
        self,
        state: torch.Tensor,
        grid_shape: torch.Tensor,  # Changed from 'shape' to 'grid_shape' for consistency
        dropout_eval: bool = False
    ) -> torch.Tensor:
        """
        Creates the token embeddings for:
          - Colors
          - Positions (row/col)
          - Channels
          - Grid shape
          - CLS
        Returns:
          x of shape (batch, 1 + 2 + R * C, emb_dim)
        """
        #------------------------
        # Get necessary inputs
        #------------------------

        batch_size = state.shape[0]
        R = state.shape[1]
        C = state.shape[2]

        #--------------------------------------
        # Color tokens embedding
        # Shape (B, R*C) -> (B, R*C, emb_dim)
        #--------------------------------------
        colors_embed = self.colors_embed(state.long())

        #--------------------------------------
        # Position Embedding
        #--------------------------------------
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

        #--------------------------------------
        # Channel Embedding
        #--------------------------------------
        # shape [2, emb_dim], then broadcast to [R, C, 2, emb_dim].
        channel_ids = torch.arange(2, device=DEVICE, dtype=torch.long)
        channel_emb = self.channels_embed(channel_ids)  # [2, emb_dim]
        # reshape to [1, 1, 2, emb_dim] for broadcast
        channel_emb = channel_emb.unsqueeze(0).unsqueeze(0)
        

        #--------------------------------------
        # First pre-CLS embedding
        # Combine: [B, R, C, 2, emb_dim] + [R, C, 1, emb_dim] + [R, C, 2, emb_dim]
        # We can reshape pos_embed to [R, C, 1, emb_dim], it will broadcast across batch & channel.
        # channel_emb is [1, 1, 2, emb_dim], it will broadcast across R, C, batch.
        #--------------------------------------      
        x = colors_embed + pos_embed + channel_emb  # [B, R, C, 2, emb_dim]
        # Flatten [R, C, 2] -> single sequence dim
        x = x.view(batch_size, R * C * 2, self.config.emb_dim)  # (B, 2*R*C, emb_dim)

        # --------------------------------------------------------------------
        # Grid shape tokens embedding
        # grid_shapes has shape (B, 2, 2): e.g. [[R_in,R_out],[C_in,C_out]].
        # We'll embed each row/col count. Then add channel_emb again for clarity.
        # --------------------------------------------------------------------
        # grid_shapes[..., 0, :] => shape (B, 2) for row [R_in, R_out].
        # grid_shapes[..., 1, :] => shape (B, 2) for col [C_in, C_out].
        # They are in [1..max_rows] or [1..max_cols], but the embedding is 0-based -> subtract 1.
        grid_shapes = grid_shape.long()
        row_part = self.grid_shapes_row_embed(grid_shapes[:, 0, :] - 1)  # [B, 2, emb_dim]
        col_part = self.grid_shapes_col_embed(grid_shapes[:, 1, :] - 1)  # [B, 2, emb_dim]

        #--------------------------------------
        # Seond pre-CLS embedding
        # Shape embedding + channel embedding for consistent channel vocabulary:
        # channel_emb for shape tokens => shape [2, emb_dim].
        #--------------------------------------    
        row_part = row_part + channel_emb.squeeze(0)  # channel_emb.squeeze(0) => shape [1, 2, emb_dim]
        col_part = col_part + channel_emb.squeeze(0)  # same

        # Concat along "sequence" dimension => [B, 4, emb_dim]
        grid_shapes_embed = torch.cat([row_part, col_part], dim=1)  # [B, 4, emb_dim]

        #--------------------------------------
        # Final pre-CLS embedding
        # final shape [B, 4 + 2*R*C, emb_dim]
        #--------------------------------------    
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
        x = self.embed_dropout(x) if not dropout_eval else x
        return x
        

    def make_pad_mask(self, grid_shapes: torch.Tensor) -> torch.Tensor:
        """
        Creates a key_padding_mask of shape (B, T), where True indicates padding tokens.
        
        Args:
            grid_shapes: shape (B, 2, 2), representing [[R_in, R_out], [C_in, C_out]].
        
        Returns:
            key_padding_mask: shape (B, T), where T = 1 + 4 + 2 * max_rows * max_cols.
                            True indicates positions that are padding.
        """
        B = grid_shapes.shape[0]
        T = 1 + 4 + 2 * (self.config.max_rows * self.config.max_cols)

        # Compute used tokens: 1 (CLS) + 4 grid shape tokens + 2*(R * C)
        rows_used = torch.max(grid_shapes[:, 0, :], dim=-1)[0]  # shape (B,)
        cols_used = torch.max(grid_shapes[:, 1, :], dim=-1)[0]  # shape (B,)
        used_tokens = 1 + 4 + 2 * (rows_used * cols_used)  # shape (B,)

        # Create a tensor with positions from 0 to T-1, shape (B, T)
        positions = torch.arange(T, device=grid_shapes.device).unsqueeze(0).expand(B, T)
        # The mask is True for positions that are greater than or equal to used_tokens for each batch element.
        key_padding_mask = positions >= used_tokens.unsqueeze(1)
        return key_padding_mask
        
    # Gattino della buona sperenza =^.^=