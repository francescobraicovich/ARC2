# transformer.py

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from utils.util_transformer import TransformerLayerConfig, EncoderTransformerConfig

# Define a single Transformer layer
class TransformerLayer(nn.Module):
    def __init__(self, config: TransformerLayerConfig):
        super().__init__()
        # Multi-head self-attention mechanism
        self.attn = nn.MultiheadAttention(
            embed_dim=config.emb_dim,
            num_heads=config.num_heads,
            dropout=config.dropout_rate,
            bias=config.use_bias,
            batch_first=True
        )

        # Layer normalization after attention
        self.norm = nn.LayerNorm(config.emb_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        embeddings: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        dropout_eval: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer layer that supports both:
        - embeddings of shape [B, T, emb_dim], or
        - embeddings of shape [N, B, T, emb_dim],
        and similarly for any number of leading dims (*leading_dims, T, emb_dim).

        Args:
            embeddings: input tensor of shape [*leading_dims, T, emb_dim].
            pad_mask:   optional mask tensor of shape [*leading_dims, 1, T, T] or [*leading_dims, T], etc.
            dropout_eval: if True, skip dropout; if False, apply dropout.

        Returns:
            Tensor of shape [*leading_dims, T, emb_dim], same as the input shape.
        """

        # ------------------------------------------------
        # 1) Flatten all leading dims for MultiheadAttention
        # ------------------------------------------------
        orig_shape = embeddings.shape  # e.g. [N, B, T, emb_dim] or [B, T, emb_dim]
        leading_dims = orig_shape[:-2]  # everything except the last 2 dims (T, emb_dim)
        seq_len, emb_dim = orig_shape[-2], orig_shape[-1]

        # Compute 'combined_batch_size' by multiplying all leading dims
        combined_batch_size = 1
        for d in leading_dims:
            combined_batch_size *= d

        # Reshape embeddings => [combined_batch_size, T, emb_dim]
        embeddings_2d = embeddings.view(combined_batch_size, seq_len, emb_dim)

        # ------------------------------------------------
        # 2) Handle pad_mask (flatten if it exists)
        # ------------------------------------------------
        attn_mask = None
        key_padding_mask = None
        if pad_mask is not None:
            # pad_mask might be e.g. [*leading_dims, 1, T, T] or [*leading_dims, T] etc.
            # Flatten the leading dims as well.
            pm_shape = pad_mask.shape
            pad_leading_dims = pm_shape[:-2]  # everything except the last 2 dims (T, T) if 4D
            combined_pm_batch_size = 1
            for d in pad_leading_dims:
                combined_pm_batch_size *= d

            # Example: if pad_mask is [N, B, 1, T, T], reshape => [N*B, 1, T, T].
            # Then pass it as attn_mask or key_padding_mask. 
            # You’ll need to adapt this to your MHA usage. Below is a placeholder:
            pad_mask_2d = pad_mask.view(combined_pm_batch_size, *pm_shape[-2:])

            # For simplicity, we won’t convert pad_mask_2d further, but typically:
            #   - if using `key_padding_mask` (shape [batch_size, seq_len]), 
            #     you'd reduce the [1, T, T] to [T] somehow, etc.
            #   - if using `attn_mask` (shape [batch_size, T, T]),
            #     you can pass it directly as `attn_mask=pad_mask_2d.squeeze(1)`.
            #
            # We'll just demonstrate passing it as an attention mask:
            attn_mask = pad_mask_2d.squeeze(1)  # shape => [combined_pm_batch_size, T, T]

            # If you need a key_padding_mask: you'd create something like
            #   key_padding_mask = ~pad_mask_2d.all(dim=-2)  # or some custom logic

        # ------------------------------------------------
        # 3) Run self-attention
        # ------------------------------------------------
        attn_out, _ = self.attn(
            query=embeddings_2d,
            key=embeddings_2d,
            value=embeddings_2d,
            attn_mask=attn_mask,          # or None
            key_padding_mask=key_padding_mask  # or None
        )

        # Dropout + residual
        if not dropout_eval:
            attn_out = self.dropout(attn_out)

        embeddings_2d = embeddings_2d + attn_out
        embeddings_2d = self.norm(embeddings_2d)

        # ------------------------------------------------
        # 4) Reshape back to original shape
        # ------------------------------------------------
        out = embeddings_2d.view(*leading_dims, seq_len, emb_dim)
        return out

    def forward(self, embeddings, pad_mask=None, dropout_eval=False):
        """
        Forward pass of the Transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim)
            key_padding_mask (torch.Tensor): Mask tensor of shape (batch_size, seq_len)
            dropout_eval (bool): Whether to skip dropout
        """
        # Self-Attention sub-block
        if pad_mask is not None:
            # For MultiheadAttention with batch_first=True, we need a key_padding_mask: [B, T].
            # Or an attn_mask: [T, T]. Adjust below as needed for your code.
            # We'll do a simple demonstration ignoring the exact mask shape details:
            attn_mask = None
            key_padding_mask = None
            # If your pad_mask is shape [B, 1, T, T], you might transform it to a 2D or 3D mask.
            # This part is placeholder; adapt it to match your actual MHA usage.
        else:
            attn_mask = None
            key_padding_mask = None

        print('Attention input shape: ', embeddings.shape)
        # Self-attention
        attn_out, _ = self.attn(
            query=embeddings,
            key=embeddings,
            value=embeddings,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        print('Attention output shape: ', attn_out.shape)
        # Residual + dropout
        if not dropout_eval:
            attn_out = self.dropout(attn_out)
        embeddings = embeddings + attn_out
        embeddings = self.norm(embeddings)
        return embeddings

# Define the Encoder Transformer
class EncoderTransformer(nn.Module):
    def __init__(self, config: EncoderTransformerConfig):
        super().__init__()
        # Initialize device
        self.device = self._get_device()
        print(f"Using device: {self.device} for Transformer")

        self.config = config

        # Initialize embedding layers
        self._initialize_embedding_layers()

        # Initialize transformer layers
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(config.transformer_layer) for _ in range(config.num_layers)]
        )

        # Initialize CLS token layer normalization
        self.cls_layer_norm = nn.LayerNorm(
            config.emb_dim, elementwise_affine=config.transformer_layer.use_bias
        )

        # Initialize latent space layers
        self.latent_mu, self.latent_logvar = self._initialize_latent_layers()

        # Initialize dropout
        self.dropout = nn.Dropout(config.transformer_dropout)

    def _get_device(self) -> torch.device:
        """Automatically determine and return the device to use."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _initialize_embedding_layers(self):
        """Initialize embedding layers."""
        self.colors_embed = nn.Embedding(self.config.vocab_size, self.config.emb_dim)
        self.channels_embed = nn.Embedding(2, self.config.emb_dim)
        self.pos_row_embed = nn.Embedding(
            1 if self.config.scaled_position_embeddings else self.config.max_rows,
            self.config.emb_dim
        )
        self.pos_col_embed = nn.Embedding(
            1 if self.config.scaled_position_embeddings else self.config.max_cols,
            self.config.emb_dim
        )
        self.grid_shapes_row_embed = nn.Embedding(self.config.max_rows, self.config.emb_dim)
        self.grid_shapes_col_embed = nn.Embedding(self.config.max_cols, self.config.emb_dim)
        self.cls_token = nn.Embedding(1, self.config.emb_dim)

    def _initialize_latent_layers(self):
        """Initialize layers for latent space projections."""
        latent_mu = nn.Linear(
            self.config.emb_dim, self.config.latent_dim, bias=self.config.latent_projection_bias
        )
        latent_logvar = (
            nn.Linear(
                self.config.emb_dim, self.config.latent_dim, bias=self.config.latent_projection_bias
            ) if self.config.variational else None
        )
        return latent_mu, latent_logvar

    def forward(
        self,
        pairs: torch.Tensor,
        grid_shapes: torch.Tensor,
        dropout_eval: bool = True,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Applies Transformer Encoder on the (input, output) pairs.

        Args:
            pairs: Input data as tokens. Shape (*B, R, C, 2).
                   - R: number of rows
                   - C: number of columns
                   - 2: two channels (input and output)
            grid_shapes: Shapes of the grids (e.g. 30x30). Shape (*B, 2, 2). The last two dimension
                         represent (rows, columns) of two channels: [[R_input, R_output], [C_input, C_output]].
                         Expects grid shapes values to be in [1, max_rows] and [1, max_cols].
            dropout_eval: if True, no dropout is applied; if False, dropout is applied.

        Returns:
            latent_mu:       shape (*B, latent_dim) = the mean latent embedding
            latent_logvar:   shape (*B, latent_dim) = the log-variance embedding (if variational=True),
                             otherwise None.
        """
        # 1) Embed
        x = self.embed_grids(pairs, grid_shapes, dropout_eval)
        
        squeeze = False
        if x.dim() == 4:
            original_size0, original_size1, original_size2, original_size3 = x.size()
            x = x.view(original_size0 * original_size1, original_size2, original_size3)
            squeeze = True

        # 2) Make pad mask
        #pad_mask = self.make_pad_mask(grid_shapes)
        pad_mask = None

        # 3) Pass through Transformer layers
        for layer in self.transformer_layers:
            x = layer(embeddings=x, dropout_eval=self.dropout, pad_mask=pad_mask)
            print("After transformer layer, x shape: ", x.shape)

        # 4) Extract the CLS embedding (x[..., 0, :] => shape [*B, emb_dim])
        cls_embed = x[..., 0, :]

        # 5) Layer norm on CLS
        cls_embed = self.cls_layer_norm(cls_embed)

        # 6) Project to latent space (mu, logvar if variational)
        latent_mu = self.latent_mu(cls_embed).float()  # shape [*B, latent_dim]

        if squeeze:
            latent_mu = latent_mu.view(original_size0, original_size1, -1)

        return latent_mu
  
    def embed_grids(self, pairs: torch.Tensor, grid_shapes: torch.Tensor, dropout_eval: bool) -> torch.Tensor:
        """
        Args:
            pairs: Input data as tokens. 
                - Case A: shape (B, R, C, 2)
                - Case B: shape (N, B, R, C, 2)
                More generally, (*leading_dims, R, C, 2).
            grid_shapes: Shapes of the grids (e.g. 30x30). 
                - Case A: shape (B, 2, 2)
                - Case B: shape (N, B, 2, 2)
                More generally, (*leading_dims, 2, 2).
                The last two dims represent (rows, columns) for two channels 
                (e.g. [[R_in, R_out], [C_in, C_out]]).
            dropout_eval: If True, do not apply dropout; if False, apply dropout.
        """
        config = self.config
        #print("\nDebugging embed_grids:")
        #print("Pairs shape: ", pairs.shape)
        #print("Grid shapes shape: ", grid_shapes.shape)

        # ---------------------------------------------------------------------
        # 1. Position Embeddings
        # ---------------------------------------------------------------------
        if config.scaled_position_embeddings:
            #print("Performing scaled position embeddings")
            pos_row_embed_layer = nn.Embedding(num_embeddings=1, embedding_dim=config.emb_dim).to(self.device)
            # shape: [max_rows, emb_dim]
            pos_row_embeds = pos_row_embed_layer(
                torch.zeros(config.max_rows, dtype=torch.long, device=self.device)
            )
            # multiply each row embedding by its 1..max_rows index
            row_range = torch.arange(1, config.max_rows + 1, device=self.device).unsqueeze(-1)
            pos_row_embeds = pos_row_embeds * row_range  # [max_rows, emb_dim]

            pos_col_embed_layer = nn.Embedding(num_embeddings=1, embedding_dim=config.emb_dim).to(self.device)
            # shape: [max_cols, emb_dim]
            pos_col_embeds = pos_col_embed_layer(
                torch.zeros(config.max_cols, dtype=torch.long, device=self.device)
            )
            col_range = torch.arange(1, config.max_cols + 1, device=self.device).unsqueeze(-1)
            pos_col_embeds = pos_col_embeds * col_range  # [max_cols, emb_dim]

            # Reshape to broadcast: [max_rows, 1, 1, emb_dim] + [1, max_cols, 1, emb_dim]
            pos_row_embeds = pos_row_embeds.view(config.max_rows, 1, 1, config.emb_dim)
            pos_col_embeds = pos_col_embeds.view(1, config.max_cols, 1, config.emb_dim)
            pos_embed = pos_row_embeds + pos_col_embeds  # [max_rows, max_cols, 1, emb_dim]
        else:
            #print("Performing non-scaled position embeddings")
            pos_row_embed_layer = nn.Embedding(num_embeddings=config.max_rows, embedding_dim=config.emb_dim).to(self.device)
            row_indices = torch.arange(config.max_rows, dtype=torch.long, device=self.device)  # [max_rows]
            pos_row_embeds = pos_row_embed_layer(row_indices)  # [max_rows, emb_dim]

            pos_col_embed_layer = nn.Embedding(num_embeddings=config.max_cols, embedding_dim=config.emb_dim).to(self.device)
            col_indices = torch.arange(config.max_cols, dtype=torch.long, device=self.device)  # [max_cols]
            pos_col_embeds = pos_col_embed_layer(col_indices)  # [max_cols, emb_dim]

            # Reshape for broadcast
            pos_row_embeds = pos_row_embeds.view(config.max_rows, 1, 1, config.emb_dim)
            pos_col_embeds = pos_col_embeds.view(1, config.max_cols, 1, config.emb_dim)
            pos_embed = pos_row_embeds + pos_col_embeds  # [max_rows, max_cols, 1, emb_dim]

        #print("After position embeddings:")
        #print("Position embeddings shape: ", pos_embed.shape)

        # ---------------------------------------------------------------------
        # 2. Colors Embedding
        # ---------------------------------------------------------------------
        colors_embed_layer = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.emb_dim).to(self.device)
        colors_embed = colors_embed_layer(pairs.to(self.device).long())
        # shape: [*leading_dims, R, C, 2, emb_dim]
        #print("After colors embeddings:")
        #print("Colors embeddings shape: ", colors_embed.shape)

        # ---------------------------------------------------------------------
        # 3. Channels Embedding
        # ---------------------------------------------------------------------
        channels_embed_layer = nn.Embedding(num_embeddings=2, embedding_dim=config.emb_dim).to(self.device)
        channels_base = torch.arange(2, dtype=torch.long, device=self.device)
        # shape: [2, emb_dim]
        channels_embed_raw = channels_embed_layer(channels_base)
        # We'll reshape for broadcast-add later.

        #print("After channels embeddings:")
        #print("Channels embeddings shape: ", channels_embed_raw.shape, "(unreshaped)")

        # ---------------------------------------------------------------------
        # 4. Combine colors + positions + channels
        #    - Slice pos_embed for the actual R, C
        #    - Add them up with broadcast
        #    - Flatten (R*C*2) into one dimension
        # ---------------------------------------------------------------------
        R, C = pairs.shape[-3], pairs.shape[-2]  # actual row, col from input
        # pos_embed: [max_rows, max_cols, 1, emb_dim] => slice to [R, C, 1, emb_dim]
        pos_embed_slice = pos_embed[:R, :C, :, :]

        # channels_embed_raw => shape [1,1,1,2,emb_dim] for broadcast
        # We'll insert enough leading dims to match colors_embed
        # E.g. if colors_embed is [N, B, R, C, 2, emb_dim], we want
        # channels_embed_raw to be [1, 1, 1, 1, 2, emb_dim].
        # So let's do:
        channels_embed_broadcast = channels_embed_raw.view(
            *([1] * (colors_embed.dim() - 2)), 2, config.emb_dim
        )
        #print("Channels embeddings shape: ", channels_embed_broadcast.shape)
        # Now channels_embed_broadcast has shape [1, ..., 1, 2, emb_dim]
        # matching the rank of colors_embed.

        # Now do the broadcast add:
        # colors: [*leading_dims, R, C, 2, emb_dim]
        # pos:    [       R,       C,   1, emb_dim] => broadcast on leading_dims & 2
        # channel:[... 1, 2, emb_dim]
        x = colors_embed + pos_embed_slice + channels_embed_broadcast
        # shape => [*leading_dims, R, C, 2, emb_dim]

        # 1) Identify leading dimensions (everything before R, C, 2, emb_dim).
        leading_dims = x.shape[:-4]

        # 2) Flatten (R*C*2) into one dimension, keep emb_dim as is.
        x = x.view(*leading_dims, -1, x.shape[-1])

        #print("After combining colors + positions + channels:")
        #print("Combined embeddings shape: ", x.shape)

        # ---------------------------------------------------------------------
        # 5. Embed the grid shape tokens
        #    (e.g. rows in [*leading_dims, 2], columns in [*leading_dims, 2])
        # ---------------------------------------------------------------------
        grid_shapes_row_embed_layer = nn.Embedding(num_embeddings=config.max_rows, embedding_dim=config.emb_dim).to(self.device)
        # row_indices => shape [*leading_dims, 2]
        row_indices = (grid_shapes[..., 0, :].long().to(self.device)) - 1
        # embed => [*leading_dims, 2, emb_dim]
        row_embed = grid_shapes_row_embed_layer(row_indices)

        # Similarly for columns
        grid_shapes_col_embed_layer = nn.Embedding(num_embeddings=config.max_cols, embedding_dim=config.emb_dim).to(self.device)
        col_indices = (grid_shapes[..., 1, :].long().to(self.device)) - 1
        col_embed = grid_shapes_col_embed_layer(col_indices)

        # We also want to add the channel embeddings to row/col embeddings.
        # row_embed: [*leading_dims, 2, emb_dim]
        # channels_embed_raw: [2, emb_dim]. We'll shape it => [1,...,1,2,emb_dim]
        # so it broadcasts with row_embed.
        channels_embed_grid = channels_embed_raw.view(
            *([1] * (row_embed.dim() - 2)), 2, config.emb_dim
        )
        # Add
        grid_shapes_row_embed = row_embed + channels_embed_grid
        grid_shapes_col_embed = col_embed + channels_embed_grid

        #print("After row embeddings:")
        #print("Row embeddings shape: ", grid_shapes_row_embed.shape)
        #print("After column embeddings:")
        #print("Column embeddings shape: ", grid_shapes_col_embed.shape)

        # Concatenate rows+cols => [*leading_dims, 4, emb_dim]
        grid_shapes_embed = torch.cat([grid_shapes_row_embed, grid_shapes_col_embed], dim=-2)

        # Now cat with x => [*leading_dims, 4 + R*C*2, emb_dim]
        x = torch.cat([grid_shapes_embed, x], dim=-2)

        #print("After combining grid shape embeddings:")
        #print("Combined embeddings shape: ", x.shape)

        # ---------------------------------------------------------------------
        # 6. Add the cls token => shape [*leading_dims, 1 + 4 + R*C*2, emb_dim]
        # ---------------------------------------------------------------------
        cls_token_layer = nn.Embedding(num_embeddings=1, embedding_dim=config.emb_dim).to(self.device)

        # shape [*leading_dims, 1]
        cls_index = torch.zeros_like(x[..., 0:1, 0], dtype=torch.long, device=self.device)
        # embed => [*leading_dims, 1, emb_dim]
        cls_token = cls_token_layer(cls_index)

        x = torch.cat([cls_token, x], dim=-2)

        # ---------------------------------------------------------------------
        # 7. Apply Dropout
        # ---------------------------------------------------------------------
        if dropout_eval:
            with torch.no_grad():
                out = x
        else:
            out = self.dropout(x)

        return out
    
    def make_pad_mask(self, grid_shapes: torch.Tensor) -> torch.Tensor:
        """
        Make the pad mask False outside of the grid shapes and True inside.

        Args:
            grid_shapes: shapes of the grids. Shape can be (B, 2, 2) or (n, B, 2, 2).
                The last two dimensions represent (rows, columns) of two channels, e.g. 
                [[R_input, R_output], [C_input, C_output]].

        Returns:
            pad mask of shape (*grid_shapes.shape[:-2], 1, T, T) with 
            T = 1 + 4 + 2 * max_rows * max_cols.
        """
        batch_ndims = len(grid_shapes.shape[:-2])  # Account for batch dimensions
        
        # Generate row mask
        row_arange_broadcast = torch.arange(self.config.max_rows, device=grid_shapes.device).view(
            *([1] * batch_ndims), self.config.max_rows, 1
        )
        row_mask = row_arange_broadcast < grid_shapes[..., 0:1, :]
        
        # Generate column mask
        col_arange_broadcast = torch.arange(self.config.max_cols, device=grid_shapes.device).view(
            *([1] * batch_ndims), self.config.max_cols, 1
        )
        col_mask = col_arange_broadcast < grid_shapes[..., 1:2, :]
        
        # Combine row and column masks
        pad_mask = row_mask[..., :, None, :] & col_mask[..., None, :, :]
        
        # Flatten rows, columns, and channels
        pad_mask = pad_mask.view(*pad_mask.shape[:-3], 1, -1)
        
        # Add the masks corresponding to the cls token and grid shape tokens
        additional_tokens = torch.ones((*pad_mask.shape[:-1], 1 + 4), dtype=torch.bool, device=grid_shapes.device)
        pad_mask = torch.cat([additional_tokens, pad_mask], dim=-1)
        
        # Outer product to create the self-attention mask
        pad_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(-2)
        
        return pad_mask

    
    def embed_grids_non_adapted(self, pairs: torch.Tensor, grid_shapes: torch.Tensor, dropout_eval: bool) -> torch.Tensor:
        """
        Args:
            pairs: Input data as tokens. Shape (*B, R, C, 2).
                   - R: number of rows
                   - C: number of columns
                   - 2: two channels (input and output)
            grid_shapes: Shapes of the grids (e.g. 30x30). Shape (*B, 2, 2).
                         The last two dimensions represent (rows, columns) of two channels:
                         e.g. [[R_input, R_output], [C_input, C_output]]
            dropout_eval: If True, use evaluation (no dropout), else train (use dropout).
        """
        config = self.config
        print('\n Debugging embed_grids:')
        print('Pairs shape: ', pairs.shape)
        print('Grid shapes shape: ', grid_shapes.shape)

        # ---------------------------------------------------------------------
        # 1. Position Embeddings
        # ---------------------------------------------------------------------
        if config.scaled_position_embeddings:
            print('Performing scaled position embeddings')
            pos_row_embed_layer = nn.Embedding(num_embeddings=1, embedding_dim=config.emb_dim).to(self.device)
            pos_row_embeds = pos_row_embed_layer(torch.zeros(config.max_rows, dtype=torch.long, device=self.device))
            row_range = torch.arange(1, config.max_rows + 1, device=self.device).unsqueeze(-1)
            pos_row_embeds = pos_row_embeds * row_range

            pos_col_embed_layer = nn.Embedding(num_embeddings=1, embedding_dim=config.emb_dim).to(self.device)
            pos_col_embeds = pos_col_embed_layer(torch.zeros(config.max_cols, dtype=torch.long, device=self.device))
            col_range = torch.arange(1, config.max_cols + 1, device=self.device).unsqueeze(-1)
            pos_col_embeds = pos_col_embeds * col_range

            pos_row_embeds = pos_row_embeds.view(config.max_rows, 1, 1, config.emb_dim)
            pos_col_embeds = pos_col_embeds.view(1, config.max_cols, 1, config.emb_dim)
            pos_embed = pos_row_embeds + pos_col_embeds

        else:
            print('Performing non-scaled position embeddings')
            pos_row_embed_layer = nn.Embedding(num_embeddings=config.max_rows, embedding_dim=config.emb_dim).to(self.device)
            row_indices = torch.arange(config.max_rows, dtype=torch.long, device=self.device)
            pos_row_embeds = pos_row_embed_layer(row_indices)

            pos_col_embed_layer = nn.Embedding(num_embeddings=config.max_cols, embedding_dim=config.emb_dim).to(self.device)
            col_indices = torch.arange(config.max_cols, dtype=torch.long, device=self.device)
            pos_col_embeds = pos_col_embed_layer(col_indices)

            pos_row_embeds = pos_row_embeds.view(config.max_rows, 1, 1, config.emb_dim)
            pos_col_embeds = pos_col_embeds.view(1, config.max_cols, 1, config.emb_dim)
            pos_embed = pos_row_embeds + pos_col_embeds

        print('After position embeddings:')
        print('Position embeddings shape: ', pos_embed.shape)

        # ---------------------------------------------------------------------
        # 2. Colors Embedding
        # ---------------------------------------------------------------------
        colors_embed_layer = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.emb_dim).to(self.device)
        colors_embed = colors_embed_layer(pairs.to(self.device).long())
        print('After colors embeddings:')
        print('Colors embeddings shape: ', colors_embed.shape)

        # ---------------------------------------------------------------------
        # 3. Channels Embedding
        # ---------------------------------------------------------------------
        channels_embed_layer = nn.Embedding(num_embeddings=2, embedding_dim=config.emb_dim).to(self.device)
        channels_base = torch.arange(2, dtype=torch.long, device=self.device)
        channels_embed_raw = channels_embed_layer(channels_base).view(1, 1, 1, 2, config.emb_dim)
        print('After channels embeddings:')
        print('Channels embeddings shape: ', channels_embed_raw.shape)

        # ---------------------------------------------------------------------
        # 4. Combine colors + positions + channels
        # ---------------------------------------------------------------------
        R, C = pairs.shape[-3], pairs.shape[-2]
        pos_embed_slice = pos_embed[:R, :C, :, :]
        x = colors_embed + pos_embed_slice + channels_embed_raw
        num_batch_dims = x.dim() - 3  # how many leading batch dims
        x = x.view(x.shape[0], -1, x.shape[-1])
        print('After combining colors + positions + channels:')
        print('Combined embeddings shape: ', x.shape)

        # ---------------------------------------------------------------------
        # 5. Embed the grid shape tokens
        # ---------------------------------------------------------------------
        grid_shapes_row_embed_layer = nn.Embedding(num_embeddings=config.max_rows, embedding_dim=config.emb_dim).to(self.device)
        row_indices = grid_shapes[..., 0, :].to(self.device).long() - 1
        grid_shapes_row_embed = grid_shapes_row_embed_layer(row_indices) + channels_embed_raw.view(2, config.emb_dim)
        print('After row embeddings:')
        print('Row embeddings shape: ', grid_shapes_row_embed.shape)

        grid_shapes_col_embed_layer = nn.Embedding(num_embeddings=config.max_cols, embedding_dim=config.emb_dim).to(self.device)
        col_indices = grid_shapes[..., 1, :].to(self.device).long() - 1
        grid_shapes_col_embed = grid_shapes_col_embed_layer(col_indices) + channels_embed_raw.view(2, config.emb_dim)
        print('After column embeddings:')
        print('Column embeddings shape: ', grid_shapes_col_embed.shape)

        grid_shapes_embed = torch.cat([grid_shapes_row_embed, grid_shapes_col_embed], dim=-2)
        x = torch.cat([grid_shapes_embed, x], dim=-2)
        print('After combining grid shape embeddings:')
        print('Combined embeddings shape: ', x.shape)

        # ---------------------------------------------------------------------
        # 6. Add the cls token
        # ---------------------------------------------------------------------
        cls_token_layer = nn.Embedding(num_embeddings=1, embedding_dim=config.emb_dim).to(self.device)
        cls_index = torch.zeros_like(x[..., 0:1, 0], dtype=torch.long, device=self.device)
        cls_token = cls_token_layer(cls_index)
        x = torch.cat([cls_token, x], dim=-2)

        # ---------------------------------------------------------------------
        # 7. Apply Dropout
        # ---------------------------------------------------------------------
        if dropout_eval:
            with torch.no_grad():
                out = x
        else:
            out = self.dropout(x)

        return out

    def old_forward(
        self,
        pairs: torch.Tensor,
        grid_shapes: torch.Tensor,
        dropout_eval: bool = False
    ) -> torch.Tensor:
        raise DeprecationWarning('This function is deprecated. Use forward instead.')
        
        # Embed the input grids
        x = self.embed_grids(pairs, grid_shapes, dropout_eval)

        # Create padding mask for attention
        key_padding_mask = self.make_pad_mask(grid_shapes)
        print('Key padding mask shape: ', key_padding_mask.shape)

        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, dropout_eval=dropout_eval)

        # Extract CLS token and project to latent space
        cls_embed = x[..., 0, :]
        cls_embed = self.cls_layer_norm(cls_embed)
        latent_mu = self.latent_mu(cls_embed).float()
        #latent_logvar = self.latent_logvar(cls_embed).float() if self.latent_logvar else None
        return latent_mu
    

    def embed_grids_old(self, pairs, grid_shapes, dropout_eval):
        raise DeprecationWarning('This function is deprecated. Use embed_grids instead.')
        config = self.config
        print('\n Embedding grids:')
        
        # Calculate position embeddings
        if config.scaled_position_embeddings:
            print('Scaled position embeddings')
            r_emb = self.pos_row_embed(torch.zeros(config.max_rows, dtype=torch.long, device=pairs.device))
            c_emb = self.pos_col_embed(torch.zeros(config.max_cols, dtype=torch.long, device=pairs.device))
            row_factors = torch.arange(1, config.max_rows + 1, device=pairs.device).unsqueeze(-1) * r_emb
            col_factors = torch.arange(1, config.max_cols + 1, device=pairs.device).unsqueeze(-1) * c_emb
            pos_embed = row_factors[:, None, None, :] + col_factors[None, :, None, :]
        else:
            print('Non-scaled position embeddings')
            r_emb = self.pos_row_embed(torch.arange(config.max_rows, dtype=torch.long, device=pairs.device))
            c_emb = self.pos_col_embed(torch.arange(config.max_cols, dtype=torch.long, device=pairs.device))
            pos_embed = r_emb[:, None, None, :] + c_emb[None, :, None, :]

        # Embed colors and channels
        colors = self.colors_embed(pairs)  # [batch, R, C, 2, emb_dim]
        channels = self.channels_embed(torch.arange(2, dtype=torch.long, device=pairs.device))  # [2, emb_dim]

        # Combine embeddings
        x = colors + pos_embed + channels  # [batch, R, C, 2, emb_dim]

        # Flatten the grid dimensions
        x = x.view(x.shape[0], -1, x.shape[-1])  # [batch, R*C*2, emb_dim]

        # Embed grid shapes
        row_embed = self.grid_shapes_row_embed(grid_shapes[..., 0, :] - 1) + channels  # [batch, 1, emb_dim]
        col_embed = self.grid_shapes_col_embed(grid_shapes[..., 1, :] - 1) + channels  # [batch, 1, emb_dim]
        shape_embed = torch.cat([row_embed, col_embed], dim=-2)  # [batch, 2, emb_dim]

        # Concatenate shape embeddings with grid embeddings
        x = torch.cat([shape_embed, x], dim=-2)  # [batch, 2 + R*C*2, emb_dim]

        # Add CLS token
        cls_tok = self.cls_token(torch.zeros(x.shape[0], 1, dtype=torch.long, device=pairs.device))
        x = torch.cat([cls_tok, x], dim=-2)  # [batch, 3 + R*C*2, emb_dim]

        # Apply dropout if not in evaluation mode
        x = self.dropout(x) if not dropout_eval else x
        return x
    
    
    def make_pad_mask_old(self, grid_shapes):
        """
        Creates a padding mask for the attention mechanism.
        
        Args:
            grid_shapes (torch.Tensor): Shape [batch_size, 2, 2]
        
        Returns:
            torch.Tensor: Padding mask of shape [batch_size, seq_len]
        """
        raise DeprecationWarning('This function is deprecated. Use make_pad_mask instead.')
        if len(grid_shapes.shape) == 3:
            batch_size = grid_shapes.shape[0]
            seq_len = 1 + 4 + 2 * (self.config.max_rows * self.config.max_cols)  # Example calculation
            # Initialize mask with no padding
            key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=grid_shapes.device)
        else:
            batch_size = grid_shapes.shape[1]
        return key_padding_mask