import torch
import torch.nn as nn

# go up one directory to import utils
from utils.util import set_device
from world_model.transformer import TransformerLayer
import os

# Determine the device: CUDA -> MPS -> CPU
DEVICE = set_device('action_space_embed.py')

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
        self.target_state = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long).to(DEVICE) # dummy tensor that gets updated

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

        # Bring self to the right device
        self.to(DEVICE)

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
        # assert that all state values are between 0 and vocab_size
        assert torch.all(state >= 0) and torch.all(state < self.config.vocab_size), \
            f"Input colors should be between 0 and {self.config.vocab_size - 1}, got {state.min()} and {state.max()}"
        # assert that all shape values are between 0 and max_rows/max_cols
        assert torch.all(shape[:, 0] >= 0) and torch.all(shape[:, 0] < self.config.max_rows), \
            f"Input rows should be between 0 and {self.config.max_rows - 1}, got {shape[:, 0].min()} and {shape[:, 0].max()}"
        # assert that all shape values are between 0 and max_rows/max_cols
        assert torch.all(shape[:, 1] >= 0) and torch.all(shape[:, 1] < self.config.max_cols), \
            f"Input cols should be between 0 and {self.config.max_cols - 1}, got {shape[:, 1].min()} and {shape[:, 1].max()}"

        x = self.embed_grids(state, shape, dropout_eval)

        pad_mask = self.make_pad_mask(shape)
        for layer in self.transformer_layers:
            x = layer(embeddings=x, dropout_eval=dropout_eval, pad_mask=pad_mask)
            # TODO: Check the pad mask is used correctly in the layer (not inverted).
        # Extract the CLS token (first token).
        cls_embed = x[:, 0, :]
        cls_embed = self.cls_layer_norm(cls_embed)
        latent_mu = self.latent_mu(cls_embed).to(torch.float32)

        if self.config.variational:
            latent_logvar = self.latent_logvar(cls_embed).to(torch.float32)
        else:
            latent_logvar = None
        return latent_mu

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
            assert torch.all(pos_row_indices < config.max_rows), \
                f"Position indices should be between 0 and {config.max_rows - 1}, got {pos_row_indices.min()} and {pos_row_indices.max()}"
            assert torch.all(pos_row_indices >= 0), \
                f"Position indices should be between 0 and {config.max_rows - 1}, got {pos_row_indices.min()} and {pos_row_indices.max()}"
            pos_row_embed = self.pos_row_embed(pos_row_indices)  # (max_rows, emb_dim)
            pos_col_indices = torch.arange(config.max_cols, dtype=torch.long, device=DEVICE)
            assert torch.all(pos_col_indices < config.max_cols), \
                f"Position indices should be between 0 and {config.max_cols - 1}, got {pos_col_indices.min()} and {pos_col_indices.max()}"
            assert torch.all(pos_col_indices >= 0), \
                f"Position indices should be between 0 and {config.max_cols - 1}, got {pos_col_indices.min()} and {pos_col_indices.max()}"
            pos_col_embed = self.pos_col_embed(pos_col_indices)  # (max_cols, emb_dim)
            pos_embed = pos_row_embed.unsqueeze(1) + pos_col_embed.unsqueeze(0)

        # Colors embedding block.
        # pairs: (B, R, C) -> colors_embed: (B, R, C, emb_dim)
        colors_embed = self.colors_embed(state)

        # Combine all embeddings.
        # Broadcasting: pos_embed (max_rows, max_cols, 1, emb_dim) will broadcast to (R, C, 2, emb_dim) if R==max_rows and C==max_cols.
        x = colors_embed + pos_embed  # (B, R, C, emb_dim)

        # Flatten the spatial and channel dimensions.
        B = x.shape[0]
        x = x.view(B, -1, x.shape[-1])  # (B, R*C, emb_dim)

        # Embed grid shape tokens.
        # grid_shapes: (B, 2, 2)
        grid_shapes_row = shape[:, 0].long()  # (B)
        assert torch.all(grid_shapes_row >= 0) and torch.all(grid_shapes_row < config.max_rows), \
            f"Grid shape row indices should be between 0 and {config.max_rows - 1}, got {grid_shapes_row.min()} and {grid_shapes_row.max()}"
        grid_shapes_row_embed = self.grid_shapes_row_embed(grid_shapes_row)  # (B, emb_dim)
        grid_shapes_row_embed = grid_shapes_row_embed.unsqueeze(1)  # (B, 1, emb_dim)

        grid_shapes_col = shape[:, 1].long()  # (B)
        grid_shapes_col_embed = self.grid_shapes_col_embed(grid_shapes_col)  # (B, emb_dim)
        assert torch.all(grid_shapes_col >= 0) and torch.all(grid_shapes_col < config.max_cols), \
            f"Grid shape column indices should be between 0 and {config.max_cols - 1}, got {grid_shapes_col.min()} and {grid_shapes_col.max()}"
        grid_shapes_col_embed = grid_shapes_col_embed.unsqueeze(1)  # (B, 1, emb_dim) 

        # Concatenate row and column grid tokens.
        grid_shapes_embed = torch.cat([grid_shapes_row_embed, grid_shapes_col_embed], dim=1)  # (B, 2, emb_dim)
        x = torch.cat([grid_shapes_embed, x], dim=1)  # (B, 2 + R*C, emb_dim)

        # Add the CLS token.
        cls_token = self.cls_token(torch.zeros(x.shape[0], 1, dtype=torch.long, device=DEVICE))  # (B, 1, emb_dim)
        x = torch.cat([cls_token, x], dim=1)  # (B, 1 + 2 + 2*R*C, emb_dim)

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
        B = shape.shape[0]
        # Create a row mask of shape (B, max_rows)
        row_arange = torch.arange(self.config.max_rows, device=DEVICE).view(1, self.config.max_rows) # (1, max_rows)
        row_mask = row_arange < shape[:, 0].view(B, 1)  # (B, max_rows)

        # Create a column mask of shape (B, max_cols)
        col_arange = torch.arange(self.config.max_cols, device=DEVICE).view(1, self.config.max_cols)  # (1, max_cols)
        col_mask = col_arange < shape[:, 1].view(B, 1)  # (B, max_cols)
        # Combine to get a spatial mask.
        row_mask = row_mask.unsqueeze(2)  # (B, max_rows, 1)
        col_mask = col_mask.unsqueeze(1)  # (B, 1, max_cols)
        pad_mask = row_mask & col_mask  # (B, max_rows, max_cols)
        pad_mask = pad_mask.view(B, 1, -1)  # (B, 1, max_rows*max_cols)

        # Prepend ones for the CLS token and grid shape tokens (1+4 tokens).
        ones_mask = torch.ones(B, 1, 1 + 2, dtype=torch.bool, device=DEVICE)
        pad_mask = torch.cat([ones_mask, pad_mask], dim=-1)  # (B, 1, 1+2+max_rows*max_cols)

        # Outer product to build a full attention mask.
        pad_mask = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(3)  # (B, 1, T, T)
        return pad_mask
    
    def encode(self, state, shape, new_episode=False):
        current_state = state[:, :, 1] + 1 # ensure values are between 0 and 10 (padding tokens go from -1 to 0)
        target_state = state[:, :, 0] + 1 # ensure values are between 0 and 10 (padding tokens go from -1 to 0)
        current_shape = shape[0, :] - 1 # ensure values are between 0 and 29 (1 to 30 would break the embedding)
        target_shape = shape[1, :] - 1 # ensure values are between 0 and 29 (1 to 30 would break the embedding)

        # unsqueeze to add batch dimension
        current_state = current_state.unsqueeze(0)
        target_state = target_state.unsqueeze(0)
        current_shape = current_shape.unsqueeze(0)
        target_shape = target_shape.unsqueeze(0)

        if torch.equal(target_state, self.target_state):
            embedded_target_state = self.target_state_embed
        else:
            with torch.no_grad():
                target_state = target_state.long()
                target_shape = target_shape.long()
                embedded_target_state = self.forward(target_state, target_shape, dropout_eval=True)
                self.target_state_embed = embedded_target_state

        with torch.no_grad():
            current_state = current_state.long()
            current_shape = current_shape.long()
            current_state_embed = self.forward(current_state, current_shape, dropout_eval=True)
            x = torch.cat([current_state_embed, self.target_state_embed], dim=1)
            x = x.squeeze(0)        
        return x
    
    def save_weights(self, path: str):
        """
        Save the embedding weights to a file.
        
        Args:
            path (str): Path to save the weights.
        """
        # append the 'encoder.pt' to the path
        path = path + '/encoder.pt'
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        """
        Load the embedding weights from a file.
        
        Args:
            path (str): Path to load the weights from.
        """
        # append the 'encoder.pt' to the path
        path = path + '/encoder.pt'
        self.load_state_dict(torch.load(path))

    @property
    def num_parameters(self):
        """
        Returns the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)