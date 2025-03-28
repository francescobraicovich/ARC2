import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
from utils.util import set_device

DEVICE = set_device('world_model/transformer.py')

class EncoderTransformerConfig:
    """
    Stand-in for the real config. Adjust fields to match your usage.
    """
    def __init__(
        self,
        vocab_size=11,
        max_rows=30,
        max_cols=30,
        emb_dim=32,
        latent_dim=32,
        num_layers=1,
        scaled_position_embeddings=False,
        variational=False,
        latent_projection_bias=False,
        dtype=torch.float32,
        transformer_layer=None
    ):
        self.vocab_size = vocab_size
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.scaled_position_embeddings = scaled_position_embeddings
        self.variational = variational
        self.latent_projection_bias = latent_projection_bias
        self.dtype = dtype
        self.transformer_layer = transformer_layer
        # For convenience:
        self.max_len = max_rows * max_cols
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

    def forward(self, embedded_action, embedded_state, dropout_eval: bool):
       
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
        shape_row_logits = self.shape_row_proj(x[:, embedded_state.size(1)+1, :])
        shape_col_logits = self.shape_col_proj(x[:, embedded_state.size(1)+2, :])
        grid_logits = self.grid_proj(x[:, embedded_state.size(1)+3:, :])
        return shape_row_logits, shape_col_logits, grid_logits
