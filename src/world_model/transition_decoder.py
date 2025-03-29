# =^ . ^=
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
from utils.util import set_device

DEVICE = set_device('world_model/transformer.py')

#NOTE: to adjust
class DecoderTransformerConfig:
        def __init__(
            self,
            emb_dim: int,
            num_heads: int, # ADDED
            attention_dropout_rate: float, # ADDED
            dropout_rate: float, # ADDED
            mlp_dim_factor: int, # ADDED
            activation: str, # ADDED
            use_bias: bool, # ADDED
            max_rows: int = 30,
            max_cols: int = 30,
            vocab_size: int = 10, # NOTE: vocab_size is the number of unique tokens in the vocabulary
            num_layers: int = 2 # NOTE: number of transformer layers
                    ):
            self.emb_dim = emb_dim
            self.max_rows = max_rows
            self.max_cols = max_cols
            self.vocab_size = vocab_size
            self.num_layers = num_layers
            self.num_heads = num_heads # ADDED
            self.attention_dropout_rate = attention_dropout_rate # ADDED
            self.dropout_rate = dropout_rate # ADDED
            self.mlp_dim_factor = mlp_dim_factor # ADDED
            self.activation = activation # ADDED
            self.use_bias = use_bias # ADDED
            

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
    Now accepts two separate embedded inputs that are projected to a common space.
    '''
    def __init__(self, config: DecoderTransformerConfig, emb_action_dim: int, emb_state_dim: int):
        super().__init__()
        self.config = config
        
        # Here we assume a simple split: each projection outputs half of config.emb_dim;
        self.emb_action_proj = nn.Linear(emb_action_dim, config.emb_dim // 2).to(DEVICE)
        self.emb_state_proj = nn.Linear(emb_state_dim, config.emb_dim // 2).to(DEVICE)
    
        # Create transformer layers using the unified config.
        self.layers = nn.ModuleList([TransformerLayer(config).to(DEVICE) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.emb_dim).to(DEVICE)
        
        # Projections to logits.
        self.shape_row_proj = nn.Linear(config.emb_dim, config.max_rows).to(DEVICE)
        self.shape_col_proj = nn.Linear(config.emb_dim, config.max_cols).to(DEVICE)
        # grid_proj now outputs max_rows * max_cols * vocab_size logits.
        self.grid_proj = nn.Linear(config.emb_dim, config.max_rows * config.max_cols * config.vocab_size).to(DEVICE)

    def forward(self, embedded_action, embedded_state, dropout_eval: bool):
        '''
        Args:
            embedded_action: shape (B, emb_action_dim)
            embedded_state: shape (B, emb_state_dim)
            dropout_eval: bool, if True, disables dropout for evaluation mode
            
        Returns:
            shape_row_logits: shape (B, R), the logits for grid shape row
            shape_col_logits: shape (B, C), the logits for grid shape col
            grid_logits: shape (B, R*C, vocab_size), the logits for grid tokens
        '''
        assert len(embedded_action.shape) == 2 and len(embedded_state.shape) == 2

        # Project each input to half of config.emb_dim.
        x_action = self.emb_action_proj(embedded_action)
        x_state = self.emb_state_proj(embedded_state)
        # Concatenate along the embedding (feature) dimension
        x = torch.cat([x_action, x_state], dim=1)
        print("DEBUG: after projection and concatenation, x shape:", x.shape)

        # Pass through transformer layers.
        for i, layer in enumerate(self.layers):
            x = layer(x, dropout_eval=dropout_eval)
            print(f"DEBUG: after transformer layer {i}, x shape:", x.shape)
        
        # Apply layer normalization.
        x = self.layer_norm(x)
        print("DEBUG: after layer normalization, x shape:", x.shape)
        
        # Project to logits.
        shape_row_logits = self.shape_row_proj(x)
        shape_col_logits = self.shape_col_proj(x)
        grid_logits = self.grid_proj(x)  # (B, max_rows * max_cols * vocab_size)
        print("DEBUG: after grid projection, grid_logits shape:", grid_logits.shape)
        
        B = grid_logits.size(0)
        grid_logits = grid_logits.view(B, self.config.max_rows * self.config.max_cols, self.config.vocab_size)
        print("DEBUG: after reshaping grid_logits, grid_logits shape:", grid_logits.shape)
        
        print("DEBUG: shape_row_logits shape:", shape_row_logits.shape)
        print("DEBUG: shape_col_logits shape:", shape_col_logits.shape)
        
        return shape_row_logits, shape_col_logits, grid_logits



