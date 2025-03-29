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



class DecoderTransformer(nn.Module):
    def __init__(self, config):
        """
        config should have at least:
          - max_rows, max_cols: maximum grid dimensions.
          - emb_dim: embedding dimension.
          - vocab_size: size of the color vocabulary.
          - max_len: maximum number of grid tokens (typically R * C).
          - latent_dim: dimension of the latent context.
          - output_vocab_size: vocabulary size for grid tokens.
          - scaled_position_embeddings: bool, whether to use scaled positional embeddings.
          - next_position_embeddings: bool, whether to use a different mechanism for positions.
          - next_position_embeddings_new_input_embeds: bool, whether to create new input position embeddings.
          - logits_projection_bias: bool, whether to use bias in the projection layers.
          - num_layers: number of decoder transformer layers.
          - transformer_layer: a sub-config with fields:
              * dropout_rate
              * num_heads
              * use_bias
              * ffn_dim
        """
        super(DecoderTransformer, self).__init__()
        self.config = config

        # ----------------------------
        # Context embedding: project latent context to emb_dim.
        # ----------------------------
        self.context_embed = nn.Linear(config.latent_dim, config.emb_dim, bias=config.transformer_layer.use_bias)

        # ----------------------------
        # Positional embeddings (shared for input/output tokens).
        # If scaled, we use an embedding with a single token that is later scaled.
        # ----------------------------
        if config.scaled_position_embeddings:
            self.pos_row_embed = nn.Embedding(1, config.emb_dim)
            self.pos_col_embed = nn.Embedding(1, config.emb_dim)
        else:
            self.pos_row_embed = nn.Embedding(config.max_rows, config.emb_dim)
            self.pos_col_embed = nn.Embedding(config.max_cols, config.emb_dim)

        # ----------------------------
        # Next position embeddings for input tokens (if enabled).
        # ----------------------------
        if config.next_position_embeddings and config.next_position_embeddings_new_input_embeds:
            if config.scaled_position_embeddings:
                self.input_pos_row_embed = nn.Embedding(1, config.emb_dim)
                self.input_pos_col_embed = nn.Embedding(1, config.emb_dim)
            else:
                self.input_pos_row_embed = nn.Embedding(config.max_rows, config.emb_dim)
                self.input_pos_col_embed = nn.Embedding(config.max_cols, config.emb_dim)

        # ----------------------------
        # Grid shapes embedding: for both input and output grid tokens.
        # ----------------------------
        self.grid_shapes_row_embed = nn.Embedding(config.max_rows, config.emb_dim)
        self.grid_shapes_col_embed = nn.Embedding(config.max_cols, config.emb_dim)

        # ----------------------------
        # Colors embedding: for the grid cell tokens.
        # ----------------------------
        self.colors_embed = nn.Embedding(config.vocab_size, config.emb_dim)

        # ----------------------------
        # Input/output type embedding: one embedding for input tokens and one for output tokens.
        # We create a 2-token embedding and later split it.
        # ----------------------------
        self.input_output_embed = nn.Embedding(2, config.emb_dim)

        # ----------------------------
        # Dropout for embeddings.
        # ----------------------------
        self.embed_dropout = nn.Dropout(config.transformer_layer.dropout_rate)

        # ----------------------------
        # Transformer decoder layers.
        # ----------------------------
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config.transformer_layer) for _ in range(config.num_layers)
        ])

        # ----------------------------
        # For extracting logits, we build three layer norms and projection layers.
        # (Note: Flax’s use_scale=False is simulated by freezing the weight to 1.)
        # ----------------------------
        self.row_logits_layer_norm = nn.LayerNorm(config.emb_dim, elementwise_affine=True)
        self.col_logits_layer_norm = nn.LayerNorm(config.emb_dim, elementwise_affine=True)
        self.grid_logits_layer_norm = nn.LayerNorm(config.emb_dim, elementwise_affine=True)
        with torch.no_grad():
            self.row_logits_layer_norm.weight.fill_(1.0)
            self.col_logits_layer_norm.weight.fill_(1.0)
            self.grid_logits_layer_norm.weight.fill_(1.0)
        if not config.transformer_layer.use_bias:
            self.row_logits_layer_norm.bias.data.zero_()
            self.col_logits_layer_norm.bias.data.zero_()
            self.grid_logits_layer_norm.bias.data.zero_()
            self.row_logits_layer_norm.bias.requires_grad = False
            self.col_logits_layer_norm.bias.requires_grad = False
            self.grid_logits_layer_norm.bias.requires_grad = False

        self.shape_row_logits_proj = nn.Linear(config.emb_dim, config.max_rows, bias=config.logits_projection_bias)
        self.shape_col_logits_proj = nn.Linear(config.emb_dim, config.max_cols, bias=config.logits_projection_bias)
        self.grid_logits_proj = nn.Linear(config.emb_dim, config.output_vocab_size, bias=config.logits_projection_bias)

    def forward(self, input_seq, output_seq, context, dropout_eval):
        """
        Args:
          input_seq: Tensor of shape (B, 2+R*C) – first 2 tokens are grid shapes.
          output_seq: Tensor of shape (B, 2+R*C) – first 2 tokens are grid shapes.
          context: Tensor of shape (B, latent_dim) representing the task latent.
          dropout_eval: bool – if True, dropout is disabled.
        Returns:
          grid_shape_row_logits: (B, max_rows)
          grid_shape_col_logits: (B, max_cols)
          output_grid_logits: (B, R*C, output_vocab_size)
        """
        x = self.embed_inputs(input_seq, output_seq, context, dropout_eval)
        # Build a causal mask from the grid shape tokens (first two tokens of each sequence).
        causal_pad_mask = self.make_causal_pad_mask(input_seq[:, :2], output_seq[:, :2])
        for layer in self.transformer_layers:
            x = layer(embeddings=x, dropout_eval=dropout_eval, pad_mask=causal_pad_mask)
        grid_shape_row_logits, grid_shape_col_logits, output_grid_logits = self.extract_logits(x, input_seq.shape[-1])
        grid_shape_row_logits = grid_shape_row_logits.to(torch.float32)
        grid_shape_col_logits = grid_shape_col_logits.to(torch.float32)
        output_grid_logits = output_grid_logits.to(torch.float32)
        return grid_shape_row_logits, grid_shape_col_logits, output_grid_logits

    # -------------------------------------------------------------------------
    # embed_inputs: Combines context, positional, grid shape, and colors embeddings.
    # -------------------------------------------------------------------------
    def embed_inputs(self, input_seq, output_seq, context, dropout_eval):
        config = self.config
        device = input_seq.device

        # --- Context embedding ---
        # Project the latent context to emb_dim.
        context_embed = self.context_embed(context)  # (B, emb_dim)

        # --- Position embedding block ---
        if config.scaled_position_embeddings:
            pos_row_indices = torch.zeros(config.max_rows, dtype=torch.long, device=device)
            pos_row_embed = self.pos_row_embed(pos_row_indices)  # (max_rows, emb_dim)
            pos_col_indices = torch.zeros(config.max_cols, dtype=torch.long, device=device)
            pos_col_embed = self.pos_col_embed(pos_col_indices)  # (max_cols, emb_dim)
            pos_row_factors = torch.arange(1, config.max_rows + 1, device=device).unsqueeze(1).type_as(pos_row_embed)
            pos_row_embeds = pos_row_factors * pos_row_embed  # (max_rows, emb_dim)
            pos_col_factors = torch.arange(1, config.max_cols + 1, device=device).unsqueeze(1).type_as(pos_col_embed)
            pos_col_embeds = pos_col_factors * pos_col_embed  # (max_cols, emb_dim)
            pos_embed = pos_row_embeds.unsqueeze(1) + pos_col_embeds.unsqueeze(0)  # (max_rows, max_cols, emb_dim)
        else:
            pos_row_indices = torch.arange(config.max_rows, dtype=torch.long, device=device)
            pos_row_embed = self.pos_row_embed(pos_row_indices)
            pos_col_indices = torch.arange(config.max_cols, dtype=torch.long, device=device)
            pos_col_embed = self.pos_col_embed(pos_col_indices)
            pos_embed = pos_row_embed.unsqueeze(1) + pos_col_embed.unsqueeze(0)  # (max_rows, max_cols, emb_dim)

        # --- Next position embeddings (if enabled) ---
        if config.next_position_embeddings:
            # input_seq and output_seq: first token is grid shape token; token index 1 holds number of columns.
            input_num_cols = input_seq[:, 1]  # (B,)
            output_num_cols = output_seq[:, 1]  # (B,)
            # Shift the position embedding horizontally.
            shifted_left_pos_embed = torch.roll(pos_embed, shifts=-1, dims=1)
            first_col_embed = pos_embed[:, 0, :]  # (max_rows, emb_dim)
            shifted_up_first_col_embed = torch.roll(first_col_embed, shifts=-1, dims=0)
            arange_broadcast = torch.arange(config.max_cols, device=device).view(1, config.max_cols)
            if config.next_position_embeddings_new_input_embeds:
                if config.scaled_position_embeddings:
                    input_pos_row_indices = torch.zeros(config.max_rows, dtype=torch.long, device=device)
                    input_pos_row_embed = self.input_pos_row_embed(input_pos_row_indices)
                    input_pos_col_indices = torch.zeros(config.max_cols, dtype=torch.long, device=device)
                    input_pos_col_embed = self.input_pos_col_embed(input_pos_col_indices)
                    input_pos_row_factors = torch.arange(1, config.max_rows + 1, device=device).unsqueeze(1).type_as(input_pos_row_embed)
                    input_pos_row_embeds = input_pos_row_factors * input_pos_row_embed
                    input_pos_col_factors = torch.arange(1, config.max_cols + 1, device=device).unsqueeze(1).type_as(input_pos_col_embed)
                    input_pos_col_embeds = input_pos_col_factors * input_pos_col_embed
                    input_pos_embeds = input_pos_row_embeds.unsqueeze(1) + input_pos_col_embeds.unsqueeze(0)
                else:
                    input_pos_row_indices = torch.arange(config.max_rows, dtype=torch.long, device=device)
                    input_pos_row_embed = self.input_pos_row_embed(input_pos_row_indices)
                    input_pos_col_indices = torch.arange(config.max_cols, dtype=torch.long, device=device)
                    input_pos_col_embed = self.input_pos_col_embed(input_pos_col_indices)
                    input_pos_embeds = input_pos_row_embed.unsqueeze(1) + input_pos_col_embed.unsqueeze(0)
            else:
                input_pos_embeds = pos_embed

            # For output tokens, choose between a shifted version for the last column and the default otherwise.
            # (This mimics the jnp.where logic in the Flax code.)
            output_num_cols_exp = output_num_cols.unsqueeze(1)  # (B,1)
            mask = (arange_broadcast == (output_num_cols_exp - 1))
            # Expand shifted embeddings for broadcasting.
            shifted_left = shifted_left_pos_embed.unsqueeze(0).expand(input_seq.size(0), -1, -1, -1)
            shifted_up = shifted_up_first_col_embed.unsqueeze(0).expand(input_seq.size(0), config.max_rows, config.max_cols, -1)
            output_pos_embeds = torch.where(mask.unsqueeze(0).unsqueeze(-1), shifted_up, shifted_left)
            # Flatten the spatial dimensions.
            input_pos_embeds = input_pos_embeds.view(-1, config.emb_dim)
            output_pos_embeds = output_pos_embeds.view(-1, config.emb_dim)
        else:
            pos_embeds = pos_embed.view(-1, config.emb_dim)
            input_pos_embeds = pos_embeds
            output_pos_embeds = pos_embeds

        # --- Grid shapes embedding block ---
        # The first token in each sequence (index 0) is the grid row token and index 1 is the grid col token.
        input_grid_shapes_row_embed = self.grid_shapes_row_embed(input_seq[:, 0].long() - 1)
        output_grid_shapes_row_embed = self.grid_shapes_row_embed(output_seq[:, 0].long() - 1)
        input_grid_shapes_col_embed = self.grid_shapes_col_embed(input_seq[:, 1].long() - 1)
        output_grid_shapes_col_embed = self.grid_shapes_col_embed(output_seq[:, 1].long() - 1)

        # --- Colors embedding block ---
        # Tokens from index 2 onward.
        input_colors_embed = self.colors_embed(input_seq[:, 2:].long())
        output_colors_embed = self.colors_embed(output_seq[:, 2:].long())
        # Get the two learned embeddings for input vs. output tokens.
        io_embed = self.input_output_embed(torch.arange(2, device=device))
        input_io_embed = io_embed[0]  # (emb_dim,)
        output_io_embed = io_embed[1]  # (emb_dim,)

        # --- Combining all the embeddings ---
        # Expand dims where needed so that each token is a “sequence element.”
        x_input_shape_row = (input_grid_shapes_row_embed + input_io_embed).unsqueeze(1)  # (B, 1, emb_dim)
        x_input_shape_col = (input_grid_shapes_col_embed + input_io_embed).unsqueeze(1)
        x_input_colors = input_colors_embed + input_pos_embeds + input_io_embed  # (B, R*C, emb_dim)
        x_context = context_embed.unsqueeze(1)  # (B, 1, emb_dim)
        x_output_shape_row = (output_grid_shapes_row_embed + output_io_embed).unsqueeze(1)
        x_output_shape_col = (output_grid_shapes_col_embed + output_io_embed).unsqueeze(1)
        x_output_colors = output_colors_embed + output_pos_embeds + output_io_embed
        # Concatenate along the sequence dimension.
        # The final sequence will have length: 1 + 2*(2 + max_len) = 5 + 2*max_len.
        x = torch.cat([
            x_input_shape_row,
            x_input_shape_col,
            x_input_colors,
            x_context,
            x_output_shape_row,
            x_output_shape_col,
            x_output_colors,
        ], dim=1)
        expected_seq_len = 1 + 2 * (2 + config.max_len)
        assert x.size(1) == expected_seq_len, f"Expected sequence length {expected_seq_len}, got {x.size(1)}"
        if not dropout_eval:
            x = self.embed_dropout(x)
        return x

    # -------------------------------------------------------------------------
    # make_causal_pad_mask: Build a combined (input/input, input/output, output/input, output/output)
    # attention mask for the decoder.
    # -------------------------------------------------------------------------
    def make_causal_pad_mask(self, input_grid_shape, output_grid_shape):
        """
        Args:
          input_grid_shape: (B, 2) with [rows, cols] for the input grid.
          output_grid_shape: (B, 2) with [rows, cols] for the output grid.
        Returns:
          A boolean mask of shape (B, 1, T, T) where T = 1+2*(2+max_rows*max_cols)
        """
        config = self.config
        device = input_grid_shape.device
        B = input_grid_shape.size(0)
        max_rows = config.max_rows
        max_cols = config.max_cols

        # --- Input pad mask ---
        row_arange = torch.arange(max_rows, device=device).unsqueeze(0)  # (1, max_rows)
        input_row_mask = row_arange < input_grid_shape[:, 0].unsqueeze(1)  # (B, max_rows)
        col_arange = torch.arange(max_cols, device=device).unsqueeze(0)  # (1, max_cols)
        input_col_mask = col_arange < input_grid_shape[:, 1].unsqueeze(1)  # (B, max_cols)
        input_spatial_mask = input_row_mask.unsqueeze(2) & input_col_mask.unsqueeze(1)  # (B, max_rows, max_cols)
        input_spatial_mask_flat = input_spatial_mask.view(B, -1)  # (B, max_rows*max_cols)
        # Prepend 2 tokens for the grid shape tokens.
        input_pad_mask = torch.cat([torch.ones(B, 2, device=device, dtype=torch.bool), input_spatial_mask_flat], dim=1)  # (B, T1)
        T1 = input_pad_mask.size(1)
        input_input_mask = input_pad_mask.unsqueeze(2) & input_pad_mask.unsqueeze(1)  # (B, T1, T1)

        # --- Output pad mask ---
        output_row_mask = row_arange < output_grid_shape[:, 0].unsqueeze(1)  # (B, max_rows)
        output_col_mask = col_arange < output_grid_shape[:, 1].unsqueeze(1)  # (B, max_cols)
        output_spatial_mask = output_row_mask.unsqueeze(2) & output_col_mask.unsqueeze(1)  # (B, max_rows, max_cols)
        output_spatial_mask_flat = output_spatial_mask.view(B, -1)  # (B, max_rows*max_cols)
        # Prepend (1+2)=3 tokens for the grid shape tokens and the context.
        output_pad_mask = torch.cat([torch.ones(B, 3, device=device, dtype=torch.bool), output_spatial_mask_flat], dim=1)  # (B, T2)
        T2 = output_pad_mask.size(1)
        output_output_mask = output_pad_mask.unsqueeze(2) & output_pad_mask.unsqueeze(1)  # (B, T2, T2)
        # Apply causal (lower triangular) mask to the output-output block.
        causal = torch.tril(torch.ones(T2, T2, device=device, dtype=torch.bool))
        output_output_mask = output_output_mask & causal

        # --- Input/Output cross masks ---
        # Input tokens should not attend to output tokens (to enforce causality) except for the first output token.
        input_output_mask = torch.zeros(B, T1, T2, device=device, dtype=torch.bool)
        input_output_mask[:, :, 0] = input_pad_mask  # allow input tokens to see the context (first output token)
        # Output tokens can attend to input tokens if they are valid.
        output_input_mask = output_pad_mask.unsqueeze(2) & input_pad_mask.unsqueeze(1)  # (B, T2, T1)

        # --- Assemble full mask ---
        top = torch.cat([input_input_mask, input_output_mask], dim=2)  # (B, T1, T1+T2)
        bottom = torch.cat([output_input_mask, output_output_mask], dim=2)  # (B, T2, T1+T2)
        full_mask = torch.cat([top, bottom], dim=1)  # (B, T1+T2, T1+T2)
        full_mask = full_mask.unsqueeze(1)  # (B, 1, T, T)
        return full_mask

    # -------------------------------------------------------------------------
    # extract_logits: From the transformer output, select and project the tokens
    # corresponding to grid shape rows, columns, and the output grid.
    # -------------------------------------------------------------------------
    def extract_logits(self, x, input_seq_length):
        """
        Args:
          x: Tensor of shape (B, T, emb_dim) (T = 1+2*(2+max_len))
          input_seq_length: int (should equal 2+max_len)
        Returns:
          shape_row_logits: (B, max_rows)
          shape_col_logits: (B, max_cols)
          grid_logits: (B, (T - (input_seq_length+2) - 1), output_vocab_size)
        """
        # According to the original logic:
        # - Token at index input_seq_length → row logits.
        # - Token at index input_seq_length+1 → column logits.
        # - Tokens from index input_seq_length+2 up to the penultimate token → grid logits.
        shape_row_embeds = self.row_logits_layer_norm(x[:, input_seq_length, :])
        shape_col_embeds = self.col_logits_layer_norm(x[:, input_seq_length + 1, :])
        grid_embeds = self.grid_logits_layer_norm(x[:, input_seq_length + 2:-1, :])
        shape_row_logits = self.shape_row_logits_proj(shape_row_embeds)
        shape_col_logits = self.shape_col_logits_proj(shape_col_embeds)
        grid_logits = self.grid_logits_proj(grid_embeds)
        return shape_row_logits, shape_col_logits, grid_logits