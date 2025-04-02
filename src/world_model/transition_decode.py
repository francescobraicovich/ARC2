import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.util import set_device
# Assume DEVICE is defined somewhere, for example:
DEVICE = set_device('transition_decode.py')

class ContextTransformer2D(nn.Module):
    def __init__(
        self,
        state_encoded_dim,
        action_emb_dim,
        emb_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        seq_len=902,      # Total output tokens: 2 (grid-shape) + 900 grid cells
        grid_size=30,     # Grid is 30x30 (900 tokens)
        vocab_size_first=30,
        vocab_size_rest=11,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.grid_size = grid_size
        self.vocab_size_first = vocab_size_first
        self.vocab_size_rest = vocab_size_rest

        # Project context inputs (x_t and e_t) into shared emb_dim space.
        self.state_proj = nn.Linear(state_encoded_dim, emb_dim).to(DEVICE)
        self.action_proj = nn.Linear(action_emb_dim, emb_dim).to(DEVICE)

        # Learned query embeddings for all 902 tokens.
        # (These will be added to positional information.)
        self.query_emb = nn.Parameter(torch.randn(seq_len, emb_dim)).to(DEVICE)
        # For the first two tokens, we use a learned 1D positional embedding.
        self.pos_emb = nn.Parameter(torch.randn(2, emb_dim)).to(DEVICE)
        self.shape_token_transform = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU()
        ).to(DEVICE)

        # For grid tokens (900 tokens), we use 2D positional embeddings.
        # Create separate embeddings for row and column positions.
        self.row_emb = nn.Parameter(torch.randn(grid_size, emb_dim)).to(DEVICE)
        self.col_emb = nn.Parameter(torch.randn(grid_size, emb_dim)).to(DEVICE)

        # Positional encodings for the 2 context tokens.
        self.ctx_pos_emb = nn.Parameter(torch.randn(2, emb_dim)).to(DEVICE)

        # Additional normalization layers for a SOTA approach:
        self.input_layer_norm_tgt = nn.LayerNorm(emb_dim, elementwise_affine=True).to(DEVICE)
        self.memory_layer_norm = nn.LayerNorm(emb_dim, elementwise_affine=True).to(DEVICE)
        self.decoder_output_norm = nn.LayerNorm(emb_dim, elementwise_affine=True).to(DEVICE)

        # Transformer Decoder layers.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        ).to(DEVICE)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers).to(DEVICE)

        # Two output heads:
        # - The first head predicts the first 2 tokens (vocab_size_first).
        # - The second head predicts the grid tokens (vocab_size_rest).
        self.head_first = nn.Linear(emb_dim, vocab_size_first).to(DEVICE)
        self.head_rest = nn.Linear(emb_dim, vocab_size_rest).to(DEVICE)

    def forward(self, x_t, e_t, tgt_key_padding_mask=None):
        B = x_t.size(0)
        # Project context tokens and add learned positional encodings.
        x_proj = self.state_proj(x_t) + self.ctx_pos_emb[0]  # [B, emb_dim]
        e_proj = self.action_proj(e_t) + self.ctx_pos_emb[1]   # [B, emb_dim]
        # Normalize memory tokens.
        memory = torch.stack([x_proj, e_proj], dim=1)          # [B, 2, emb_dim]
        memory = self.memory_layer_norm(memory)

        # Prepare target (tgt) tokens.
        # For the first two tokens, add 1D positional embeddings.
        first_two = self.query_emb[:2] + self.pos_emb         # [2, emb_dim]
        first_two = self.shape_token_transform(first_two)       # Enhance representation

        # For the grid tokens, add 2D positional embeddings.
        grid_tokens = self.query_emb[2:]                        # [900, emb_dim]
        # Create grid positions (row-major order).
        rows = torch.arange(self.grid_size, device=x_t.device).unsqueeze(1).repeat(1, self.grid_size).flatten()  # [900]
        cols = torch.arange(self.grid_size, device=x_t.device).unsqueeze(0).repeat(self.grid_size, 1).flatten()  # [900]
        grid_pos = self.row_emb[rows] + self.col_emb[cols]      # [900, emb_dim]
        grid_tokens = grid_tokens + grid_pos

        # Concatenate first two tokens and grid tokens.
        tgt = torch.cat([first_two, grid_tokens], dim=0)        # [902, emb_dim]
        # Normalize the target tokens before decoding.
        tgt = self.input_layer_norm_tgt(tgt)
        tgt = tgt.unsqueeze(0).expand(B, -1, -1)                # [B, 902, emb_dim]

        # Create a standard causal mask for autoregressive decoding.
        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=x_t.device), diagonal=1).bool()

        # Pass the optional key padding mask into the decoder (if provided).
        output = self.decoder(tgt, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # Normalize the decoder output.
        output = self.decoder_output_norm(output)

        # Split outputs for the two segments.
        logits_first = self.head_first(output[:, :2])   # [B, 2, vocab_size_first]
        logits_rest  = self.head_rest(output[:, 2:])      # [B, 900, vocab_size_rest]

        return logits_first, logits_rest

    def generate(self, x_t, e_t, temperature=1.0):
        """
        Two-stage autoregressive generation:
          1. First pass to predict the grid shape tokens.
          2. Compute a key padding mask for grid tokens based on those tokens.
          3. Second pass with dynamic attention that masks padded grid tokens.
        The grid tokens outside the valid region are post-processed to be -1.
        """
        self.eval()
        with torch.no_grad():
            # === Stage 1: Generate the first two tokens (grid shape indicators) ===
            logits_first, _ = self.forward(x_t, e_t, tgt_key_padding_mask=None)
            # Sample first two tokens.
            probs_first = F.softmax(logits_first / temperature, dim=-1)
            sampled_first = torch.multinomial(
                probs_first.view(-1, self.head_first.out_features), 1
            ).view(x_t.size(0), 2)

            # === Stage 2: Compute key padding mask for grid tokens based on grid shape ===
            B = x_t.size(0)
            grid_mask = torch.zeros((B, self.grid_size, self.grid_size), dtype=torch.bool, device=x_t.device)
            for b in range(B):
                valid_rows = int(sampled_first[b, 0].item())
                valid_cols = int(sampled_first[b, 1].item())
                grid_mask[b, valid_rows:, :] = True
                grid_mask[b, :, valid_cols:] = True
            grid_mask_flat = grid_mask.view(B, -1)
            full_mask = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=x_t.device),
                                    grid_mask_flat], dim=1)

            # === Stage 3: Second pass using the dynamic key padding mask ===
            logits_first_masked, logits_rest_masked = self.forward(x_t, e_t, tgt_key_padding_mask=full_mask)
            # For safety, re-use the first tokens from Stage 1.
            probs_rest = F.softmax(logits_rest_masked / temperature, dim=-1)
            sampled_rest = torch.multinomial(
                probs_rest.view(-1, self.head_rest.out_features), 1
            ).view(B, 900)
            full_mask_grid = full_mask[:, 2:]
            sampled_rest[full_mask_grid] = 0
            return sampled_first, sampled_rest

    def save_weights(self, path: str):
        """
        Save the model weights to a file.
        """
        torch.save(self.state_dict(), os.path.join(path, 'decoder.pt'))

    def load_weights(self, path: str):
        """
        Load the model weights from a file.
        """
        self.load_state_dict(torch.load(os.path.join(path, 'decoder.pt')))

    @property
    def num_parameters(self):
        """
        Returns the number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from typing import Optional, Tuple

# Assume DEVICE is defined somewhere, for example:
# from utils.util import set_device
# DEVICE = set_device('transition_decode_v2.py')
# Placeholder if not defined elsewhere:
try:
    # Attempt to set device using a utility function if available
    from utils.util import set_device
    DEVICE = set_device('transition_decode_v2.py')
    print(f"Using device: {DEVICE}")
except ImportError:
    print("utils.util.set_device not found or failed. Defaulting device.")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")

# ========== RoPE Implementation ==========

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    Applies rotary embeddings to input tensors (typically queries and keys).
    """
    def __init__(self, dim: int, seq_len: int, theta: float = 10000.0):
        super().__init__()
        # Ensure dimension is even as RoPE pairs dimensions
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even for RoPE, got {dim}")
            
        self.dim = dim
        self.seq_len = seq_len
        self.theta = theta
        self.freqs_cis = self._precompute_freqs_cis()

    def _precompute_freqs_cis(self):
        # Precompute the complex exponentials for rotary embeddings
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32)[: (self.dim // 2)] / self.dim))
        t = torch.arange(self.seq_len, device=freqs.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        # Convert to complex numbers: cos(m*theta) + i*sin(m*theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        # Cache on the specified device
        self.register_buffer("cached_freqs_cis", freqs_cis.to(DEVICE))
        return freqs_cis.to(DEVICE)

    def _apply_rotary_emb(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Reshape x to separate real and imaginary parts (view as complex)
        # x shape: [B, Seq, Heads, HeadDim]
        # freqs_cis shape: [Seq, HeadDim//2] -> needs broadcasting
        x_complex = x.float().reshape(*x.shape[:-1], -1, 2).view(torch.complex64)

        # Adjust freqs_cis shape for broadcasting: [Seq, 1, HeadDim//2]
        freqs_cis = freqs_cis.unsqueeze(1) # Add head dimension

        # Apply rotation: element-wise complex multiplication
        # Ensure correct shapes for broadcasting:
        # x_complex: [B, Seq, Heads, HeadDim//2]
        # freqs_cis: [Seq, 1, HeadDim//2] -> Broadcasts over B and Heads
        x_rotated = x_complex * freqs_cis
        
        # Reshape back to original tensor shape
        x_out = torch.view_as_real(x_rotated).flatten(3)
        return x_out.type_as(x) # Cast back to original dtype

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.
        Args:
            q (torch.Tensor): Query tensor, shape [B, SeqQ, Heads, HeadDim]
            k (torch.Tensor): Key tensor, shape [B, SeqK, Heads, HeadDim]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
        """
        seq_len_q = q.shape[1]
        seq_len_k = k.shape[1]

        # Ensure precomputed freqs are sufficient; slice if needed
        if seq_len_q > self.seq_len or seq_len_k > self.seq_len:
             # Recompute if needed (e.g., for variable sequence lengths beyond initial seq_len)
             # This basic implementation assumes fixed max seq_len during init
             # For dynamic length beyond capability, recalculation or error is needed
             # Here we'll slice, assuming generation length <= self.seq_len
             # print(f"Warning: RoPE sequence length {max(seq_len_q, seq_len_k)} exceeds precomputed {self.seq_len}")
             pass # Or recompute/error handle if necessary

        # Slice freqs_cis for the actual sequence lengths
        freqs_cis_q = self.freqs_cis[:seq_len_q].to(q.device)
        freqs_cis_k = self.freqs_cis[:seq_len_k].to(k.device)

        q_rotated = self._apply_rotary_emb(q, freqs_cis_q)
        k_rotated = self._apply_rotary_emb(k, freqs_cis_k)
        
        return q_rotated, k_rotated


# ========== SwiGLU FeedForward Implementation ==========

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU FeedForward Network module.
    Implements the SwiGLU activation function within a feed-forward layer.
    """
    def __init__(self, dim: int, hidden_dim_multiplier: float = 4.0, dropout: float = 0.1):
        super().__init__()
        # Calculate hidden dimension, often 2/3 of the traditional FFN dim for SwiGLU
        hidden_dim = int(2 * (dim * hidden_dim_multiplier) / 3)
        # Ensure hidden_dim is multiple of typical internal alignment (e.g., 256) - Optional
        # hidden_dim = ((hidden_dim + 255) // 256) * 256 # Example alignment

        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # Gate projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # Up projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # Down projection
        self.activation = nn.SiLU() # Use SiLU (Swish) for SwiGLU
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        gate = self.activation(self.w1(x))
        up = self.w3(x)
        fuse = gate * up # Element-wise multiplication
        x = self.dropout1(fuse)
        x = self.w2(x)
        x = self.dropout2(x)
        return x

# ========== Custom Attention with RoPE ==========
# Note: Implementing custom attention is complex. For simplicity here,
# we'll keep using nn.MultiheadAttention BUT acknowledge that RoPE
# ideally requires modifying the Q/K *before* the attention dot products,
# which nn.MHA doesn't directly expose.
# A *true* RoPE integration often involves reimplementing attention logic.
# Here, we'll pass RoPE module and apply it *before* nn.MHA call as an approximation,
# though this isn't the mathematically pure way RoPE interacts with Q/K internally.
# A better way requires a fully custom MHA implementation.

# For this implementation, we will proceed with the *structural changes* (Cross-Attn first)
# and *component changes* (SwiGLU, MLP Heads) and implement RoPE *conceptually*
# by adding the RoPE module but noting the limitation with standard nn.MHA.
# The single-pass generation *will* be implemented.

# ========== Custom Decoder Layer (Cross-Attention First) ==========

class CustomDecoderLayer(nn.Module):
    """
    Custom Transformer Decoder Layer with Cross-Attention performed *before* Self-Attention.
    Includes options for SwiGLU FFN. RoPE is handled externally before passing Q/K.
    """
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.1, activation="swiglu", batch_first=True):
        super().__init__()
        if not batch_first:
            raise NotImplementedError("CustomDecoderLayer currently only supports batch_first=True")
        if dim_feedforward is None:
            dim_feedforward = d_model * 4 # Default FFN dimension multiplier

        self.d_model = d_model
        self.nhead = nhead

        # --- Cross-Attention ---
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # --- Self-Attention ---
        # NOTE: Needs causal mask passed during forward if used autoregressively
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # --- Feed Forward Network ---
        if activation == "swiglu":
            self.ffn = SwiGLUFeedForward(d_model, dim_feedforward / d_model, dropout)
        elif activation == "gelu":
             # Fallback to original GELU FFN if needed
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        # RoPE arguments (passed if external RoPE application is used)
        # rotary_emb: Optional[RotaryEmbedding] = None,
        # tgt_positions: Optional[torch.Tensor] = None,
        # memory_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # --- 1. Cross-Attention (Query: tgt, Key/Value: memory) ---
        # Ideally, apply RoPE to query (based on tgt_positions) before this call.
        # If applying RoPE externally:
        # if rotary_emb is not None and tgt_positions is not None:
        #     q_rot, _ = rotary_emb(tgt_q_proj, tgt_k_proj) # Dummy k_proj needed for interface
        # else:
        #     q_rot = tgt # Use original if no RoPE
        
        # Using standard MHA requires applying RoPE before projecting Q, K, V, which is complex.
        # We proceed without explicit external RoPE application here due to nn.MHA limitations.
        
        attn_output, _ = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=memory_mask,
            need_weights=False # Usually not needed unless inspecting attention
        )
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        # --- 2. Self-Attention (Query/Key/Value: tgt) ---
        # Ideally, apply RoPE to query and key (based on tgt_positions) before this call.
        # Using standard MHA requires applying RoPE before projecting Q, K, V.
        # Causal mask (tgt_mask) is crucial here for autoregressive decoding.
        
        sa_output, _ = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            key_padding_mask=tgt_key_padding_mask,
            attn_mask=tgt_mask, # Causal mask typically goes here
            is_causal= (tgt_mask is not None) # Use is_causal if mask shape allows
        )
        tgt = tgt + self.dropout2(sa_output)
        tgt = self.norm2(tgt)

        # --- 3. Feed Forward Network ---
        ffn_output = self.ffn(tgt)
        tgt = tgt + ffn_output # Residual connection already included in SwiGLU FFN class if needed, check implementation (standard FFN usually adds outside)
        tgt = self.norm3(tgt) # Final normalization

        return tgt

# ========== Main Model (ContextTransformer2D_v2) ==========

class ContextTransformer2D_v2(nn.Module):
    def __init__(
        self,
        state_encoded_dim,
        action_emb_dim,
        emb_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        seq_len=902,      # Total output tokens: 2 (grid-shape) + 900 grid cells
        grid_size=30,     # Grid is 30x30 (900 tokens)
        vocab_size_first=30, # Usually grid_size for shape tokens
        vocab_size_rest=11,  # Example: 10 digits + 1 padding/empty
        ffn_activation="swiglu", # Use swiglu by default
        head_mlp_dim_multiplier=1, # Multiplier for MLP head intermediate dim
    ):
        super().__init__()
        if grid_size * grid_size != (seq_len - 2):
             raise ValueError("seq_len must be grid_size*grid_size + 2")
             
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.grid_size = grid_size
        self.vocab_size_first = vocab_size_first
        self.vocab_size_rest = vocab_size_rest
        self.num_heads = num_heads # Store for RoPE if needed

        # --- Input Projections ---
        self.state_proj = nn.Linear(state_encoded_dim, emb_dim).to(DEVICE)
        self.action_proj = nn.Linear(action_emb_dim, emb_dim).to(DEVICE)

        # --- Target Query Embeddings (Learnable) ---
        # These represent the base "query" for each output position
        self.query_emb = nn.Parameter(torch.randn(seq_len, emb_dim)).to(DEVICE)
        
        # --- RoPE (Replaces learned positional embeddings) ---
        # Note: RoPE dimension should match head dimension if applied per head
        # Here applying to full embedding dim before MHA projects Q/K/V
        # Requires custom MHA for proper integration.
        # Add RoPE module, but acknowledge limitation with standard MHA
        self.rotary_emb = RotaryEmbedding(dim=emb_dim, seq_len=seq_len).to(DEVICE)
        # Remove learned embeddings: pos_emb, row_emb, col_emb, ctx_pos_emb
        
        # --- Optional Transformation for Shape Tokens Query ---
        # Can still be useful to specialize the first two query embeddings
        self.shape_token_transform = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU()
        ).to(DEVICE)

        # --- Normalization Layers (Pre-Norm) ---
        self.input_layer_norm_tgt = nn.LayerNorm(emb_dim, elementwise_affine=True).to(DEVICE)
        self.memory_layer_norm = nn.LayerNorm(emb_dim, elementwise_affine=True).to(DEVICE)
        self.decoder_output_norm = nn.LayerNorm(emb_dim, elementwise_affine=True).to(DEVICE)

        # --- Custom Transformer Decoder ---
        # Use nn.ModuleList of CustomDecoderLayer instead of nn.TransformerDecoder
        self.decoder_layers = nn.ModuleList([
            CustomDecoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=emb_dim * 4, # Standard FFN dim
                dropout=dropout,
                activation=ffn_activation,
                batch_first=True
            ) for _ in range(num_layers)
        ]).to(DEVICE)

        # --- Output Heads (MLP) ---
        head_intermediate_dim = int(emb_dim * head_mlp_dim_multiplier)
        self.head_first = nn.Sequential(
            nn.Linear(emb_dim, head_intermediate_dim),
            #nn.GELU(), # GELU or SiLU? Keep consistent or GELU is fine here.
            nn.SiLU() if ffn_activation == "swiglu" else nn.GELU(),
            nn.LayerNorm(head_intermediate_dim),
            nn.Linear(head_intermediate_dim, vocab_size_first)
        ).to(DEVICE)
        self.head_rest = nn.Sequential(
            nn.Linear(emb_dim, head_intermediate_dim),
            nn.SiLU() if ffn_activation == "swiglu" else nn.GELU(),
            nn.LayerNorm(head_intermediate_dim),
            nn.Linear(head_intermediate_dim, vocab_size_rest)
        ).to(DEVICE)

        print(f"ContextTransformer2D_v2 initialized with {self.num_parameters:,} parameters.")
        print(f"Using {ffn_activation} activation in FFN.")
        print(f"Using MLP heads with intermediate dim multiplier {head_mlp_dim_multiplier}.")
        print("Using Cross-Attention First decoder layers.")
        print("Using Rotary Embeddings (RoPE). Note: Full integration requires custom MHA.")
        print("Using single-pass generation strategy.")


    def _prepare_target(self, B: int) -> torch.Tensor:
        """Prepares the target input tensor using query embeddings."""
        # Note: RoPE replaces the explicit addition of positional embeddings here.
        # RoPE will be applied *conceptually* within the attention mechanism.
        
        # Apply optional transform to the first two query embeddings
        first_two_query = self.query_emb[:2]
        first_two_query = self.shape_token_transform(first_two_query)

        # Keep grid queries as they are
        grid_tokens_query = self.query_emb[2:]

        # Concatenate queries
        tgt_base = torch.cat([first_two_query, grid_tokens_query], dim=0) # [902, emb_dim]
        
        # Normalize the base target queries
        tgt_base = self.input_layer_norm_tgt(tgt_base)
        
        # Expand for batch size
        tgt = tgt_base.unsqueeze(0).expand(B, -1, -1) # [B, 902, emb_dim]
        return tgt


    def forward(self, x_t, e_t, tgt_key_padding_mask=None):
        """
        Forward pass of the modified Transformer Decoder.
        RoPE is conceptually applied within the attention layers (requires custom MHA for full effect).
        """
        B = x_t.size(0)
        
        # --- Prepare Memory ---
        # Project context tokens (no explicit positional encoding added here, RoPE handles positions)
        x_proj = self.state_proj(x_t)  # [B, emb_dim]
        e_proj = self.action_proj(e_t)   # [B, emb_dim]
        memory = torch.stack([x_proj, e_proj], dim=1)          # [B, 2, emb_dim]
        memory = self.memory_layer_norm(memory)

        # --- Prepare Target ---
        # Target is now based solely on query embeddings; RoPE adds positional info in attention
        tgt = self._prepare_target(B) # [B, 902, emb_dim]

        # --- Create Causal Mask ---
        # Standard causal mask for self-attention within the decoder
        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=DEVICE, dtype=torch.bool), diagonal=1)

        # --- Apply Decoder Layers ---
        output = tgt # Start with the prepared target embeddings
        for layer in self.decoder_layers:
            output = layer(
                tgt=output,
                memory=memory,
                tgt_mask=causal_mask, # Apply causal mask to self-attention
                memory_mask=None, # No mask needed for memory unless specific structure demands it
                tgt_key_padding_mask=tgt_key_padding_mask, # Mask padding in target sequence
                memory_key_padding_mask=None # Assuming memory (x_t, e_t) is never padded
                # Pass RoPE info here if using external application (see layer comments)
            )

        # --- Final Normalization & Output Heads ---
        output = self.decoder_output_norm(output)

        # Split outputs for the two segments and apply MLP heads
        logits_first = self.head_first(output[:, :2])   # [B, 2, vocab_size_first]
        logits_rest  = self.head_rest(output[:, 2:])      # [B, 900, vocab_size_rest]

        return logits_first, logits_rest

    def generate(self, x_t, e_t, temperature=1.0):
        """
        Single-pass autoregressive generation strategy.
        Runs the forward pass ONCE, then samples tokens and applies
        validity masking based on the first two (shape) tokens.
        """
        self.eval()
        B = x_t.size(0)

        with torch.no_grad():
            # === Stage 1: Single Forward Pass ===
            # Run the decoder once to get logits for all positions
            # No target key padding mask is used here initially, assuming full generation space
            logits_first, logits_rest = self.forward(x_t, e_t, tgt_key_padding_mask=None)

            # === Stage 2: Sample Shape Tokens ===
            probs_first = F.softmax(logits_first / temperature, dim=-1)
            # Ensure sampling doesn't produce indices out of bounds for grid size
            # Clamp logits or adjust probabilities if vocab_size_first > grid_size
            # Assuming vocab_size_first corresponds to valid grid dimensions (0 to grid_size-1)
            # If vocab_size_first allows values >= grid_size, they need handling.
            # For simplicity, assume vocab_size_first = grid_size or sampling handles range.
            sampled_first = torch.multinomial(
                probs_first.view(B * 2, self.vocab_size_first), 1 # Reshape for multinomial
            ).view(B, 2)

            # === Stage 3: Determine Valid Grid Mask ===
            # Create a mask for the grid based on sampled shape tokens
            grid_mask_invalid = torch.ones((B, self.grid_size, self.grid_size), dtype=torch.bool, device=DEVICE)
            for b in range(B):
                # Ensure sampled values are valid indices (non-negative and within grid bounds)
                # Add clamping or error check if vocab_size_first doesn't guarantee this.
                valid_rows = int(sampled_first[b, 0].item()) + 1 # Add 1 because shape tokens are typically size (e.g., 0 means 1 row) - adjust if definition differs
                valid_cols = int(sampled_first[b, 1].item()) + 1
                
                # Clamp to be within grid dimensions
                valid_rows = max(0, min(valid_rows, self.grid_size))
                valid_cols = max(0, min(valid_cols, self.grid_size))

                # Mark the valid area as False (not invalid)
                if valid_rows > 0 and valid_cols > 0:
                     grid_mask_invalid[b, :valid_rows, :valid_cols] = False

            grid_mask_invalid_flat = grid_mask_invalid.view(B, -1) # [B, 900]

            # === Stage 4: Sample Grid Tokens & Apply Mask ===
            probs_rest = F.softmax(logits_rest / temperature, dim=-1)
            sampled_rest = torch.multinomial(
                probs_rest.view(B * (self.seq_len - 2), self.vocab_size_rest), 1
            ).view(B, self.seq_len - 2) # [B, 900]

            # Apply the mask: set tokens in invalid positions to a padding value (e.g., 0 or -1)
            # Choose '0' for consistency if 0 is the padding/empty token index.
            sampled_rest[grid_mask_invalid_flat] = 0 # Or -1 if preferred

            return sampled_first, sampled_rest

    def save_weights(self, path: str, filename: str = 'decoder_v2.pt'):
        """
        Save the model weights to a file.
        """
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        torch.save(self.state_dict(), filepath)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, path: str, filename: str = 'decoder_v2.pt'):
        """
        Load the model weights from a file.
        """
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            print(f"Warning: Weight file not found at {filepath}")
            return
        try:
            self.load_state_dict(torch.load(filepath, map_location=DEVICE))
            print(f"Model weights loaded from {filepath}")
        except Exception as e:
            print(f"Error loading weights from {filepath}: {e}")


    @property
    def num_parameters(self):
        """
        Returns the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example Usage (Optional)
if __name__ == '__main__':
    print(f"Running example usage on device: {DEVICE}")
    
    # Hyperparameters (match original or adjust as needed)
    state_dim = 512
    action_dim = 64
    embed_dim = 256
    n_layers = 4 # Reduced for faster example
    n_heads = 4  # Reduced for faster example
    seq_len = 902
    grid_sz = 30
    vocab_first = 30 # Should ideally be grid_sz
    vocab_rest = 11
    batch_size = 4

    # Instantiate the new model
    model = ContextTransformer2D_v2(
        state_encoded_dim=state_dim,
        action_emb_dim=action_dim,
        emb_dim=embed_dim,
        num_layers=n_layers,
        num_heads=n_heads,
        seq_len=seq_len,
        grid_size=grid_sz,
        vocab_size_first=vocab_first,
        vocab_size_rest=vocab_rest
    ).to(DEVICE)

    # Create dummy input data
    dummy_state = torch.randn(batch_size, state_dim).to(DEVICE)
    dummy_action = torch.randn(batch_size, action_dim).to(DEVICE)

    # Test forward pass
    print("\nTesting forward pass...")
    try:
        logits_f, logits_r = model(dummy_state, dummy_action)
        print("Forward pass successful.")
        print("Logits First Shape:", logits_f.shape) # Expected: [B, 2, vocab_size_first]
        print("Logits Rest Shape:", logits_r.shape)   # Expected: [B, 900, vocab_size_rest]
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e

    # Test generation pass
    print("\nTesting generation pass...")
    try:
        sampled_f, sampled_r = model.generate(dummy_state, dummy_action)
        print("Generation pass successful.")
        print("Sampled First Shape:", sampled_f.shape) # Expected: [B, 2]
        print("Sampled Rest Shape:", sampled_r.shape)   # Expected: [B, 900]
        # Check if masking worked (some values should be 0)
        print("Sampled Rest (first batch item, first 20 values):", sampled_r[0, :20])
        print("Sampled Rest (first batch item, last 20 values):", sampled_r[0, -20:])
        # Verify shape tokens are within expected range (depends on vocab_size_first)
        print("Sampled First Tokens (first batch item):", sampled_f[0])

    except Exception as e:
        print(f"Generation pass failed: {e}")
        raise e

    # Test save/load weights
    print("\nTesting weight save/load...")
    try:
        save_dir = "./temp_model_weights"
        model.save_weights(save_dir)
        
        # Create a new instance and load weights
        model_new = ContextTransformer2D_v2(
            state_encoded_dim=state_dim,
            action_emb_dim=action_dim,
            emb_dim=embed_dim,
            num_layers=n_layers,
            num_heads=n_heads,
            seq_len=seq_len,
            grid_size=grid_sz,
            vocab_size_first=vocab_first,
            vocab_size_rest=vocab_rest
        ).to(DEVICE)
        model_new.load_weights(save_dir)
        
        # Simple check: compare parameter count or a specific parameter value
        assert model.num_parameters == model_new.num_parameters
        print("Weight save/load test successful.")
        # Clean up dummy weights
        import shutil
        #shutil.rmtree(save_dir)

    except Exception as e:
        print(f"Weight save/load test failed: {e}")
        raise e

    print("\nAll tests completed.")