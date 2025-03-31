import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import set_device
DEVICE = set_device('world_model/transition_decode.py')

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

        # Stack the two context tokens to form the memory for the decoder.
        memory = torch.stack([x_proj, e_proj], dim=1)          # [B, 2, emb_dim]

        # Prepare target (tgt) tokens.
        # For the first two tokens, add 1D positional embeddings.
        # In forward, modify first_two tokens:
        first_two = self.query_emb[:2] + self.pos_emb  # [2, emb_dim]
        first_two = self.shape_token_transform(first_two)  # Enhance representation

        # For the grid tokens, add 2D positional embeddings.
        grid_tokens = self.query_emb[2:]  # [900, emb_dim]
        # Create grid positions (row-major order).
        rows = torch.arange(self.grid_size, device=x_t.device).unsqueeze(1).repeat(1, self.grid_size).flatten()  # [900]
        cols = torch.arange(self.grid_size, device=x_t.device).unsqueeze(0).repeat(self.grid_size, 1).flatten()  # [900]
        grid_pos = self.row_emb[rows] + self.col_emb[cols]  # [900, emb_dim]
        grid_tokens = grid_tokens + grid_pos

        # Concatenate first two tokens and grid tokens.
        tgt = torch.cat([first_two, grid_tokens], dim=0)  # [902, emb_dim]
        tgt = tgt.unsqueeze(0).expand(B, -1, -1)           # [B, 902, emb_dim]

        # Create a standard causal mask for autoregressive decoding.
        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=x_t.device), diagonal=1).bool()

        # Pass the optional key padding mask into the decoder (if provided).
        output = self.decoder(tgt, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_key_padding_mask)

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
            # Assume the first token indicates the number of valid rows,
            # and the second token indicates the number of valid columns.
            B = x_t.size(0)
            grid_mask = torch.zeros((B, self.grid_size, self.grid_size), dtype=torch.bool, device=x_t.device)
            for b in range(B):
                valid_rows = int(sampled_first[b, 0].item())
                valid_cols = int(sampled_first[b, 1].item())
                # Mark positions beyond the valid region as padded.
                grid_mask[b, valid_rows:, :] = True
                grid_mask[b, :, valid_cols:] = True
            # Flatten the grid mask: shape [B, 900]
            grid_mask_flat = grid_mask.view(B, -1)
            # Full key padding mask: first two tokens (grid shape) are always valid.
            full_mask = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=x_t.device),
                                    grid_mask_flat], dim=1)

            # === Stage 3: Second pass using the dynamic key padding mask ===
            logits_first_masked, logits_rest_masked = self.forward(x_t, e_t, tgt_key_padding_mask=full_mask)
            # For safety, re-use the first tokens from Stage 1.
            # Now sample grid tokens.
            probs_rest = F.softmax(logits_rest_masked / temperature, dim=-1)
            sampled_rest = torch.multinomial(
                probs_rest.view(-1, self.head_rest.out_features), 1
            ).view(B, 900)

            # Post-process: for grid positions that are padded, force token to -1.
            full_mask_grid = full_mask[:, 2:]  # [B, 900]
            sampled_rest[full_mask_grid] = -1

            return sampled_first, sampled_rest
        
    def save_weights(self, path: str):
        """
        Save the model weights to a file.
        
        Args:
            path (str): Path to save the weights.
        """
        # append 'decoder.pt' to the path
        path = path + '/decoder.pt'
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        """
        Load the model weights from a file.
        
        Args:
            path (str): Path to load the weights from.
        """
        # append 'decoder.pt' to the path
        path = path + '/decoder.pt'
        self.load_state_dict(torch.load(path))

    @property
    def num_parameters(self):
        """
        Returns the number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
"""
# === Testing the Model with Debug Print Statements ===
if __name__ == "__main__":
    # Dummy inputs.
    batch_size = 2
    state_encoded_dim = 128
    action_emb_dim = 64

    x_t = torch.randn(batch_size, state_encoded_dim)
    e_t = torch.randn(batch_size, action_emb_dim)

    # Instantiate the model.
    model = ContextTransformer2D(state_encoded_dim, action_emb_dim, emb_dim=256, num_layers=4, num_heads=8)
    
    print("\n=== Forward Pass (without dynamic mask) ===")
    logits_first, logits_rest = model(x_t, e_t)
    
    print("\n=== Generation (with dynamic masking) ===")
    sampled_first, sampled_rest = model.generate(x_t, e_t, temperature=0.8)
    print("Sampled first tokens (grid shape):", sampled_first)
    print("Sampled grid tokens shape:", sampled_rest.shape)
    # Optionally, reshape grid tokens to [B, grid_size, grid_size] for visualization.
    sampled_grid = sampled_rest.view(batch_size, model.grid_size, model.grid_size)
    print("Sampled grid tokens (reshaped):", sampled_grid)"""
