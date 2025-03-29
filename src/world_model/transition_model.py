import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextTransformer(nn.Module):
    def __init__(
        self,
        state_emb_dim,
        action_emb_dim,
        d_model=256,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        seq_len=902,
        vocab_size_first=30,
        vocab_size_rest=11,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.vocab_size_first = vocab_size_first
        self.vocab_size_rest = vocab_size_rest

        # Project context inputs into a shared d_model space.
        self.state_proj = nn.Linear(state_emb_dim, d_model)
        self.action_proj = nn.Linear(action_emb_dim, d_model)

        # Learned query embeddings and positional encodings for the 902 tokens to be generated.
        self.query_emb = nn.Parameter(torch.randn(seq_len, d_model))
        self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model))

        # Positional encoding for the 2 context tokens.
        self.ctx_pos_emb = nn.Parameter(torch.randn(2, d_model))

        # Transformer Decoder with configurable layers.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output heads for two segments.
        self.head_first = nn.Linear(d_model, vocab_size_first)
        self.head_rest = nn.Linear(d_model, vocab_size_rest)

    def forward(self, x_t, e_t):
        B = x_t.size(0)
        print("Input x_t shape:", x_t.shape)
        print("Input e_t shape:", e_t.shape)

        # Project and add context positional embeddings.
        x_proj = self.state_proj(x_t) + self.ctx_pos_emb[0]  # [B, d_model]
        e_proj = self.action_proj(e_t) + self.ctx_pos_emb[1]   # [B, d_model]
        print("x_proj shape:", x_proj.shape)
        print("e_proj shape:", e_proj.shape)

        # Stack the two context tokens.
        memory = torch.stack([x_proj, e_proj], dim=1)          # [B, 2, d_model]
        print("Memory shape (stacked context):", memory.shape)

        # Prepare the query tokens with positional embeddings.
        tgt = self.query_emb.unsqueeze(0).expand(B, -1, -1) + self.pos_emb.unsqueeze(0)  # [B, 902, d_model]
        print("Target (query) shape:", tgt.shape)

        # Create a causal mask so that each position can only attend to previous positions.
        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool().to(x_t.device)
        print("Causal mask shape:", causal_mask.shape)

        # Transformer decoding
        output = self.decoder(tgt, memory, tgt_mask=causal_mask)  # [B, 902, d_model]
        print("Transformer output shape:", output.shape)

        # Split the output logits.
        logits_first = self.head_first(output[:, :2])  # [B, 2, vocab_size_first]
        logits_rest = self.head_rest(output[:, 2:])      # [B, 900, vocab_size_rest]
        print("Logits for first 2 tokens shape:", logits_first.shape)
        print("Logits for next 900 tokens shape:", logits_rest.shape)

        return logits_first, logits_rest

    def generate(self, x_t, e_t, temperature=1.0):
        """
        Generation method:
        - Runs the forward pass.
        - Applies softmax (with temperature) to obtain probabilities.
        - Samples tokens from the distributions.
        Returns a tuple of sampled tokens for the first segment and the rest.
        """
        logits_first, logits_rest = self.forward(x_t, e_t)

        # For the first two tokens.
        probs_first = F.softmax(logits_first / temperature, dim=-1)
        sampled_first = torch.multinomial(probs_first.view(-1, self.vocab_size_first), num_samples=1)
        sampled_first = sampled_first.view(logits_first.shape[0], 2)
        print("Sampled first tokens shape:", sampled_first.shape)

        # For the next 900 tokens.
        probs_rest = F.softmax(logits_rest / temperature, dim=-1)
        sampled_rest = torch.multinomial(probs_rest.view(-1, self.vocab_size_rest), num_samples=1)
        sampled_rest = sampled_rest.view(logits_rest.shape[0], 900)
        print("Sampled rest tokens shape:", sampled_rest.shape)

        # Optionally, concatenate both parts to form the full sequence.
        full_sequence = torch.cat([sampled_first, sampled_rest], dim=1)
        print("Full generated sequence shape:", full_sequence.shape)

        return full_sequence

# ----------------------- Testing the Model -----------------------
if __name__ == '__main__':
    # Example dimensions.
    batch_size = 2
    state_emb_dim = 128
    action_emb_dim = 64
    d_model = 256
    seq_len = 902

    # Create dummy inputs.
    x_t = torch.randn(batch_size, state_emb_dim)
    e_t = torch.randn(batch_size, action_emb_dim)

    # Initialize the model.
    model = ContextTransformer(
        state_emb_dim=state_emb_dim,
        action_emb_dim=action_emb_dim,
        d_model=d_model,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        seq_len=seq_len,
        vocab_size_first=30,
        vocab_size_rest=11,
    )

    # Run a forward pass.
    print("\n--- Forward Pass ---")
    logits_first, logits_rest = model(x_t, e_t)

    # Run generation.
    print("\n--- Generation ---")
    generated_tokens = model.generate(x_t, e_t, temperature=1.0)
    print("\nFinal Generated Tokens:\n", generated_tokens)
