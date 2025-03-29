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

        # Project context inputs
        self.state_proj = nn.Linear(state_emb_dim, d_model)
        self.action_proj = nn.Linear(action_emb_dim, d_model)

        # Learned query embeddings (902 tokens to predict)
        self.query_emb = nn.Parameter(torch.randn(seq_len, d_model))
        self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model))

        # Positional encoding for the 2 context tokens
        self.ctx_pos_emb = nn.Parameter(torch.randn(2, d_model))

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output heads
        self.head_first = nn.Linear(d_model, vocab_size_first)
        self.head_rest = nn.Linear(d_model, vocab_size_rest)

    def forward(self, x_t, e_t):
        B = x_t.size(0)

        # Project context
        x_proj = self.state_proj(x_t) + self.ctx_pos_emb[0]  # [B, D]
        e_proj = self.action_proj(e_t) + self.ctx_pos_emb[1]  # [B, D]
        memory = torch.stack([x_proj, e_proj], dim=1)         # [B, 2, D]

        # Prepare queries
        tgt = self.query_emb.unsqueeze(0).expand(B, -1, -1) + self.pos_emb.unsqueeze(0)  # [B, 902, D]

        # Causal mask for autoregressive decoding
        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool().to(x_t.device)  # [902, 902]

        # Decode
        output = self.decoder(tgt, memory, tgt_mask=causal_mask)  # [B, 902, D]

        # Split logits
        logits_first = self.head_first(output[:, :2])   # [B, 2, 30]
        logits_rest = self.head_rest(output[:, 2:])     # [B, 900, 11]

        return logits_first, logits_rest