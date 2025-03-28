import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import set_device
from cnn import CNNFeatureExtractor

# Determine the device: CUDA -> MPS -> CPU
DEVICE = set_device('model.py')

class Actor(nn.Module):
    def __init__(self, state_emb_dim, action_emb_dim, h1_dim=256, h2_dim=128):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_emb_dim, h1_dim).to(DEVICE)
        self.fc2 = nn.Linear(h1_dim, h2_dim).to(DEVICE)
        self.fc3 = nn.Linear(h2_dim, action_emb_dim).to(DEVICE)

        self.gelu = nn.GELU()
       
        self.init_weights()
        self.to(DEVICE) 

    def init_weights(self):
        """
        Custom weight initialization for the layers.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x_t):
        """
        Forward pass of the Actor network.
        """
        print('Using the actor network')
        print(f"Input shape: {x_t.shape}")
        x = self.gelu(self.fc1(x_t))
        x = self.gelu(self.fc2(x))
        out = self.fc3(x)
        return out

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_emb_dim, action_emb_dim, h1_dim=256, h2_dim=128):
        super(Critic, self).__init__()
        
        # Project both state and action embeddings to a common dimension for attention.
        self.state_proj = nn.Linear(state_emb_dim, h1_dim)
        self.action_proj = nn.Linear(action_emb_dim, h1_dim)
        
        # Multi-head cross-attention: treat projected state embeddings as queries and projected action embeddings as keys and values.
        self.cross_attn = nn.MultiheadAttention(embed_dim=h1_dim, num_heads=4, batch_first=True)
        
        # Feedforward network for further processing after attention.
        self.fc1 = nn.Linear(h1_dim, h2_dim)
        self.fc2 = nn.Linear(h2_dim, 1)  # Output a scalar (e.g., Q-value) per (state, action) pair.
        
        self.gelu = nn.GELU()
        
        self.init_weights()
        self.to(DEVICE)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, a):
        """
        Args:
            x: [batch, k_neighrest, state_emb_dim]
            a: [batch, k_neighrest, action_emb_dim]
            
        The forward pass flattens the batch and neighborhood dimensions into a meta-batch
        so that the cross-attention layer can be applied in parallel over both dimensions.
        """
        batch, k, _ = x.shape
        
        # Project the state and action embeddings.
        x_proj = self.state_proj(x)   # [batch, k, h1_dim]
        a_proj = self.action_proj(a)    # [batch, k, h1_dim]
        
        # Flatten the first two dimensions (batch and k) into a meta-batch dimension.
        # This allows the attention layer to process all (state, action) pairs in parallel.
        x_meta = x_proj.view(batch * k, 1, -1)  # [batch*k, 1, h1_dim] as queries.
        a_meta = a_proj.view(batch * k, 1, -1)    # [batch*k, 1, h1_dim] as keys and values.
        
        # Apply cross-attention: state tokens attend to the corresponding action tokens.
        attn_out, _ = self.cross_attn(query=x_meta, key=a_meta, value=a_meta)  # [batch*k, 1, h1_dim]
        attn_out = attn_out.squeeze(1)  # [batch*k, h1_dim]
        
        # Process with a feedforward network.
        ff_out = self.gelu(self.fc1(attn_out))  # [batch*k, h2_dim]
        out = self.fc2(ff_out)                  # [batch*k, 1]
        
        # Reshape back to [batch, k, 1]
        print(f"Output shape before view: {out.shape}")
        out = out.view(batch, k, -1)
        print(f"Output shape after view: {out.shape}")
        out = out.squeeze(-1)
        print(f"Output shape after view: {out.shape}")
        return out



