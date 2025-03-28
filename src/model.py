import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import set_device

# Determine the device: CUDA -> MPS -> CPU
DEVICE = set_device('model.py')

class Actor(nn.Module):
    def __init__(self, state_emb_dim, action_emb_dim, h1_dim=256, h2_dim=128):
        super(Actor, self).__init__()

        self.state_emb_dim = state_emb_dim

        self.concat_fc = nn.Linear(state_emb_dim * 2, h1_dim).to(DEVICE)
        self.fc1 = nn.Linear(h1_dim, h2_dim).to(DEVICE)
        self.fc2 = nn.Linear(h2_dim, action_emb_dim).to(DEVICE)

        self.gelu = nn.GELU()
       
        self.init_weights()
        self.to(DEVICE) 

    def init_weights(self):
        """
        Custom weight initialization for the layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x_t):
        """
        Forward pass of the Actor network.
        """    
        x = self.gelu(self.concat_fc(x_t))  # [batch, h1_dim]
        x = self.gelu(self.fc1(x))  # [batch, h2_dim]
        out = self.fc2(x)  # [batch, action_emb_dim]
        return out

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_emb_dim, action_emb_dim, h1_dim=256, h2_dim=128):
        super(Critic, self).__init__()

        self.state_emb_dim = state_emb_dim
        
        self.state_concat_fc = nn.Linear(state_emb_dim * 2, h1_dim).to(DEVICE)
        self.action_proj = nn.Linear(action_emb_dim, h1_dim).to(DEVICE)
        self.action_concat_fc = nn.Linear(h1_dim * 2, h2_dim).to(DEVICE)
        
        self.fc1 = nn.Linear(h2_dim, 1)  # Output a scalar (e.g., Q-value) per (state, action) pair.
        
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
        # Concatenate the two state embeddings
        x = self.gelu(self.state_concat_fc(x))
        a_proj = self.action_proj(a)
        x_concat = torch.cat((x, a_proj), dim=-1)  # Concatenate state and action embeddings
        x = self.gelu(self.action_concat_fc(x_concat))
        out = self.fc1(x)
        out.squeeze_(-1)
        return out



