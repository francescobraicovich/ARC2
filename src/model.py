import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util_transformer import EncoderTransformerConfig
from transformer import EncoderTransformer

# Custom weight initialization function
def fanin_init(size, fanin=None):
    """
    Initialize weights for layers with uniform distribution based on the fan-in size.
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

# Actor Network
class Actor(nn.Module):
    def __init__(self, encoder_config: EncoderTransformerConfig, nb_actions=3, init_w=3e-3):
        """
        Actor network using an EncoderTransformer for policy prediction.

        Args:
            encoder_config (EncoderTransformerConfig): Configuration for the EncoderTransformer.
            nb_actions (int): Dimension of action space (e.g., 3 for 3D point).
            init_w (float): Initialization range for the output layer.
        """
        super(Actor, self).__init__()
        self.encoder = EncoderTransformer(encoder_config)
        self.action_head = nn.Linear(encoder_config.latent_dim, nb_actions)
        self.tanh = nn.Tanh()  # For bounded action space
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        """
        Custom weight initialization for the output layer.
        """
        nn.init.kaiming_uniform_(self.action_head.weight, nonlinearity='relu')
        self.action_head.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        """
        Forward pass of the Actor network.

        Args:
            x (tuple): A tuple containing:
                - state (torch.Tensor): Tensor of shape [batch_size, R, C, 2].
                - shape (torch.Tensor): Tensor of shape [batch_size, 2, 2].

        Returns:
            torch.Tensor: Action tensor of shape [batch_size, 3].
        """
        state, shape = x  # Unpack the input tuple
        latent_mu, latent_logvar = self.encoder(state, shape, dropout_eval=False)
        action = self.action_head(latent_mu)
        action = torch.sigmoid(action)  # Assuming actions are bounded between -1 and 1
        return action

# Critic Network
class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=128, init_w=3e-3):
        """
        Critic network for state-action value estimation.
        Args:
            nb_states: Dimension of state space.
            nb_actions: Dimension of action space.
            hidden1: Number of neurons in the first hidden layer.
            hidden2: Number of neurons in the second hidden layer.
            init_w: Initialization range for the output layer.
        """
        super(Critic, self).__init__()
        self.nb_states = nb_states
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)  # Combine state and action
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        """
        Custom weight initialization for the layers.
        """
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x, a):
        """
        Forward pass of the Critic network.
        Args:
            x: State input.
            a: Action input.
        """
        x = torch.reshape(x, (x.shape[0],self.nb_states))
        out = self.relu(self.fc1(x))
        out = torch.cat([out, a], dim=-1)  # Concatenate state and action
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out