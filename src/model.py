import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import set_device
from cnn import CNNFeatureExtractor

# Determine the device: CUDA -> MPS -> CPU
DEVICE = set_device()

from utils.util_transformer import EncoderTransformerConfig, TransformerLayerConfig
from transformer import EncoderTransformer

# Custom weight initialization function
def fanin_init(size, fanin=None):
    """
    Initialize weights for layers with uniform distribution based on the fan-in size.
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v).to(DEVICE)

def config_encoder(latent_dim=32):
    # Define configuration
    transformer_layer_config = TransformerLayerConfig(
        num_heads=1,
        emb_dim_per_head=8,
        mlp_dim_factor=1.0,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        use_bias=False,
        activation="silu",
        dtype=torch.float32
    )

    cfg = EncoderTransformerConfig(
        vocab_size=10,
        max_rows=30,
        max_cols=30,
        emb_dim=32,
        latent_dim=latent_dim,
        num_layers=1,
        scaled_position_embeddings=False,
        variational=True,
        latent_projection_bias=False,
        dtype=torch.float32,
        transformer_layer=transformer_layer_config
    )

    # Initialize the encoder
    encoder = EncoderTransformer(cfg)
    return encoder

class CustomSoftsign(nn.Module):
    def __init__(self, min_val, max_val):
        super(CustomSoftsign, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.softsign = nn.Softsign()

    def forward(self, x):
        # Apply Softsign and scale to [min_val, max_val]
        return self.min_val + (self.max_val - self.min_val) * (self.softsign(x) + 1) / 2



def process_in_chunks(encoder, state, shape, chunk_size):
    """
    Processes large batches by splitting into smaller chunks.

    Args:
        encoder: The encoder module.
        state: Input tensor for states of shape (B, R, C, 2).
        shape: Input tensor for shapes of shape (B, 2, 2).
        chunk_size: Number of samples to process at a time.

    Returns:
        latent_mu: Concatenated latent mean tensor of shape (B, latent_dim).
        latent_logvar: Concatenated latent logvar tensor of shape (B, latent_dim) if variational=True, else None.
    """
    num_samples = state.size(0)  # Total number of samples in the batch
    latent_mu_list = []

    for i in range(0, num_samples, chunk_size):
        # Get the current chunk
        state_chunk = state[i:i + chunk_size]
        shape_chunk = shape[i:i + chunk_size]

        # Process the chunk through the encoder
        latent_mu = encoder(state_chunk, shape_chunk)

        # Store the results
        latent_mu_list.append(latent_mu)
    
    # Concatenate results along the batch dimension
    latent_mu = torch.cat(latent_mu_list, dim=0)
    return latent_mu

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, latent_dim=32, type='lpn',
                 hidden1=256, hidden2=128, init_w=3e-3, chunk_size=100, min_val=-1, max_val=1):
        """
        Actor network for policy prediction.
        Args:
            nb_states: Dimension of state space (used for MLP).
            nb_actions: Dimension of action space.
            hidden1: Number of neurons in the first hidden layer.
            hidden2: Number of neurons in the second hidden layer.
            init_w: Initialization range for the output layer.
        """
        super(Actor, self).__init__()
        self.type = type
        self.chunk_size = chunk_size
        
        if self.type == 'lpn':
            self.encoder = config_encoder(latent_dim).to(DEVICE)
            self.fc1 = nn.Linear(latent_dim, hidden1)
        
        elif self.type == 'cnn':
            self.cnn = CNNFeatureExtractor(hidden1).to(DEVICE)

        else:
            # Default MLP setup if not LPN or CNN
            self.fc1 = nn.Linear(nb_states, hidden1).to(DEVICE)

        # Shared layers (for MLP or after CNN feature extraction)
        self.fc2 = nn.Linear(hidden1, hidden2).to(DEVICE)
        self.fc3 = nn.Linear(hidden2, nb_actions).to(DEVICE)
        self.bn2 = nn.BatchNorm1d(hidden2).to(DEVICE)



        self.relu = nn.LeakyReLU()
        self.final_activation = CustomSoftsign(min_val, max_val)
        
        self.init_weights(init_w)
        self.to(DEVICE)  # Move the network to the selected device

    def init_weights(self, init_w):
        """
        Custom weight initialization for the layers based on network type.
        Args:
            init_w: Initialization range for the output layer.
        """
        if self.type == 'cnn':
            # CNNFeatureExtractor already initialized its own weights
            pass  # No additional initialization needed here
        else:
            if hasattr(self, 'fc1') and self.fc1.weight is not None:
                nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
                if self.fc1.bias is not None:
                    nn.init.zeros_(self.fc1.bias)
        
        # Initialize shared layers
        if hasattr(self, 'fc2') and self.fc2.weight is not None:
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)
        if hasattr(self, 'fc3') and self.fc3.weight is not None:
            nn.init.uniform_(self.fc3.weight, -init_w, init_w)
            if self.fc3.bias is not None:
                nn.init.zeros_(self.fc3.bias)

    def forward(self, x):

        """
        Forward pass of the Actor network.

        Shapes of arguments:
            state: input data as tokens. Shape (B, R, C, 2).
            shape: shape of the data. Shape (B, 2, 2).
        """
        state, shape = x
        
        if self.type == 'lpn':
            # Use transformer encoder
            latent = process_in_chunks(self.encoder, state, shape, self.chunk_size)
            latent = self.fc1(latent)

        elif self.type == 'cnn':
            latent = self.cnn(state)  # Pass through CNN
        else:
            # Plain MLP approach
            latent = self.relu(self.fc1(state))

        # Shared part of the network
        out = self.fc2(latent)
        out = self.bn2(out)  # Batch Normalization here
        out = self.relu(out)
        out = self.fc3(out)
        out = self.final_activation(out)
        return out

# Critic Network
class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, type='lpn', latent_dim = 32, hidden1=256, hidden2=128, init_w=3e-3, chunk_size=100):
        """
        Critic network for state-action value estimation.
        Args:
            nb_states: Dimension of state space (for MLP).
            nb_actions: Dimension of action space.
            hidden1: Number of neurons in the first hidden layer.
            hidden2: Number of neurons in the second hidden layer.
            init_w: Initialization range for the output layer.
        """
        super(Critic, self).__init__()
        self.type = type
        self.chunk_size = chunk_size
        
        if self.type == 'lpn':
            self.encoder = config_encoder(latent_dim).to(DEVICE)
            self.fc1 = nn.Linear(latent_dim, hidden1).to(DEVICE)

        elif self.type == 'cnn':
            self.cnn = CNNFeatureExtractor(hidden1).to(DEVICE)
        else:
            # Default MLP setup if not LPN or CNN
            self.fc1 = nn.Linear(nb_states, hidden1).to(DEVICE)

        # In the second layer, we combine state-latent + action
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2).to(DEVICE)
        self.fc3 = nn.Linear(hidden2, 1).to(DEVICE)
        self.relu = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(hidden2).to(DEVICE)  # BatchNorm after fc2


        self.init_weights(init_w)
        self.to(DEVICE)  # Move the network to the selected device
    
    def init_weights(self, init_w):
        """
        Custom weight initialization for the layers based on network type.
        Args:
            init_w: Initialization range for the output layer.
        """
        if self.type == 'cnn':
            # CNNFeatureExtractor already initialized its own weights
            pass  # No additional initialization needed here
        else:
            if hasattr(self, 'fc1') and self.fc1.weight is not None:
                nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
                if self.fc1.bias is not None:
                    nn.init.zeros_(self.fc1.bias)
        
        # Initialize shared layers
        if hasattr(self, 'fc2') and self.fc2.weight is not None:
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)
        if hasattr(self, 'fc3') and self.fc3.weight is not None:
            nn.init.uniform_(self.fc3.weight, -init_w, init_w)
            if self.fc3.bias is not None:
                nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x, a):

        state, shape = x
        reshape = False

        if len(a.shape) == 3:
            reshape = True
            B, N, nb_actions = a.shape
            a = torch.reshape(a, (B * N, nb_actions))

        if self.type == 'lpn':
            latent = process_in_chunks(self.encoder, state, shape, self.chunk_size)
            latent = self.fc1(latent)

        elif self.type == 'cnn':
            latent = self.cnn(state)  # Pass through CNN
        else:
            # MLP approach
            latent = self.relu(self.fc1(state))
        
        # Combine latent + action
        concatenated = torch.cat([latent, a], dim=-1)  # => (N*B, hidden1 + nb_actions)
        out = self.fc2(concatenated)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        if reshape:
            out = out.reshape(B, N, -1)
        out = torch.squeeze(out)
        return out



