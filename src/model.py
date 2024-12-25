import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import to_tensor

# Determine the device: CUDA -> MPS -> CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print("Using device: {} for actor-critic model".format(DEVICE))

from util_transformer import EncoderTransformerConfig
from transformer import EncoderTransformer

# Custom weight initialization function
def fanin_init(size, fanin=None):
    """
    Initialize weights for layers with uniform distribution based on the fan-in size.
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v).to(DEVICE)

encoder_config = EncoderTransformerConfig()

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, encoder_config=encoder_config, type='lpn',
                 hidden1=256, hidden2=128, init_w=3e-3):
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
        
        if self.type == 'lpn':
            self.encoder = EncoderTransformer(encoder_config).to(DEVICE)
            self.fc1 = nn.Linear(nb_states, hidden1)
        
        elif self.type == 'cnn':
            # Simple CNN example
            self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce to (B, 64, 1, 1)
            # Now flatten => (B, 64)
            # Then map to hidden1 so it can replace the usage of fc1 for the next layers.
            self.fc1 = nn.Linear(64, hidden1)

        else:
            # Default MLP setup if not LPN or CNN
            self.fc1 = nn.Linear(nb_states, hidden1)

        # Shared layers (for MLP or after CNN feature extraction)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # For bounded action space
        
        self.init_weights(init_w)
        self.to(DEVICE)  # Move the network to the selected device

    def init_weights(self, init_w):
        """
        Custom weight initialization for the layers.
        """
        # If using CNN, conv layers typically initialize well with default Kaiming,
        # but you can also manually init them if you like:
        # nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        # ...
        
        if hasattr(self.fc1, 'weight'):
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.fc3.weight.data.uniform_(-init_w, init_w)

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
            latent = self.encoder(state, shape)

        elif self.type == 'cnn':
            # state shape: (B, R, C, 2) -> (B, 2, R, C)
            state = state.permute(0, 3, 1, 2)  # reorder axes
            state = state.contiguous()
            # Pass through CNN
            x = self.relu(self.conv1(state))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pool(x)          # (B, 64, 1, 1)
            x = x.reshape(x.size(0), -1) # Flatten to (B, 64)
            # Map to hidden1 dimension
            latent = self.relu(self.fc1(x))

        else:
            # Plain MLP approach
            latent = self.relu(self.fc1(state))

        # Shared part of the network
        out = self.relu(self.fc2(latent))
        out = self.tanh(self.fc3(out))  # Use Tanh for bounded outputs
        return out

# Critic Network
class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, type='lpn', hidden1=256, hidden2=128, init_w=3e-3):
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
        
        if self.type == 'lpn':
            self.encoder = EncoderTransformer(encoder_config).to(DEVICE)
            self.fc1 = nn.Linear(nb_states, hidden1)

        elif self.type == 'cnn':
            # Simple CNN example
            self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # shape => (N*B, 64, 1, 1)
            # Flatten => (N*B, 64)
            self.fc1 = nn.Linear(64, hidden1)  # Merge with action later

        else:
            # Default MLP setup if not LPN or CNN
            self.fc1 = nn.Linear(nb_states, hidden1)

        # In the second layer, we combine state-latent + action
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

        self.init_weights(init_w)
        self.to(DEVICE)  # Move the network to the selected device
    
    def init_weights(self, init_w):
        """
        Custom weight initialization for the layers.
        """
        if hasattr(self.fc1, 'weight'):
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x, a):
        """
        Forward pass of the Critic network.
        Args:
            x: State input as (state, shape).
            a: Action input.

        Shapes of arguments for the CNN case:
            state: (N, B, R, C, 2) -> we eventually reshape to (N*B, 2, R, C).
            a: (N, B, nb_actions) -> we reshape to (N*B, nb_actions).
        """
        state, shape = x
        
        if self.type == 'lpn':
            latent = self.encoder(state, shape)

        elif self.type == 'cnn':
            # Reshape: (N, B, R, C, 2) -> (N, B, 2, R, C) -> (N*B, 2, R, C)
            # If your data is just (B, R, C, 2), adjust accordingly (similar to the Actor).
            N, B, R, C, CH = state.shape
            state = state.permute(0, 1, 4, 2, 3)   # => (N, B, 2, R, C)
            state = state.contiguous()
            state = state.reshape(-1, 2, R, C)    # => (N*B, 2, R, C)

            
            # Forward through CNN
            x = self.relu(self.conv1(state))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pool(x)                # => (N*B, 64, 1, 1)
            x = x.reshape(x.size(0), -1)       # => (N*B, 64)
            
            latent = self.relu(self.fc1(x)) # => (N*B, hidden1)

            # Reshape latent back to (N, B, hidden1)
            latent = latent.reshape(N, B, -1)
        else:
            # MLP approach
            latent = self.relu(self.fc1(state))

        # Combine latent + action
        concatenated = torch.cat([latent, a], dim=-1)  # => (N*B, hidden1 + nb_actions)
        out = self.relu(self.fc2(concatenated))
        out = self.fc3(out)
        return out


class Actor_old(nn.Module):
    
    def __init__(self, nb_states, nb_actions, encoder_config=encoder_config, type='cnn', hidden1=256, hidden2=128, init_w=3e-3):
        raise DeprecationWarning("This class is deprecated. Use the new Actor class instead.")
        """
        Actor network for policy prediction.
        Args:
            nb_states: Dimension of state space.
            nb_actions: Dimension of action space.
            hidden1: Number of neurons in the first hidden layer.
            hidden2: Number of neurons in the second hidden layer.
            init_w: Initialization range for the output layer.
        """
        super(Actor, self).__init__()
        self.type = type
        if self.type == 'lpn':
            self.encoder = EncoderTransformer(encoder_config).to(DEVICE)
        elif self.type == 'cnn':
            # set up cnn layers
            pass
        
        #self.action_head = nn.Linear(encoder_config.latent_dim, nb_actions)
        self.nb_states = nb_states
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # For bounded action space
        self.init_weights(init_w)
        self.to(DEVICE)  # Move the network to the selected device
    
    def init_weights(self, init_w):
        """
        Custom weight initialization for the layers.
        """
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        """
        Forward pass of the Actor network.

        Shapes of arguments:
            state: input data as tokens. Shape (B, R, C, 2).
                - B: batch size.
                - R: number of rows.
                - C: number of columns.
                - 2: two channels (input and output)
            shape: shape of the data. Shape (B, 2, 2).
                - B: batch size.
                - The last two dimension represents (rows, columns) of two channels, e.g. [[R_input, R_output], [C_input, C_output]].
        """
        state, shape = x
        if self.type == 'lpn':
            latent = self.encoder(state, shape)
        elif self.type == 'cnn':
            # Rehshape (Batch, Rows, Columns, Channels) -> (Batch, Channels, Rows, Columns)
            # pass through cnn layers
            pass

        out = self.relu(self.fc2(latent))
        out = self.tanh(self.fc3(out))  # Use Tanh for bounded outputs
        return out

# Critic Network
class Critic_old(nn.Module):
    def __init__(self, nb_states, nb_actions, type='cnn', hidden1=256, hidden2=128, init_w=3e-3):
        raise DeprecationWarning("This class is deprecated. Use the new Critic class instead.")

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
        self.type = type
        
        if self.type == 'lpn':
            self.encoder = EncoderTransformer(encoder_config).to(DEVICE)
        elif self.type == 'cnn':
            # set up cnn layers
            pass

        self.nb_states = nb_states
        self.encoder = EncoderTransformer(encoder_config)
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)  # Combine state and action
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
        self.to(DEVICE)  # Move the network to the selected device
    
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
        
        Shapes of arguments:
            state: input data as tokens. Shape (K, B, R, C, 2).
                - K: k nearest neighbors.
                - B: batch size.
                - R: number of rows.
                - C: number of columns.
                - 2: two channels (input and output)
            shape: shape of the data. Shape (K, B, 2, 2).
                - K: k nearest neighbors.
                - B: batch size.
                - The last two dimension represents (rows, columns) of two channels, e.g. [[R_input, R_output], [C_input, C_output]].
            a: action tensor. Shape (N, B, nb_action):
                - N: number of actions.
                - B: batch size.
                - nb_action: action space dimension.
        """
        state, shape = x

        if self.type == 'lpn':
            latent = self.encoder(state, shape)
        if self.type == 'cnn':
            # Rehshape (N, Batch, Rows, Columns, Channels) -> (N, Batch, Channels, Rows, Columns)
            # View as (N * Batch, Channels, Rows, Columns)
            # pass through cnn layers
            # View as (N, Batch, -1)
            pass
    
        concatenated = torch.cat([latent, a], dim=-1)  # Concatenate state and action
        out = self.relu(self.fc2(concatenated))
        out = self.fc3(out)
        return out
    