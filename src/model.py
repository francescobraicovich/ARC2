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

# Custom weight initialization function
def fanin_init(size, fanin=None):
    """
    Initialize weights for layers with uniform distribution based on the fan-in size.
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v).to(DEVICE)


# Actor Network
class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=128, init_w=3e-3):
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
        """
        state, shape = x

        # Reshape state and shape tensors
        state_flat = state.reshape(state.shape[0], self.nb_states - 5)
        shape_flat = shape.reshape(shape.shape[0], 4)

        # Add a zero to the end of the shape tensor for each batch
        shape_flat = torch.cat([shape_flat, torch.zeros((shape_flat.size(0), 1), device=DEVICE)], dim=-1)

        # Concatenate the tensors
        x = torch.cat([state_flat, shape_flat], dim=-1)

        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.tanh(self.fc3(out))  # Use Tanh for bounded outputs
        return out


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
        """
        state, shape = x
        # Reshape state and shape tensors
        state_flat = state.reshape(state.shape[0], state.shape[1], self.nb_states - 5)
        shape_flat = shape.reshape(shape.shape[0], shape.shape[1], 4)

        # Add a zero to the end of the shape tensor for each batch
        shape_flat = torch.cat([shape_flat, torch.zeros((shape_flat.size(0), shape_flat.size(1), 1), device=DEVICE)], dim=-1)

        # Concatenate the tensors
        x = torch.cat([state_flat, shape_flat], dim=-1)

        out = self.relu(self.fc1(x))
        concatenated = torch.cat([out, a], dim=-1)  # Concatenate state and action
        out = self.relu(self.fc2(concatenated))
        out = self.fc3(out)
        return out
