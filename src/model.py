import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # NOTE: this is current implementation waiting to pass the state and shape into the encoder
        state_flat = torch.reshape(state, (state.shape[0],self.nb_states - 5))
        shape_flat = torch.reshape(shape, (shape.shape[0], 4))
        # add a zero on the end of the shape tensor for each batch
        shape_flat = torch.cat([shape_flat, torch.zeros((shape_flat.shape[0], 1))], dim=-1)
        print('state shape:', state_flat.shape)
        print('shape shape:', shape_flat.shape)
        x = torch.cat([state_flat, shape_flat], dim=-1)
        print('x shape:', x.shape)
        
        # print the dtype of x
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