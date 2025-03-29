import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import os
from utils.util import set_device

DEVICE=set_device('world_model/memory_data_load.py')

class WorldModelDataset(Dataset):
    """
    Loads a saved memory file, prepares next_state and next_shape by shifting the data,
    removes the last row (which has no next state/shape), and filters out all transitions
    where terminal==True. Also provides a train-test split method.
    """
    def __init__(self, file_path="../output/memory/sequential_memory.pkl"):
        # Load the saved data
        # append the file path to the current working directory
        file_path = os.path.join(os.getcwd(), file_path)
        print('Loading data from:', file_path)

        with open(file_path, "rb") as f:
            data = pickle.load(f)
        # Convert lists to numpy arrays (assuming each entry is a numpy array or a scalar)

        state = torch.stack(data['state']).to(DEVICE)
        shape = torch.stack(data['shape']).to(DEVICE)
        action = torch.stack(data['action']).to(DEVICE)
        terminal = torch.tensor(data['terminal'], dtype=torch.bool).to(DEVICE)
        
        # Process the states into current and target states
        current_state, target_state = self.process_states(state)
        current_shape, target_shape = self.process_shapes(shape)
        
        # Create next_state and next_shape by shifting the arrays one element ahead.
        next_current_state, next_target_state = current_state[1:], target_state[1:]
        next_current_shape, next_target_shape = current_shape[1:], target_shape[1:]
        
        # Remove the last row from current data since it has no corresponding next state/shape.
        current_state = current_state[:-1]
        current_shape = current_shape[:-1]
        target_state = target_state[:-1]
        target_shape = target_shape[:-1]
        action = action[:-1]
        terminal = terminal[:-1]

        
        # Remove transitions where terminal==True (i.e. where an episode ended)
        valid_mask = (terminal == False)
        self.current_state = current_state[valid_mask]
        self.current_shape = current_shape[valid_mask]
        self.target_state = target_state[valid_mask]
        self.target_shape = target_shape[valid_mask]
        self.action = action[valid_mask]
        self.terminal = terminal[valid_mask]

        self.current_state = torch.tensor(self.current_state, dtype=torch.float32).to(DEVICE).long()
        self.current_shape = torch.tensor(self.current_shape, dtype=torch.float32).to(DEVICE).long()
        self.target_state = torch.tensor(self.target_state, dtype=torch.float32).to(DEVICE).long()
        self.target_shape = torch.tensor(self.target_shape, dtype=torch.float32).to(DEVICE).long()
        self.action = torch.tensor(self.action, dtype=torch.float32).to(DEVICE).long()
        self.terminal = torch.tensor(self.terminal, dtype=torch.bool).to(DEVICE).long()
        print('Data loaded successfully.')
        
    def __len__(self):
        return len(self.current_state)
    
    def __getitem__(self, idx):
        """
        Returns a dictionary with:
          - state, shape, action, terminal (current)
          - next_state, next_shape (target features)
        """
        return {
            'current_state': self.current_state[idx],
            'current_shape': self.current_shape[idx],
            'target_state': self.target_state[idx],
            'target_shape': self.target_shape[idx],
            'action': self.action[idx],
        }
    
    def process_states(self, state):
        current_state = state[:, :, :, 0]
        target_state = state[:, :, :, 1]
        target_state = target_state.reshape(-1, 900) + 1
        return current_state, target_state
    
    def process_shapes(self, shape):
        current_shape = shape[:, :, 0]
        target_shape = shape[:, :, 1]
        return current_shape, target_shape
 
    def train_test_split(self, test_ratio=0.2, shuffle=True):
        """
        Splits the dataset indices into training and test subsets.
        Returns two torch.utils.data.Subset objects.
        """
        dataset_size = len(self)
        indices = list(range(dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        split = int(np.floor(test_ratio * dataset_size))
        test_indices = indices[:split]
        train_indices = indices[split:]
        train_subset = torch.utils.data.Subset(self, train_indices)
        test_subset = torch.utils.data.Subset(self, test_indices)
        return train_subset, test_subset