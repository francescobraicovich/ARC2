import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import os

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
        state = np.array(data['state'])
        shape = np.array(data['shape'])
        action = np.array(data['action'])
        terminal = np.array(data['terminal'])
        print('state shape:', state.shape)
        
        # Create next_state and next_shape by shifting the arrays one element ahead.
        next_state = state[1:]
        next_shape = shape[1:]
        
        # Remove the last row from current data since it has no corresponding next state/shape.
        state = state[:-1]
        shape = shape[:-1]
        action = action[:-1]
        terminal = terminal[:-1]
        
        # Remove transitions where terminal==True (i.e. where an episode ended)
        valid_mask = (terminal == False)
        self.state = torch.tensor(state[valid_mask])
        self.next_state = torch.tensor(next_state[valid_mask])
        self.shape = torch.tensor(shape[valid_mask])
        self.next_shape = torch.tensor(next_shape[valid_mask])
        self.action = torch.tensor(action[valid_mask])
        self.terminal = torch.tensor(terminal[valid_mask])
        
    def __len__(self):
        return len(self.state)
    
    def __getitem__(self, idx):
        """
        Returns a dictionary with:
          - state, shape, action, terminal (current)
          - next_state, next_shape (target features)
        """
        return {
            'state': self.state[idx],
            'shape': self.shape[idx],
            'action': self.action[idx],
            'terminal': self.terminal[idx],
            'next_state': self.next_state[idx],
            'next_shape': self.next_shape[idx]
        }
    
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