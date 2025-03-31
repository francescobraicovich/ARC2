import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lzma
import os
from utils.util import set_device

DEVICE=set_device('world_model/memory_data_load.py')

class WorldModelDataset(Dataset):
    """
    Loads a saved memory file, prepares next_state and next_shape by shifting the data,
    removes the last row (which has no next state/shape), and filters out all transitions
    where terminal==True. Also provides a train-test split method.
    """
    def __init__(self, file_path="../output/memory/sequential_memory.pt.xz"):
        # Load the saved data
        # append the file path to the current working directory
        file_path = os.path.join(os.getcwd(), file_path)
        print('Loading data from:', file_path)

        with lzma.open(file_path, "rb") as f:
            data = torch.load(f)

        self.current_state = data['current_state']
        self.current_shape = data['current_shape']
        self.target_state = data['target_state']
        self.target_shape = data['target_shape']
        self.action = data['action']
        self.terminal = data['terminal']

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