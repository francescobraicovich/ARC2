import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import lzma
import os
from utils.util import set_device

DEVICE = set_device('world_model/memory_data_load.py')

class WorldModelDataset(Dataset):
    """
    Loads a saved memory file and (optionally) an evaluation memory file. 
    Prepares next_state and next_shape by shifting the data, removes the last row 
    (which has no next state/shape), and filters out all transitions where terminal==True.
    Provides a train-test split method that uses evaluation memory (if available) for testing.
    """
    def __init__(self, 
                 sequential_file_path="../output/memory/sequential_memory.pt.xz", 
                 evaluation_file_path="../output/memory/evaluation_memory.pt.xz"):
        # Load sequential memory data
        self.sequential_file_path = os.path.join(os.getcwd(), sequential_file_path)
        print('Loading sequential data from:', self.sequential_file_path)
        sequential_data = self._load_data(self.sequential_file_path)
        self.current_state = sequential_data['current_state']
        self.current_shape = sequential_data['current_shape']
        self.target_state = sequential_data['target_state']
        self.target_shape = sequential_data['target_shape']
        self.action = sequential_data['action']
        self.terminal = sequential_data['terminal']

        # Try to load evaluation memory data
        self.eval_data = None
        evaluation_file_full_path = os.path.join(os.getcwd(), evaluation_file_path)
        if os.path.exists(evaluation_file_full_path):
            print("Evaluation file found. Loading evaluation data from:", evaluation_file_full_path)
            self.eval_data = self._load_data(evaluation_file_full_path)
        else:
            print("Evaluation file not found. WARNING: Evaluation will be performed using a train_test split on sequential memory, which may introduce bias.")
        print('Data loaded successfully.')

    def _load_data(self, file_path):
        with lzma.open(file_path, "rb") as f:
            data = torch.load(f)
        # Convert and move tensors to the device with the desired dtypes
        data['current_state'] = torch.tensor(data['current_state'], dtype=torch.float32).to(DEVICE).long()
        data['current_shape'] = torch.tensor(data['current_shape'], dtype=torch.float32).to(DEVICE).long()
        assert torch.all(data['current_state'] >= 0) and torch.all(data['current_state'] <= 10), "State values out of bounds"
        assert torch.all(data['current_shape'] >= 0) and torch.all(data['current_shape'] <= 29), "Shape values out of bounds"

        data['target_state'] = torch.tensor(data['target_state'], dtype=torch.float32).to(DEVICE).long()
        data['target_shape'] = torch.tensor(data['target_shape'], dtype=torch.float32).to(DEVICE).long()
        assert torch.all(data['target_state'] >= 0) and torch.all(data['target_state'] <= 10), "State values out of bounds"
        assert torch.all(data['target_shape'] >= 0) and torch.all(data['target_shape'] <= 29), "Shape values out of bounds"

        data['action'] = torch.tensor(data['action'], dtype=torch.float32).to(DEVICE).long()
        data['terminal'] = torch.tensor(data['terminal'], dtype=torch.bool).to(DEVICE).long()
        return data

    def __len__(self):
        return len(self.current_state)

    def __getitem__(self, idx):
        """
        Returns a dictionary with:
          - current state, shape, action, terminal
          - target state, shape (i.e. next state features)
        """
        return {
            'current_state': self.current_state[idx],
            'current_shape': self.current_shape[idx],
            'target_state': self.target_state[idx],
            'target_shape': self.target_shape[idx],
            'action': self.action[idx],
        }

    def train_test_split(self, test_ratio=0.2, shuffle=True):
        """
        Splits the dataset into training and testing subsets.
        If evaluation memory is available, returns the sequential memory as the training dataset 
        and the evaluation memory as the test dataset.
        Otherwise, performs a random train_test split on the sequential memory.
        """
        if self.eval_data is not None:
            print("Using evaluation file for testing. Evaluation data is loaded separately.")
            train_dataset = self
            test_dataset = _MemoryDataset(self.eval_data)
            return train_dataset, test_dataset
        else:
            print("No evaluation file found. Performing train_test split on sequential memory. WARNING: This may lead to bias in evaluation.")
            dataset_size = len(self)
            indices = list(range(dataset_size))
            if shuffle:
                np.random.shuffle(indices)
            split = int(np.floor(test_ratio * dataset_size))
            test_indices = indices[:split]
            train_indices = indices[split:]
            train_subset = Subset(self, train_indices)
            test_subset = Subset(self, test_indices)
            return train_subset, test_subset

class _MemoryDataset(Dataset):
    """
    Helper Dataset class to wrap pre-loaded memory data (used for evaluation memory).
    """
    def __init__(self, data):
        self.current_state = data['current_state']
        self.current_shape = data['current_shape']
        self.target_state = data['target_state']
        self.target_shape = data['target_shape']
        self.action = data['action']
        self.terminal = data['terminal']

    def __len__(self):
        return len(self.current_state)

    def __getitem__(self, idx):
        return {
            'current_state': self.current_state[idx],
            'current_shape': self.current_shape[idx],
            'target_state': self.target_state[idx],
            'target_shape': self.target_shape[idx],
            'action': self.action[idx],
        }
