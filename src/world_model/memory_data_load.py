import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset # Import both
import lzma
import os
import glob # To find chunk files
import random # To shuffle chunk files
# from utils.util import set_device # Assuming this exists and works

# --- Device Setup ---
# Re-add device setup or ensure it's handled externally
# Example fallback:
try:
    from utils.util import set_device
    DEVICE = set_device('world_model/memory_data_load_iterable.py')
except (ImportError, NameError):
    print("Warning: set_device not found or failed. Falling back to CPU.")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- End Device Setup ---


class _MapStyleEvaluationDataset(Dataset):
    """
    Helper Map-Style Dataset class to wrap pre-loaded evaluation memory data.
    This is used because the evaluation dataset is assumed to be a single file
    and benefits from standard Dataset features (e.g., known length).
    """
    def __init__(self, data, device):
        # Data is expected to be the dictionary loaded by _load_data
        self.device = device
        self.current_state = data['current_state'].to(device)
        self.current_shape = data['current_shape'].to(device)
        self.target_state = data['target_state'].to(device)
        self.target_shape = data['target_shape'].to(device)
        self.action = data['action'].to(device)
        # Terminal might not be strictly needed for __getitem__ but keep if useful
        self.terminal = data['terminal'].to(device)

    def __len__(self):
        return len(self.current_state)

    def __getitem__(self, idx):
        """
        Returns a dictionary for a single evaluation transition.
        """
        return {
            'current_state': self.current_state[idx],
            'current_shape': self.current_shape[idx],
            'target_state': self.target_state[idx],
            'target_shape': self.target_shape[idx],
            'action': self.action[idx],
        }

class IterableWorldModelDataset(IterableDataset):
    """
    Loads and streams training data iteratively from chunked memory files
    (e.g., sequential_memory_chunk_*.pt.xz) saved in a directory.

    Loads a separate, single evaluation memory file entirely for evaluation purposes.
    """
    def __init__(self,
                 memory_chunk_dir="../output/", # Directory with training chunks
                 evaluation_file_path="../../memory/evaluation_memory.pt.xz",
                 chunk_shuffle=True): # Option to shuffle chunk file order each epoch
        """
        Args:
            memory_chunk_dir (str): Path to the directory containing training chunk files.
                                    Files should be named like 'sequential_memory_chunk_*.pt.xz'.
            evaluation_file_path (str): Path to the single evaluation memory file.
            chunk_shuffle (bool): If True, shuffle the order of chunk files processed in each iteration.
        """
        super(IterableWorldModelDataset).__init__()

        memory_chunk_dir = '../output/' + memory_chunk_dir

        # add '/memory_chunks' to the path if not already present
        memory_chunk_dir = os.path.join(memory_chunk_dir, "memory_chunks")
        self.memory_chunk_dir = os.path.join(os.getcwd(), memory_chunk_dir)
        
        self.evaluation_file_path = os.path.join(memory_chunk_dir, evaluation_file_path)

        # for debugging print the files in the directory
        self.chunk_shuffle = chunk_shuffle
        self.device = DEVICE # Use the globally defined device

        self.chunk_files = sorted(glob.glob(os.path.join(self.memory_chunk_dir, "sequential_memory_chunk_*.pt.xz")))

        if not self.chunk_files:
            raise FileNotFoundError(f"No training chunk files found matching 'sequential_memory_chunk_*.pt.xz' in {self.memory_chunk_dir}")
        else:
             print(f"Found {len(self.chunk_files)} training chunk files.")

        self.chunk_files = sorted(glob.glob(os.path.join(self.memory_chunk_dir, "sequential_memory_chunk_*.pt.xz")))
        self.current_file_index = -1 # Start at -1, will be 0 for the first file
        self.total_files = len(self.chunk_files)

        # --- Load Evaluation Data ---
        print(f"Attempting to load evaluation data from: {self.evaluation_file_path}")
        if os.path.exists(self.evaluation_file_path):
            eval_data_loaded = self._load_data(self.evaluation_file_path)
            self.eval_dataset = _MapStyleEvaluationDataset(eval_data_loaded, self.device)
            print(f"Evaluation data loaded successfully ({len(self.eval_dataset)} samples).")
        else:
            # Since the user stated eval is always one file, we might error if it's missing
            # Or return None and let the user handle it. Let's raise an error for clarity.
            raise FileNotFoundError(f"Evaluation file not found at: {self.evaluation_file_path}. An evaluation file is required.")
            # Alternatively:
            # print("Warning: Evaluation file not found. No evaluation dataset will be available.")
            # self.eval_dataset = None


    def _load_data(self, file_path):
        """Loads and processes data from a single .pt.xz file."""
        try:
            with lzma.open(file_path, "rb") as f:
                # Load directly to the target device if possible and sensible,
                # otherwise load to CPU and move later. Loading to CPU is safer.
                data = torch.load(f, map_location='cpu') # Load to CPU first

            # Basic validation
            expected_keys = ['current_state', 'current_shape', 'target_state', 'target_shape', 'action', 'terminal']
            if not all(key in data for key in expected_keys):
                 raise ValueError(f"File {file_path} is missing one or more expected keys: {expected_keys}")

            # Process and move tensors to the target device
            # Convert tensors - Ensure they are tensors before conversion if needed
            data['current_state'] = torch.as_tensor(data['current_state'], dtype=torch.float32).long()
            data['current_shape'] = torch.as_tensor(data['current_shape'], dtype=torch.float32).long()
            # Assertions after conversion
            assert torch.all(data['current_state'] >= 0) and torch.all(data['current_state'] <= 10), f"State values out of bounds in {file_path}"
            assert torch.all(data['current_shape'] >= 0) and torch.all(data['current_shape'] <= 29), f"Shape values out of bounds in {file_path}"

            data['target_state'] = torch.as_tensor(data['target_state'], dtype=torch.float32).long()
            data['target_shape'] = torch.as_tensor(data['target_shape'], dtype=torch.float32).long()
            assert torch.all(data['target_state'] >= 0) and torch.all(data['target_state'] <= 10), f"State values out of bounds in {file_path}"
            assert torch.all(data['target_shape'] >= 0) and torch.all(data['target_shape'] <= 29), f"Shape values out of bounds in {file_path}"

            data['action'] = torch.as_tensor(data['action'], dtype=torch.float32).long()
            data['terminal'] = torch.as_tensor(data['terminal'], dtype=torch.bool).long()

            # Return data still on CPU, move to device happens during iteration or in eval dataset
            return data
        except Exception as e:
            print(f"Error loading or processing file {file_path}: {e}")
            # Decide how to handle: raise error, return None, return empty dict?
            # Raising error is often clearest.
            raise e


    def __iter__(self):
        """
        Iterator for streaming training data.
        Loads chunk files one by one, yields samples from each chunk.
        Optionally shuffles the order of chunk files.
        """
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.chunk_files
        self.current_file_index = -1
        self.total_files = len(files_to_process) # Update total in case workers change the list length

        if self.chunk_shuffle:
            # Shuffle differently in each epoch and potentially each worker
            g = torch.Generator()
            # Seed generator based on epoch if available (requires passing epoch info)
            # or just use random seed each time. Using random is simpler here.
            # g.manual_seed(seed + worker_id) # More complex setup
            indices = torch.randperm(len(files_to_process), generator=g).tolist()
            files_to_process = [files_to_process[i] for i in indices]
            # print(f"Worker {worker_info.id if worker_info else 0}: Shuffled chunk order: {[os.path.basename(f) for f in files_to_process[:5]]}...") # Debug print
        
        # Basic multi-worker support: Each worker processes a subset of files
        if worker_info is not None:
            # Split files among workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            files_to_process = files_to_process[worker_id::num_workers]
            # print(f"Worker {worker_id}: Processing {len(files_to_process)} files.")


        for file_idx, file_path in enumerate(files_to_process):
            # Update the index *before* loading/processing the file
            self.current_file_index = file_idx 

            try:
                # Load data for the current chunk (on CPU)
                chunk_data = self._load_data(file_path)
                num_samples_in_chunk = len(chunk_data['current_state'])

                # Iterate through samples within the loaded chunk
                for idx in range(num_samples_in_chunk):
                    # Construct the sample dictionary
                    sample = {
                        # Move data to the target device just before yielding
                        'current_state': chunk_data['current_state'][idx].to(self.device),
                        'current_shape': chunk_data['current_shape'][idx].to(self.device),
                        'target_state': chunk_data['target_state'][idx].to(self.device),
                        'target_shape': chunk_data['target_shape'][idx].to(self.device),
                        'action': chunk_data['action'][idx].to(self.device),
                        # 'terminal': chunk_data['terminal'][idx].to(self.device) # Optional
                    }
                    yield sample

            except FileNotFoundError:
                 print(f"Warning: Chunk file {file_path} not found during iteration. Skipping.")
                 continue
            except Exception as e:
                 print(f"Warning: Error processing chunk {file_path} during iteration: {e}. Skipping.")
                 continue # Skip to the next file if a chunk is corrupted/unreadable

    def get_evaluation_dataset(self):
        """
        Returns the loaded evaluation dataset (map-style).
        Returns None if the evaluation file was not found during initialization.
        """
        if not hasattr(self, 'eval_dataset'):
             # This case should ideally be prevented by the __init__ error handling
             print("Error: Evaluation dataset was not loaded successfully during initialization.")
             return None
        return self.eval_dataset