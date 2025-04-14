from __future__ import absolute_import
import warnings
import random
from collections import deque, namedtuple
import numpy as np
import torch
import os, lzma


from utils.util import set_device

# If you have a custom 'to_tensor' function, you can import it. 
# (Code below does not strictly require it.)
# from utils.util import to_tensor

# ---------------------------------------------------------------
# Device selection: CUDA -> MPS -> CPU
# ---------------------------------------------------------------
DEVICE = set_device('memory.py')

# ---------------------------------------------------------------
# Experience namedtuple
# ---------------------------------------------------------------
Experience = namedtuple("Experience", ["state0", "state_embedded_0", "action", 
                                       "reward", "state1", "state_embedded_1",
                                        "terminal1", "shape0", "shape1"])

# ---------------------------------------------------------------
# Helper for sampling batch indexes
# ---------------------------------------------------------------
def sample_batch_indexes(low, high, size):
    """
    Draw `size` unique samples from [low, high) if possible.
    Otherwise, allow duplicates (with a warning).
    """
    if high - low >= size:
        # Enough data to sample without replacement
        r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data, sample with replacement
        warnings.warn("Not enough entries to sample without replacement. "
                      "Consider increasing your warm-up phase to avoid oversampling!")
        batch_idxs = np.random.randint(low, high, size=size)
    assert len(batch_idxs) == size
    return batch_idxs

# ---------------------------------------------------------------
# Ring buffer for fast appends and random access
# ---------------------------------------------------------------
class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None] * maxlen

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError("Index out of valid range in RingBuffer.")
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            # Overwrite the oldest entry
            self.start = (self.start + 1) % self.maxlen
        else:
            # Should never happen
            raise RuntimeError("RingBuffer in invalid state.")
        self.data[(self.start + self.length - 1) % self.maxlen] = v

# ---------------------------------------------------------------
# Zeroed observation helper (used only if you wanted >1 window)
# ---------------------------------------------------------------
def zeroed_observation(observation):
    """
    Returns a zero array of the same shape/type as `observation`.
    If window_length=1, you won't really need this.
    """
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape, dtype=observation.dtype)
    elif hasattr(observation, '__iter__'):
        # If observation is a nested structure
        return [zeroed_observation(x) for x in observation]
    else:
        return 0.

# ---------------------------------------------------------------
# Base Memory class (kept minimal)
# ---------------------------------------------------------------
class Memory(object):
    def __init__(self, ignore_episode_boundaries=False):
        self.ignore_episode_boundaries = ignore_episode_boundaries

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, shape, action, reward, terminal, training=True):
        raise NotImplementedError()

    def get_config(self):
        return {'ignore_episode_boundaries': self.ignore_episode_boundaries}

# ---------------------------------------------------------------
# SequentialMemory
#   - stores transitions and samples them
#   - we set window_length=1 for simplicity
# ---------------------------------------------------------------
class SequentialMemory(Memory):
    def __init__(self, limit, ignore_episode_boundaries=False):
        super(SequentialMemory, self).__init__(ignore_episode_boundaries=ignore_episode_boundaries)
        
        self.limit = limit
        
        # RingBuffers for each part of the transition
        self.observations = RingBuffer(limit)
        self.embedded_observations = RingBuffer(limit)
        self.shapes       = RingBuffer(limit)
        self.actions      = RingBuffer(limit)
        self.rewards      = RingBuffer(limit)
        self.terminals    = RingBuffer(limit)
        self.num_actions  = RingBuffer(limit)

        # Initialize chunk count for saving memory chunks
        self.chunk_count = 0 


    @property
    def nb_entries(self):
        return len(self.observations)

    def append(self, observation, embedded_observation, shape, action, reward, terminal, num_actions, training=True):
        """
        Add a new transition to memory:
          observation -> (take action) -> reward, terminal
        The *next* state is the subsequent observation in the ring buffer.
        """
        if training:
            self.observations.append(observation)
            self.embedded_observations.append(embedded_observation)
            self.shapes.append(shape)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            self.num_actions.append(num_actions)

    def sample(self, batch_size, batch_idxs=None):
        """
        Sample a batch of transitions from memory, skipping transitions
        that cross episode boundaries if ignore_episode_boundaries=False.
        
        Because window_length=1, we define:
            state0 = observations[idx - 1]
            state1 = observations[idx]
        """
        if self.nb_entries < 2:
            raise ValueError("Not enough entries in memory to sample a batch. "
                             f"Need >=2, have {self.nb_entries}.")

        # If user hasn't passed in specific indices, sample them
        if batch_idxs is None:
            # We want to sample from [0..nb_entries-2], because we'll do +1
            # below. So let's sample from [0..nb_entries - 2].
            high_value = self.nb_entries - 1
            batch_idxs = sample_batch_indexes(0, high_value, batch_size)

        # We will actually use idx+1 for next state, so do that shift
        batch_idxs = np.array(batch_idxs) + 1

        experiences = []
        for idx in batch_idxs:
            # If ignoring episode boundaries, skip transitions where the
            # previous step was terminal (that would cross the boundary).
            terminal0 = self.terminals[idx - 1] if (idx >= 1) else False

            
            """
            while terminal0:
                # Resample a new index
                idx = sample_batch_indexes(1, self.nb_entries, 1)[0]
                terminal0 = self.terminals[idx - 1] if (idx >= 1) else False
            """
            
            # Now gather transition info
            state0   = self.observations[idx - 1]
            shape0   = self.shapes[idx - 1]
            x_t0 = self.embedded_observations[idx - 1]
            action   = self.actions[idx - 1]
            reward   = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]
            num_actions = self.num_actions[idx - 1]

            state1   = self.observations[idx]
            shape1   = self.shapes[idx]
            x_t1 = self.embedded_observations[idx]

            exp = Experience(state0=state0,  state_embedded_0=x_t0, action=action, reward=reward,
                             state1=state1, state_embedded_1=x_t1, terminal1=terminal0,
                             shape0=shape0, shape1=shape1)
            experiences.append(exp)

        return experiences

    def sample_and_split(self, batch_size, batch_idxs=None):
        """
        Sample from memory and return separate tensors (state0, shape0, action, etc.).
        """
        experiences = self.sample(batch_size, batch_idxs)

        state0_batch   = []
        shape0_batch   = []
        state_embedded_0_batch = []
        
        state1_batch   = []
        shape1_batch   = []
        state_embedded_1_batch = []
        
        action_batch   = []
        reward_batch   = []
        terminal1_batch = []

        for exp in experiences:
            state0_batch.append(exp.state0)
            shape0_batch.append(exp.shape0)
            state_embedded_0_batch.append(exp.state_embedded_0)
            state1_batch.append(exp.state1)
            shape1_batch.append(exp.shape1)
            state_embedded_1_batch.append(exp.state_embedded_1)
            action_batch.append(exp.action)
            reward_batch.append(exp.reward)
            terminal1_batch.append(exp.terminal1)
        
        # Example, we assume each state0/state1 is a Tensor of shape (C,H,W) or (D,) etc.
        state0_batch = torch.stack(state0_batch).to(DEVICE)
        shape0_batch = torch.stack(shape0_batch).to(DEVICE)
        state_embedded_0_batch = torch.stack(state_embedded_0_batch).to(DEVICE)
        
        state1_batch = torch.stack(state1_batch).to(DEVICE)
        shape1_batch = torch.stack(shape1_batch).to(DEVICE)
        state_embedded_1_batch = torch.stack(state_embedded_1_batch).to(DEVICE)
        action_batch = torch.stack(action_batch).to(DEVICE)

        # For reward, action, terminal, we typically convert from Python scalars
        reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=DEVICE)
        # Terminal can be bool or float. Here we make it bool:
        terminal1_batch = torch.tensor(terminal1_batch, dtype=torch.bool, device=DEVICE)

        return (state0_batch, shape0_batch, state_embedded_0_batch,
                state1_batch, shape1_batch, state_embedded_1_batch,
                action_batch, reward_batch, terminal1_batch)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

    def save_memory_for_world_model(self, directory):
        """Efficiently save memory for world model training."""
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, "sequential_memory.pt.xz")

        data = {
            'state': [self.observations[i] for i in range(len(self.observations))],
            'shape': [self.shapes[i] for i in range(len(self.shapes))],
            'action': [self.actions[i] for i in range(len(self.actions))],
            'terminal': [self.terminals[i] for i in range(len(self.terminals))],
            'num_actions': [self.num_actions[i] for i in range(len(self.num_actions))]
        }

        data = self.process_data_for_world_model(data)
        keys = ['current_state', 'current_shape', 'target_state', 'target_shape', 'action', 'terminal']

        with lzma.open(file_path, "wb") as f:
            torch.save(dict(zip(keys, data)), f)

        print(f"Memory saved (compressed) to {file_path}")

    def process_states(self, state):
        """ Copied from original for completeness """
        current_state = state[:, :, :, 0]
        target_state = state[:, :, :, 1]
        target_state = target_state.reshape(-1, 900)
        return current_state, target_state
    
    def process_shapes(self, shape):
        """ Copied from original for completeness """
        current_shape = shape[:, 0, :]
        target_shape = shape[:, 1, :]
        return current_shape, target_shape
        
    def process_data_for_world_model(self, data):
        """ Copied from original for completeness """
        state = torch.stack(data['state']).to(DEVICE) + 1 # add 1 to get values between 0 and 10
        shape = torch.stack(data['shape']).to(DEVICE) - 1 # subtract 1 to get values between 0 and 29

        # assert all values are between 0 and 10
        assert torch.all(state >= 0) and torch.all(state <= 10), "State values out of bounds"
        assert torch.all(shape >= 0) and torch.all(shape <= 29), "Shape values out of bounds"
        action = torch.stack(data['action']).to(DEVICE)
        terminal = torch.tensor(data['terminal'], dtype=torch.bool).to(DEVICE)
        num_actions = torch.tensor(data['num_actions'], dtype=torch.float).to(DEVICE) # Changed dtype to float

        # Process the states into current and target states
        current_state, target_state = self.process_states(state)
        current_shape, target_shape = self.process_shapes(shape)
        
        # Create next_state and next_shape by shifting the arrays one element ahead.
        # Note: Original code references next_num_actions but doesn't seem to use it
        # beyond an assertion. If needed for filtering, ensure it's handled correctly.
        # Let's calculate it for the assertion:
        next_num_actions = torch.cat((num_actions[1:], torch.tensor([0.0]).to(DEVICE))) # Add dummy value for last element shift

        # Remove the last row from current data since it has no corresponding next state/shape.
        current_state_proc = current_state[:-1]
        current_shape_proc = current_shape[:-1]
        target_state_proc = target_state[:-1]
        target_shape_proc = target_shape[:-1]
        action_proc = action[:-1]
        terminal_proc = terminal[:-1]
        num_actions_proc = num_actions[:-1] # Use num_actions for the *current* step of the transition
        next_num_actions_proc = next_num_actions[:-1] # Use shifted for the *next* step comparison

        # Remove transitions where terminal==True (i.e. where an episode ended)
        valid_mask = ~terminal_proc # Use ~ for boolean tensor negation

        # Assertion logic from original code
        for i in range(current_state_proc.shape[0]):
            num_actions_i = num_actions_proc[i].item()
            next_num_actions_i = next_num_actions_proc[i].item()
            if valid_mask[i].item() == True:
                # Adding a small tolerance for float comparison if necessary
                assert abs(next_num_actions_i - (num_actions_i + 1)) < 1e-6 , \
                       f"num_actions mismatch: {next_num_actions_i} != {num_actions_i + 1} at index {i}"
            else:
                # If terminal is True, the *next* step's num_actions should reset to 1 (or start value)
                 assert abs(next_num_actions_i - 1.0) < 1e-6, \
                       f"Expected next_num_actions to be 1 after terminal, got {next_num_actions_i} at index {i}"


        current_state_final = current_state_proc[valid_mask]
        current_shape_final = current_shape_proc[valid_mask]
        target_state_final = target_state_proc[valid_mask]
        target_shape_final = target_shape_proc[valid_mask]
        action_final = action_proc[valid_mask]
        terminal_final = terminal_proc[valid_mask] # Keep terminal state for potential future use? Original returned it.

        # Ensure tensors are contiguous before converting to numpy if needed, or directly use torch tensors
        # Convert to CPU tensors as requested in the original save function
        current_state_final = current_state_final.contiguous().to('cpu', dtype=torch.float32)
        current_shape_final = current_shape_final.contiguous().to('cpu', dtype=torch.float32)
        target_state_final = target_state_final.contiguous().to('cpu', dtype=torch.float32)
        target_shape_final = target_shape_final.contiguous().to('cpu', dtype=torch.float32)
        action_final = action_final.contiguous().to('cpu', dtype=torch.float32)
        terminal_final = terminal_final.contiguous().to('cpu', dtype=torch.bool)


        return current_state_final, current_shape_final, target_state_final, target_shape_final, action_final, terminal_final
    # --- End of assumed methods ---


    # --- New Function: save_memory_for_world_model_chunk ---
    def save_memory_for_world_model_chunk(self, directory):
        """
        Efficiently save the *entire* current memory content as a numbered chunk
        for world model training. Increments the chunk counter.
        """
        if self.nb_entries == 0:
            print("Warning: Attempted to save an empty memory chunk. Skipping.")
            return
            
        os.makedirs(directory, exist_ok=True)
        # Use self.chunk_count in the filename
        file_path = os.path.join(directory, f"sequential_memory_chunk_{self.chunk_count}.pt.xz")

        # Gather data from the entire buffer *up to current length*
        # Important: Access RingBuffer correctly using indices from 0 to len-1
        current_length = len(self.observations)
        data = {
            'state': [self.observations[i] for i in range(current_length)],
            'shape': [self.shapes[i] for i in range(current_length)],
            'action': [self.actions[i] for i in range(current_length)],
            'terminal': [self.terminals[i] for i in range(current_length)],
            'num_actions': [self.num_actions[i] for i in range(current_length)]
        }
        
        # Check if enough data exists to form at least one transition
        if len(data['state']) < 2:
             print(f"Warning: Not enough data ({len(data['state'])} entries) "
                   f"in memory to process transitions for chunk {self.chunk_count}. Skipping save.")
             # Don't increment chunk_count if nothing is saved
             return

        # Process the gathered data exactly like the original function
        try:
            processed_data = self.process_data_for_world_model(data)
            # Ensure process_data_for_world_model returns the expected 6 tensors
            if len(processed_data) != 6:
                raise ValueError(f"process_data_for_world_model returned {len(processed_data)} items, expected 6.")
                
            # Check if processing resulted in empty tensors (e.g., all transitions were terminal)
            if processed_data[0].shape[0] == 0:
                print(f"Warning: Processing resulted in zero valid transitions for chunk {self.chunk_count}. Skipping save.")
                # Don't increment chunk_count if nothing is saved
                return
                
        except Exception as e:
            print(f"Error processing data for chunk {self.chunk_count}: {e}")
            # Don't increment chunk_count if processing fails
            return

        keys = ['current_state', 'current_shape', 'target_state', 'target_shape', 'action', 'terminal']
        save_dict = dict(zip(keys, processed_data))

        try:
            with lzma.open(file_path, "wb") as f:
                torch.save(save_dict, f)
            
            print(f"Memory chunk {self.chunk_count} saved (compressed) to {file_path} "
                  f"({processed_data[0].shape[0]} transitions)")
            # Increment chunk count *only after successful save*
            self.chunk_count += 1
        except Exception as e:
             print(f"Error saving memory chunk {self.chunk_count} to {file_path}: {e}")


