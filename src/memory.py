from __future__ import absolute_import
import warnings
import random
from collections import deque, namedtuple
import numpy as np
import torch
import os, pickle

from utils.util import set_device

# If you have a custom 'to_tensor' function, you can import it. 
# (Code below does not strictly require it.)
# from utils.util import to_tensor

# ---------------------------------------------------------------
# Device selection: CUDA -> MPS -> CPU
# ---------------------------------------------------------------
DEVICE = set_device()
print("Using device for memory:", DEVICE)

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

    @property
    def nb_entries(self):
        return len(self.observations)

    def append(self, observation, embedded_observation, shape, action, reward, terminal, training=True):
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

            while terminal0:
                # Resample a new index
                idx = sample_batch_indexes(1, self.nb_entries, 1)[0]
                terminal0 = self.terminals[idx - 1] if (idx >= 1) else False

            # Now gather transition info
            state0   = self.observations[idx - 1]
            shape0   = self.shapes[idx - 1]
            x_t0 = self.embedded_observations[idx - 1]
            action   = self.actions[idx - 1]
            reward   = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            state1   = self.observations[idx]
            shape1   = self.shapes[idx]
            x_t1 = self.embedded_observations[idx]

            exp = Experience(state0=state0,  state_embedded_0=x_t0, action=action, reward=reward,
                             state1=state1, state_embedded_1=x_t1, terminal1=terminal1,
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
        action_batch   = []
        reward_batch   = []
        state1_batch   = []
        shape1_batch   = []
        state_embedded_1_batch = []
        terminal1_batch = []

        for exp in experiences:
            state0_batch.append(exp.state0)
            shape0_batch.append(exp.shape0)
            state_embedded_0_batch.append(exp.state_embedded_0)
            action_batch.append(exp.action)
            reward_batch.append(exp.reward)
            state1_batch.append(exp.state1)
            shape1_batch.append(exp.shape1)
            state_embedded_1_batch.append(exp.state_embedded_1)
            # Convert terminal to 0/1 or bool as you prefer
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

        return (state0_batch, shape0_batch,
                action_batch, reward_batch,
                state1_batch, shape1_batch,
                terminal1_batch)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

    def save_memory(self, directory):
        """Save the memory to a specified directory."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, "sequential_memory.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Memory saved to {file_path}")

    @classmethod
    def load_memory(cls, directory):
        """Load the memory from a specified directory."""
        import os, pickle
        file_path = os.path.join(directory, "sequential_memory.pkl")
        with open(file_path, "rb") as f:
            memory = pickle.load(f)
        print(f"Memory loaded from {file_path}")
        return memory
