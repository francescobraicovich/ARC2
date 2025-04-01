from __future__ import absolute_import
import warnings
import random
from collections import deque, namedtuple
import numpy as np
import torch
import os
import lzma
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, TensorStorage, LazyTensorStorage
import torchrl
from utils.util import set_device # Assuming you still need this for device selection

# If you have a custom 'to_tensor' function, you might still need it
# if your inputs to 'append' aren't already tensors.
# from utils.util import to_tensor

# ---------------------------------------------------------------
# Device selection: CUDA -> MPS -> CPU
# ---------------------------------------------------------------
# Use a function scope for device determination if preferred
# Or keep it global if set_device() is designed for that
# DEVICE = set_device()
# print("Using device for memory:", DEVICE)

# Removed Experience namedtuple - TensorDict replaces it

# Removed sample_batch_indexes - TorchRL sampler handles it

# Removed RingBuffer - TorchRL storage handles it

# Removed zeroed_observation - Not needed with window_length=1 and TorchRL

# Removed Base Memory class

# ---------------------------------------------------------------
# TorchRL Sequential Memory Implementation
# - uses ReplayBuffer with TensorStorage
# - maintains the required external API
# ---------------------------------------------------------------
class SequentialMemory:
    def __init__(self, limit, device=None):
        """
        Initializes the memory buffer using TorchRL's ReplayBuffer.

        Args:
            limit (int): The maximum number of transitions to store.
            device (str or torch.device, optional): The device to store tensors on.
                                                  Defaults to auto-detection (CUDA > MPS > CPU).
        """
        if device is None:
            self.device = set_device()
        else:
            self.device = torch.device(device)
        print(f"Using device for TorchRLMemory: {self.device}")

        self.limit = int(limit) # Ensure limit is an integer

        # Using LazyTensorStorage initially allows flexibility with input shapes/dtypes.
        # It will be converted to TensorStorage upon first insertion.
        # Specify `device` during storage creation.
        self.storage = LazyTensorStorage(max_size=self.limit, device=self.device)

        # Use the default RandomSampler and RoundRobinWriter
        self.buffer = ReplayBuffer(
            storage=self.storage,
            sampler=torchrl.data.RandomSampler(), # Default sampler
            writer=torchrl.data.RoundRobinWriter(), # Default writer ensures sequential overwrite
            collate_fn=lambda x: x, # Pass-through collation, sample() handles batching
            pin_memory=False, # Set based on your device and needs
            prefetch=None,
        )
        # We need to track the 'done' status of the *previous* step
        # to replicate the original sampling constraint (avoid transitions *starting*
        # at the first step of a new episode).
        self._last_done_status = True # Assume start is like end of a previous episode

    @property
    def nb_entries(self):
        """Returns the current number of entries in the buffer."""
        # buffer.__len__() gives the number of *valid* samples available,
        # respecting buffer state (e.g., writer position).
        return len(self.buffer)

    def append(self, observation, embedded_observation, shape, action, reward, terminal, num_actions, training=True):
        """
        Add a new transition to memory.

        Args:
            observation: The state observation.
            embedded_observation: The embedded state observation.
            shape: Shape information related to the observation.
            action: The action taken.
            reward: The reward received.
            terminal (bool): Whether the *next* state is terminal.
            num_actions: Custom counter.
            training (bool): Flag indicating if adding during training (always True for buffer).
        """
        if not training:
             # Original code had this flag, but didn't seem to use it to skip appending.
             # If skipping is desired, uncomment the following line:
             # return
             pass # Always append if called during training phase logic

        # --- Convert inputs to Tensors ---
        # We assume inputs might be numpy arrays or lists and need conversion.
        # If they are already tensors on the correct device, this is less critical
        # but ensures consistency.
        # Use torch.as_tensor to avoid copying if already a tensor.
        obs_tensor = torch.as_tensor(observation, device=self.device)
        emb_obs_tensor = torch.as_tensor(embedded_observation, device=self.device)
        shape_tensor = torch.as_tensor(shape, device=self.device)
        action_tensor = torch.as_tensor(action, device=self.device) # Ensure correct dtype if needed (e.g., long for discrete)
        reward_tensor = torch.as_tensor([reward], dtype=torch.float32, device=self.device) # Wrap scalar in list/tensor
        terminal_tensor = torch.as_tensor([terminal], dtype=torch.bool, device=self.device) # Wrap scalar in list/tensor
        num_actions_tensor = torch.as_tensor([num_actions], dtype=torch.int64, device=self.device) # Assuming int counter
        prev_done_tensor = torch.as_tensor([self._last_done_status], dtype=torch.bool, device=self.device)

        # --- Create TensorDict for the step ---
        # TorchRL expects data structured with current step keys and 'next' keys
        # for the subsequent step's data.
        # We add the data for the *current* transition. TorchRL's writer and buffer
        # logic handle linking s_t with s_{t+1} internally based on sequential adds.
        # We also add 'prev_done' to help with sampling logic later.
        step_data = TensorDict({
            "observation": obs_tensor,
            "state_embedded": emb_obs_tensor,
            "shape": shape_tensor,
            "action": action_tensor,
            "reward": reward_tensor,
            "done": terminal_tensor, # 'done' marks the end of the *current* transition
            "num_actions": num_actions_tensor,
            "prev_done": prev_done_tensor, # Record if the *previous* state was terminal
            # Define structure for 'next' keys. TorchRL will populate these automatically
            # when the *next* step is added, using the data from that next step.
            # Provide dummy/repeated values initially; they get overwritten.
            ("next", "observation"): obs_tensor,
            ("next", "state_embedded"): emb_obs_tensor,
            ("next", "shape"): shape_tensor,
            ("next", "done"): terminal_tensor,
            ("next", "num_actions"): num_actions_tensor,
            # Note: 'reward' and 'action' typically aren't needed under 'next'.
            # 'prev_done' is also not needed under 'next'.
        }, batch_size=[]) # Indicate single step data (batch size is empty)

        self.buffer.add(step_data)

        # Update the last done status for the *next* append call
        self._last_done_status = terminal # terminal here refers to terminal1 in original logic

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the buffer.

        This reimplements the original sampling logic:
        - It avoids sampling transitions where the *starting* state (`state0`)
          was the first state after an episode termination.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            TensorDict: A TensorDict containing the batch of experiences.
        """
        if len(self.buffer) < batch_size:
             # Original code checked for >= 2 entries total. TorchRL's len checks
             # usable entries. A common check is if len >= batch_size.
             # You might need a larger minimum length depending on your warmup phase.
             raise ValueError(f"Not enough entries in memory to sample a batch. "
                              f"Need >= {batch_size}, have {len(self.buffer)} usable entries.")

        valid_samples = []
        collected_count = 0

        # Keep sampling until we have enough *valid* samples
        # This loop replaces the `while terminal0:` resampling in the original code.
        # It's less efficient than TorchRL's direct sampling if filtering wasn't needed,
        # but necessary to match the specified constraint.
        attempts = 0
        max_attempts = batch_size * 10 # Safety break to prevent infinite loops

        while collected_count < batch_size and attempts < max_attempts:
            # Sample slightly more than needed to improve chances of getting valid ones
            sample_size = max(1, int(1.2 * (batch_size - collected_count))) # Sample at least 1
            try:
                raw_batch = self.buffer.sample(sample_size)
            except RuntimeError as e:
                 # Can happen if buffer is modified during sampling or becomes too small
                 print(f"Warning: Error during buffer sampling: {e}. Trying again.")
                 attempts += sample_size
                 continue


            # Identify transitions where the *previous* step was terminal ('prev_done' is True).
            # These are the transitions we want to *exclude*.
            is_valid = ~raw_batch["prev_done"].squeeze(-1) # Squeeze potential trailing dim if added

            valid_batch = raw_batch[is_valid]
            num_valid_in_batch = valid_batch.shape[0]

            if num_valid_in_batch > 0:
                # Add the valid samples, ensuring not to exceed batch_size
                num_to_add = min(num_valid_in_batch, batch_size - collected_count)
                valid_samples.append(valid_batch[:num_to_add])
                collected_count += num_to_add

            attempts += sample_size
            # Add a small sleep if running into contention issues, though unlikely here
            # import time; time.sleep(0.001)


        if collected_count < batch_size:
            warnings.warn(f"Could only collect {collected_count} valid samples after {attempts} attempts "
                          f"(requested {batch_size}). Buffer size: {len(self.buffer)}. "
                          "There might be too many terminal states close together or buffer size is too small.")
            if not valid_samples:
                 raise RuntimeError(f"Failed to sample any valid transitions after {attempts} attempts.")

        # Concatenate the collected valid samples
        batch = torch.cat(valid_samples, dim=0)

        # Ensure final batch size is correct (it might be less if warning occurred)
        if batch.shape[0] != batch_size and collected_count == batch_size:
            # This case should ideally not happen if logic is correct
             raise RuntimeError(f"Internal error: Collected count matches batch size, but final batch shape {batch.shape[0]} is wrong.")
        elif batch.shape[0] > batch_size:
            # Trim if we somehow collected slightly too many (due to sample_size estimation)
            batch = batch[:batch_size]


        # We don't need the 'prev_done' key in the final output
        if "prev_done" in batch.keys():
             batch = batch.exclude("prev_done")

        return batch


    def sample_and_split(self, batch_size):
        """
        Samples a batch from memory and returns separated tensors on the correct device.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            tuple: A tuple containing tensors for (state0, shape0, state_embedded_0,
                   state1, shape1, state_embedded_1, action, reward, terminal1).
                   Tensors are on the device specified during initialization.
        """
        # Use the custom sample method that handles the 'prev_done' filtering
        batch_td = self.sample(batch_size)

        # Extract tensors from the TensorDict. They are already on self.device.
        state0_batch = batch_td["observation"]
        shape0_batch = batch_td["shape"]
        state_embedded_0_batch = batch_td["state_embedded"]
        action_batch = batch_td["action"]
        reward_batch = batch_td["reward"].squeeze(-1) # Remove trailing dim often added for scalars
        # Access next state information using the ("next", "key") convention
        state1_batch = batch_td[("next", "observation")]
        shape1_batch = batch_td[("next", "shape")]
        state_embedded_1_batch = batch_td[("next", "state_embedded")]
        terminal1_batch = batch_td[("next", "done")].squeeze(-1) # 'done' refers to the state *after* the action


        # --- Assertions (Optional but Recommended) ---
        # Verify shapes and types if needed, e.g.:
        assert state0_batch.shape[0] == batch_size
        assert shape0_batch.shape[0] == batch_size
        assert state_embedded_0_batch.shape[0] == batch_size
        assert action_batch.shape[0] == batch_size
        assert reward_batch.shape[0] == batch_size
        assert state1_batch.shape[0] == batch_size
        assert shape1_batch.shape[0] == batch_size
        assert state_embedded_1_batch.shape[0] == batch_size
        assert terminal1_batch.shape[0] == batch_size

        assert reward_batch.dtype == torch.float32
        assert terminal1_batch.dtype == torch.bool

        return (
            state0_batch,
            shape0_batch,
            state_embedded_0_batch,
            state1_batch,
            shape1_batch,
            state_embedded_1_batch,
            action_batch,
            reward_batch,
            terminal1_batch,
        )

    def get_config(self):
        """Returns configuration information."""
        # Basic config matching the original structure
        return {'limit': self.limit, 'device': str(self.device)}

    # --- World Model Saving Logic ---
    # Keep the helper processing functions internal or move to utils if preferred

    def _process_states(self, state_batch):
        """Helper to process state batch for world model format."""
        # Assuming state_batch has shape [N, C, H, W] or similar from buffer
        # Adapt the slicing/reshaping based on your actual state structure stored
        # Example based on original: state was (C, H, W, 2) ? -> N, C, H, W, 2
        if state_batch.ndim == 5 and state_batch.shape[-1] == 2: # N, C, H, W, 2
             current_state = state_batch[..., 0] # N, C, H, W
             target_state_flat = state_batch[..., 1] # N, C, H, W
             # Example reshape based on original code's 900 target size
             # You MUST adapt this to your specific state dimensions
             if target_state_flat.numel() > 0 :
                 target_state = target_state_flat.reshape(target_state_flat.shape[0], -1) # N, (C*H*W)
                 # Original code added 1? Ensure this logic is correct for your model
                 target_state = target_state #+ 1 # Example: removed +1 unless needed
             else:
                 target_state = torch.empty((0, 900), device=state_batch.device, dtype=state_batch.dtype) # Handle empty case
        else:
            # Fallback/error if shape is unexpected
            raise ValueError(f"Unexpected state shape for processing: {state_batch.shape}. "
                             "Expected something like (N, ..., 2) based on original code.")

        return current_state, target_state

    def _process_shapes(self, shape_batch):
        """Helper to process shape batch for world model format."""
        # Assuming shape_batch has shape [N, 2, D] based on original
        if shape_batch.ndim == 3 and shape_batch.shape[1] == 2: # N, 2, D
            current_shape = shape_batch[:, 0, :] # N, D
            target_shape = shape_batch[:, 1, :] # N, D
        else:
             # Fallback/error if shape is unexpected
             raise ValueError(f"Unexpected shape format for processing: {shape_batch.shape}. "
                              "Expected (N, 2, D) based on original code.")
        # Original code had a loop that seemingly did nothing:
        # for i in range(current_shape.shape[0]):
        #     current_shape[i] = current_shape[i]
        #     target_shape[i] = target_shape[i]
        # This is likely unnecessary.
        return current_shape, target_shape

    def save_memory_for_world_model(self, directory):
        """
        Extracts, processes, and saves all valid transitions from the buffer
        in a compressed format suitable for world model training.

        Filters out transitions where the episode ended (terminal=True).
        Performs assertions on num_actions continuity for valid transitions.
        """
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, "torchrl_memory.pt.xz")

        if len(self.buffer) == 0:
            print("Memory buffer is empty. Nothing to save.")
            # Optionally save an empty file or structure
            with lzma.open(file_path, "wb") as f:
                 torch.save({}, f)
            return

        # --- Extract all data ---
        # Use storage directly to get the full sequence including incomplete final step if any
        # Note: This loads *everything* into memory. For extremely large buffers,
        # consider iterating or using buffer.sample(len(buffer)) if sufficient.
        try:
            # Access internal storage. This is less ideal than a public API
            # but sometimes necessary for full sequence access.
            # The exact structure might depend on TorchRL version.
            # Let's try sampling all available data first, which is safer API-wise.
            all_data = self.buffer.sample(len(self.buffer), return_info=False) # Sample valid transitions
            print(f"Extracted {len(all_data)} transitions for saving.")
            if len(all_data) == 0:
                 print("No valid transitions could be sampled. Saving empty file.")
                 with lzma.open(file_path, "wb") as f:
                    torch.save({}, f)
                 return

        except Exception as e:
             print(f"Error sampling all data from buffer: {e}. Trying storage access (less safe).")
             # Fallback to storage access (use with caution)
             # Make sure storage isn't Lazy anymore
             if isinstance(self.buffer.storage, LazyTensorStorage):
                 raise RuntimeError("Cannot directly access LazyTensorStorage. Ensure buffer has data.")
             # This might include data beyond the writer's current valid cycle
             all_data = self.buffer.storage[:] # Potentially includes invalid/overwritten data

        # --- Extract relevant tensors ---
        # Use .get() with default=None for safety in case keys are missing
        state = all_data.get("observation")
        next_state = all_data.get(("next", "observation"))
        shape = all_data.get("shape")
        next_shape = all_data.get(("next", "shape"))
        action = all_data.get("action")
        terminal = all_data.get("done") # This is terminal1 in original terms (end of current step)
        num_actions = all_data.get("num_actions")
        next_num_actions = all_data.get(("next", "num_actions"))

        # --- Basic validation ---
        required_keys = [state, next_state, shape, next_shape, action, terminal, num_actions, next_num_actions]
        if any(k is None for k in required_keys):
            missing = [name for name, k in zip(
                ["state", "next_state", "shape", "next_shape", "action", "terminal", "num_actions", "next_num_actions"],
                required_keys) if k is None]
            raise KeyError(f"Could not extract required keys from buffer data for saving: {missing}")

        # Squeeze scalar dimensions if present
        terminal = terminal.squeeze(-1)
        num_actions = num_actions.squeeze(-1)
        next_num_actions = next_num_actions.squeeze(-1)


        # --- Assertions on num_actions (similar to original) ---
        # Check continuity only for non-terminal transitions
        non_terminal_mask = ~terminal

        if torch.any(non_terminal_mask):
            num_actions_t_valid = num_actions[non_terminal_mask]
            next_num_actions_t_valid = next_num_actions[non_terminal_mask]
            try:
                 # Using torch.testing.assert_close for robust comparison
                 torch.testing.assert_close(
                    next_num_actions_t_valid,
                    num_actions_t_valid + 1,
                    msg="num_actions mismatch for non-terminal steps"
                )
            except AssertionError as e:
                print(f"Assertion Warning: {e}") # Log warning instead of crashing
                # Find indices of mismatch
                mismatch_indices = torch.where(next_num_actions_t_valid != num_actions_t_valid + 1)[0]
                print(f"First few mismatches (expected next = current + 1):")
                for i in mismatch_indices[:5]:
                     print(f"  Index (in non-terminal): {i}, Current: {num_actions_t_valid[i]}, Next: {next_num_actions_t_valid[i]}")


        # Check reset condition for terminal transitions
        terminal_mask = terminal
        if torch.any(terminal_mask):
            next_num_actions_term = next_num_actions[terminal_mask]
            try:
                # Assuming counter resets to 1 after termination, as in original assert
                torch.testing.assert_close(
                    next_num_actions_term,
                    torch.ones_like(next_num_actions_term),
                    msg="num_actions did not reset to 1 after terminal step"
                )
            except AssertionError as e:
                print(f"Assertion Warning: {e}") # Log warning
                mismatch_indices = torch.where(next_num_actions_term != 1)[0]
                print(f"First few mismatches (expected next = 1):")
                for i in mismatch_indices[:5]:
                     print(f"  Index (in terminal): {i}, Next Num Actions: {next_num_actions_term[i]}")


        # --- Filter out transitions where terminal == True ---
        # We keep transitions *leading* to a non-terminal state.
        valid_mask = ~terminal # same as non_terminal_mask
        if not torch.any(valid_mask):
             print("No valid (non-terminal) transitions found to save.")
             with lzma.open(file_path, "wb") as f:
                 torch.save({}, f)
             return

        state_valid = state[valid_mask]
        next_state_valid = next_state[valid_mask] # target state is the next state for valid transitions
        shape_valid = shape[valid_mask]
        next_shape_valid = next_shape[valid_mask] # target shape
        action_valid = action[valid_mask]
        terminal_valid = terminal[valid_mask] # Should be all False now

        # --- Process states and shapes for the valid transitions ---
        # Apply the original processing logic
        # Need to handle potential dimension changes from _process_states/_process_shapes
        current_state, target_state = self._process_states(state_valid, next_state_valid) # Pass both for clarity
        current_shape, target_shape = self._process_shapes(shape_valid, next_shape_valid) # Pass both for clarity


        # --- Final Type Conversion (as in original) ---
        # Convert to float32 then long() - check if this sequence is intended
        # It's unusual to convert float to long directly unless for specific embedding lookups.
        # Casting directly to long might be more appropriate if they represent indices.
        # Assuming direct cast to long is needed:
        current_state = current_state.long()
        current_shape = current_shape.long()
        target_state = target_state.long()
        target_shape = target_shape.long()
        action_valid = action_valid.long()
        terminal_valid = terminal_valid.long() # Cast boolean False to 0

        # --- Prepare data for saving ---
        keys = ['current_state', 'current_shape', 'target_state', 'target_shape', 'action', 'terminal']
        data_to_save = dict(zip(keys, [
            current_state.cpu(), # Move to CPU before saving
            current_shape.cpu(),
            target_state.cpu(),
            target_shape.cpu(),
            action_valid.cpu(),
            terminal_valid.cpu()
        ]))

        # --- Save compressed data ---
        with lzma.open(file_path, "wb") as f:
            torch.save(data_to_save, f)

        print(f"Memory saved ({len(current_state)} valid transitions) compressed to {file_path}")


    # Adjusted processing functions to accept both current and next items explicitly
    def _process_states(self, current_state_batch, next_state_batch):
        """Helper to process state batch for world model format."""
        # Adapt based on your actual state structure and processing needs.
        # This version assumes 'current_state_batch' is S_t and 'next_state_batch' is S_{t+1}
        # Example: If original `state` was (..., 2) containing both:
        # current_state = current_state_batch # Assuming this is already S_t
        # target_state = next_state_batch # Assuming this is already S_{t+1}

        # Replicating the *specific* reshaping and +1 from original code if needed:
        # This part is highly dependent on what your world model expects.
        # If state comes directly from buffer as S_t, then:
        current_state = current_state_batch # Use as is
        target_state_flat = next_state_batch # Use the actual next state

        if target_state_flat.numel() > 0:
             # Example reshape based on original code's 900 target size
             # Adapt C, H, W calculation if needed
             target_state = target_state_flat.reshape(target_state_flat.shape[0], -1)
             # target_state = target_state + 1 # Apply offset only if model requires it
        else:
             num_features = 900 # Replace with actual expected feature count
             target_state = torch.empty((0, num_features), device=target_state_flat.device, dtype=target_state_flat.dtype)

        # Make sure current_state also matches expected format if model needs preprocessing
        # e.g., current_state = current_state.reshape(current_state.shape[0], -1)

        return current_state, target_state

    def _process_shapes(self, current_shape_batch, next_shape_batch):
        """Helper to process shape batch for world model format."""
        # Assuming current_shape_batch is Shape_t and next_shape_batch is Shape_{t+1}
        current_shape = current_shape_batch
        target_shape = next_shape_batch
        # The loop in the original did nothing, so it's omitted here.
        return current_shape, target_shape

