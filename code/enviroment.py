import numpy as np
from dsl.utilities.padding import pad_grid, unpad_grid
from dsl.utilities.plot import plot_grid, plot_grid_3d, plot_selection
import gymnasium as gym
from gymnasium import spaces
from collections import deque

def extract_states(previous_state, current_state,  target_state):
    """
    Extract the current and target states from the training grids.
    """
    # Remove padding
    previous_state_not_padded = unpad_grid(previous_state)
    current_state_not_padded = unpad_grid(current_state)
    target_state_not_padded = unpad_grid(target_state)

    return previous_state_not_padded, current_state_not_padded, target_state_not_padded

def maximum_overlap_regions(array1, array2):
        """
        Vectorized calculation of maximum overlap between two 2D arrays.
        """
        shape1 = array1.shape
        shape2 = array2.shape
        
        # Calculate possible positions for sliding array2 over array1
        offsets_i = np.arange(-shape2[0] + 1, shape1[0])
        offsets_j = np.arange(-shape2[1] + 1, shape1[1])
        
        # Create grids for all possible offsets
        grid_i, grid_j = np.meshgrid(offsets_i, offsets_j, indexing='ij')
        
        # Calculate the valid overlap regions for each position
        row_start1 = np.maximum(0, grid_i)
        row_end1 = np.minimum(shape1[0], grid_i + shape2[0])
        col_start1 = np.maximum(0, grid_j)
        col_end1 = np.minimum(shape1[1], grid_j + shape2[1])
        
        row_start2 = np.maximum(0, -grid_i)
        row_end2 = row_start2 + (row_end1 - row_start1)
        col_start2 = np.maximum(0, -grid_j)
        col_end2 = col_start2 + (col_end1 - col_start1)
        
        # Calculate overlap scores for all positions
        max_overlap_score = 0
        best_overlap1 = None
        best_overlap2 = None
        
        for idx in np.ndindex(grid_i.shape):
            r1s, r1e = row_start1[idx], row_end1[idx]
            c1s, c1e = col_start1[idx], col_end1[idx]
            r2s, r2e = row_start2[idx], row_end2[idx]
            c2s, c2e = col_start2[idx], col_end2[idx]
            
            region1 = array1[r1s:r1e, c1s:c1e]
            region2 = array2[r2s:r2e, c2s:c2e]
            
            overlap_score = np.sum(region1 == region2)
            
            if overlap_score > max_overlap_score:
                max_overlap_score = overlap_score
                best_overlap1 = (slice(r1s, r1e), slice(c1s, c1e))
                best_overlap2 = (slice(r2s, r2e), slice(c2s, c2e))
        
        return best_overlap1, best_overlap2

class ARC_Env(gym.Env):
    def __init__(self, challenge_dictionary, action_space, dim=30, seed=None):
        super(ARC_Env, self).__init__()
        
        # Set the seed
        if seed:
            np.random.seed(seed)

        self.challenge_dictionary = challenge_dictionary # dictionary of challenges
        self.dictionary_keys = list(challenge_dictionary.keys()) # list of keys in the dictionary
        self.num_challenges = len(challenge_dictionary) # number of challenges in the dictionary
        self.dim = dim # maximum dimension of the problem
        self.observation_shape = (1, dim, 2*dim) # shape of the grid

        # reward variables
        self.step_penalty = 1
        self.maximum_similarity = 50
        self.completed_challenge_reward = 25

        # Define the action space: a sequence of 9 integers
        self.action_space = action_space
        
        # Define the observation space: a 60x30 image with pixel values between 0 and 255
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)

        # Define the state of the environment
        self.new_states = None
        self.infos = None
        
        self.state = None
        self.done = False
        self.info = None

    def get_random_challenge(self):
        """
        Get a random challenge from the challenge dictionary. 
        """

        challenge_key = np.random.choice(self.dictionary_keys) 
        challenge = self.challenge_dictionary[challenge_key]
        challenge_train = challenge['train']
        challenge_length = len(challenge_train)
        random_index = np.random.randint(0, challenge_length-1)
        random_challenge = challenge_train[random_index]
        random_input = np.array(random_challenge['input'])
        random_output = np.array(random_challenge['output'])
        
        # Pad the grids
        training_grid = np.zeros(self.observation_shape, dtype=np.int32)
        training_grid[:, :, :self.dim] = pad_grid(random_input)
        training_grid[:, :, self.dim:] = pad_grid(random_output)
        return training_grid, challenge_key
    
    def reset(self):
        # Reset the enviroment variables
        self.new_states = deque()
        self.infos = deque()
        self.done = False

        # Get a new challenge
        new_state, new_key = self.get_random_challenge()
        info = {'key': new_key, 'actions': [], 'action_strings':[], 'num_actions': 0, 'solved': False}
        
        # Update the state of the environment
        self.new_states.append(new_state)
        self.infos.append(info)
        self.state = self.new_states.popleft()
        self.info = self.infos.popleft()
        return self.state
    
    def shape_reward(self, previous_state_unpadded, current_state_unpadded, target_state):
        """
        Reward the agent based on the shape of the grid.
        Outputs:
        - reward: the reward for the agent based on the shape of the grid
        - shapes_agree: a boolean indicating if the shapes of the 3 grids agree
        """
        raise DeprecationWarning("This method is deprecated. Use best_overlap_reward instead.")
        current_shape0 = np.shape(previous_state_unpadded)
        current_shape1 = np.shape(current_state_unpadded)
        target_shape = np.shape(target_state)
        
        # If the transformation does not change the shape of the grid
        if np.any(current_shape0 == current_shape1):
            # If the shape of the grid is different from the target shape: small penalty
            # The agent did not change the shape of the grid, but it did not mistakely change it either
            if np.any(current_shape1 != target_shape): 
                return -1, False
            
            # If the shape of the grid is the same as the target shape: no reward
            # The agent did not change the shape of the grid, and the shape is correct
            else:
                return 0, True
        
        # If the transformation changes the shape of the grid
        if np.any(current_shape0 != current_shape1):
            # If the shape of the starting grid is the same as the target shape: big penalty
            # The agent changed a correct shape to an incorrect shape
            if np.all(current_shape0 == target_shape):
                return -5, False
        
            # If the shape of the grid is different from the target shape: medium penalty
            # The agent changed the shape of the grid, and it is not the correct shape
            if np.any(current_shape1 != target_shape):
                return -2, False
            
            # If the shape of the grid is the same as the target shape: big reward
            # The agent changed the shape of the grid, and it is the correct shape
            else:
                return 5, False
          
    def similarity_reward(self, previous_state_unpadded, current_state_unpadded, target_state):
        """
        Reward the agent based on the similarity of the grid.
        """ 
        raise DeprecationWarning("This method is deprecated. Use best_overlap_reward instead.")
        size = np.size(target_state)
        # If the shapes agree, we compute the similarity reward
        similarity_0 = np.sum(previous_state_unpadded == target_state) / size
        similarity_1 = np.sum(current_state_unpadded == target_state) / size

        similarity_difference = similarity_1 - similarity_0
        return self.maximum_similarity * similarity_difference

    def total_reward(self, previous_state_unpadded, current_state_unpadded, target_state):
        """
        Compute the total reward for the agent.
        """
        raise DeprecationWarning("This method is deprecated. Use best_overlap_reward instead.")

        shape_reward, shapes_agree = self.shape_reward(previous_state_unpadded, current_state_unpadded, target_state)
        if not shapes_agree: # If the shapes do not agree, we return the shape reward
            return shape_reward
        
        # If the shapes agree, we compute the similarity reward
        similarity_reward = self.similarity_reward(previous_state_unpadded, current_state_unpadded, target_state)
        
        # If the agent has completed the challenge, we return the completed challenge reward
        if np.all(current_state_unpadded == target_state):
            return shape_reward + similarity_reward + self.completed_challenge_reward, True
        
        # If the agent has not completed the challenge, we return the sum of the shape and similarity rewards
        return shape_reward + similarity_reward + self.step_penalty, False
    
    
    def best_overlap_reward(self, previous_state_unpadded, current_state_unpadded, target_state_unpadded):
        """
        Reward the agent based on the best overlap between the current state and the target state.
        """
        num_cells_target_state = target_state_unpadded.size
        
        
        # Calculate the overlap between the previous state and the target state
        best_overlap_previous, best_overlap_target_with_previous = maximum_overlap_regions(previous_state_unpadded, target_state_unpadded)
        previous_score = 0
        if np.any(best_overlap_previous) and np.any(best_overlap_target_with_previous):
            previous_score = np.sum(previous_state_unpadded[best_overlap_previous] == target_state_unpadded[best_overlap_target_with_previous])
        previous_score = previous_score / num_cells_target_state

        # Calculate the overlap between the current state and the target state
        best_overlap_current, best_overlap_target_with_current = maximum_overlap_regions(current_state_unpadded, target_state_unpadded)
        current_score = 0
        if np.any(best_overlap_current) and np.any(best_overlap_target_with_current):
            current_score = np.sum(current_state_unpadded[best_overlap_current] == target_state_unpadded[best_overlap_target_with_current])
        current_score = current_score / num_cells_target_state

        step_penalty = self.step_penalty
        curent_shape = current_state_unpadded.shape
        target_shape = target_state_unpadded.shape
        
        if curent_shape != target_shape:
            step_penalty += 1
        else:
            if np.all(current_state_unpadded == target_state_unpadded):
                return self.completed_challenge_reward, True


        similarity_reward = (current_score - previous_score) * self.maximum_similarity
        reward = similarity_reward - step_penalty 
        return reward, False

    
    def act(self, previous_state, action):
        """
        Apply the action to the previous state.
        """
        # Extract the color selection, selection, and transformation keys
        color_selection_key = int(action[0])
        selection_key = int(action[1])
        transformation_key = int(action[2])

        # Extract the color selection, selection, and transformation
        color_selection = self.action_space.color_selection_dict[color_selection_key]
        selection = self.action_space.selection_dict[selection_key]
        transformation = self.action_space.transformation_dict[transformation_key]

        # Apply the color selection, selection, and transformation to the previous state
        color = color_selection(grid = previous_state)
        selected = selection(grid = previous_state, color = color)
        if np.any(selected):
            transformed = transformation(grid = previous_state, selection = selected)
        else:
            transformed = np.expand_dims(previous_state, axis=0)
        return transformed

    def step(self, action, max_states_per_action=3):
        
        # Update the info dictionary
        info = self.info
        info['actions'].append(action)
        action_string = self.action_space.action_to_string(action)
        info['action_strings'].append(action_string)
        info['num_actions'] += 1
        
        # Extract the previous and target states
        previous_state = self.state[:, :, :self.dim] # make the current state into the previous state
        previous_state_not_padded = unpad_grid(previous_state) # remove the padding from the previous state
        
        # Extract the current and target states
        target_state = self.state[:, :, self.dim:] # get the target state
        target_state_not_padded = unpad_grid(target_state) # remove the padding from the target state

        # Apply the action to the previous state
        current_state_tensor = self.act(previous_state_not_padded, action) # apply the action to the previous state

        # Initialize the rewards, dones, and current states tensors to store the results of the step function
        rewards = np.zeros(current_state_tensor.shape[0]) # initialize the rewards
        current_states = np.zeros((current_state_tensor.shape[0], self.dim, self.dim*2)) # initialize the current states
        solveds = np.zeros(current_state_tensor.shape[0], dtype=bool) # initialize the successes

        # Loop over the first dimension of the tensor
        for i in range(current_state_tensor.shape[0]):
            current_state_not_padded = current_state_tensor[i, :, :] # get the current state
            reward, solved = self.best_overlap_reward(previous_state_not_padded, current_state_not_padded, target_state_not_padded) # compute the reward
            current_state_padded = pad_grid(current_state_not_padded) # pad the current state
            
            # Store the results
            rewards[i] = reward # store the reward
            solveds[i] = solved
            current_states[i, :, :self.dim] = current_state_padded
            current_states[i, :, self.dim:] = target_state
        
        num_states_to_evaluate = min(max_states_per_action, current_state_tensor.shape[0])
        top_n_indices = np.argsort(rewards)[-num_states_to_evaluate:] # get the top n indices
        reward = np.max(rewards[top_n_indices]) # get the maximum reward

        # if the agent has completed the challenge, we update the info dictionary
        done = np.any(solveds)
        if done:
            index_of_solved = np.where(solveds)[0][0]
            self.state = current_states[index_of_solved, :, :]
            info['solved'] = True
            return self.state, reward, done, info
            
        # Add the top n states to the new states and infos
        for i in top_n_indices:
            state_to_append = current_states[i, :, :]
            state_to_append = np.expand_dims(state_to_append, axis=0)
            self.new_states.append(state_to_append)
            self.infos.append(info)
        
        # End the episode with a small probability if not ended already
        if not done and np.random.random() < 0.005:
            done = True

        # Update the state of the environment
        self.state = self.new_states.popleft()
        self.info = self.infos.popleft()

        return self.state, reward, done, self.info
