import numpy as np
import random
from dsl.utilities.padding import pad_grid, unpad_grid
import gymnasium as gym
from gymnasium import spaces

def extract_states(previous_state, current_state,  target_state):
    """
    Extract the current and target states from the training grids.
    """
    # Remove padding
    previous_state_not_padded = unpad_grid(previous_state)
    current_state_not_padded = unpad_grid(current_state)
    target_state_not_padded = unpad_grid(target_state)

    return previous_state_not_padded, current_state_not_padded, target_state_not_padded

class ARC_Env(gym.Env):
    def __init__(self, challenge_dictionary, dim=30):
        super(ARC_Env, self).__init__()
        
        self.challenge_dictionary = challenge_dictionary # dictionary of challenges
        self.dictionary_keys = list(challenge_dictionary.keys()) # list of keys in the dictionary
        self.num_challenges = len(challenge_dictionary) # number of challenges in the dictionary
        self.dim = dim # maximum dimension of the problem
        self.observation_shape = (1, dim, 2*dim) # shape of the grid

        # reward variables
        self.step_penalty = -1
        self.maximum_similarity = 50
        self.completed_challenge_reward = 25

        # Define the action space: a sequence of 9 integers
        self.action_space = spaces.Box(low=0, high=255, shape=(9,), dtype=np.int32)
        
        # Define the observation space: a 60x30 image with pixel values between 0 and 255
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)

        # Define the state of the environment
        self.state = np.zeros(self.observation_shape, dtype=np.uint8)
        self.done = False

    def get_random_challenge(self):
        """
        Get a random challenge from the challenge dictionary. 
        """
        challenge_key = random.choice(self.dictionary_keys) 
        challenge = self.challenge_dictionary[challenge_key]
        challenge_train = challenge['train']
        challenge_length = len(challenge_train)
        random_index = random.randint(0, challenge_length-1)
        random_challenge = challenge_train[random_index]
        random_input = np.array(random_challenge['input'])
        random_output = np.array(random_challenge['output'])
        
        # Pad the grids
        training_grid = np.zeros(self.observation_shape, dtype=np.uint8)
        training_grid[:, :, :self.dim] = pad_grid(random_input)
        training_grid[:, :, self.dim:] = pad_grid(random_output)
        return training_grid
    
    def reset(self):
        self.state = self.get_random_challenge()
        self.done = False
        return self.state
    
    def shape_reward(self, previous_state_unpadded, current_state_unpadded, target_state):
        """
        Reward the agent based on the shape of the grid.
        Outputs:
        - reward: the reward for the agent based on the shape of the grid
        - shapes_agree: a boolean indicating if the shapes of the 3 grids agree
        """
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
    
    def act(self, previous_state, action):
        """
        Apply the action to the previous state.
        """
        color_selection = action[0]
        selection = action[1:3]
        selection_parameters = action[3:5]
        transformation = action[5:7]
        transformation_parameters = action[7:]

        

    def step(self, action):
        previous_state = self.state[:, :self.dim] # make the current state into the previous state
        current_state = self.act(previous_state, action) # apply the action to the previous state
        target_state = self.state[:, self.dim:] # get the target state

        # Extract the states without padding
        previous_state_not_padded, current_state_not_padded, target_state_not_padded = extract_states(previous_state, current_state, target_state)
        reward, done = self.total_reward(previous_state_not_padded, current_state_not_padded, target_state_not_padded) # compute the reward
        
        # End the episode with a small probability if not ended already
        if not done and random.random() < 0.01:
            done = True
    
        # Update the state and done variables
        self.state[:, :self.dim] = current_state
        self.done = done
        return self.state, reward, done, {}
