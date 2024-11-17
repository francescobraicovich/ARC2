import numpy as np
import random
from dsl.utilities.padding import pad_grid, unpad_grid

def extract_states(training_grid0, training_grid1, dim):
    """
    Extract the current and target states from the training grids.
    """
    current_state0 = training_grid0[:, :dim] # get the current state before the action
    current_state1 = training_grid1[:, :dim] # get the current state after the action
    target_state = training_grid0[:, dim:] # get the target state
    
    # Remove padding
    current_state0 = unpad_grid(current_state0)
    current_state1 = unpad_grid(current_state1)
    target_state = unpad_grid(target_state)

    return current_state0, current_state1, target_state


class ARCEnviroment():
    def __init__(self, challenge_dictionary, dim=30):
        self.challenge_dictionary = challenge_dictionary
        self.num_challenges = len(challenge_dictionary)
        self.dim = dim # maximum dimension of the problem

        # reward variables
        self.step_penalty = -1
        self.maximum_similarity = 50
        self.completed_challenge_reward = 25

        # Initialize current training grid
        self.current_training_grid = self.get_random_challenge()

    def get_random_challenge(self):
        """
        Get a random challenge from the challenge dictionary. 
        """
        challenge_key = random.choice(list(self.challenge_dictionary.keys())) 
        challenge = self.challenge_dictionary[challenge_key]
        challenge_train = challenge['train']
        challenge_length = len(challenge_train)
        random_index = random.randint(0, challenge_length-1)
        random_challenge = challenge_train[random_index]
        random_input = np.array(random_challenge['input'])
        random_output = np.array(random_challenge['output'])
        random_input = pad_grid(random_input)
        random_output = pad_grid(random_output)
        training_grid = np.hstack((random_input, random_output))
        return training_grid
    
    def reset(self):
        self.current_training_grid = self.get_random_challenge()
        return self.current_training_grid
    
    def shape_reward(self, current_state0, current_state1, target_state):
        """
        Reward the agent based on the shape of the grid.
        Outputs:
        - reward: the reward for the agent based on the shape of the grid
        - shapes_agree: a boolean indicating if the shapes of the 3 grids agree
        """

        current_shape0 = np.shape(current_state0)
        current_shape1 = np.shape(current_state1)
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
          
    def similarity_reward(self, current_state0, current_state1, target_state):
        """
        Reward the agent based on the similarity of the grid.
        """ 
        size = np.size(target_state)
        # If the shapes agree, we compute the similarity reward
        similarity_0 = np.sum(current_state0 == target_state) / size
        similarity_1 = np.sum(current_state1 == target_state) / size

        similarity_difference = similarity_1 - similarity_0
        return self.maximum_similarity * similarity_difference

    def total_reward(self, current_state0, current_state1, target_state):
        """
        Compute the total reward for the agent.
        """

        shape_reward, shapes_agree = self.shape_reward(current_state0, current_state1, target_state)
        if not shapes_agree: # If the shapes do not agree, we return the shape reward
            return shape_reward
        
        # If the shapes agree, we compute the similarity reward
        similarity_reward = self.similarity_reward(current_state0, current_state1, target_state)
        
        # If the agent has completed the challenge, we return the completed challenge reward
        if np.all(current_state1 == target_state):
            return shape_reward + similarity_reward + self.completed_challenge_reward
        
        # If the agent has not completed the challenge, we return the sum of the shape and similarity rewards
        return shape_reward + similarity_reward + self.step_penalty

    def step(self, action):
        dim = self.dim        
        current_state0 = self.current_training_grid[:, :dim]

        # Apply the action to the current state
        current_state1 = action(current_state0)

        

        pass