"""
Module for computing approximate similarity matrices and embeddings for ARC action spaces.

This module defines functions to:
    - Generate random ARC problems and arrays.
    - Compute similarity matrices for color selections, grid selections, and transformations.
    - Filter actions based on the change they induce in the environment.
    - Create an overall approximate similarity matrix by combining the component similarities.
    - Embed the similarity matrix using Multi-Dimensional Scaling (MDS).

Dependencies:
    - sys, os
    - numpy
    - json
    - scikit-learn (MDS)
    - Custom modules: enviroment (ARC_Env, maximum_overlap_regions),
                      dsl.utilities.padding (pad_grid, unpad_grid)
"""

import sys
import os
import numpy as np
from rearc.main import *

# Add parent directory to sys.path to allow relative imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local module imports.
from enviroment import ARC_Env
from dsl.utilities.padding import unpad_grid

def random_arc_problem(env):
    """
    Generate a random ARC problem using the environment.

    Args:
        env: The ARC environment instance.

    Returns:
        np.ndarray: The unpadded training grid extracted from the environment's state.
    """
    random_state = env.reset()
    grid, shape = random_state
    training_grid = grid[:, :, 0]
    unpadded_grid = unpad_grid(training_grid)
    return unpadded_grid

def filter_by_change(action_space, num_experiments, threshold):
    """
    Filter actions that induce change in the environment.

    For each action in the action space, the function tests it on random ARC problems.
    If the action leaves the state unchanged too often (equal ratio above threshold),
    it is filtered out. Additionally, actions that always turn the entire grid 
    into a single unique color are also removed.

    Args:
        action_space: The action space object.
        env: The environment instance.
        num_experiments (int): The number of experiments to run per action.
        threshold (float): The threshold for the fraction of unchanged outcomes.

    Returns:
        np.ndarray: An array of actions that meet the change criteria.
    """
    # Load challenge data for the ARC environment.
    challenge_dictionary = json.load(
        open('data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json')
    )
    env = ARC_Env(
        path_to_challenges='data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json',
        action_space=action_space
    )

    actions = action_space.get_space()
    n = len(actions)
    equal_ratios = np.zeros(n, dtype=np.float64)
    always_single_color = np.zeros(n, dtype=bool)  # Track actions that always result in a single color

    for i in range(n):
        action = actions[i]
        num_equal = 0
        num_single_color = 0

        for _ in range(num_experiments):
            random_problem = random_arc_problem(env)
            new_state = env.act(random_problem, action)[0, :, :]

            # Check if the action leaves the state unchanged
            if np.array_equal(random_problem, new_state):
                num_equal += 1

            # Check if the entire grid is converted to a single unique color
            if np.unique(new_state).size == 1:
                num_single_color += 1

        equal_ratio = num_equal / num_experiments
        equal_ratios[i] = equal_ratio

        # If an action *always* results in a single color, mark it for removal
        if num_single_color == num_experiments:
            always_single_color[i] = True

        if i % 500 == 0:
            # Clear the line
            print('\r' + ' ' * 50, end='', flush=True)
            # Print the new message
            print(f'\rFiltered {i} out of {n} actions', end='', flush=True)

    print('Average equal ratio:', np.mean(equal_ratios))
    print('Transformations that always result in a single color:', np.sum(always_single_color))
    
    # Actions that have too high an "unchanged" ratio are filtered out
    change_ratios = 1 - equal_ratios
    mask = (change_ratios > threshold) & ~always_single_color  # Also remove actions that always turn the grid into a single color

    print(f'Out of {n} actions, only {np.sum(mask)} are used.')
    cleaned_actions = actions[mask]

    assert len(cleaned_actions) == np.sum(mask), 'Error in filtering actions.'
    return cleaned_actions