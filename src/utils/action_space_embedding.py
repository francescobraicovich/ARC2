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
import json
import numpy as np
import random
from rearc.main import *
from sklearn.manifold import MDS

# Add parent directory to sys.path to allow relative imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local module imports.
from enviroment import maximum_overlap_regions, ARC_Env
from dsl.utilities.padding import pad_grid, unpad_grid


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


def random_array(env, arc_prob=1):
    """
    Generate a random 2D array for testing purposes.
    
    With probability `arc_prob`, an ARC problem is generated via the environment.
    Otherwise, a re-arc  array is used .

    Args:
        env: The environment instance.
        arc_prob (float): The probability of generating an ARC problem instead of a random array.

    Returns:
        np.ndarray: A random 2D array.
    """
    if np.random.rand() < arc_prob:
        return random_arc_problem(env)

    print("generating random array")
    print('generators: ', generators)
    random_challenge = random.choice(list(generators.keys()))
    print(random_challenge)
    result = demo_generator(random_challenge) 
    result = np.array(result[0]['input'] )
    return result


def create_color_similarity_matrix(action_space, env, num_experiments=10):
    """
    Create a similarity matrix for color selection functions.

    For each pair of color selections, the function runs multiple experiments on a random problem
    and increments similarity if both selections yield the same result.

    Args:
        action_space: The action space containing color selection functions.
        env: The environment instance.
        num_experiments (int): The number of experiments to run per pair.

    Returns:
        np.ndarray: A similarity matrix (size: number of color selections).
    """
    color_selections = list(action_space.color_selection_dict.keys())
    n = len(color_selections)
    similarity_matrix = np.identity(n)

    for _ in range(num_experiments):
        random_problem = random_array(env, arc_prob=1)
        for i in range(n):
            for j in range(i + 1, n):
                selection1 = color_selections[i]
                selection2 = color_selections[j]
                col1 = action_space.color_selection_dict[selection1](random_problem)
                col2 = action_space.color_selection_dict[selection2](random_problem)

                # Increment similarity if both selections yield the same color.
                similarity = 1 if col1 == col2 else 0
                similarity_matrix[i, j] += similarity
                similarity_matrix[j, i] += similarity

    # Average out the similarity scores over experiments (ignoring the diagonal).
    mask = np.identity(n) == 0
    similarity_matrix[mask] /= num_experiments
    return similarity_matrix


def compute_selection_similarity(sel1, sel2):
    """
    Compute the maximum similarity between two selection masks.

    The similarity is computed as the fraction of identical elements (over the total number of elements)
    for each pair of masks from sel1 and sel2. The maximum similarity over all pairs is returned.

    Args:
        sel1 (np.ndarray): A 3D array representing the first selection.
        sel2 (np.ndarray): A 3D array representing the second selection.

    Returns:
        float: The maximum similarity value found.
    """
    total_elements = np.size(sel1)
    max_similarity = 0.0
    for i in range(sel1.shape[0]):
        for j in range(sel2.shape[0]):
            # Compute the fraction of matching elements.
            similarity = np.sum(sel1[i, :, :] == sel2[j, :, :]) / total_elements
            if similarity > max_similarity:
                max_similarity = similarity
    return max_similarity


def create_selection_similarity_matrix(action_space, env, num_experiments=10):
    """
    Create a similarity matrix for grid selection functions.

    For each pair of selection functions, the function runs experiments on a random problem
    and computes similarity using `compute_selection_similarity`.

    Args:
        action_space: The action space containing selection functions.
        env: The environment instance.
        num_experiments (int): The number of experiments to run per pair.

    Returns:
        np.ndarray: A similarity matrix (size: number of selection functions).
    """
    selections = list(action_space.selection_dict.keys())
    n = len(selections)
    similarity_matrix = np.identity(n)

    for _ in range(num_experiments):
        random_problem = random_array(env, arc_prob=1)
        # Choose a random color present in the problem for selection functions.
        random_color_in_problem = int(np.random.choice(np.unique(random_problem)))
        for i in range(n):
            for j in range(i + 1, n):
                selection1 = selections[i]
                selection2 = selections[j]
                sel1 = action_space.selection_dict[selection1](random_problem, color=random_color_in_problem)
                sel2 = action_space.selection_dict[selection2](random_problem, color=random_color_in_problem)
                similarity = compute_selection_similarity(sel1, sel2)
                similarity_matrix[i, j] += similarity
                similarity_matrix[j, i] += similarity

    # Average out the similarity scores over experiments (ignoring the diagonal).
    mask = np.identity(n) == 0
    similarity_matrix[mask] /= num_experiments
    return similarity_matrix


def create_transformation_similarity_matrix(action_space, env, num_experiments=10):
    """
    Create a similarity matrix for transformation functions.

    For each pair of transformation functions, the function selects a random problem and a random
    non-empty selection, then applies each transformation and computes similarity using
    `maximum_overlap_regions`.

    Args:
        action_space: The action space containing transformation functions.
        env: The environment instance.
        num_experiments (int): The number of experiments to run per pair.

    Returns:
        np.ndarray: A similarity matrix (size: number of transformation functions).
    """
    transformations = list(action_space.transformation_dict.keys())
    selections = list(action_space.selection_dict.keys())
    n = len(transformations)
    similarity_matrix = np.identity(n)

    for _ in range(num_experiments):
        random_problem = random_array(env, arc_prob=1)
        # Find a non-empty selection by repeatedly sampling until a valid selection is found.
        while True:
            random_color_in_problem = int(np.random.choice(np.unique(random_problem)))
            random_selection_key = np.random.choice(selections)
            random_selection = action_space.selection_dict[random_selection_key](random_problem, color=random_color_in_problem)
            random_action = np.array([1, random_selection_key, 0])
            if np.any(random_selection):
                break

        for i in range(n):
            for j in range(i + 1, n):
                transformation1 = transformations[i]
                transformation2 = transformations[j]

                random_action[2] = transformation1
                out1 = env.act(random_problem, random_action, fixed_color=random_color_in_problem)

                random_action[2] = transformation2
                out2 = env.act(random_problem, random_action, fixed_color=random_color_in_problem)

                # Compute similarity using maximum_overlap_regions.
                _, _, similarity = maximum_overlap_regions(out1, out2)
                similarity_matrix[i, j] += similarity
                similarity_matrix[j, i] += similarity

    # Average out the similarity scores over experiments (ignoring the diagonal).
    mask = np.identity(n) == 0
    similarity_matrix[mask] /= num_experiments
    return similarity_matrix


def filter_by_change(action_space, env, num_experiments, threshold):
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


def create_approximate_similarity_matrix(action_space, num_experiments_filter, filter_threshold, num_experiments_similarity):
    """
    Create an approximate similarity matrix for the entire action space.

    The function performs the following steps:
        1. Loads challenge data and creates an ARC environment.
        2. Filters out actions that do not cause significant changes.
        3. Computes similarity matrices for color selections, selections, and transformations.
        4. Combines these component matrices to produce a final similarity matrix.

    Args:
        action_space: The action space object.
        num_experiments_filter (int): Number of experiments for filtering actions.
        filter_threshold (float): Threshold for filtering actions based on change.
        num_experiments_similarity (int): Number of experiments to compute similarity.

    Returns:
        tuple: (cleaned_actions, similarity_matrix)
            - cleaned_actions (np.ndarray): The filtered set of actions.
            - similarity_matrix (np.ndarray): The combined similarity matrix.
    """
    # Load challenge data for the ARC environment.
    challenge_dictionary = json.load(
        open('data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json')
    )
    env = ARC_Env(
        path_to_challenges='data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json',
        action_space=action_space
    )

    # Filter actions based on their ability to change the environment.
    cleaned_actions = filter_by_change(action_space, env, num_experiments_filter, filter_threshold)

    # Compute similarity matrices for each action component.
    color_similarity = create_color_similarity_matrix(action_space, env, num_experiments_similarity)
    print('Similarity matrix created for: color selections.')
    selection_similarity = create_selection_similarity_matrix(action_space, env, num_experiments_similarity)
    print('Similarity matrix created for: selections.')
    transformation_similarity = create_transformation_similarity_matrix(action_space, env, num_experiments_similarity)
    print('Similarity matrix created for: transformations.')

    # Retrieve keys lists for mapping.
    color_selections = list(action_space.color_selection_dict.keys())
    selections = list(action_space.selection_dict.keys())
    transformations = list(action_space.transformation_dict.keys())

    n = len(cleaned_actions)
    similarity_matrix = np.identity(n, dtype=np.float32)

    # Combine component similarities for each pair of cleaned actions.
    for i in range(n):
        col_sel1 = color_selections.index(cleaned_actions[i][0])
        sel1 = selections.index(cleaned_actions[i][1])
        trn1 = transformations.index(cleaned_actions[i][2])
        for j in range(i + 1, n):
            col_sel2 = color_selections.index(cleaned_actions[j][0])
            sel2 = selections.index(cleaned_actions[j][1])
            trn2 = transformations.index(cleaned_actions[j][2])
            similarity = (
                color_similarity[col_sel1, col_sel2] *
                selection_similarity[sel1, sel2] *
                transformation_similarity[trn1, trn2]
            )
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

        if i % 500 == 0:
            print(f"Processed {i}/{n} actions.", end="\r")

    print('Similarity matrix shape: ', similarity_matrix.shape)

    return cleaned_actions, similarity_matrix


def mds_embed(similarity_matrix, n_components=20):
    """
    Embed the similarity matrix into a lower-dimensional space using MDS.

    Args:
        similarity_matrix (np.ndarray): The precomputed similarity (or dissimilarity) matrix.
        n_components (int): The number of dimensions for the embedding.

    Returns:
        np.ndarray: The embedded representation of the actions.
    """
    print('Embedding with MDS...')

    embedding = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=42,
        n_jobs=-1,
        metric=False,
        normalized_stress=True,
        n_init=1
    )

    embedding.fit(similarity_matrix)
    stress = embedding.stress_
    print(f'MDS stress: {stress}')
    return embedding.embedding_
