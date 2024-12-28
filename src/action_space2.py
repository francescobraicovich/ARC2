import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
from action_space import ARCActionSpace
from enviroment import maximum_overlap_regions

test_array = np.random.randint(0, 9, (30, 30))

def create_similarity_matrix(input_matrix, output_matrix):
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32)
    output_tensor = torch.tensor(output_matrix, dtype=torch.float32)

    action_space = ARCActionSpace()
    actions = action_space.get_space()
    similarities = []

    def apply_action(grid, action):
        # Extract action components
        color_key, selection_key, transform_key = action
    
        # Get functions from action space dictionaries
        color_func = action_space.color_selection_dict[color_key]
        selection_func = action_space.selection_dict[selection_key]
        transform_func = action_space.transformation_dict[transform_key]
    
        # Apply color selection first
        color = color_func(grid)
        
        # Apply selection using color
        selected = selection_func(grid, color=color)
    
        # Only transform if something was selected
        if np.any(selected):
            result = transform_func(grid, selection=selected)
        else:
            result = np.expand_dims(grid, axis=0)
            
        return result

    for action in actions:
        try:
            transformed = apply_action(input_tensor, action)
            #match = (transformed == output_tensor).sum().item()
            #similarities.append(match / output_tensor.numel())
            similarities.append(maximum_overlap_regions(transformed, output_tensor))
        except:
            similarities.append(0)

    return np.array(similarities)

def apply_action(grid, action):
    # Extract action components
    color_key, selection_key, transform_key = action

    # Get functions from action space dictionaries
    color_func = action_space.color_selection_dict[color_key]
    selection_func = action_space.selection_dict[selection_key]
    transform_func = action_space.transformation_dict[transform_key]

    # Apply color selection first
    color = color_func(grid)
    
    # Apply selection using color
    selected = selection_func(grid, color=color)

    # Only transform if something was selected
    if np.any(selected):
        result = transform_func(grid, selection=selected)
    else:
        result = np.expand_dims(grid, axis=0)
        
    return result

def build_action_action_similarity_matrix(input_matrix):
    print("Building action-action similarity matrix...")
    action_space = ARCActionSpace()
    actions = action_space.get_space()
    n = len(actions)
    print(f"Number of actions: {n}")
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        print(f"Processing row action {i+1}/{n}")
        out_i = apply_action(input_tensor, actions[i])
        for j in range(n):
            out_j = apply_action(input_tensor, actions[j])
            similarity_matrix[i, j] = maximum_overlap_regions(out_i, out_j)

    print("Similarity matrix built.")
    return similarity_matrix

def compute_action_action_similarity(num_experiments=10):
    action_space = ARCActionSpace()
    actions = action_space.get_space()
    similarities_accum = np.zeros((len(actions),))

    for _ in range(num_experiments):
        rand_matrix = np.random.randint(0, 9, (10, 10))
        idx = np.random.randint(0, len(actions))
        chosen_action = actions[idx]
        output_matrix = apply_action(rand_matrix, chosen_action)
        sim_vector = create_similarity_matrix(rand_matrix, output_matrix)
        similarities_accum += sim_vector

    return similarities_accum / num_experiments
