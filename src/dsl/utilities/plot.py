import matplotlib.pyplot as plt
import json
from matplotlib import colors
import numpy as np
import torch

def to_numpy(var, device=None):
    """
    Convert a PyTorch tensor to a NumPy array, handling the device.

    Args:
        var (torch.Tensor): The PyTorch tensor to convert.
        device (torch.device or None): The device type (e.g., 'cuda', 'mps', 'cpu').

    Returns:
        np.ndarray: The NumPy array.
    """
    if not isinstance(var, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")

    # Move tensor to the specified device if provided
    if device is not None:
        var = var.to(device)

    # Ensure the tensor is on CPU before converting to NumPy
    if var.device.type != 'cpu':
        var = var.cpu()

    # Detach from the computational graph and convert to NumPy
    return var.detach().numpy()


# Define the colormap: -1 maps to white, other values follow the colors list
cmap = colors.ListedColormap(
    ['#FFFFFF',  # -1 corresponds to white
     '#000000',  # 0
     '#0074D9',  # 1
     '#FF4136',  # 2
     '#2ECC40',  # 3
     '#FFDC00',  # 4
     '#AAAAAA',  # 5
     '#F012BE',  # 6
     '#FF851B',  # 7
     '#7FDBFF',  # 8
     '#870C25']  # 9
)

# Define the boundaries for each color in the colormap
bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

#training_challenge_dict = json.load(open('../../../data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json'))
#training_solutions_dict = json.load(open('../../../data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_solutions.json'))

def display_challenge(challenge_key, solution=None, color_map=cmap, transformations=None, kwargs=None):
    """
    Display the challenge inputs and outputs as a grid of images.

    This function visualizes the training examples from a challenge, and optionally
    the test input and solution. It creates a matplotlib figure with subplots for
    each input-output pair.

    Parameters:
    -----------
    challenge : dict
        A dictionary containing 'test' and 'train' keys. The 'train' value should be
        a list of dictionaries, each with 'input' and 'output' keys.
    solution : list, optional
        If provided, should contain the output for the test input. Default is None.
    color_map : str, optional
        The colormap to use for displaying the images. Default is 'inferno'.

    Returns:
    --------
    None
        The function displays the plot using plt.show().
    """
    challenge = training_challenge_dict[challenge_key]
    solution = training_solutions_dict[challenge_key]       

    test = challenge['test']
    train = challenge['train']

    # find how many examples there are in the train
    num_train_examples = len(train)
    num_rows = num_train_examples

    # Create a new figure and axis
    if solution is not None:
        num_rows += 1
    
    fig, axs = plt.subplots(num_rows, 2, figsize=(8, 3*num_rows))

    for i in range(num_train_examples):
        input = train[i]['input']
        output = train[i]['output']

        ax_input = axs[i, 0]
        ax_output = axs[i, 1]

        if transformations is not None:
            #input = apply_transformations(input, transformations, kwargs)
            #output = apply_transformations(output, transformations, kwargs)
            pass
        
        # Plot the array as an image
        im_input = ax_input.imshow(input, cmap=color_map, norm=norm)
        im_output = ax_output.imshow(output, cmap=color_map, norm=norm)

        # remove labels
        ax_input.set_xticks([])
        ax_input.set_yticks([])
        ax_output.set_xticks([])
        ax_output.set_yticks([])

        # Set title and labels
        ax_input.set_title(f"Train Input {i}")
        ax_output.set_title(f"Train Output {i}")

    if solution is not None:
        input = test[0]['input']
        output = solution[0]

        ax_input = axs[-1, 0]
        ax_output = axs[-1, 1]

        im_input = ax_input.imshow(input, cmap=color_map, norm=norm)
        im_output = ax_output.imshow(output, cmap=color_map, norm=norm)

        ax_input.set_title(f"Test Input")
        ax_output.set_title(f"Test Output")

        # remove labels
        ax_input.set_xticks([])
        ax_input.set_yticks([])
        ax_output.set_xticks([])
        ax_output.set_yticks([])

    # Show the plot
    plt.show()

def plot_selection(selection_mask):
    cmap = plt.cm.gray # cmap with 0 as black and 1 as white
    num_selections = len(selection_mask) # Number of selections to plot

    # Calculate the number of rows and columns for the subplots
    num_cols = min(5, num_selections)  # Max 5 columns
    num_rows = (num_selections - 1) // num_cols + 1

    ig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    axs = axs.flatten() if num_selections > 1 else [axs]

    for idx, selection in enumerate(selection_mask):
        axs[idx].imshow(selection, cmap=cmap)
        axs[idx].set_title(f'Geometry {idx}')
        axs[idx].axis('off')

    # Hide any unused subplots
    for idx in range(num_selections, len(axs)):
        axs[idx].axis('off')
    plt.show()

def plot_grid_3d(grid_3d, title=None):
    num_transformations = grid_3d.shape[0] # Number of transformations to plot

    # Calculate the number of rows and columns for the subplots
    num_cols = min(5, num_transformations)  # Max 5 columns
    num_rows = (num_transformations - 1) // num_cols + 1

    ig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    axs = axs.flatten() if num_transformations > 1 else [axs]

    for idx, grid in enumerate(grid_3d):
        axs[idx].imshow(grid, cmap=cmap, norm=norm)
        #axs[idx].set_title(f'Transformation on selection {idx}')
        axs[idx].axis('off')

    # Hide any unused subplots
    for idx in range(num_transformations, len(axs)):
        axs[idx].axis('off')

    if title is not None:
        plt.suptitle(title)

    plt.show()

def plot_grid(grid, title=None):
    # add a dimension to the grid to make it 3D
    grid_3d = np.expand_dims(grid, axis=0)
    plot_grid_3d(grid_3d, title)

def plot_step(state0, state1, shape0, shape1, r_t, info):
    # make the figure to contain 4 square subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    state0 = to_numpy(state0)
    state1 = to_numpy(state1)

    current0 = state0[:, :, 0]
    current1 = state1[:, :, 0]

    target0 = state0[:, :, 1]
    target1 = state1[:, :, 1]

    # plot the current state
    axs[0, 0].imshow(current0, cmap=cmap, norm=norm)
    axs[0, 1].imshow(target0, cmap=cmap, norm=norm)
    axs[1, 0].imshow(current1, cmap=cmap, norm=norm)
    axs[1, 1].imshow(target1, cmap=cmap, norm=norm)

    last_action = info['action_strings'][-1]
    action_values = [last_action[k] for k in last_action.keys()]
    string = '\n'.join(action_values)
    reward_string = 'reward: {}'.format(np.round(r_t, 2))
    string += '\n' + reward_string + '\n\n.'
  
    # set string as title
    fig.suptitle(string)
    shape0_string = 'shape0: {}'.format(torch.flatten(shape0).tolist())
    shape1_string = 'shape1: {}'.format(torch.flatten(shape1).tolist())

    axs[0, 0].set_title(shape0_string)
    axs[1, 0].set_title(shape1_string)

    plt.show()
