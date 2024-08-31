import matplotlib.pyplot as plt
import json

training_challenge_dict = json.load(open('../data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json'))
training_solutions_dict = json.load(open('../data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_solutions.json'))

def display_challenge(challenge_key, solution=None, color_map='inferno'):
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
        
        # Plot the array as an image
        im_input = ax_input.imshow(input, cmap=color_map)
        im_output = ax_output.imshow(output, cmap=color_map)

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

        im_input = ax_input.imshow(input, cmap=color_map)
        im_output = ax_output.imshow(output, cmap=color_map)

        ax_input.set_title(f"Test Input")
        ax_output.set_title(f"Test Output")

        # remove labels
        ax_input.set_xticks([])
        ax_input.set_yticks([])
        ax_output.set_xticks([])
        ax_output.set_yticks([])

    # Show the plot
    plt.show()