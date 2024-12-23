import matplotlib.pyplot as plt
import numpy as np
import json
import random
import inspect
import torch
from joblib import Parallel, delayed

from dsl.utilities.plot import plot_grid, plot_grid_3d, plot_selection, display_challenge
from action_space import ARCActionSpace
from dsl.transform import Transformer
from enviroment import ARC_Env


def cleaner():
    print("Loading dataset...")
    with open("../data/RAW_DATA_DIR/arc-prize-2024/arc-agi_training_challenges.json", "r") as f:
        dataset = json.load(f)
    print(f"Loaded dataset with {len(dataset)} challenges.")

    print("Initializing action space...")
    action_space = ARCActionSpace()
    all_actions = action_space.get_space()
    print(f"Initialized action space with {len(all_actions)} actions.")

    # Set device to MPS if available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    ineffective_actions = {}

    for action_idx, action in enumerate(all_actions):
        print(f"Processing action {action_idx + 1}/{len(all_actions)}: {action}")

        color_selection_key = torch.tensor(action[0], device=device)
        selection_key = torch.tensor(action[1], device=device)
        transformation_key = torch.tensor(action[2], device=device)

        always_ineffective = True

        for challenge_key, challenge_data in dataset.items():
            for example in challenge_data["train"]:
                grid = torch.tensor(example["input"], dtype=torch.float32, device=device)
                try:
                    # Perform operations on the GPU
                    selected = grid > 0  # Example operation; replace with your logic

                    if selected.sum() > 0:
                        transformed = grid + 1  # Example transformation
                    else:
                        transformed = grid.clone()

                    # Move to CPU for comparison
                    if not torch.equal(transformed.cpu(), grid.cpu()):
                        always_ineffective = False
                        break

                except Exception as e:
                    print(f"Error: {e}")
                    always_ineffective = False
                    break

            if not always_ineffective:
                break

        if always_ineffective:
            ineffective_actions[action_idx] = {
                "action": action,
                "color_selection": action[0],
                "selection": action[1],
                "transformation": action[2],
            }

    print(f"Found {len(ineffective_actions)} ineffective actions.")
    return ineffective_actions