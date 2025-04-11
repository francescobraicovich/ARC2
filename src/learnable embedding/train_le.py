import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import copy

from utils.util import to_tensor, set_device
from errors_optimizer_le import max_overlap_loss, get_grad_norm, update_with_adamw

from enviroment import ARC_Env
from rearc.main import random_challenge
from action_space import ARCActionSpace

from model_le import FullTransitionModel

# Determine the device: CUDA -> MPS -> CPU
DEVICE = set_device()
print("Using device for model:", DEVICE)

def random_action():
    a_s = ARCActionSpace.create_action_space()
    random_action = np.random.choice(a_s)
    return random_action

def pretrain_embedding(
    model,            # Instance of FullTransitionModel
    num_epochs,
    lr,
    device,
    batch_size,
):
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    '''
    # For continuous next-state images, we use MSE loss.
    criterion_next_state = nn.MSELoss()
    # For discrete action reconstruction, use cross-entropy loss.
    criterion_action_recon = nn.CrossEntropyLoss()
    '''
    #TODO randomize initial weights for action embeddder

    state_img = None
    action_idx = None
    next_state_img = None

    for epoch in range(num_epochs):
        for batch in batch_size:
            if state_img == None:
                state_img = random_challenge()
            else:
                state_img = state_img
            
            action_idx = 




# Example usage:
# Assume you have a dataset that yields tuples (state_img, action_idx, next_state_img)
# and a DataLoader constructed from it.
#
# from your_dataset_module import YourDataset
# dataset = YourDataset(...)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# Instantiate your model:
num_actions = 50000        # example number of actions
action_embed_dim = 32      # dimension for action embeddings
state_embed_dim = 64       # dimension for state embeddings (from Vision Transformer)
diff_in_channels = 3       # input channels for the U-net (e.g., RGB)
diff_out_channels = 3      # output channels (same as input)

model = FullTransitionModel(
    num_actions,
    action_embed_dim,
    state_embed_dim,
    diff_in_channels,
    diff_out_channels
)

# Run the pretraining loop:
pretrained_model = pretrain_embedding(model, dataloader, num_epochs=10, noise_level=0.0, lr=1e-4, device='cuda')

if __name__ == "__main__":
    from arg_parser_le import init_parser_le
    parser = init_parser_le()
    args = parser.parse_args()

    # Create model using the specified hyperparameters.
    # Convert observation_shape list to tuple.
    obs_shape = tuple(args.observation_shape)
    model = FullTransitionModel(
        args.num_actions,
        args.action_embed_dim,
        args.state_embed_dim,
        args.diff_in_channels,
        args.diff_out_channels,
        obs_shape
    )

    # TODO: define or import your DataLoader here.
    dataloader = None  # ...existing code to initialize your dataloader...

    # Run pretraining.
    # (Note: pretrain_embedding currently expects noise_level as an extra parameter.)
    pretrained_model = pretrain_embedding(
        model,
        dataloader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        max_episode_length=args.max_episode_length,
        max_episode=args.max_episode,
        max_actions=args.max_actions,
        max_steps=args.max_steps,
        noise_level=args.noise_level
    )
    # ...existing code (e.g., saving model, additional training steps)...

