import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import copy

from utils.util import to_tensor

from dsl.utilities.plot import plot_step

from model_le import FullTransitionModel


def pretrain_embedding(
    model,            # Instance of FullTransitionModel
    dataloader,       # DataLoader yielding (state_img, action_idx, next_state_img)
    num_epochs,
    lr,
    device,
    max_episode_length,
    max_episode,
    max_actions,
    max_steps,     
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
    

    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_batches = 0
        
        for state_img, action_idx, next_state_img in dataloader:
            # Move data to device
            state_img = state_img.to(device)           # [B, 3, H, W]
            action_idx = action_idx.to(device)           # [B]
            next_state_img = next_state_img.to(device)   # [B, 3, H, W]
            
            # For a U-net based transition model you can choose:
            # - to use the clean next state directly, or
            # - add a small noise as data augmentation (optional)
            if noise_level > 0.0:
                input_next_state = add_noise(next_state_img, noise_level=noise_level)
            else:
                input_next_state = next_state_img
            
            # Forward pass through the model:
            # The model returns:
            #   predicted_next_state: prediction of next state image.
            #   reconstructed_action_logits: logits for the discrete action.
            predicted_next_state, reconstructed_action_logits = model(
                state_img,
                action_idx,
                input_next_state
            )
            
            # Compute next-state loss.
            loss_next_state = criterion_next_state(predicted_next_state, next_state_img)
            # Compute action reconstruction loss.
            loss_action_recon = criterion_action_recon(reconstructed_action_logits, action_idx)
            
            # Total loss is the sum of both losses.
            loss = loss_next_state + loss_action_recon
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
        
        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return model

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