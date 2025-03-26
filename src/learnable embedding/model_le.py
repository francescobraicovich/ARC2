import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.functional import relu
import numpy as np  # added import
from dsl.utilities.padding import pad_grid



class ActionEmbedding(nn.Module):
    def __init__(self, num_actions, embed_dim):
        """
        num_actions: Total number of discrete actions (e.g., 50,000)
        embed_dim: Dimensionality of the action embedding vector.
        """
        super(ActionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_actions, embed_dim)
        
    def forward(self, action_idx):
        # action_idx: tensor of shape [batch_size] containing action indices.
        return self.embedding(action_idx)  # returns shape [batch_size, embed_dim]

class StateEncoderViT(nn.Module):
    def __init__(self, embed_dim):
        """
        embed_dim: Desired output embedding dimension for the state.
        Uses a Vision Transformer backbone from torchvision.
        """
        super(StateEncoderViT, self).__init__()
        # Using torchvision's ViT (vit_b_16) without pretrained weights for demonstration.
        self.vit = models.vit_b_16(pretrained=False)
        # Replace the classification head with a new Linear layer for our embedding.
        in_features = self.vit.heads.head.in_features  # input dimension of the original head
        self.vit.heads.head = nn.Linear(in_features, embed_dim)
        
    def forward(self, state_img):
        # state_img: tensor of shape [batch_size, 3, H, W] (e.g., 224x224 images)
        # Returns: state embedding of shape [batch_size, embed_dim]
        return self.vit(state_img)

# Modified UNet version for transitions:
class TransitionUNETModel(nn.Module):
    def __init__(self, in_channels, cond_dim, out_channels):
        """
        in_channels: Number of channels in the training grid (e.g., 3).
        cond_dim: Dimension of the conditioning vector (state_embed_dim + action_embed_dim).
        out_channels: Number of channels in the predicted next state.
        """
        super(TransitionUNETModel, self).__init__()
        # Encoder path: Downsampling layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder path with skip connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256 + cond_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, training_grid, cond):
        """
        training_grid: tensor of shape [B, H, W, C] representing the padded grid.
        cond: Conditioning vector [B, cond_dim].
        """
        # Convert to channels-first [B, C, H, W]
        x = training_grid.permute(0, 3, 1, 2)
        B, _, H, W = x.shape
        # Encoder
        x1 = self.enc1(x)         # [B, 64, H, W]
        x2 = self.enc2(x1)        # [B, 128, H/2, W/2]
        x3 = self.enc3(x2)        # [B, 256, H/4, W/4]
        # Bottleneck
        x_b = self.bottleneck(x3)  # [B, 256, H/4, W/4]
        # Expand cond to spatial dims
        cond_expanded = cond.unsqueeze(-1).unsqueeze(-1).expand(B, cond.shape[-1], H//4, W//4)
        x_b_cond = torch.cat([x_b, cond_expanded], dim=1)
        # Decoder with skip connections
        d3 = self.dec3(x_b_cond)      # [B, 128, H/2, W/2]
        d3_cat = torch.cat([d3, x2], dim=1)  # [B, 128+128, H/2, W/2]
        d2 = self.dec2(d3_cat)        # [B, 64, H, W]
        d2_cat = torch.cat([d2, x1], dim=1)  # [B, 64+64, H, W]
        out = self.dec1(d2_cat)       # [B, out_channels, H, W]
        return out

class ActionReconstructionEmbedding(nn.Module):
    def __init__(self, num_actions, action_embed_dim):
        """
        num_actions: Total number of discrete actions.
        action_embed_dim: Dimension of the action embedding.
        """
        super(ActionReconstructionEmbedding, self).__init__()
        # Create an embedding table for mapping back to discrete actions.
        self.embedding = nn.Embedding(num_actions, action_embed_dim)
        
    def forward(self, predicted_action_emb):
        """
        predicted_action_emb: tensor of shape [B, action_embed_dim] from the policy output.
        The function computes similarity scores between the predicted action embedding and the embedding table.
        """
        # Compute dot-product similarity between predicted embedding and each entry in the embedding table.
        # This yields logits for each discrete action.
        logits = torch.matmul(predicted_action_emb, self.embedding.weight.t())  # [B, num_actions]
        return logits



class FullTransitionModel(nn.Module):
    def __init__(self, num_actions, action_embed_dim, state_embed_dim,
                 unet_in_channels, unet_out_channels, observation_shape):
        """
        observation_shape: Tuple defining the padded grid shape (H, W, C)
        """
        super(FullTransitionModel, self).__init__()
        self.observation_shape = observation_shape
        self.action_embedding = ActionEmbedding(num_actions, action_embed_dim)
        self.state_encoder = StateEncoderViT(state_embed_dim)
        cond_dim = state_embed_dim + action_embed_dim
        # Replace diffusion_model with transition_model (UNet)
        self.transition_model = TransitionUNETModel(unet_in_channels, cond_dim, unet_out_channels)
        self.action_reconstruction = ActionReconstructionEmbedding(num_actions, action_embed_dim)
        
    def forward(self, state_img, action_idx, predicted_action_emb=None):
        """
        state_img: tensor [B, 3, H, W] or np.ndarray representing the current state as a training grid.
        action_idx: tensor [B] with discrete action indices.
        """
        # If state_img is a NumPy array, preprocess it.
        if isinstance(state_img, np.ndarray):
            state_img = preprocess_grid(state_img)
            state_img = state_img.unsqueeze(0)
        # Obtain state embedding from the Vision Transformer.
        state_emb = self.state_encoder(state_img)  # [B, state_embed_dim]
        action_emb = self.action_embedding(action_idx)  # [B, action_embed_dim]
        cond = torch.cat([state_emb, action_emb], dim=1)
        # Pass the training grid directly into the UNet model.
        predicted_next_state = self.transition_model(state_img, cond)
        if predicted_action_emb is None:
            predicted_action_emb = action_emb
        reconstructed_action_logits = self.action_reconstruction(predicted_action_emb)
        return predicted_next_state, reconstructed_action_logits

# New helper to preprocess the grid:
def preprocess_grid(state):
    """
    Pads the input grid and adds extra channels with the original (n_rows, n_cols).
    
    Args:
        state (np.ndarray): 2D input grid.
        
    Returns:
        torch.Tensor: Preprocessed grid tensor.
    """
    # Store original shape before padding
    n_rows, n_cols = state.shape
    # Assume pad_grid is defined elsewhere and returns a padded 2D grid
    padded = pad_grid(state)  # external function
    observation_shape = (padded.shape[0], padded.shape[1], 3)
    training_grid = np.zeros(observation_shape, dtype=np.int16)
    training_grid[:, :, 0] = padded
    training_grid[:, :, 1] = n_rows
    training_grid[:, :, 2] = n_cols
    return torch.tensor(training_grid, dtype=torch.float32)