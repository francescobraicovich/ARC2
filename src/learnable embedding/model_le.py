import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.functional import relu

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

class TransitionDiffusionModel(nn.Module):
    def __init__(self, in_channels, cond_dim, out_channels):
        """
        in_channels: Number of channels in the noisy next-state input.
        cond_dim: Dimension of the conditioning vector (state_embed_dim + action_embed_dim).
        out_channels: Number of channels in the predicted next state.
        """
        super(TransitionDiffusionModel, self).__init__()
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
        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder path: Upsampling with transposed convolutions and skip connections
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

    def forward(self, noisy_state, cond):
        """
        noisy_state: tensor [B, in_channels, H, W] representing the noisy next state.
        cond: Conditioning vector [B, cond_dim] (concatenation of state and action embeddings).
        """
        B, _, H, W = noisy_state.shape
        # Encoder
        x1 = self.enc1(noisy_state)      # [B, 64, H, W]
        x2 = self.enc2(x1)               # [B, 128, H/2, W/2]
        x3 = self.enc3(x2)               # [B, 256, H/4, W/4]
        # Bottleneck
        x_b = self.bottleneck(x3)        # [B, 256, H/4, W/4]
        # Expand conditioning vector to match bottleneck spatial dimensions
        cond_expanded = cond.unsqueeze(-1).unsqueeze(-1).expand(B, cond.shape[-1], H//4, W//4)
        # Concatenate condition with bottleneck output
        x_b_cond = torch.cat([x_b, cond_expanded], dim=1)  # [B, 256+cond_dim, H/4, W/4]
        # Decoder with skip connections
        d3 = self.dec3(x_b_cond)         # [B, 128, H/2, W/2]
        d3_cat = torch.cat([d3, x2], dim=1)  # [B, 128+128, H/2, W/2]
        d2 = self.dec2(d3_cat)           # [B, 64, H, W]
        d2_cat = torch.cat([d2, x1], dim=1)  # [B, 64+64, H, W]
        out = self.dec1(d2_cat)          # [B, out_channels, H, W]
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
                 diffusion_in_channels, diffusion_out_channels):
        """
        num_actions: Total discrete actions (e.g., 50,000)
        action_embed_dim: Dimension for the action embedding.
        state_embed_dim: Dimension for the state embedding (output from the Vision Transformer).
        diffusion_in_channels: Channels of the noisy next state input.
        diffusion_out_channels: Channels of the predicted next state.
        
        The conditioning vector for the diffusion model is the concatenation of state and action embeddings.
        """
        super(FullTransitionModel, self).__init__()
        self.action_embedding = ActionEmbedding(num_actions, action_embed_dim)
        self.state_encoder = StateEncoderViT(state_embed_dim)
        
        cond_dim = state_embed_dim + action_embed_dim  # Conditioning dimension.
        self.diffusion_model = TransitionDiffusionModel(diffusion_in_channels, cond_dim, diffusion_out_channels)
        
        # Function f implemented as an nn.Embedding for action reconstruction.
        self.action_reconstruction = ActionReconstructionEmbedding(num_actions, action_embed_dim)
        
    def forward(self, state_img, action_idx, noisy_next_state, predicted_action_emb=None):
        """
        state_img: tensor [B, 3, H, W] representing the current state as an image.
        action_idx: tensor [B] with discrete action indices.
        noisy_next_state: tensor [B, diffusion_in_channels, H, W] representing the noisy next state.
        predicted_action_emb (optional): if provided, use this embedding as the output of the internal policy.
                                        Otherwise, use the embedding from the action lookup.
        """
        # Obtain state embedding from the Vision Transformer.
        state_emb = self.state_encoder(state_img)  # [B, state_embed_dim]
        
        # Obtain action embedding via the embedding lookup.
        action_emb = self.action_embedding(action_idx)  # [B, action_embed_dim]
        
        # For conditioning, we can use the current action embedding.
        cond = torch.cat([state_emb, action_emb], dim=1)  # [B, state_embed_dim + action_embed_dim]
        
        # Predict the next state using the diffusion model.
        predicted_next_state = self.diffusion_model(noisy_next_state, cond)
        
        # Use the predicted action embedding to reconstruct the original action.
        # If predicted_action_emb is provided, it is used (e.g., from the policy); else, we use the one from the lookup.
        if predicted_action_emb is None:
            predicted_action_emb = action_emb
        reconstructed_action_logits = self.action_reconstruction(predicted_action_emb)
        
        return predicted_next_state, reconstructed_action_logits

