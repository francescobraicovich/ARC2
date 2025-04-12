#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import set_device
DEVICE = set_device('world_model/action_embed.py')
class ActionEmbedding(nn.Module):
    def __init__(self, num_actions: int, embed_dim: int, normalize: bool = True, dropout: float = 0.0):
        """
        Args:
            num_actions (int): Total number of discrete actions (e.g., 50,000).
            embed_dim (int): Dimensionality of the action embedding vector.
            normalize (bool): If True, applies L2 normalization to embeddings.
            dropout (float): Dropout probability applied after embedding.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embed_dim).to(DEVICE)
        
        # SOTA weight initialization: truncated normal initialization with std=0.02
        # This is widely used in state-of-the-art transformer models (e.g., BERT, GPT)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        
        self.normalize = normalize
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, action):
        """
        Args:
            action (int | torch.Tensor): Either a single int index or a tensor of indices.
        
        Returns:
            torch.Tensor: Embedded action tensor.
                - shape: [embed_dim] for scalar input
                - shape: [batch_size, embed_dim] for batched input
        """
        if not torch.is_tensor(action):
            action = torch.tensor([action], dtype=torch.long, device=self.embedding.weight.device)
        elif action.dim() == 0:
            action = action.unsqueeze(0)

        embedded = self.embedding(action)
        embedded = self.dropout(embedded)

        if self.normalize:
            embedded = F.normalize(embedded, p=2, dim=-1)

        # Return [embed_dim] if input was a scalar
        if embedded.size(0) == 1 and action.dim() == 1:
            return embedded.squeeze(0)

        return embedded
    
    def export_weights(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The embedding matrix of shape [num_actions, embed_dim].
                         If normalization is enabled, returns normalized embeddings.
        """
        weights = self.embedding.weight.data.clone()
        if self.normalize:
            weights = F.normalize(weights, p=2, dim=-1)
        return weights
    
    def save_weights(self, path: str):
        """
        Save the embedding weights to a file.
        
        Args:
            path (str): Path to save the weights.
        """
        # append the 'action_embedding.pt' to the path
        path = path + '/action_embedding.pt'
        torch.save(self.export_weights(), path)

    def load_weights(self, path: str):
        """
        Load the embedding weights from a file.
        
        Args:
            path (str): Path to load the weights from.
        """
        # append the 'action_embedding.pt' to the path
        path = path + '/action_embedding.pt'
        self.embedding.weight.data.copy_(
        torch.load(path, map_location=torch.device('cpu'))
        )
        self.to(DEVICE)

    @property
    def num_parameters(self) -> int:
        """
        Returns:
            int: Number of parameters in the embedding layer.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
