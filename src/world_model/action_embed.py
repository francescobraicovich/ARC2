import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionEmbedding(nn.Module):
    def __init__(self, num_actions: int, embed_dim: int, normalize: bool = False, dropout: float = 0.0):
        """
        Args:
            num_actions (int): Total number of discrete actions (e.g., 50,000).
            embed_dim (int): Dimensionality of the action embedding vector.
            normalize (bool): If True, applies L2 normalization to embeddings.
            dropout (float): Dropout probability applied after embedding.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embed_dim)
        self.normalize = normalize
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Optional: initialize with better weight strategy
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

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
