import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Self-Attention Module for Spatial Feature Maps.
    Enhances feature maps by modeling spatial dependencies.
    """
    def __init__(self, in_dim):
        """
        Args:
            in_dim (int): Number of input channels.
        """
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        # Query, Key, and Value convolutions
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Softmax for attention scores
        self.softmax = nn.Softmax(dim=-1)

        # Learnable scaling factor for residual connection
        self.gamma = nn.Parameter(torch.ones(1))  # Initialize gamma to 1

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Initialize weights for stability
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.xavier_uniform_(self.key_conv.weight)

    def forward(self, x):
        """
        Forward pass for self-attention.

        Args:
            x (torch.Tensor): Input feature maps (B, C, H, W).

        Returns:
            torch.Tensor: Output feature maps after applying self-attention.
        """
        m_batchsize, C, width, height = x.size()
        
        # Compute Query, Key, and Value matrices
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height)  # (B, C', N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # (B, C', N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # (B, C, N)

        # Normalize Query and Key to prevent large values
        proj_query = F.normalize(proj_query, p=2, dim=-1)  # L2 Normalize along last dim
        proj_key = F.normalize(proj_key, p=2, dim=-1)

        # Compute attention energy and scale
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # (B, N, N)
        energy = energy / (self.temperature + 1e-6)  # Learnable temperature scaling

        # Residual connection for stability
        identity_matrix = torch.eye(energy.shape[-1], device=x.device).expand_as(energy)
        energy = energy + identity_matrix  # Helps smooth out extreme values

        # Softmax for attention scores
        attention = self.softmax(energy)

        # Apply attention to value
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(m_batchsize, C, width, height)  # (B, C, H, W)

        # Scale and add residual
        out = self.gamma * out + x

        return out

class CNNFeatureExtractor(nn.Module):
    def __init__(self, hidden1=512, dropout_prob=0.1, pretrained=True):
        super(CNNFeatureExtractor, self).__init__()
        
        # Load ResNet-18 model
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify the first convolutional layer to accept 2 input channels
        self._modify_first_conv_layer()
        
        # Modify layer2 and layer3 to avoid downsampling
        self._modify_resnet_architecture()

        # Remove the final fully connected layer
        self.resnet.fc = nn.Identity()
        
        # Self-Attention Module
        self.self_attention = SelfAttention(in_dim=256)  # Change to match layer3 output
        
        # Fully connected layer to match hidden1
        self.fc1 = nn.Linear(256, hidden1)  # Adjust for new spatial size
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Activation function
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # Initialize weights for new layers
        self._initialize_weights()
    
    def _modify_first_conv_layer(self):
        """
        Modify the first convolutional layer of ResNet-18:
        - Change kernel size to 3x3
        - Change stride to 1
        - Reduce padding
        """
        self.resnet.conv1 = nn.Conv2d(
            in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )

        del self.resnet.layer4  # Remove the final layer

    def _modify_resnet_architecture(self):
        """
        Modify layer2 and layer3 to avoid downsampling.
        """
        # Prevent layer2 from reducing spatial dimensions
        self.resnet.layer2[0].conv1.stride = (2, 2)
        self.resnet.layer2[0].downsample[0].stride = (2, 2)
        
        # Prevent layer3 from reducing spatial dimensions
        self.resnet.layer3[0].conv1.stride = (2, 2)
        self.resnet.layer3[0].downsample[0].stride = (2, 2)
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)

    def forward(self, state):
        """
        Forward pass through the ResNet-18 + Self-Attention feature extractor.
        """
        x = state.permute(0, 3, 1, 2).contiguous()
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)  # Output shape should now be (B, 256, 8, 8)

        # Apply Self-Attention
        x = self.self_attention(x)  # Shape: (B, 256, H, W)
        
        # Global Average Pooling
        x = nn.AdaptiveAvgPool2d((1, 1))(x) # Shape: (B, 256, 1, 1)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1) 
        
        # Fully Connected Layer with Activation and Dropout
        x = self.activation(self.fc1(x))  # Shape: (B, hidden1)
        x = self.dropout(x)
        
        return x