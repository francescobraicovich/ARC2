import torch
import torch.nn as nn


class MultiScaleConv(nn.Module):
    """
    Multi-Scale Convolutional Module.
    Applies parallel convolutions with different kernel sizes and concatenates their outputs.
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels for each convolutional branch.
        """
        super(MultiScaleConv, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, bias=False)
        
        # Batch normalization for concatenated output
        self.bn = nn.BatchNorm2d(out_channels * 3)
        
        # Activation function
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
    def forward(self, x):
        # Apply different convolutional filters
        out1 = self.activation(self.conv3x3(x))
        out2 = self.activation(self.conv5x5(x))
        out3 = self.activation(self.conv7x7(x))
        
        # Concatenate along the channel dimension
        out = torch.cat([out1, out2, out3], dim=1)  # Shape: (B, out_channels*3, H, W)
        
        # Apply batch normalization
        out = self.bn(out)
        
        return out

class CNNFeatureExtractor(nn.Module):
    """
    Enhanced CNN Feature Extractor with Multi-Scale Convolutions, Batch Normalization,
    Dropout, and Advanced Activation Functions.
    """
    def __init__(self, hidden1=512, dropout_prob=0.3):
        """
        Args:
            hidden1 (int): Number of neurons in the final fully connected layer.
            dropout_prob (float): Dropout probability.
        """
        super(CNNFeatureExtractor, self).__init__()
        
        # Multi-Scale Convolutional Layers
        self.multi_scale1 = MultiScaleConv(in_channels=2, out_channels=16)    # Output channels: 48
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                  # Downsample by 2
        
        self.multi_scale2 = MultiScaleConv(in_channels=48, out_channels=32)   # Output channels: 96
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.multi_scale3 = MultiScaleConv(in_channels=96, out_channels=64)   # Output channels: 192
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.multi_scale4 = MultiScaleConv(in_channels=192, out_channels=128) # Output channels: 384
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))                            # Global average pooling
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(384, hidden1)
        
        # Dropout Layer
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Activation Function
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def init_weights(self):
        """
        Initialize weights for convolutional and fully connected layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        """
        Forward pass through the CNN module.
        
        Args:
            state (torch.Tensor): Input tensor of shape (B, R, C, 2).
            
        Returns:
            torch.Tensor: Latent feature representation of shape (B, hidden1).
        """
        # Rearrange tensor to (B, 2, R, C)
        x = state.permute(0, 3, 1, 2).contiguous()
        
        # Multi-Scale Block 1
        x = self.multi_scale1(x)  # (B, 48, R, C)
        x = self.pool1(x)         # (B, 48, R/2, C/2)
        
        # Multi-Scale Block 2
        x = self.multi_scale2(x)  # (B, 96, R/2, C/2)
        x = self.pool2(x)         # (B, 96, R/4, C/4)
        
        # Multi-Scale Block 3
        x = self.multi_scale3(x)  # (B, 192, R/4, C/4)
        x = self.pool3(x)         # (B, 192, R/8, C/8)
        
        # Multi-Scale Block 4
        x = self.multi_scale4(x)  # (B, 384, R/8, C/8)
        x = self.pool4(x)         # (B, 384, 1, 1)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)  # (B, 384)
        
        # Fully Connected Layer with Activation and Dropout
        x = self.activation(self.fc1(x))  # (B, hidden1)
        x = self.dropout(x)

        return x

"""
class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim):
        super(CNNFeatureExtractor, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=2, 
            out_channels=8, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )  # Output: (B, 8, R, C)
        self.bn1 = nn.BatchNorm2d(8)  # BatchNorm for conv1
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=8, 
            out_channels=16, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )  # Output: (B, 16, R, C)
        self.bn2 = nn.BatchNorm2d(16)  # BatchNorm for conv2
        
        # Adaptive average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (B, 16, 1, 1)
         
        # Fully connected layer
        self.fc = nn.Linear(16, output_dim)  # Output: (B, output_dim)
        self.bn_fc = nn.BatchNorm1d(output_dim)  # Optional BatchNorm for FC layer
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        Forward pass through the CNN with Batch Normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, R, C, 2).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, output_dim).
        # Rearrange dimensions from (B, R, C, 2) to (B, 2, R, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # First convolutional block: Conv -> BatchNorm -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second convolutional block: Conv -> BatchNorm -> ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Adaptive average pooling
        x = self.pool(x)  # Shape: (B, 16, 1, 1)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)  # Shape: (B, 16)
        
        # Fully connected layer
        x = self.fc(x)  # Shape: (B, output_dim)
        
        # Optional: Apply BatchNorm and activation after FC layer
        x = self.bn_fc(x)
        # You can choose to add an activation here if needed, e.g., self.relu(x)
        
        return x
"""

import torch
import torch.nn as nn
import torchvision.models as models

    

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