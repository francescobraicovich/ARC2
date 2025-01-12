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