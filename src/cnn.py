import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.temperature = nn.Parameter(torch.tensor(0.5))  # Lower initial value for stability

        self.layer_norm = nn.LayerNorm([in_dim, 8, 8])  # Match output shape after layer3

        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.xavier_uniform_(self.key_conv.weight)

    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = F.normalize(self.query_conv(x).view(B, -1, W * H), p=2, dim=-1)
        proj_key = F.normalize(self.key_conv(x).view(B, -1, W * H), p=2, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, W * H)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key) / (self.temperature + 1e-6)
        attention = self.softmax(energy)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, W, H)
        out = self.gamma * out + (1 - self.gamma) * x  # Weighted residual connection

        return self.layer_norm(out)  # Apply LayerNorm

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads=4):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        head_dim = in_dim // num_heads
        self.query_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.temperature = nn.Parameter(torch.tensor(0.5))
        self.proj = nn.Conv2d(in_dim, in_dim, 1)
        # You could also keep a LayerNorm if you wish to normalize across the heads:
        self.layer_norm = nn.LayerNorm([in_dim, 8, 8]) # Match output shape after layer3
    
    def forward(self, x):
        B, C, W, H = x.size()
        head_dim = C // self.num_heads
        # Project and reshape for multi-head
        q = self.query_conv(x).view(B, self.num_heads, head_dim, W * H)  # (B, heads, head_dim, N)
        k = self.key_conv(x).view(B, self.num_heads, head_dim, W * H)
        v = self.value_conv(x).view(B, self.num_heads, head_dim, W * H)
        
        # Normalize queries and keys for stability
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # Compute scaled dot-product attention per head
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) / (self.temperature + 1e-6)
        attn = self.softmax(attn)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        
        # Concatenate heads
        out = out.contiguous().view(B, C, W * H).view(B, C, W, H)
        out = self.proj(out)
        out = self.gamma * out + (1 - self.gamma) * x  # Residual connection
        
        return self.layer_norm(out)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, hidden1=256, dropout_prob=0, pretrained=True):
        super(CNNFeatureExtractor, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self._modify_first_conv_layer()
        self._modify_resnet_architecture()

        # Remove the final classification layer
        self.resnet.fc = nn.Identity()

        # Self-attention, fully connected layers, etc.
        self.self_attention = SelfAttention(in_dim=256)
        self.fc1 = nn.Linear(256 * 4 * 4, hidden1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.activation = nn.GELU()

        self._initialize_weights()

    def _modify_first_conv_layer(self):
        # Now the input will have 4 channels:
        #   2 channels from the original state data, plus 2 mask channels.
        self.resnet.conv1 = nn.Conv2d(
            in_channels=4, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )

    def _modify_resnet_architecture(self):
        # Modify layer2: keep downsampling and adjust dilation/padding as needed
        layer2 = self.resnet.layer2
        for block in layer2:
            block.conv2.dilation = (2, 2)
            block.conv2.padding = (2, 2)

        # Modify layer3: prevent downsampling in the first block and adjust dilation/padding
        layer3 = self.resnet.layer3
        for i, block in enumerate(layer3):
            if i == 0:
                block.conv1.stride = (1, 1)  # Prevent downsampling
                if block.downsample is not None:
                    block.downsample[0].stride = (1, 1)
            block.conv2.dilation = (2, 2)
            block.conv2.padding = (2, 2)

        del self.resnet.layer4  # Remove layer4

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)

    def forward(self, state):
        # state shape is expected to be (B, H, W, 2)
        # Permute to (B, C, H, W) as in your original code
        x = state.permute(0, 3, 1, 2).contiguous()  # Now x has shape (B, 2, H, W)

        # Create the mask:
        # For every position, if the value is not -1, mark it as 1.0; otherwise, 0.0.
        # This produces a mask with the same shape as x, i.e., (B, 2, H, W)
        mask = (x != -1).float()

        # Concatenate the original input and the mask along the channel dimension.
        # The result will have 4 channels: 2 original channels and 2 mask channels.
        x = torch.cat([x, mask], dim=1)  # x now has shape (B, 4, H, W)

        # Continue with the forward pass through ResNet and self-attention:
        x = self.resnet.conv1(x)  # Now uses the modified conv1 that accepts 4 channels
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.self_attention(x)  # (B, 256, H', W')
        x = nn.AdaptiveAvgPool2d((4, 4))(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        return x