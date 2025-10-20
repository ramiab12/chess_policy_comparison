# Created by: Rami Abu Mukh, Omar Gharra, Ameer Khalaily

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# 1 residual block - 2 Convolutional layers with batch normalization and dropout
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout_rate: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = out + identity
        out = F.relu(out)
        
        return out


class ChessCNNPolicy(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 18,
        num_filters: int = 256,
        num_blocks: int = 15,
        dropout_rate: float = 0.1
    ):
        super(ChessCNNPolicy, self).__init__()
        
        self.num_input_channels = num_input_channels
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        
        # Initial convolution - as in AlphaZero paper 
        self.conv_initial = nn.Conv2d(
            num_input_channels, num_filters,
            kernel_size=3, padding=1, bias=False
        )
        self.bn_initial = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])
        
        # pooling removed - by omar
        # From-square head
        self.from_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.from_bn = nn.BatchNorm2d(2)
        
        # To-square head
        self.to_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.to_bn = nn.BatchNorm2d(2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = self.conv_initial(x)
        x = self.bn_initial(x)
        x = F.relu(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # From-square head
        from_out = self.from_conv(x)
        from_out = self.from_bn(from_out)
        from_out = F.relu(from_out)
        from_logits = from_out[:, 0, :, :].reshape(batch_size, 64)
        
        # To-square head
        to_out = self.to_conv(x)
        to_out = self.to_bn(to_out)
        to_out = F.relu(to_out)
        to_logits = to_out[:, 0, :, :].reshape(batch_size, 64)
        
        return from_logits, to_logits
    
    def predict_move(self, x: torch.Tensor, legal_moves: list) -> torch.Tensor:
        from_logits, to_logits = self.forward(x)
        
        from_probs = F.softmax(from_logits, dim=1)
        to_probs = F.softmax(to_logits, dim=1)
        
        # Score legal moves
        move_scores = []
        for move in legal_moves:
            from_idx = move.from_square
            to_idx = move.to_square
            score = from_probs[0, from_idx] * to_probs[0, to_idx]
            move_scores.append(score.item())
        
        best_idx = torch.tensor(move_scores).argmax()
        return best_idx
