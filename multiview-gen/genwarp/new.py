import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNUpsampler(nn.Module):
    """
    A CNN-based upsampler that takes a tensor of shape (B, 37, 37, 1024)
    and produces a tensor of shape (B, 64, 64, 128), using four Conv2d layers
    which reduce channels as [1024 → 1024 → 512 → 256 → 128].
    """
    def __init__(
        self,
        in_channels: int = 1024,
        block_channels: tuple = (1024, 512, 256, 128),
        target_size: tuple = (64, 64),
    ):
        super().__init__()
        self.target_size = target_size
        
        # First conv keeps channels at 1024
        self.conv_in = nn.Conv2d(in_channels, block_channels[0], kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(block_channels[0])
        self.relu = nn.ReLU(inplace=True)
        
        # Then three more conv layers stepping down through block_channels
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for ch_in, ch_out in zip(block_channels[:-1], block_channels[1:]):
            self.convs.append(nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm2d(ch_out))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 37, 37, 1024) → (B, 1024, 37, 37)
        x = x.permute(0, 3, 1, 2)
        
        # Upsample spatial dims to (64, 64)
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # Conv #1: 1024 → 1024
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)
        
        # Conv #2: 1024 → 512; Conv #3: 512 → 256; Conv #4: 256 → 128
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
        
        # (B, 128, 64, 64) → (B, 64, 64, 128)
        x = x.permute(0, 2, 3, 1)
        return x

# Example usage:
if __name__ == "__main__":
    B = 4
    input_tensor = torch.randn(B, 37, 37, 1024)
    upsampler = CNNUpsampler()
    output_tensor = upsampler(input_tensor)
    import pdb; pdb.set_trace()
    print("Output shape:", output_tensor.shape)
    # Should print: Output shape: torch.Size([4, 64, 64, 128])