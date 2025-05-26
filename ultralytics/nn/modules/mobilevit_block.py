import torch
import torch.nn as nn
import sys
import os

# Add the vit directory to the path to import MobileViT dependencies
vit_path = os.path.join(os.path.dirname(__file__), '..', 'vit')
if vit_path not in sys.path:
    sys.path.insert(0, vit_path)

# Import with proper error handling
try:
    from ultralytics.nn.vit.mobilevit_v2_block import MobileViTBlockv2
except ImportError:
    # Fallback to direct import if the above fails
    import mobilevit_v2_block
    MobileViTBlockv2 = mobilevit_v2_block.MobileViTBlockv2

__all__ = ['MobileViTBlock']

class MobileViTBlock(nn.Module):
    """
    MobileViT block wrapper for YOLO integration.
    Can be used as a drop-in replacement for C3k2 blocks.
    """
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize MobileViT block.
        
        Args:
            c1: Input channels
            c2: Output channels  
            n: Number of blocks (number of attention blocks in MobileViT)
            shortcut: Whether to use shortcut connection
            g: Groups (for compatibility)
            e: Expansion ratio (for compatibility)
        """
        super().__init__()
        
        self.c = c2
        self.shortcut = shortcut and c1 == c2
        
        # Channel adjustment if needed
        if c1 != c2:
            self.channel_adjust = nn.Conv2d(c1, c2, 1, bias=False)
        else:
            self.channel_adjust = nn.Identity()
            
        # Calculate appropriate dimensions for MobileViT
        attn_unit_dim = max(64, c2 // 4)  # Attention dimension
        ffn_multiplier = 2
        
        # MobileViT block
        self.mobilevit = MobileViTBlockv2(
            in_channels=c2,
            attn_unit_dim=attn_unit_dim,
            ffn_multiplier=ffn_multiplier,
            n_attn_blocks=max(1, n),  # Use n parameter for number of attention blocks
            patch_h=2,
            patch_w=2,
            dropout=0.0,
            ffn_dropout=0.0,
            attn_dropout=0.0,
            conv_ksize=3,
            dilation=1,
        )
        
    def forward(self, x):
        """Forward pass through MobileViT block."""
        # Adjust channels if needed
        x_adjusted = self.channel_adjust(x)
        
        # Apply MobileViT block
        out = self.mobilevit(x_adjusted)
        
        # Apply shortcut connection if enabled and dimensions match
        if self.shortcut:
            # Check if dimensions match before applying shortcut
            if out.shape == x_adjusted.shape:
                out = out + x_adjusted
            else:
                # If dimensions don't match due to spatial resizing in MobileViT,
                # resize x_adjusted to match the output dimensions
                if out.shape[2:] != x_adjusted.shape[2:]:
                    x_adjusted = torch.nn.functional.interpolate(
                        x_adjusted, 
                        size=out.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                out = out + x_adjusted
            
        return out 