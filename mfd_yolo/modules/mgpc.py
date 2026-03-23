"""
MGPC: Multi-scale Group Pointwise Convolution module.

Replaces the standard Bottleneck inside C2f at backbone layers 6 and 8.
Splits input channels into 4 groups, applies different kernel sizes
(1x1, 3x3, 5x5, 7x7) to each group, concatenates, and fuses with 1x1 conv.

Reference: Paper Fig 2, page 6, Eq 1-3.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class MGPC(nn.Module):
    """Multi-scale Group Pointwise Convolution bottleneck.
    
    Splits channels into 4 groups and applies heterogeneous kernel sizes
    (1, 3, 5, 7) to capture multi-level features, then fuses via
    concatenation + pointwise convolution.
    
    Args:
        c1: Input channels.
        c2: Output channels.
        shortcut: Whether to use residual shortcut (only if c1 == c2).
        g: Groups for the group convolutions (default 1, each sub-conv
           operates on c2//4 channels independently).
        e: Expansion ratio (unused, kept for API compatibility with Bottleneck).
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        assert c2 % 4 == 0, f"MGPC requires c2 divisible by 4, got {c2}"
        c_group = c2 // 4

        # 4 parallel group convolutions with heterogeneous kernels
        # Each takes c1//4 channels (from chunk) and outputs c_group channels
        c_in_group = c1 // 4
        self.gconv1 = nn.Conv2d(c_in_group, c_group, 1, 1, 0, bias=False)
        self.gconv3 = nn.Conv2d(c_in_group, c_group, 3, 1, 1, bias=False)
        self.gconv5 = nn.Conv2d(c_in_group, c_group, 5, 1, 2, bias=False)
        self.gconv7 = nn.Conv2d(c_in_group, c_group, 7, 1, 3, bias=False)

        # Pointwise fusion: 1x1 conv with BN + SiLU (Ultralytics Conv)
        self.pw_conv = Conv(c2, c2, 1, 1)

        # Residual shortcut
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass: split → heterogeneous convs → concat → pointwise → residual."""
        # Split input into 4 equal channel groups
        x1, x2, x3, x4 = x.chunk(4, dim=1)

        # Apply different kernel sizes to each group
        y1 = self.gconv1(x1)
        y2 = self.gconv3(x2)
        y3 = self.gconv5(x3)
        y4 = self.gconv7(x4)

        # Concatenate and fuse
        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.pw_conv(y)

        # Residual connection
        return x + y if self.add else y


class C2f_MGPC(nn.Module):
    """C2f module with MGPC bottleneck replacing standard Bottleneck.
    
    Same split-process-concat structure as Ultralytics C2f, but uses
    MGPC blocks instead of standard Bottleneck blocks.
    
    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of MGPC bottleneck blocks (scaled by model depth factor).
        shortcut: Whether to use residual shortcuts in MGPC blocks.
        g: Groups (passed to MGPC).
        e: Expansion ratio for hidden channels (default 0.5).
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 1x1 conv to expand then split
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # final 1x1 fusion

        # Stack of MGPC bottleneck blocks
        self.m = nn.ModuleList(
            MGPC(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass: cv1 → split → sequential MGPC blocks → concat all → cv2."""
        # cv1 produces 2*c channels, split into two halves
        y = list(self.cv1(x).chunk(2, 1))

        # Each MGPC block takes the last output and appends its result
        for m in self.m:
            y.append(m(y[-1]))

        # Concatenate all chunks and fuse
        return self.cv2(torch.cat(y, 1))
