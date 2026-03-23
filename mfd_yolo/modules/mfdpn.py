"""
MFDPN: Multi-scale Feature Diffusion Pyramid Network.

Contains three components:
  - ADown_MFD: Detail-preserving downsampling (from YOLOv9).
  - MFF: Multi-scale Feature Focus — aggregates P3/P4/P5 features with
    multi-kernel depthwise convolutions.
  - MFDPN: Two-pass bidirectional pyramid network using MFF for
    aggregation and diffusion across detection scales.

Reference: Paper Fig 1, Fig 3, pages 7-8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f
from mfd_yolo.modules.mgpc import C2f_MGPC


class ADown_MFD(nn.Module):
    """Adaptive Downsampling module (YOLOv9-style).
    
    Preserves more detail than standard strided convolution by combining
    average pooling and max pooling paths with learned convolutions.
    Halves spatial dimensions.
    
    Args:
        c1: Input channels.
        c2: Output channels.
    """

    def __init__(self, c1, c2):
        super().__init__()
        c_mid = c2 // 2
        self.cv1 = Conv(c1 // 2, c_mid, 3, 2, 1)  # avg-pool path
        self.cv2 = Conv(c1 // 2, c_mid, 1, 1, 0)   # max-pool path

    def forward(self, x):
        """Forward: avg_pool → split → conv paths → concat."""
        # Average pool to smooth, then split channels
        x = F.avg_pool2d(x, 2, 1, 0, count_include_pad=False)
        x1, x2 = x.chunk(2, dim=1)

        # Path 1: strided conv for learned downsampling
        x1 = self.cv1(x1)

        # Path 2: max pool for detail preservation + 1x1 conv
        x2 = F.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)

        return torch.cat([x1, x2], dim=1)


class MFF(nn.Module):
    """Multi-scale Feature Focus module.
    
    Aggregates features from P3, P4, P5 at P4 resolution, applies
    multi-kernel depthwise convolutions (5,7,9,11) + identity, fuses
    via element-wise addition and pointwise convolution.
    
    Args:
        c3: P3 input channels.
        c4: P4 input channels.
        c5: P5 input channels.
        c_out: Output channels (default: c4 // 4).
    """

    def __init__(self, c3, c4, c5, c_out=None):
        super().__init__()
        if c_out is None:
            c_out = c4 // 4
        self.c_out = c_out
        c_agg = 3 * c_out  # channels after concat of 3 aligned features

        # Scale alignment: bring P3, P4, P5 to same resolution (P4) and channels
        self.p3_down = ADown_MFD(c3, c_out)          # 80→40, c3→c_out
        self.p4_conv = Conv(c4, c_out, 1, 1)          # channel adjust only
        self.p5_conv = Conv(c5, c_out, 1, 1)          # channel adjust (upsample in forward)

        # Multi-kernel depthwise convolutions on aggregated features
        self.dw5 = nn.Conv2d(c_agg, c_agg, 5, 1, 2, groups=c_agg, bias=False)
        self.dw7 = nn.Conv2d(c_agg, c_agg, 7, 1, 3, groups=c_agg, bias=False)
        self.dw9 = nn.Conv2d(c_agg, c_agg, 9, 1, 4, groups=c_agg, bias=False)
        self.dw11 = nn.Conv2d(c_agg, c_agg, 11, 1, 5, groups=c_agg, bias=False)

        # Pointwise fusion: c_agg → c_out
        self.fuse_conv = Conv(c_agg, c_out, 1, 1)

    def forward(self, p3, p4, p5):
        """Forward: align scales → concat → DWConvs + identity → add → PWConv."""
        p3_d = self.p3_down(p3)
        p4_a = self.p4_conv(p4)
        p5_u = self.p5_conv(
            F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        )

        # Aggregate via concatenation
        f_agg = torch.cat([p3_d, p4_a, p5_u], dim=1)

        # Multi-kernel DW convs + identity
        f_identity = f_agg
        f5 = self.dw5(f_agg)
        f7 = self.dw7(f_agg)
        f9 = self.dw9(f_agg)
        f11 = self.dw11(f_agg)

        # Fuse all branches via element-wise addition
        f_multi = f_identity + f5 + f7 + f9 + f11

        # Pointwise conv to reduce channels
        f_mff = self.fuse_conv(f_multi)
        return f_mff


class MFDPN(nn.Module):
    """Multi-scale Feature Diffusion Pyramid Network.
    
    Two-pass bidirectional FPN that replaces YOLOv8's PAN+FPN neck.
    
    Args:
        c3: P3 input channels (e.g. 64 for YOLOv8n).
        c4: P4 input channels (e.g. 128 for YOLOv8n).
        c5: P5 input channels (e.g. 256 for YOLOv8n).
    """

    def __init__(self, c3, c4, c5):
        super().__init__()

        c_mff = c4 // 4   # MFF output channels (32 for YOLOv8n)
        c_out = c4         # Final output channels per scale (128 for YOLOv8n)

        # Store output channels for DTADH to read
        self.output_channels = [c_out, c_out, c5]

        # === Pass 1: Aggregate + Diffuse ===
        self.mff1 = MFF(c3, c4, c5, c_mff)

        # Diffuse to P5: MFF→Conv(3,2,1)→Concat(P5)→C2f_MGPC
        self.pass1_down_conv = Conv(c_mff, c_mff * 2, 3, 2, 1)
        self.pass1_c2f_p5 = C2f_MGPC(c_mff * 2 + c5, c_out, n=1, shortcut=False)

        # Diffuse to P3: MFF→Upsample→Concat(P3)→C2f
        c_p3_pass1 = c_mff + c3  # 32 + 64 = 96
        self.pass1_c2f_p3 = C2f(c_p3_pass1, c_p3_pass1, n=1, shortcut=False)

        # === Pass 2: Re-aggregate + Diffuse ===
        self.mff2 = MFF(c_p3_pass1, c_mff, c_out, c_mff)

        # P4 output: Conv(1,1) to expand MFF output to c_out
        self.pass2_p4_conv = Conv(c_mff, c_out, 1, 1)

        # P5 output: Conv(3,2,1)→Concat(P5')→C2f_MGPC
        self.pass2_down_conv = Conv(c_mff, c_mff * 2, 3, 2, 1)
        self.pass2_c2f_p5 = C2f_MGPC(c_mff * 2 + c_out, c5, n=1, shortcut=False)

        # P3 output: Upsample→Concat(P3')→C2f_MGPC
        c_p3_pass2 = c_mff + c_p3_pass1
        self.pass2_c2f_p3 = C2f_MGPC(c_p3_pass2, c_out, n=1, shortcut=False)

    def forward(self, inputs):
        """Forward pass.
        
        Args:
            inputs: List of [P3, P4, P5] feature maps from backbone.
            
        Returns:
            List of [N3, N4, N5] feature maps for detection heads.
        """
        p3, p4, p5 = inputs

        # === Pass 1: Aggregate + Diffuse ===
        f_mff = self.mff1(p3, p4, p5)

        # Diffuse to P5 scale
        f_down = self.pass1_down_conv(f_mff)
        f_p5_cat = torch.cat([f_down, p5], dim=1)
        p5_pass1 = self.pass1_c2f_p5(f_p5_cat)

        # Diffuse to P3 scale
        f_up = F.interpolate(f_mff, size=p3.shape[2:], mode='nearest')
        f_p3_cat = torch.cat([f_up, p3], dim=1)
        p3_pass1 = self.pass1_c2f_p3(f_p3_cat)

        # === Pass 2: Re-aggregate + Diffuse ===
        f_mff2 = self.mff2(p3_pass1, f_mff, p5_pass1)

        # N4: Direct expansion
        n4 = self.pass2_p4_conv(f_mff2)

        # N5: Diffuse down
        f_down2 = self.pass2_down_conv(f_mff2)
        f_p5_cat2 = torch.cat([f_down2, p5_pass1], dim=1)
        n5 = self.pass2_c2f_p5(f_p5_cat2)

        # N3: Diffuse up
        f_up2 = F.interpolate(f_mff2, size=p3.shape[2:], mode='nearest')
        f_p3_cat2 = torch.cat([f_up2, p3_pass1], dim=1)
        n3 = self.pass2_c2f_p3(f_p3_cat2)

        return [n3, n4, n5]
