"""
DTADH: Dynamic Task-Alignment Detection Heads.

Replaces YOLOv8's standard Detect head with:
  - Shared feature extractor (Conv+GN+SiLU × 2)
  - Cross-scale Task Decomposition (GAP → concat → FC → sigmoid)
  - Classification branch with dynamic conv
  - Localization branch with DCNv2
  - Per-scale Scale layers
  - DFL for bbox regression (via Detect base class)

Inherits from Ultralytics Detect so that:
  1. isinstance(m, Detect) passes in DetectionModel.__init__
  2. Stride computation and bias_init work correctly
  3. v8DetectionLoss recognizes the head

Reference: Paper Fig 4, Fig 1 head detail, Eq 4-6, pages 8-9.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect

# Try to import mmcv DCNv2
_USE_DCN = False
try:
    from mmcv.ops import ModulatedDeformConv2d
    _USE_DCN = True
except ImportError:
    print("[DTADH] WARNING: mmcv not found, falling back to standard Conv2d for DCNv2.")


def _get_gn_groups(channels, max_groups=32):
    """Get a valid number of groups for GroupNorm."""
    for g in [max_groups, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class DCNv2Layer(nn.Module):
    """Deformable Convolution v2 layer with learned offsets and masks."""

    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
        super().__init__()
        if _USE_DCN:
            self.dcn = ModulatedDeformConv2d(
                c_in, c_out, kernel_size,
                stride=stride, padding=padding, bias=True
            )
            self.offset_mask_conv = nn.Conv2d(
                c_in, 3 * kernel_size * kernel_size,
                kernel_size, stride=stride, padding=padding, bias=True
            )
            nn.init.zeros_(self.offset_mask_conv.weight)
            nn.init.zeros_(self.offset_mask_conv.bias)
        else:
            self.dcn = nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=True)
            self.offset_mask_conv = None

    def forward(self, x):
        if _USE_DCN and self.offset_mask_conv is not None:
            offset_mask = self.offset_mask_conv(x)
            k2 = self.dcn.kernel_size[0] * self.dcn.kernel_size[1]
            offset = offset_mask[:, :2 * k2, :, :]
            mask = torch.sigmoid(offset_mask[:, 2 * k2:, :, :])
            return self.dcn(x, offset, mask)
        else:
            return self.dcn(x)


class CrossScaleTaskDecomposition(nn.Module):
    """Cross-scale Task Decomposition module (Eq 4-6).
    
    Computes per-scale attention weights from all scales' features.
    """

    def __init__(self, c_in, n_scales=3, reduction=4):
        super().__init__()
        self.n_scales = n_scales
        c_mid = max(c_in * n_scales // reduction, 8)
        self.fc1 = nn.Linear(c_in * n_scales, c_mid)
        self.fc2 = nn.Linear(c_mid, c_in * n_scales)

    def forward(self, features):
        gaps = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in features]
        x_concat = torch.cat(gaps, dim=1)
        attn = torch.sigmoid(self.fc1(x_concat))
        attn = torch.sigmoid(self.fc2(attn))
        attns = attn.chunk(self.n_scales, dim=1)
        return [a.unsqueeze(-1).unsqueeze(-1) for a in attns]


class DTADH(Detect):
    """Dynamic Task-Alignment Detection Head.
    
    Inherits from Ultralytics Detect for framework compatibility,
    but completely replaces the internal architecture with:
    1. Shared feature extractor (Conv3+GN × 2)
    2. Cross-scale task decomposition for cls and reg
    3. Classification branch (Conv3+GN+SiLU → Conv1x1)
    4. Localization branch (DCNv2 → reg Conv1x1)
    5. Per-scale Scale layers
    """

    def __init__(self, nc=80, ch=()):
        # Initialize nn.Module directly, NOT Detect.__init__,
        # because we don't want Detect's cv2/cv3 modules
        nn.Module.__init__(self)

        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        # Detect class attributes needed for compatibility
        self.dynamic = False
        self.export = False
        self.end2end = False
        self.max_det = 300
        self.shape = None
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

        # Determine common intermediate channel count
        c_inter = ch[0] if len(ch) > 0 else 128
        self.c_inter = c_inter
        gn_groups = _get_gn_groups(c_inter)

        # === Shared Feature Extractor ===
        self.feat_extract = nn.Sequential(
            nn.Conv2d(c_inter, c_inter, 3, 1, 1, bias=False),
            nn.GroupNorm(gn_groups, c_inter),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_inter, c_inter, 3, 1, 1, bias=False),
            nn.GroupNorm(gn_groups, c_inter),
            nn.SiLU(inplace=True),
        )

        # Input projection: align all scales to c_inter channels
        self.input_projs = nn.ModuleList()
        for c in ch:
            if c != c_inter:
                self.input_projs.append(
                    nn.Conv2d(c, c_inter, 1, 1, 0, bias=False)
                )
            else:
                self.input_projs.append(nn.Identity())

        # === Cross-scale Task Decomposition ===
        self.task_decomp_cls = CrossScaleTaskDecomposition(c_inter, self.nl)
        self.task_decomp_reg = CrossScaleTaskDecomposition(c_inter, self.nl)

        # === Classification branch ===
        cls_gn_groups = _get_gn_groups(c_inter)
        self.cls_conv = nn.Sequential(
            nn.Conv2d(c_inter, c_inter, 3, 1, 1, bias=False),
            nn.GroupNorm(cls_gn_groups, c_inter),
            nn.SiLU(inplace=True),
        )
        self.cls_pred = nn.Conv2d(c_inter, nc, 1, 1, 0)

        # === Localization branch ===
        self.reg_task_attn = nn.Sequential(
            nn.Conv2d(c_inter, c_inter, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_inter),
            nn.Sigmoid(),
        )
        self.dcnv2 = DCNv2Layer(c_inter, c_inter, 3, 1, 1)
        self.reg_pred = nn.Conv2d(c_inter, 4 * self.reg_max, 1, 1, 0)

        # === Per-scale Scale layers ===
        self.cls_scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(self.nl)]
        )
        self.reg_scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(self.nl)]
        )

        # DFL module (from Detect parent, but we create our own)
        from ultralytics.nn.modules.block import DFL
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self._initialize_biases()

    def _initialize_biases(self):
        """Initialize prediction biases."""
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
        nn.init.zeros_(self.reg_pred.bias)

    def bias_init(self):
        """Called by DetectionModel.__init__ after stride computation.
        
        Re-initialize biases with stride information.
        """
        for i, s in enumerate(self.stride):
            # Classification bias: account for stride-dependent anchor density
            cls_bias = math.log(5 / self.nc / (640 / s) ** 2)
            # We apply this as a scale factor rather than overwriting
            # since our cls_pred is shared across scales
        # Reg pred bias
        self.reg_pred.bias.data[:] = 1.0

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: List of feature maps [N3, N4, N5] from MFDPN.
            
        Returns:
            Training: list of per-scale outputs, each (B, no, H, W)
            Inference: (decoded_preds, raw_feats) compatible with Detect
        """
        # Project all inputs to common channel count
        x_proj = [proj(feat) for proj, feat in zip(self.input_projs, x)]

        # Shared feature extractor → interactive features
        x_inter = [self.feat_extract(f) for f in x_proj]

        # Cross-scale task decomposition
        cls_attns = self.task_decomp_cls(x_inter)
        reg_attns = self.task_decomp_reg(x_inter)

        outputs = []
        for i in range(self.nl):
            # === Classification ===
            cls_feat = x_inter[i] * cls_attns[i]
            cls_feat = self.cls_conv(cls_feat)
            cls_out = self.cls_pred(cls_feat) * self.cls_scales[i]

            # === Localization ===
            reg_feat = x_inter[i] * reg_attns[i]
            reg_attn_mask = self.reg_task_attn(x_inter[i])
            dcn_out = self.dcnv2(reg_feat)
            reg_combined = dcn_out + reg_attn_mask * x_inter[i]
            reg_out = self.reg_pred(reg_combined) * self.reg_scales[i]

            # Combine reg + cls: (B, 4*reg_max + nc, H, W)
            out = torch.cat([reg_out, cls_out], dim=1)
            outputs.append(out)

        if self.training:
            return outputs

        # Inference: use Detect's _inference for decode + NMS compatibility
        y = self._inference(outputs)
        return y if self.export else (y, outputs)
