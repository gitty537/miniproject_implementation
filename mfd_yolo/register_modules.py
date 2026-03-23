"""
Register all custom MFD-YOLO modules with Ultralytics.

Must be imported before creating a YOLO model from the custom YAML.
Patches the Ultralytics parse_model function so it correctly handles:
  - C2f_MGPC: treated like C2f (channel scaling + repeat injection)
  - MFDPN: multi-input module, builds with (c3,c4,c5), stores output_channels
  - DTADH: Detect-like head, reads MFDPN.output_channels when f points to MFDPN
"""

from mfd_yolo.modules.mgpc import MGPC, C2f_MGPC
from mfd_yolo.modules.mfdpn import ADown_MFD, MFF, MFDPN
from mfd_yolo.modules.dtadh import DTADH

# All custom module classes
_CUSTOM_MODULES = {
    'MGPC': MGPC,
    'C2f_MGPC': C2f_MGPC,
    'ADown_MFD': ADown_MFD,
    'MFF': MFF,
    'MFDPN': MFDPN,
    'DTADH': DTADH,
}


def register_all():
    """Register all custom modules with Ultralytics' module namespace."""
    import ultralytics.nn.modules as ult_modules
    import ultralytics.nn.tasks as tasks

    # Inject into the modules namespace
    for name, cls in _CUSTOM_MODULES.items():
        setattr(ult_modules, name, cls)

    # Patch parse_model
    tasks.parse_model = _make_patched_parse_model(tasks.parse_model)


def _make_patched_parse_model(original_parse_model):
    """Create a patched parse_model that handles MFD-YOLO custom modules."""

    def patched_parse_model(d, ch, verbose=True):
        """Extended parse_model that handles MFD-YOLO custom modules.
        
        Key differences from original:
          - C2f_MGPC gets C2f-like treatment (width scaling, repeat injection)
          - MFDPN gets multi-input handling, builds with (c3,c4,c5)
          - DTADH reads MFDPN.output_channels for its channel list
        """
        import ast
        import contextlib
        import torch
        import torch.nn as nn

        from ultralytics.nn.modules.conv import Conv, Concat
        from ultralytics.nn.modules.block import (
            C2f, Bottleneck, BottleneckCSP, SPP, SPPF,
            C1, C2, C3, C3TR, C3Ghost, C3x, RepC3,
        )
        from ultralytics.nn.modules.conv import (
            ConvTranspose, GhostConv, DWConv, DWConvTranspose2d, Focus,
        )
        from ultralytics.nn.modules.head import Detect, Segment, Pose, OBB
        from ultralytics.utils.torch_utils import make_divisible
        from ultralytics.utils import LOGGER, colorstr

        # Try importing optional modules (may not exist in all versions)
        _optional = {}
        for name in ['RepNCSPELAN4', 'ELAN1', 'AConv', 'SPPELAN', 'C2fAttn',
                      'PSA', 'SCDown', 'C2fCIB', 'GhostBottleneck']:
            try:
                _optional[name] = getattr(__import__('ultralytics.nn.modules.block', fromlist=[name]), name)
            except (AttributeError, ImportError):
                pass
        for name in ['WorldDetect', 'v10Detect']:
            try:
                _optional[name] = getattr(__import__('ultralytics.nn.modules.head', fromlist=[name]), name)
            except (AttributeError, ImportError):
                pass
        try:
            _optional['Classify'] = getattr(__import__('ultralytics.nn.modules.conv', fromlist=['Classify']), 'Classify')
        except (AttributeError, ImportError):
            pass
        try:
            from ultralytics.nn.modules.block import ADown as UltADown
            _optional['UltADown'] = UltADown
        except ImportError:
            pass

        # Build sets for module category matching
        std_conv_set = {
            Conv, ConvTranspose, GhostConv, Bottleneck,
            SPP, SPPF, DWConv, Focus, BottleneckCSP,
            C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3,
        }
        for name in ['GhostBottleneck', 'RepNCSPELAN4', 'ELAN1', 'AConv', 'SPPELAN',
                      'C2fAttn', 'PSA', 'SCDown', 'C2fCIB', 'Classify', 'UltADown']:
            if name in _optional:
                std_conv_set.add(_optional[name])

        c2f_like_set = {BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3}
        for name in ['C2fAttn', 'C2fCIB']:
            if name in _optional:
                c2f_like_set.add(_optional[name])

        detect_like_set = {Detect, Segment, Pose, OBB}
        for name in ['WorldDetect', 'v10Detect']:
            if name in _optional:
                detect_like_set.add(_optional[name])

        # --- Parse args ---
        max_channels = float("inf")
        nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
        depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
        if scales:
            scale = d.get("scale")
            if not scale:
                scale = tuple(scales.keys())[0]
                LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            Conv.default_act = eval(act)
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")

        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

        ch = [ch]
        layers, save, c2 = [], [], ch[-1]

        for i, (f, n, m_name, args) in enumerate(d["backbone"] + d["head"]):
            # --- Resolve module class ---
            if "nn." in m_name:
                m = getattr(nn, m_name[3:])
            elif m_name in _CUSTOM_MODULES:
                m = _CUSTOM_MODULES[m_name]
            else:
                # Search Ultralytics modules
                m = None
                for search_set in [std_conv_set, detect_like_set, {Concat}]:
                    for cls in search_set:
                        if cls.__name__ == m_name:
                            m = cls
                            break
                    if m is not None:
                        break
                if m is None:
                    # Try ultralytics module namespaces
                    import ultralytics.nn.modules as _um
                    m = getattr(_um, m_name, None)
                if m is None:
                    raise ValueError(f"Module '{m_name}' not found")

            # --- Parse string args ---
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

            n = n_ = max(round(n * depth), 1) if n > 1 else n

            # === Custom module handling ===
            if m is C2f_MGPC:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                args = [c1, c2, *args[1:]]
                args.insert(2, n)  # inject repeat count
                n = 1

            elif m is MFDPN:
                # Multi-input: f is a list like [4, 6, 9]
                ch_list = [ch[x] for x in f]
                c3_in, c4_in, c5_in = ch_list
                args = [c3_in, c4_in, c5_in]
                # c2 for channel tracking: report c4 (uniform output size)
                c2 = c4_in  # 128 for YOLOv8n

            elif m is DTADH:
                # f is a scalar pointing to MFDPN layer
                if isinstance(f, int):
                    mfdpn_module = layers[f]
                    if hasattr(mfdpn_module, 'output_channels'):
                        ch_list = mfdpn_module.output_channels
                    else:
                        ch_list = [ch[f]] * 3
                else:
                    ch_list = [ch[x] for x in f]
                args = [nc, ch_list]
                c2 = None  # no output to track

            elif m is ADown_MFD:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                args = [c1, c2]

            # === Standard Ultralytics modules ===
            elif m in std_conv_set:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                if m is _optional.get('C2fAttn'):
                    args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                    args[2] = int(
                        max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                    )
                args = [c1, c2, *args[1:]]
                if m in c2f_like_set:
                    args.insert(2, n)
                    n = 1

            elif m is Concat:
                c2 = sum(ch[x] for x in f)

            elif m in detect_like_set:
                args.append([ch[x] for x in f])

            elif m is nn.BatchNorm2d:
                args = [ch[f]]

            elif m is nn.Upsample:
                c2 = ch[f]

            else:
                c2 = ch[f]

            # --- Build module instance ---
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")
            m.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2 if c2 is not None else 0)

        return nn.Sequential(*layers), sorted(save)

    return patched_parse_model
