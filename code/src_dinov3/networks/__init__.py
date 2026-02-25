# DINOv3 Networks
# Uses official DINOv3 package from Meta

# UperNet version (legacy)
from .dinov3_upernet_model import (
    build_dinov3_upernet, 
    DinoV3UperNet, 
    load_checkpoint as load_checkpoint_upernet,
    DINOV3_CONFIGS,
    # Aliases for backward compatibility
    DinoV3UNet,
    build_dinov3_unet,
)

# DPT version (recommended for dense prediction)
from .dinov3_dpt_model import (
    build_dinov3_dpt,
    DinoV3DPT,
    DinoV3Encoder,
    DPTDecoder,
    load_checkpoint as load_checkpoint_dpt,
)

# Default: use DPT version
build_model = build_dinov3_dpt
load_checkpoint = load_checkpoint_dpt
