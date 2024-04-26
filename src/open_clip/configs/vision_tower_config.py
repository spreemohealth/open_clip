from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List
import torch.nn as nn


@dataclass
class ViT_Config:
    vision_tower: Optional[str] = "ViT_3D"
    image_size: Optional[int] = 256
    frames: Optional[int] = 512
    image_patch_size: Optional[int] = 32
    frame_patch_size: Optional[int] = 4
    dim: Optional[int] = 768
    depth: Optional[int] = 12
    heads: Optional[int] = 8
    mlp_dim: Optional[int] = 2048
    dropout: Optional[float] = 0.1
    emb_dropout: Optional[float] = 0.1
    channels: Optional[int] = 1
    lora_enable: bool = False


@dataclass
class Resnet_Config:

    block: nn.Module
    layers: List[int] = field(default_factory=[3, 4, 6, 3])  # [2,2,2,2] for resnet 18
    num_vol_channel: Optional[int] = 1
    bShare_encoder: Optional[bool] = True
    num_expansion: Optional[int] = 1
    shortcut_type: Optional[str] = "B"
    num_classes_list: Optional[List[int]] = field(default_factory=[3, 2, 3])
    add_ReLU_in_fc: Optional[bool] = False
    init_num_filters: Optional[int] = 64
    outputDim: int = None
    enc_type = "isoVox_opt1"  # 'isoVox_opt1' or 'Regular',
    zero_init_residual: bool = True
    drop_path = 0
    layer_4_stride = 2
