from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple
from . import ViT_Config, Perceiver_Config


@dataclass
class Vision_Embedding_Config:
    vision_tower_config: ViT_Config
    perceiver_config: Perceiver_Config
    vision_tower: str = "resnet"
    vocab_size: Optional[int] = 32000
    embed_dim: Optional[int] = 5120
    reduce_dim: bool = False
    vision_out_dim: int = 1024

    # @classmethod
    # def from_dict(cls, data):
    #     return cls(
    #         vision_tower_config=data.get("vision_tower_config"),
    #         perceiver_config=data.get("vision_config"),
    #         vocab_size=data.get("vocab_size"),
    #         embedding_dim=data.get("embedding_dim"),
    #         vis_dim=data.get("vis_dim"),
    #     )
