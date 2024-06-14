from dataclasses import dataclass, field
from typing import Union, List, Optional


@dataclass
class Perceiver_Config:
    dim: int = 128
    num_latents: int = 32
    max_num_media: Union[int, List[int]] = None
    max_num_frames: int = None
    depth: int = 6
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    combine_series_strategy: Optional[str] = None
