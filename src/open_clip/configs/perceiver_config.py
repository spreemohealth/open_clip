from dataclasses import dataclass


@dataclass
class Perceiver_Config:
    num_latents: int = 32
    max_num_media: int = None
    max_num_frames: int = None
    depth: int = 6
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    dim: int = 128
