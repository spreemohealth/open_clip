import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import random
from transformers import AutoTokenizer, AutoModel

from .perceiver_vision_helpers import PerceiverResampler

from open_clip.configs import ViT_Config, Perceiver_Config, Vision_Embedding_Config

# from .build_towers import _build_vision_tower

# def build_vision_tower(vision_tower_cfg, **kwargs):

#     vision_tower = getattr(
#         vision_tower_cfg,
#         "vision_tower",
#         "ViT_3D",
#     )

#     if vision_tower.startswith("microsoft/BiomedCLIP"):
#         model, preprocessor = open_clip.create_model_from_pretrained(
#             "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
#         )
#         model = model.visual.trunk

#         return model

#     elif "timm" == vision_tower.split("/")[0]:
#         timm_name = timm_name.split("/")[1]

#         model = timm.create_model(timm_name, pretrained=True)

#         return model

#     elif vision_tower.startswith("ViT"):

#         return ViT(vision_tower_cfg)

#     raise ValueError(f"Unknown vision tower: {vision_tower}")


class VisionEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()

        perceiver_config = config.perceiver_config

        self.vision_tower = config.vision_tower

        self.vision_encoder = config.vision_model

        if eval(config.reduce_dim) is True:
            self.reduce = nn.Sequential(
                nn.Linear(config.vision_out_dim, perceiver_config.dim),
                # nn.ReLU()
            )
        else:
            self.reduce = nn.Identity()

        self.perceiver = PerceiverResampler(perceiver_config)
        # self.perceiver = PerceiverResampler(dim=self.vis_dim, num_latents=perceiver_num)
        self.fc = nn.Linear(perceiver_config.dim, config.embed_dim)

    def preprocess_for_vision_backbone(self, vision_x):

        if len(vision_x.shape) == 5:
            B_S, C, H, W, D = vision_x.shape
        else:
            B_S, C, H, W = vision_x.shape

        if self.vision_tower == "ViT":
            vision_x = vision_x

        elif (
            self.vision_tower
            == "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        ):
            vision_x = vision_x.squeeze(-1).repeat(1, 3, 1, 1)

        return vision_x

    def postprocess_vision_output(self, vision_x, orig_shape):

        if len(orig_shape) == 6:
            B, S, C, H, W, D = orig_shape
        else:
            B, S, C, H, W = orig_shape

        # print("vision_tower: ", self.vision_tower)
        if "ModifiedResNet" in self.vision_tower:
            B_S, D, R, C = vision_x.shape
            vision_x = rearrange(
                vision_x, "bs d r c -> bs (r c) d", bs=B_S, d=D, r=R, c=C
            )
        elif "open_clip.transformer.VisionTransformer" in self.vision_tower:
            if type(vision_x) == tuple:
                _, vision_x = vision_x
        #     B_S, D = vision_x.shape

        vision_x = rearrange(vision_x, "(b s F) v d -> b s F v d", b=B, s=S, F=1)

        return vision_x

    def forward_vision(self, vision_x, orig_shape):

        if (
            "timm" in self.vision_tower
            or self.vision_tower
            == "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        ):
            vision_x = self.vision_encoder.forward_features(vision_x)
        else:
            vision_x = self.vision_encoder(vision_x)

        return vision_x

    def forward(self, vision_x, series_mask):

        # print("input:", vision_x.shape, vision_x.dtype)

        if len(vision_x.shape) == 6:
            B, S, C, H, W, D = vision_x.shape
        else:
            B, S, C, H, W = vision_x.shape

        # print("vision_x shape: ", vision_x.shape)
        orig_shape = vision_x.shape

        vision_x = rearrange(vision_x, "b S ... -> (b S) ...")

        vision_x = self.preprocess_for_vision_backbone(vision_x)
        # print("after preprocessing: ", vision_x.shape)
        vision_x = self.forward_vision(vision_x, orig_shape)
        # print("after vision encoder: ", vision_x.shape)
        vision_x = self.postprocess_vision_output(vision_x, orig_shape)
        # print("after postprocessing: ", vision_x.shape)

        vision_x = self.reduce(vision_x)
        # print('*' * 100)
        # print('vision_x: ', type(vision_x))
        # print(vision_x.dtype, vision_x.shape)
        # print('*' * 100)
        # print('vision_x parameters:')
        # for n, p in self.vision_encoder.named_parameters():
        #     print(n, p.dtype)

        # print('*' * 100)
        # vision_x = vision_x.to(torch.bfloat16)
        # print("after preprocessing: ", vision_x.shape)
        # vision_x = self.vision_encoder(vision_x, output_tokens=True)
        # print("after vision encoder: ", vision_x.shape)
        # vision_x = Variable(vision_x,requires_grad=True)
        # vision_x, _ = checkpoint(self.vision_encoder,vision_x)

        # vision_x = self.postprocess_vision_output(vision_x, orig_shape)

        # print('series_mask outside percever: ', series_mask.shape)
        vision_x = self.perceiver(
            vision_x, series_mask
        ).squeeze()  # reshapes to (b, S, n, d)
        # vision_x = checkpoint(self.perceiver,vision_x)
        # print("after perceiver: ", vision_x.shape)

        vision_x = self.fc(vision_x)

        # if vision_x.shape[1] == 1:
        #     n = vision_x.shape[2]
        # else:
        #     n = S * vision_x.shape[2]

        # # print('n: ',n)
        # vision_x = rearrange(vision_x, "b s n d -> (b s n) d")
        # # print('first rearrange: ', vision_x.shape)
        # vision_x = self.fc(vision_x)
        # # print('vision_x: ', vision_x.shape)
        # vision_x = rearrange(vision_x, "(b T) d -> b T d", b=B, T=n)

        # print('after applying fc and rearranging: ', vision_x.shape)

        # print(
        #     "before final lookup(text input, embedding_wt): ",
        #     text_input.shape,
        #     embedding_weight.shape,
        # )

        return vision_x


if __name__ == "__main__":

    vision_tower = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # vision_tower = "ViT"

    # 2D config
    vision_tower_config = ViT_Config(
        vision_tower=vision_tower,
        channels=1,
        image_patch_size=16,
        frame_patch_size=1,
    )

    # 3D config
    # vision_tower_config = ViT_Config(
    #     vision_tower=vision_tower,
    #     channels=1,
    #     image_patch_size=16,
    #     frame_patch_size=4,
    # )

    print(ViT_Config.image_size)

    perceiver_config = Perceiver_Config(num_latents=128, max_num_media=2)
    print(perceiver_config.num_latents)

    vision_config = Vision_Embedding_Config(
        vocab_size=32000,
        embedding_dim=2048,
        vision_tower_config=vision_tower_config,
        perceiver_config=perceiver_config,
    )

    model = VisionEmbedding(vision_config)

    # p_np = 0
    # v_np = 0
    # total = 0
    # for n,p in model.named_parameters():
    #     total += n.numel()
    #     if "perceiver" in n:
    #         p_np += n.numel()
    #     elif "vision_encoder" in n:
    #         v_np += n.numel()

    # print('vision encoder: ', v_np)
    # print('perceiver: ', p_np)
    # print('total: ', total)

    print("*" * 100)
    print(model)
    print("*" * 100)
    model = model.cuda()
    text_input = torch.randint(low=0, high=3210, size=(4, 1024))
    # image_input = torch.randn((4, 3, 3, 512, 512, 4))

    if vision_tower.startswith("ViT"):
        image_input = torch.randn((4, 2, 1, 256, 256, 1))
    elif "timm" in vision_tower:
        input_size = int(vision_tower.split("_")[-1])
        image_input = torch.randn((4, 2, 1, input_size, input_size, 1))
    else:
        # image_input = torch.randn((4, 2, 1, 256, 256, 1))
        image_input = torch.randn((4, 2, 3, 224, 224))

    print("text_input: ", text_input.shape, text_input.dtype)
    print("image_input: ", image_input.shape, image_input.dtype)

    mask = torch.where(torch.rand(4, 2) > 0.5, torch.ones(4, 2), torch.zeros(4, 2))

    print("mask: ", mask.shape, mask.dtype, mask.sum(dim=-1))

    out = model(text_input.cuda(), image_input.cuda(), mask.cuda())
    print(out.dtype, out.shape)
