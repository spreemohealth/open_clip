from collections.abc import Sequence

from functools import partial
from typing import List, Union, Tuple, Optional

import torch
import torch.nn as nn
from cvtoolsmodelrepo.misc_blocks import (
    Bottleneck,
    downsample_basic_block,
)
from torch.utils import checkpoint
import torch.nn as nn
import torch.nn.functional as F


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    # if keep_prob > 0.0 and scale_by_keep:
    #     random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Bottleneck(nn.Module):
    """A bottleneck block, with residual connections. Conv3d->bn->Conv3d->bn->Conv3d->bn->(+Res)->ReLU. Downsample nn.Module is optional.
    Despite changes, version is compatible with Jin's workflow.
        Args:
            inplanes (int): [description]
            planes (int): [description]
            num_expansion (int): [description]
            stride (int, optional): [description]. Defaults to 1.
            downsample (nn.Module, optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        num_expansion: int,
        stride=1,
        downsample: nn.Module = None,
        drop_path=0.0,
    ):

        super(Bottleneck, self).__init__()
        self.num_expansion = num_expansion
        self.block = nn.Sequential(
            *[
                nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
                nn.BatchNorm3d(planes),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
                ),
                nn.BatchNorm3d(planes),
                nn.ReLU(inplace=True),
                nn.Conv3d(planes, planes * num_expansion, kernel_size=1, bias=False),
                nn.BatchNorm3d(planes * num_expansion),
            ]
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # import pdb;pdb.set_trace()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def get_num_expansion(self):
        return self.num_expansion

    def forward(self, x):
        residual = x
        out = self.drop_path(self.block(x))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        num_vol_channel: Optional[int] = 1,
        bShare_encoder: Optional[bool] = True,
        num_expansion: Optional[int] = 1,
        shortcut_type: Optional[str] = "B",
        num_classes_list: Optional[List[int]] = [3, 2, 3],
        add_ReLU_in_fc: Optional[bool] = False,
        init_num_filters: Optional[int] = 64,
        outputDim: int = None,
        enc_type="isoVox_opt1",  # 'isoVox_opt1' or 'Regular',
        zero_init_residual: bool = True,
        drop_path=0,
        layer_4_stride=2,
    ):
        self.num_expansion = num_expansion
        self.bShare_encoder = bShare_encoder
        self.drop_path = drop_path
        super(ResNet, self).__init__()

        self.init_num_filters = init_num_filters
        inplanes = init_num_filters
        self.inplanes = inplanes
        if enc_type == "Regular":
            first_kernel = (3, 7, 7)
            first_pad = (1, 3, 3)
            layer_3_stride = (2, 2, 2)
        # Designed to begin out-plane operations when voxel is isometric, assuming
        # 3.6mm z-slice and 0.3mm in plane.
        elif enc_type == "isoVox_opt1":
            first_kernel = (1, 7, 7)
            first_pad = (0, 3, 3)
            layer_3_stride = 2
        else:
            raise Exception(
                f"Unknown 'enc_type' argument for encoder: {enc_type}. Allowed are: 'isoVox_opt1', 'Regular'. "
            )

        self.conv1 = nn.Conv3d(
            num_vol_channel,
            inplanes,
            kernel_size=first_kernel,
            stride=(1, 2, 2),
            padding=first_pad,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.layer1 = self._make_layer(
            block, inplanes, layers[0], shortcut_type, stride=1
        )
        self.layer2 = self._make_layer(
            block, inplanes * 2, layers[1], shortcut_type, stride=(2, 2, 2)
        )
        self.layer3 = self._make_layer(
            block, inplanes * 2 * 2, layers[2], shortcut_type, stride=(2, 2, 2)
        )
        self.layer4 = self._make_layer(
            block, inplanes * 2 * 2 * 2, layers[3], shortcut_type, stride=(2, 2, 2)
        )  # MHK: was stride=2. I wanted (2,2,2)
        self.out_features = inplanes * 2 * 2 * 2 * num_expansion
        self.out_size = self.out_features
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):

                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to
            # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.block[7].weight is not None:
                    nn.init.constant_(m.block[7].weight, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        """Constructs a layer with specific numbers of channels. Called by '_make_encoder'.

        Parameters:

            block (nn.Module): Basic or Bottleneck, see Resnet3d.
            planes (int): The number of planes in the layer. It is either 64, 128, 256 or 512 (multiplied by 'self.num_expansion', an int. )
            blocks (int): number of repeating blocks in this layer. On upper levels, it is stored in list 'layers'.
            shortcut_type (str): Not Sure.
            stride (int, tuple of int): you know.

        Undeclared Parameters:

            self.inplanes (int): Input planes. Meaning, the number of planes in the previous layer. Is updated here as well, so it will be read and matched to the next layer.
            self.num_expansion (int): a factor which multiplies the number of planes outputted for the layer.
        Returns:
            int:Returning value
        """

        downsample = None
        if stride != (1, 1, 1) or self.inplanes != planes * self.num_expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * self.num_expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * self.num_expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * self.num_expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                self.num_expansion,
                stride,
                downsample,
                drop_path=self.drop_path,
            )
        )
        self.inplanes = planes * self.num_expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, self.num_expansion, drop_path=self.drop_path
                )
            )

        return nn.Sequential(*layers)

    @property
    def outplanes(self):
        return self.inplanes

    def _forward_first_few_layers(self, x):
        x = self.conv1(x)
        print("conv1: ", x.shape)
        # output size 32x128x128
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print("after maxpool: ", x.shape)
        # x = checkpoint.checkpoint(self.maxpool, x)
        return x

    def forward(self, x, has_sagfs_sagpd=False):
        # direct     link
        # input size 32x256x256 ([nBs, 1, nSls, nRows, nCols])
        if not has_sagfs_sagpd:
            x = self._forward_first_few_layers(x)
        # output size 32x64x64
        print("first few layers: ", x.shape)

        l1 = self.layer1(x)
        print("layer 1: ", l1.shape)
        l2 = self.layer2(l1)
        print("layer 2: ", l2.shape)
        # output size 16x32x32
        l3 = self.layer3(l2)
        print("layer 3: ", l3.shape)
        # output size 8x16x16
        l4 = self.layer4(l3)
        print("layer 4: ", l4.shape)

        # l1 = checkpoint.checkpoint(self.layer1, x)
        # l2 = checkpoint.checkpoint(self.layer2, l1)
        # # output size 16x32x32
        # l3 = checkpoint.checkpoint(self.layer3, l2)
        # # output size 8x16x16
        # l4 = checkpoint.checkpoint(self.layer4, l3)
        return l4


def resnet18(block: Optional[nn.Module] = Bottleneck, **kwargs):
    """Constructs a ResNet-50 model, using class 'ResNet_MTL_DropOut'. see
    'ResNet_MTL_DropOut'.

    Args:

        num_vol_channel (int): number of different MRI series to run in parallel encoders. "groups"
        bShare_encoder (bool): Use parallel encoder.
        num_expansion (int): a number to multiply all number of planes (Channels) throughout all layers.
        shortcut_type (str):    'A': ?
                                'B': ? Is deafult, Jin uses it.
        num_classes_list (list): list of {num_vol_channel} integers. One int per pathology. The int tells how many classes (severities) the pathology had.

    Returns:

        Return a nn.torch
    """
    model = ResNet(block, [2, 2, 2, 2], **kwargs)
    return model


def resnet50(block: Optional[nn.Module] = Bottleneck, **kwargs):
    """Constructs a ResNet-50 model, using class 'ResNet_MTL_DropOut'. see
    'ResNet_MTL_DropOut'.

    Args:

        num_vol_channel (int): number of different MRI series to run in parallel encoders. "groups"
        bShare_encoder (bool): Use parallel encoder.
        num_expansion (int): a number to multiply all number of planes (Channels) throughout all layers.
        shortcut_type (str):    'A': ?
                                'B': ? Is deafult, Jin uses it.
        num_classes_list (list): list of {num_vol_channel} integers. One int per pathology. The int tells how many classes (severities) the pathology had.

    Returns:

        Return a nn.torch
    """
    model = ResNet(block, [3, 4, 6, 3], **kwargs)
    return model


if __name__ == "__main__":
    import torch

    arr = torch.randn(1, 1, 24, 256, 256)  # .cuda()

    # device = torch.device('cuda') if torch.cuda().is_available() else torch.device('cpy')
    model = resnet50(init_num_filters=64, num_expansion=4)  # .to(device)
    print(model)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # import pdb
    # pdb.set_trace()

    out = model(arr)  # .to(device))
    print(arr.shape)
    print(out.shape)
    # import pdb
    # pdb.set_trace()
