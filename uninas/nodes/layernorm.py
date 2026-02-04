from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNode


class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm for NCHW (2D spatial) tensors.

    Normalizes over the channel dimension.
    Equivalent in behavior to timm.norm.LayerNorm2d.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__(
            num_channels,
            eps=eps,
            elementwise_affine=affine,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)

        x = F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )

        # NHWC -> NCHW
        return x.permute(0, 3, 1, 2)


class LayerNorm(BaseNode):
    """
    LayerNorm node operating on 2D feature maps.
    """

    def __init__(self, shape: Tuple[int, ...]):
        # FLOPs: mean + variance per element (approx)
        flops = 2 * shape[0] * shape[1] * shape[2]
        num_params = 2 * shape[0]

        super().__init__(
            shape=shape,
            flops=flops,
            num_params=num_params,
        )

        self.ln = LayerNorm2d(shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)
