from torch import arange, abs
from .base import BaseNode


class Mask(BaseNode):  # simple
    def __init__(self, shape: tuple, root_shape: tuple):
        super().__init__(shape, root_shape, shape[0]*shape[1]*shape[2], 0)
        rows = arange(shape[1]).view(-1, 1)
        cols = arange(shape[2]).view(1, -1)
        mask = (abs(rows - cols) <= 5).float()  # (H, W)

        self.register_buffer("mask", mask)  # non-trainable buffer

    def forward(self, x):
        return x * self.mask.view(1, 1, *self.mask.shape)