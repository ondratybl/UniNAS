import torch.nn as nn
from .base import BaseNode


class GELU(BaseNode):  # simple
    def __init__(self, shape: tuple, root_shape: tuple):
        super().__init__(shape, root_shape, 3*shape[0]*shape[1]*shape[2], 0)  # GELU ~ sigmoid ~ 3 FLOPs
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x)