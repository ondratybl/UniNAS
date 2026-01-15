import torch.nn as nn
from .base import BaseNode


class Dropout(BaseNode):  # simple
    def __init__(self, shape: tuple, root_shape: tuple):
        super().__init__(shape, root_shape, shape[0]*shape[1]*shape[2], 0)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.drop(x)