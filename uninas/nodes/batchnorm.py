import torch.nn as nn
from .base import BaseNode


class BatchNorm(BaseNode):  # simple
    def __init__(self, shape: tuple):
        super().__init__(shape, 2*shape[0]*shape[1]*shape[2], 2*shape[0])
        self.bn = nn.BatchNorm2d(shape[0])

    def forward(self, x):
        return self.bn(x)