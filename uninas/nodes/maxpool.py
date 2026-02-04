import torch.nn as nn
from .base import BaseNode


class MaxPool(BaseNode):  # simple
    def __init__(self, shape: tuple):
        super().__init__(shape, shape[0]*shape[1]*shape[2]*9, 0)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.pool(x)